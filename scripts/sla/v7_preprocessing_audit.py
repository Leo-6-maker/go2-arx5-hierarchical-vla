"""Audit frozen Go2 normalization and real reader/provider preprocessing on Corelab CPU."""
from collections import Counter
import io
import json
from pathlib import Path
import sys
import tarfile
from tempfile import mkdtemp
from types import SimpleNamespace

import numpy as np

from m38_gpu_preflight_v1 import pin, read, sha
from scripts.sla.core_hopper_native_scope import accepted
from scripts.sla.hopper_site import pinned
from go2arx5_vla.sla_adapter.codec_v1 import Codec, Quantiles
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer, observation
from go2arx5_vla.sla_adapter.scale20k_package import Provider
from openpi_client.image_tools import resize_with_pad
from v63_adapt_data import read_added
from v65_recovery_data import dataset as recovery_dataset

INDEX="9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56"

def arrays(bundle, entry):
    with tarfile.open(pin(bundle/entry["path"],entry["archive_sha256"]),"r") as archive:
        with archive.extractfile("labels.npz") as source, np.load(io.BytesIO(source.read()),allow_pickle=False) as values:
            return {name:values[name] for name in values.files}

def summarize(states, action, support):
    normalized=states["normal"]
    bins=np.digitize(normalized,np.linspace(-1,1,257)[:-1])-1
    legal=action[support]
    return dict(windows=len(bins),state_below_bin_range_per_channel=(bins==-1).sum(0).tolist(),
                state_upper_bin_per_channel=(bins==255).sum(0).tolist(),
                state_outside_quantiles_per_channel=(np.abs(normalized)>1).sum(0).tolist(),
                state_normalized_min=normalized.min(0).tolist(),state_normalized_max=normalized.max(0).tolist(),
                action_valid_points=len(legal),action_outside_quantiles_per_channel=(np.abs(legal)>1).sum(0).tolist())

def run(w):
    bundle=Path("/mnt/local/hdd1/UserData/liuyu-nfs/Projects/GR00T-WBC/WM02/slago2-embodiment-a93-r3-hub-20261127-01")
    index=read(pin(bundle/"index.json",INDEX))
    gpu_spec=read(w/"ADAPT0250-NATIVE-PLACE-01/worker/EXECUTION_SPEC.json")
    _,_,receiver=accepted(gpu_spec)
    source=pinned(receiver["input"])["target_D1"]
    stats={"state":pinned(source["statistics"])["state"],"actions":pinned(receiver["action"])["actions"]}
    quant={name:Quantiles(v["q01"],v["q99"]) for name,v in stats.items()}
    tokenizer=Tokenizer(str(pin(receiver["tokenizer"]["path"],receiver["tokenizer"]["sha256"])))
    records=[]; pools={name:[] for name in ("old","new","recovery")}
    for number, ep in enumerate(index["episodes"]):
        labels=arrays(bundle,ep)
        codec=Codec(ep["T_policy_from_mount"])
        states=np.asarray([codec.encode(s,np.tile(s,(40,1)))[0] for s in labels["state_eef10"]])
        normal=quant["state"].normalize(states).astype(np.float32)
        groups="old" if number<52 else "new" if number<93 else "recovery"
        pools[groups].append((normal,states,quant["actions"].normalize(labels["action"]),labels["valid"]))
        bins=np.digitize(normal,np.linspace(-1,1,257)[:-1])-1
        phase={}
        for name in sorted(set(labels["phase"])):
            choose=labels["phase"]==name
            phase[str(name)]=dict(windows=int(choose.sum()),any_xyz_rotation_channel_at_outer_bin=int(np.any((bins[choose,:6]==-1)|(bins[choose,:6]==255),axis=1).sum()),
                                 jaw_lower_saturated=int((bins[choose,6]==-1).sum()))
        records.append(dict(episode=ep["episode"],group=groups,phases=phase))
    norms={}
    for name,parts in pools.items():
        normal,physical,actions,support=(np.concatenate([p[i] for p in parts]) for i in range(4))
        summary=summarize(dict(normal=normal,physical=physical),actions,support)
        bins=np.digitize(normal,np.linspace(-1,1,257)[:-1])-1
        summary["jaw_saturated_ranges_mm"]={}
        for value in (-1,255):
            widths=44*physical[bins[:,6]==value,6]
            summary["jaw_saturated_ranges_mm"][str(value)]=None if not len(widths) else [float(widths.min()),float(widths.max())]
        norms[name]=summary

    output=Path(mkdtemp(prefix="v7-preprocessing-physical.",dir=w))/"ACTUAL.json"
    output.with_name("NORMALIZATION.json").write_text(json.dumps(norms,indent=2,allow_nan=False)+"\n")
    print(output.parent,flush=True)
    print(json.dumps(norms),flush=True)
    parent_root=bundle.parent/"sla-v63-adapt-first-20261127-01"
    roots=[parent_root/"A74-ADDED",parent_root/"A78-COLD-INCREMENT",parent_root/"A83-COLD-INCREMENT",
           w/"v64-reach-train-code-01/A87-COLD-INCREMENT",w/"v64-reach-train-code-03/A87-REACH-INCREMENT"]
    readers=[]
    for path in roots:
        manifest=read(path/"MANIFEST.json")
        assert all(manifest["statistics_sha256"][k]==value for k,value in
                   (("state_statistics",source["statistics"]["sha256"]),("action_statistics",receiver["action"]["sha256"])))
        readers.append(read_added(path,manifest,stats,resize_with_pad))
    rp=w/"V65-R3-COLD-02"
    readers.append(recovery_dataset(rp,read(pin(rp/"MANIFEST.json","362c8756c7b7ea62a539a441dcfa5a6bdf175f4eb92f79088ed7a7ec7a884991")),stats,resize_with_pad))
    checks=[]
    for reader in readers:
        selected={}
        first_episode=reader.rows[0][0]["id"]
        for i,(ep,k) in enumerate(reader.rows):
            if ep["split"]!="TRAIN_RECOVERY" and ep["id"]!=first_episode:
                continue
            phase=ep["expert_stage_audit_only"][k]
            bucket=(ep["id"],str(phase))
            if bucket not in selected:
                selected[bucket]=i
        # One first window per episode and phase, plus every episode tail.
        selected.update({(ep["id"],"TAIL"):i for i,(ep,k) in enumerate(reader.rows)
                         if k==ep["windows"]-1 and (ep["split"]=="TRAIN_RECOVERY" or ep["id"]==first_episode)})
        for (_,phase), i in selected.items():
            ep,k=reader.rows[i]
            item=reader[i]
            if ep["split"]=="TRAIN_RECOVERY":
                shard,offset=reader._frame(ep,k)
                state=reader.labels[ep["id"]]["state_eef10"][k]
                images=shard["rgb"][offset]
                prompt=ep["language"]
            else:
                state,_,images,prompt=reader.raw(i)
            obs,_=observation(item,tokenizer,drop_state=False)
            received={}
            def capture(active,frozen,provided,noise):
                received["observation"]=provided
                return item["actions"][None]
            request=SimpleNamespace(active=None,frozen=None,statistics=quant,tokenizer=tokenizer,resize=resize_with_pad,infer=capture)
            Provider.predict(request,state,images,prompt,ep.get("T_policy_from_mount",ep.get("mount",{}).get("T_policy_from_mount")),noise_id=0,
                             previous_reference={"width_m":.02,"source_native_s":1.,"query_native_s":1.05})
            actual=received["observation"]
            leaves=("state","tokenized_prompt","tokenized_prompt_mask")
            assert all(np.array_equal(np.asarray(getattr(obs,key)),np.asarray(getattr(actual,key))) for key in leaves)
            assert all(np.array_equal(np.asarray(left[key]),np.asarray(right[key]))
                       for left,right in ((obs.images,actual.images),(obs.image_masks,actual.image_masks)) for key in left)
            assert item["prompt"]==prompt
            checks.append(dict(episode=ep["id"],window=int(k),phase=phase,supported=int(reader.support(i).sum()),
                               token_count=int(np.asarray(obs.tokenized_prompt_mask).sum()),state_tokens_equal=True,images_and_masks_equal=True))
    result=dict(public_index_sha256=INDEX,state_statistics_sha256=source["statistics"]["sha256"],
                action_statistics_sha256=receiver["action"]["sha256"],tokenizer_sha256=receiver["tokenizer"]["sha256"],
                normalization=norms,episodes=records,reader_provider_checks=checks,
                reader_parity_scope="A74_A78_A83_A87_A93_AND_RECOVERY3_REAL_READERS_NO_STATE_DROPOUT",
                old_52_reader_parity_measured=False,normalization_clips=False,discrete_state_uses_saturated_bins=True,
                new_model_forwards=0,new_native_integrations=0,new_optimizer_updates=0)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(output,sha(output),flush=True)
    print(json.dumps(dict(normalization=norms,input_parity_windows=len(checks))),flush=True)

if __name__=="__main__":
    run(Path(sys.argv[1]))
