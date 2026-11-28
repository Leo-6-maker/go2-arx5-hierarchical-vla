"""Prepare a data-only C1 contrast after two real early suffixes are independently admitted."""

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write


INDEX = "9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56"
BANK = "a8fde923b6453be823fd8097b8440a7ad8780d6d484bc3d882df8a5b070e64ff"
MATCHED = "ec5e7436b839e8452b61fc137f4fe6f659af2aabb522165c7604413b12d20024"
EARLY = ("PREGRASP", "DESCEND", "CLOSE", "LIFT")
STEPS = 250
DATA_SEED = 529120261


def ref(root, row):
    p = root/row["file"]
    require(p.resolve().is_relative_to(root.resolve()),"Cold dataset member outside its root")
    return pin(p,row["sha256"])


def prepare(package, source_bank, cold_roots, manifest_hashes, reader_hashes, output):
    require(not output.exists() and len(cold_roots) == len(manifest_hashes) == len(reader_hashes) == 2,
            "Require exactly two immutable admitted starts and a fresh preparation directory")
    source_cold = []
    for root,digest,reader_sha in zip(cold_roots,manifest_hashes,reader_hashes,strict=True):
        manifest = read(pin(root/"MANIFEST.json",digest))
        reader = read(pin(root/"READER-ACTUAL.json",reader_sha))
        require(reader["manifest_sha256"] == digest and reader["windows"] == manifest["windows"]
                and reader["current_training_consumption"] == 0 and manifest["independent_early_takeover_starts"] == 1,
                "Only a reader-verified real early suffix is eligible")
        ep, = manifest["episodes"]
        source_cold.append((manifest,reader,ep))
    require(len({ep["id"] for _,_,ep in source_cold}) == len({ep["reset"] for _,_,ep in source_cold}) == 2,
            "Two copies of one admitted reset cannot satisfy C1's independent-start gate")
    index = read(pin(package/"index.json",INDEX))
    source = read(pin(source_bank/"MATCHED_INPUTS.json",MATCHED))
    require(source["data_index_sha256"] == INDEX,"The source bank belongs to a different population")
    with np.load(pin(source_bank/"BANK.npz",BANK),allow_pickle=False) as f:
        original = {k:f[k][:STEPS].copy() for k in ("target","robodojo","libero","auxiliary_slots","rates")}
    require(original["target"].shape == (STEPS,96) and (original["auxiliary_slots"] == [0,1,48,49]).all(),
            "The original common prefix and auxiliary slots changed")
    rows, actions, valid, state, ticks, prior = [], [], [], [], [], []
    episodes = []
    for ep in index["episodes"]:
        with tarfile.open(package/ep["path"],"r") as tar:
            with tar.extractfile("labels.npz") as stream,np.load(io.BytesIO(stream.read()),allow_pickle=False) as f:
                labels = {k:f[k] for k in ("action","valid","state_eef10","native_step","prior_actual_width_m","phase")}
        require(len(labels["native_step"]) == ep["windows"] and dict(Counter(labels["phase"].tolist())) == ep["phases"],
                "Original fixed episode labels changed")
        group = "old" if len(rows)<4450 else "new" if len(rows)<7916 else "R3"
        offset = len(rows)
        for k in range(ep["windows"]):
            rows.append(dict(episode=ep["episode"],anchor=k,phase=str(labels["phase"][k]),task=ep["task"],
                             scene=ep["scene_group_sha256"],group=group))
        episodes.append(dict(id=ep["episode"],offset=offset,windows=ep["windows"],source="A93_R3"))
        for dest,key in ((actions,"action"),(valid,"valid"),(state,"state_eef10"),(ticks,"native_step"),(prior,"prior_actual_width_m")):
            dest.extend(labels[key])
    require(len(rows) == 8031,"Preserve every old query in the exact reader order")
    # The original bank stores its reader identity; use the same repository helper.
    from v61_source_bank import index_identity
    require(index_identity((r["episode"],r["anchor"]) for r in rows) == source["reader_order"]["target"],
            "Public episode order differs from the original training reader")
    starts, treatments, ordinary = [], [], []
    for (root,digest,reader_sha),(_,_,ep) in zip(zip(cold_roots,manifest_hashes,reader_hashes,strict=True),source_cold,strict=True):
        require(ep["split"] == "TRAIN_RECOVERY" and ep["id"] not in {r["episode"] for r in rows},
                "The proposed early start is duplicate or not TRAIN Recovery")
        with np.load(ref(root,ep["labels"]),allow_pickle=False) as f:
            labels = {k:f[k] for k in ("action","valid","state_eef10","native_step","phase")}
        with np.load(ref(root,ep["gripper"]),allow_pickle=False) as f:
            before = f["prior"].copy()
        require(len(labels["native_step"]) == ep["windows"] == len(before)
                and (np.diff(labels["native_step"]) == 250).all(),"The real early query support changed")
        offset = len(rows)
        for k in range(ep["windows"]):
            rows.append(dict(episode=ep["id"],anchor=k,phase=str(labels["phase"][k]),task=ep["task"],
                             scene=ep["scene_sha256"],group="early"))
        eligible = {phase:[offset+i for i,p in enumerate(labels["phase"]) if p == phase and labels["valid"][i].all()]
                    for phase in EARLY}
        require(all(eligible.values()),"An admitted early start lacks PREGRASP/DESCEND/CLOSE/LIFT full-H40 support")
        control = {phase:[i for i,r in enumerate(rows[:8031]) if r["group"] in ("old","new")
                    and r["task"] == ep["task"] and r["scene"] == ep["scene_sha256"] and r["phase"] == phase
                    and np.asarray(valid[i]).all()] for phase in EARLY}
        require(all(control.values()),"No exact scene/task/phase/effective-H ordinary Teacher control")
        starts.append(ep["id"])
        treatments.append(eligible)
        ordinary.append(control)
        episodes.append(dict(id=ep["id"],offset=offset,windows=ep["windows"],source=str(root),manifest_sha256=digest,
                             reader_sha256=reader_sha,native_result_sha256=ep["native_result_sha256"],reset=ep["reset"]))
        for dest,key in ((actions,"action"),(valid,"valid"),(state,"state_eef10"),(ticks,"native_step")):
            dest.extend(labels[key])
        prior.extend(before)
    require(len(set(starts)) == 2 and len({e["reset"] for e in episodes[-2:]}) == 2,
            "Two windows or copies of one actual reset are not two independent starts")
    actions,valid,state,ticks,prior = map(np.asarray,(actions,valid,state,ticks,prior))
    require(valid.shape == (len(rows),40) and valid.dtype == bool and valid[:,:10].all()
            and actions.shape == (len(rows),40,7) and np.isfinite(actions).all()
            and not actions[~valid].any(),"The common reader layout or true support changed")
    groups = np.asarray([r["group"] for r in rows])
    rng = np.random.default_rng(DATA_SEED)
    control_rows,treatment_rows,replacements,contrast = [],[],[],[]
    for step,baseline in enumerate(original["target"]):
        # Keep all original old48, new44 and R3two slots common. Inherited new
        # positions92/93 are removed; they are not auxiliary positions.
        common = np.r_[baseline[:92],baseline[94:96]].copy()
        aux_indices = set(common[original["auxiliary_slots"][step]].tolist())
        phase_choices = [EARLY[(step+k)%len(EARLY)] for k in range(2)]
        c,t = [],[]
        for k,phase in enumerate(phase_choices):
            choices = [i for i in ordinary[k][phase] if i not in aux_indices and i not in c]
            require(choices,"Cannot select a distinct phase-matched contrast without changing auxiliary samples")
            c.append(int(rng.choice(choices)))
            t.append(int(rng.choice(treatments[k][phase])))
        reserved = set(c)
        for pos in np.flatnonzero(np.isin(common,c)):
            old = int(common[pos]);r = rows[old]
            choices = [i for i,z in enumerate(rows[:8031]) if z["group"] == r["group"] and z["task"] == r["task"]
                       and z["phase"] == r["phase"] and bool(valid[i].all()) == bool(valid[old].all())
                       and i not in set(common) and i not in reserved]
            require(choices,"No same-stratum replacement for a common/contrast collision")
            common[pos] = int(rng.choice(choices))
            replacements.append(dict(step=step+1,position=int(pos),before=old,after=int(common[pos])))
        require(len(np.unique(common)) == 94 and not set(common)&reserved,"Common-prefix duplicate or contrast leakage")
        a,b = np.r_[common,c],np.r_[common,t]
        require(len(np.unique(a)) == len(np.unique(b)) == 96 and np.array_equal(a[:94],b[:94])
                and np.array_equal(a[original["auxiliary_slots"][step]],baseline[original["auxiliary_slots"][step]])
                and dict(Counter(groups[common])) == dict(old=48,new=44,R3=2),"C1 batch or shared aux contract changed")
        require(all(rows[x]["phase"] == rows[y]["phase"] and rows[x]["task"] == rows[y]["task"]
                    and rows[x]["scene"] == rows[y]["scene"] and valid[x].all() and valid[y].all()
                    for x,y in zip(c,t,strict=True)),"Contrast phase, task, scene or H40 mismatch")
        control_rows.append(a);treatment_rows.append(b)
        contrast.append(dict(step=step+1,control=c,treatment=t,phase=phase_choices))
    control_rows,treatment_rows = np.asarray(control_rows),np.asarray(treatment_rows)
    output.mkdir(mode=0o700)
    bank = output/"BANK.npz"
    with bank.open("xb") as stream:
        np.savez_compressed(stream,control_target=control_rows,treatment_target=treatment_rows,
            robodojo=original["robodojo"],libero=original["libero"],auxiliary_slots=original["auxiliary_slots"],rates=original["rates"])
    fields = output/"PREVIOUS_COMMAND.npz"
    with fields.open("xb") as stream:
        np.savez_compressed(stream,width_m=prior,query_tick=ticks,consumed_start_tick=ticks-25,
                            command_valid=np.ones(len(rows),dtype=bool),full_h40=valid.all(axis=1))
    result = dict(status="C1_TWO_ACTUAL_EARLY_STARTS_MATCHED_BANK_CPU_PREPARED_NOT_TRAINING_ADMITTED",
        fixed_original_index_sha256=INDEX,original_bank_sha256=BANK,bank_sha256=sha(bank),context_sha256=sha(fields),
        episodes=episodes,actual_early_starts=starts,new_independent_layout_claim=False,common_slots=94,
        contrast_slots=2,steps=STEPS,data_seed=DATA_SEED,FM_batch=dict(old=48,common_new=44,original_R3=2,contrast=2,robodojo=16,libero=16),
        control_actual_source_groups=dict(Counter(groups[control_rows.ravel()])),
        treatment_actual_source_groups=dict(Counter(groups[treatment_rows.ravel()])),
        contrast_phase_visits=dict(Counter(p for row in contrast for p in row["phase"])),contrast=contrast,
        common_collision_replacements=replacements,auxiliary_original_rows_exact=True,
        reader_order_sha256=index_identity((r["episode"],r["anchor"])for r in rows),
        context_current_window_count=len(rows),actual_full_combined_reader_qualified=False,
        source_noise_rule="Original 529202611 model draw rule on the new equal-position paired bank; no old control reuse",
        current_training_curriculum_changed=False,current_training_consumption=0,model_forwards=0,native_integrations=0,optimizer_updates=0)
    write(output/"ACTUAL.json",result)
    print(output/"ACTUAL.json",sha(output/"ACTUAL.json"),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("package","source_bank","first_cold","second_cold","output"):
        parser.add_argument(key,type=Path)
    parser.add_argument("--manifest-hashes",nargs=2,required=True)
    parser.add_argument("--reader-hashes",nargs=2,required=True)
    args = parser.parse_args()
    prepare(args.package,args.source_bank,[args.first_cold,args.second_cold],args.manifest_hashes,args.reader_hashes,args.output)
