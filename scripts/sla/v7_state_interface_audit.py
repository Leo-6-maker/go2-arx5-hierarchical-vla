"""Audit TRAIN-only state range counterfactuals and the real original-52 reader."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
from tempfile import mkdtemp
import tarfile
from types import SimpleNamespace

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write

INDEX='9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56'


def run(site, package):
    from go2arx5_vla.sla_adapter.codec_v1 import Codec, Quantiles, pad32
    from go2arx5_vla.sla_adapter.model_v1 import Tokenizer, observation
    from go2arx5_vla.sla_adapter.scale20k_package import Provider
    from scripts.sla.core_hopper_native_scope import accepted
    from scripts.sla.hopper_site import pinned
    from v3_main_place_learn import data_scope
    from v6_reanchored_teacher_labels import bind_dataset

    _,_,receiver=accepted(read(site/'ADAPT0250-NATIVE-PLACE-01/worker/EXECUTION_SPEC.json'))
    source=pinned(receiver['input'])['target_D1']
    state_ref=source['statistics']
    state=pinned(state_ref)['state']
    action_ref=receiver['action']
    actions=pinned(action_ref)['actions']
    statistics=dict(state=state,actions=actions)
    index=read(pin(package/'index.json',INDEX))
    old=Quantiles(state['q01'],state['q99'])
    output=Path(mkdtemp(prefix='v7-state-interface-audit.',dir=site))
    physical,groups,episodes,phases,balanced=[],[],[],[],[]
    for number,ep in enumerate(index['episodes']):
        with tarfile.open(pin(package/ep['path'],ep['archive_sha256'])) as archive:
            with np.load(io.BytesIO(archive.extractfile('labels.npz').read()),allow_pickle=False) as payload:
                values={key:payload[key] for key in ('state_eef10','phase','native_step')}
        codec=Codec(ep['T_policy_from_mount'])
        encoded=np.asarray([codec.encode(row,np.broadcast_to(row,(40,10)).copy())[0] for row in values['state_eef10']])
        physical.extend(encoded)
        group='old' if number<52 else 'new' if number<93 else 'recovery'
        groups.extend([group]*len(encoded));episodes.extend([ep['episode']]*len(encoded));phases.extend(values['phase'].tolist())
        for phase in dict.fromkeys(values['phase'].tolist()):
            available=np.flatnonzero(values['phase']==phase)
            fixed=available[np.rint(np.linspace(0,len(available)-1,8)).astype(int)]
            balanced.extend(encoded[fixed])
    physical,groups,episodes,phases=map(np.asarray,(physical,groups,episodes,phases))
    require(physical.shape==(8031,7),'The full fixed TRAIN state population changed')
    proposals={
        'GLOBAL_TRAIN_Q01_Q99':np.quantile(physical,[.01,.99],axis=0),
        'EIGHT_UNIFORM_QUERIES_PER_EPISODE_PHASE_Q01_Q99':np.quantile(balanced,[.01,.99],axis=0)}
    edges=np.linspace(-1,1,257)[:-1]
    old_bins=np.digitize(old.normalize(physical).astype(np.float32),edges)-1
    counterfactual={}
    for name,(low,high) in proposals.items():
        require(np.isfinite([low,high]).all() and (high>low).all(),'Degenerate proposed state quantiles')
        bins=np.digitize(Quantiles(low,high).normalize(physical).astype(np.float32),edges)-1
        changed=bins!=old_bins
        scope={}
        for group in ('old','new','recovery'):
            select=groups==group
            scope[group]=dict(windows=int(select.sum()),any_state_code_changed=int(changed[select].any(axis=1).sum()),
                changed_per_channel=changed[select].sum(axis=0).tolist(),
                new_lower_count=(bins[select]==-1).sum(axis=0).tolist(),
                new_upper_count=(bins[select]==255).sum(axis=0).tolist(),
                new_physical_bin_width=((high-low)/256).tolist())
        counterfactual[name]=dict(q01=low.tolist(),q99=high.tolist(),by_group=scope,
            all_changed_fraction=float(changed.any(axis=1).mean()),
            old_safe_phase_changes={phase:dict(windows=int(((groups=='old')&(phases==phase)).sum()),
                changed=int(changed[(groups=='old')&(phases==phase)].any(axis=1).sum()))
                for phase in sorted(set(phases[groups=='old']))})
    write(output/'STATE-RANGE.json',dict(index_sha256=INDEX,state_statistics_sha256=state_ref['sha256'],
        action_statistics_sha256=action_ref['sha256'],original_q01=state['q01'],original_q99=state['q99'],
        proposals=counterfactual,changed_training_or_deployment_statistics=False,
        new_model_forwards=0,new_native_integrations=0,new_optimizer_updates=0))
    print('STATE_COUNTERFACTUAL',output/'STATE-RANGE.json',sha(output/'STATE-RANGE.json'),flush=True)

    native=Path('/var/tmp/sla-v6-core-uid2009-owned-20261118')
    archive=native/'sla-core-multienv-wave-a-catalog-cutover-20261122-02/snapshots/TRAIN-069'
    commands=site.parent/'V3-MAIN-CONSUMED-DATA-02'
    corrected=site.parent/'v6-retention-20261126-01/REEXPRESSED-G7-V1'
    reader=data_scope(archive,commands,statistics)
    bind_dataset(reader,corrected,sha(corrected/'MANIFEST.json'),commands/'MANIFEST.json')
    require(len(reader)==4450 and len({ep['id'] for ep,_ in reader.rows})==52,'Require the original real training reader')
    lookup={ep['episode']:ep for ep in index['episodes'][:52]}
    require(set(lookup)=={ep['id'] for ep,_ in reader.rows},'Public original-52 and actual reader identities differ')
    tokenizer=Tokenizer(str(pin(receiver['tokenizer']['path'],receiver['tokenizer']['sha256'])))
    selected={}
    for i,(ep,k) in enumerate(reader.rows):
        phase=ep['expert_stage_audit_only'][k]
        selected.setdefault((ep['id'],phase),i)
        if k==ep['windows']-1:
            selected[ep['id'],'TAIL']=i
    # Every originally admitted episode and available phase, plus every partial tail.
    checks=[]
    for (eid,phase),i in selected.items():
        ep,k=reader.rows[i];item=reader[i]
        state10,_,rgb,prompt=reader.raw(i)
        expected,_=observation(item,tokenizer,drop_state=False)
        packet={}
        def capture(_active,_frozen,provided,_noise):
            packet['observation']=provided
            return item['actions'][None]
        request=SimpleNamespace(active=None,frozen=None,statistics=reader.statistics,tokenizer=tokenizer,
                               resize=reader.resize if hasattr(reader,'resize') else None,infer=capture)
        from openpi_client.image_tools import resize_with_pad
        request.resize=resize_with_pad
        Provider.predict(request,state10,rgb,prompt,lookup[eid]['T_policy_from_mount'],noise_id=0,
                         previous_reference=dict(width_m=.02,source_native_s=1.,query_native_s=1.05))
        observed=packet['observation']
        require(all(np.array_equal(np.asarray(getattr(expected,key)),np.asarray(getattr(observed,key)))
                    for key in ('state','tokenized_prompt','tokenized_prompt_mask')),
                'The original training reader and deployment state/prompt disagree')
        require(all(np.array_equal(np.asarray(left[key]),np.asarray(right[key]))
                for left,right in ((expected.images,observed.images),(expected.image_masks,observed.image_masks)) for key in left),
                'Original RGB or missing-camera masks disagree')
        require(item['prompt']==prompt,'The language changed')
        checks.append(dict(episode=eid,window=k,phase=phase,supported=int(reader.support(i).sum()),
                           token_count=int(np.asarray(expected.tokenized_prompt_mask).sum()),
                           real_reader_provider_parity=True))
    require({r['episode'] for r in checks}==set(lookup),'An original episode was omitted')
    write(output/'OLD52-READER-PARITY.json',dict(scope='REAL_CORRECTED_OLD52_TRAIN_READER_TO_ORIGINAL_PROVIDER_PREPROCESSING',
        reader_episodes=52,reader_total_windows=4450,checked_windows=len(checks),checks=checks,
        original_command_manifest_sha256=sha(commands/'MANIFEST.json'),corrected_manifest_sha256=sha(corrected/'MANIFEST.json'),
        tokenizer_sha256=receiver['tokenizer']['sha256'],state_statistics_sha256=state_ref['sha256'],
        action_statistics_sha256=action_ref['sha256'],drop_state=False,
        new_model_forwards=0,model_loads=0,new_native_integrations=0,new_optimizer_updates=0))
    print('OLD52_PARITY',output/'OLD52-READER-PARITY.json',sha(output/'OLD52-READER-PARITY.json'),len(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('site',type=Path);p.add_argument('package',type=Path)
    run(**vars(p.parse_args()))
