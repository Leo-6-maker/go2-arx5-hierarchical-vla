"""CPU-only exact TRAIN-plan replay, followed by read-only completed-training audit."""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
import numpy as np


def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def read(p):return json.loads(Path(p).read_text())


def inputs():
    from pi0267_training_plan import BASE,ROOT,R1,RAW,sha,write,build,batch,identity,text_for
    sys.path.append(str(BASE/'PI0264/code'))
    from pi0264_plan import NONLANGUAGE
    plan=read(R1/'design/PAIRING_PLAN.json');proc=build(read(R1/'design/statistics.json'),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    path=R1/'evidence/FULL_TRAIN_INPUT_PLAN.json'
    if path.exists():
        prior=read(path);assert prior['plan_sha256']==sha(R1/'design/PAIRING_PLAN.json');return
    families={};cache={};expected=[]
    for arm in ('N','R'):
        for u in plan['units']:
            if u['family'] not in families:
                folder=RAW/u['family'];complete=read(folder/'COMPLETE.json')
                for n in ('proposals.npz','policy_frames.npz','rows.json'):assert sha(folder/n)==complete['hashes'][n]
                families[u['family']]=(np.load(folder/'proposals.npz'),np.load(folder/'policy_frames.npz'),read(folder/'rows.json'))
            z,fr,meta=families[u['family']]
            for micro,c in enumerate(u['order']):
                i=u[arm+'_indices'][c];row=meta[i];text=text_for(plan,u,c);key=(u['family'],i,text)
                if key not in cache:
                    packet=dict(body_rgb=fr['body_rgb'][row['frame_index']],wrist_rgb=fr['wrist_rgb'][row['frame_index']],state_v1=z['state_v1'][i],language=text)
                    ins=batch(proc,packet,z['raw_action'][i])['inputs']
                    cache[key]=dict(condition_sha256=identity({k:v for k,v in ins.items() if k not in ('action','action_mask')}),
                        nonlanguage_sha256=identity({k:ins[k] for k in NONLANGUAGE}),labels_masks_sha256=identity({k:ins[k] for k in ('action','action_mask')}))
                expected.append(dict(unit=u['unit'],micro=micro,arm=arm,task=u['task'],family=u['family'],slot=u['slot'],command=c,
                    sequence=u['family'],index=i,original_unit=f"{u['task']}:{u['bank_index']}",visit=u['visit'],color=plan['source_family_colors'][u['family']][c],
                    template=u['template'],language=text,frame_key=row['frame_key'],**cache[key]))
            if u['unit']%512==0:print(json.dumps(dict(arm=arm,unit=u['unit'],unique_inputs=len(cache))),flush=True)
    assert len(expected)==16384
    write(path,dict(status='PASS_FULL_REAL_TRAIN_INPUT_PLAN_REPLAY',plan_sha256=sha(R1/'design/PAIRING_PLAN.json'),
        statistics_sha256=sha(R1/'design/statistics.json'),rows=expected,unique_inputs=len(cache),EVAL_payload_opened=False,model_queries=0))


def complete():
    import pi0264_training_audit as old
    import pi0264_archive_trainer as archive
    from pi0267_train import seed_for
    root=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0267');old.ROOT=root
    expected={(r['arm'],r['unit'],r['micro']):r for r in read(root/'FULL_TRAIN_INPUT_PLAN.json')['rows']}
    paired={};checkpoints=[]
    for arm in ('N','R'):
        assert read(root/arm/'canary_compare.json')['status']=='PASS_PI0267_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY'
        path=root/arm/'main/training';samples=rows(path/'sample_audit.jsonl');rng=rows(path/'step_rng.jsonl');s4=rows(path/'S4_audit.jsonl')
        assert len(samples)==len(rng)==len(s4)==8192
        for i,(a,b,c) in enumerate(zip(samples,rng,s4)):
            unit,micro=i//2+1,i%2;e=expected[arm,unit,micro]
            assert a==e,(arm,unit,micro,'ACTUAL_INPUT_PLAN')
            assert all(b[k]==e[k] for k in ('unit','micro','command','condition_sha256'))
            assert b['seed']==seed_for(unit) and b['mode_restored'] and c['forward_index']==i+1
            packet=tuple(b[k] for k in ('native_epsilon','native_tau','stock_initial_noise'))
            assert packet==tuple(c[k] for k in ('native_epsilon','native_tau','stock_initial_noise'))
            if micro==0:paired[arm,unit]=packet
            else:assert packet==paired[arm,unit]
        old.graph(arm,max_steps=4096,campaign='PI0267')
        for step in (2048,4096):
            checkpoint=path/f'checkpoint-{step}';state=read(checkpoint/'trainer_state.json')
            assert (state['global_step'],state['max_steps'])==(step,4096)
            assert all((checkpoint/n).is_file() for n in ('optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors.index.json'))
            entries=[]
            for p in sorted(checkpoint.rglob('*')):
                assert not p.is_symlink()
                if p.is_file():entries.append(dict(path=p.relative_to(checkpoint).as_posix(),bytes=p.stat().st_size,sha256=archive.sha(p)))
            assert len(entries)>=16
            checkpoints.append(dict(arm=arm,step=step,path=str(checkpoint),entries=entries,complete_trainer=True))
    assert all(paired['N',u]==paired['R',u] for u in range(1,4097))
    updates=rows(root/'raw_completed_updates.jsonl');heads=rows(root/'raw_training_heads.jsonl')
    assert [r['ordinal'] for r in updates]==list(range(1,len(updates)+1)) and len(updates)<=8480
    assert [r['ordinal'] for r in heads]==list(range(1,len(heads)+1)) and len(heads)<=210000
    counts=Counter((r['arm'],r['phase']) for r in updates)
    assert counts['N','main']==counts['R','main']==4096 and all(r['phase'] in ('main','continuous','split','split_recovery1') for r in updates+heads)
    reserve=len(updates)-8224;assert 0<=reserve<=256
    result=dict(status='PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY',optimizer_updates={'N':4096,'R':4096},
        complete_checkpoints=checkpoints,actual_completed_by_arm_phase=[dict(arm=a,phase=p,updates=n) for (a,p),n in sorted(counts.items())],
        raw_completed_updates=len(updates),technical_recovery_completed_updates=reserve,raw_training_heads=len(heads),
        full_TRAIN_inputs_replayed=True,actual_random_packets_equal=True,actual_S4_gradient_loss_mode_checks=True,
        completed_updates_sha256=archive.sha(root/'raw_completed_updates.jsonl'),training_heads_sha256=archive.sha(root/'raw_training_heads.jsonl'),
        model_queries_by_audit=0,gradient_queries_by_audit=0,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    old.write(root/'MAIN_TRAINING_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='complete_checkpoints'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('inputs','complete'));a=p.parse_args()
    inputs() if a.mode=='inputs' else complete()
