"""Read-only CPU replay of the fixed expert pilot ledger and real reset identity."""
from pathlib import Path
import argparse
import collections
import json
import sys
import numpy as np
from pi0255_motion3 import sha,write,TASKS


def replay(root):
    sys.path.insert(0,str(root.parent/'PI0250/runtime/reference-317a/src'))
    from go2arx5_vla.contracts import TrajectoryChunkV1
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    registry=json.loads((root/'design/task_registry_v1.json').read_text());seal=json.loads((root/'design/execution_seal.json').read_text())
    assert sha(root/'design/task_registry_v1.json')==seal['registry_sha256']
    intended={r['episode_id']:r for r in registry['episodes'] if r['split']=='PILOT'}
    starts=[json.loads(s) for s in (root/'physics_attempts.jsonl').read_text().splitlines()]
    assert [r['ordinal'] for r in starts]==list(range(1,len(starts)+1)) and len(starts)<=360
    results={};rows=[];limitations=[]
    for p in sorted((root/'episodes').glob('*/attempt-*/result.json')):
        result=json.loads(p.read_text());episode=result['episode'];identity=episode['episode_id']
        assert intended[identity]==episode and result['registry_sha256']==seal['registry_sha256']
        if result['validity']!='VALID':continue
        assert identity not in results and result.get('engineering_error') is None
        results[identity]=result;directory=p.parent
        ticks=[json.loads(s) for s in (directory/'raw_ticks.jsonl').read_text().splitlines()]
        assert sha(directory/'raw_ticks.jsonl')==result['raw_ticks_sha256']
        t=np.asarray([x['actual_sim_time_s'] for x in ticks]);assert len(t) and np.isfinite(t).all()
        assert np.all(np.diff(t)>0) and np.all(np.diff(t)<=.0201)
        assert abs(t[-1]-result['end_actual_sim_time_s'])<1e-8
        with np.load(directory/'reset_state.npz',allow_pickle=False) as state:
            assert np.isfinite(state['root']).all() and np.isfinite(state['dof']).all()
        assert sha(directory/'reset_state.npz')==result['initial_reset']['tensor_rng_sha256']
        handoffs=[json.loads(s) for s in (directory/'handoffs.jsonl').read_text().splitlines()]
        full_replayed=0
        for h in handoffs:
            if 'reference_file' not in h:continue
            path=directory/h['reference_file'];assert sha(path)==h['reference_sha256']
            with np.load(path,allow_pickle=False) as data:
                now=float(data['now']);timestamps=data['point_timestamps'];assert len(timestamps)==40
                c=TrajectoryChunkV1(chunk_id=h['chunk_id'],source='PI0255_REGISTERED_DESIRED_EXPERT',observation_timestamp=float(timestamps[0]),
                    point_timestamps=timestamps,eef_position=data['eef_position'],eef_orientation=data['eef_orientation'],
                    source_frame='REPLAY_SAME_NATIVE_NUMERICAL_FRAME',tcp_semantic_id='REPLAY_SAME_TCP_NUMERICAL_POSE',
                    validity_start=float(timestamps[0]),validity_end=float(timestamps[0])+1.95,gripper=data['gripper'])
                actual=TrajectorySafetyGateV1().evaluate(c,now=now,achieved_position=data['achieved_native'],achieved_orientation=data['achieved_orientation']).as_dict()
                assert actual['decision']==h['gate']['decision']
                assert [x['code'] for x in actual['issues']]==[x['code'] for x in h['gate']['issues']]
                def equal(a,b):
                    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
                    if isinstance(a,(float,int)):return np.isclose(a,b,rtol=0,atol=1e-8)
                    return a==b
                assert equal(actual['metrics'],h['gate']['metrics'])
                full_replayed+=1
        if full_replayed!=len(handoffs):limitations.append(identity+': original full handoff arrays unavailable; full gate/codec parity not claimed')
        else:
            own_seal=json.loads((directory/'execution_seal.json').read_text());assert own_seal['runtime_files']==result['runtime_files']
        failure=result.get('failure');angle=None;event=None
        if failure=='SETTLING_EXPERT_GATE_REJECT':
            last=handoffs[-1]['gate'];assert last['decision']=='REJECT'
            assert [x['code'] for x in last['issues']]==['INITIAL_ROTATION_DISCONTINUITY_REJECT']
            angle=last['metrics']['initial_rotation_discontinuity_rad'];assert angle>1.5
            q=np.asarray(ticks[-1]['achieved_quaternion']);reference=np.asarray(ticks[0]['desired_quaternion'])
            measured=2*np.arccos(np.clip(abs(np.dot(q/np.linalg.norm(q),reference/np.linalg.norm(reference))),0,1))
            assert abs(measured-angle)<1e-5
        elif failure=='SETTLING_PHYSICAL_TERMINATION':
            assert not ticks[-1]['physical_clean']
            event=ticks[-1]['termination_event']
            assert event or ticks[-1]['done'] or ticks[-1]['reset_buf'] or ticks[-1]['obstacle_contact_force_n']>1e-6
        else:
            # Completed task metrics have a separate scorer replay requirement.
            limitations.append(identity+': task-level metric replay still required')
        rows.append(dict(episode_id=identity,task=episode['task'],command=episode['command'],family=episode['family_id'],success=result['success'],
            failure=failure,actual_end_s=float(t[-1]),raw_ticks=len(ticks),task_ticks=result.get('task_ticks'),snapshots=result.get('snapshots'),
            rotation_handoff_error_rad=angle,termination_code=event.get('termination_code') if event else None,
            result_sha256=sha(p),raw_ticks_sha256=sha(directory/'raw_ticks.jsonl')))
        rows[-1].update(handoff_count=len(handoffs),complete_handoff_gate_replays=full_replayed,runtime_files=result['runtime_files'])
    pairs=[]
    for family in registry['families']:
        if family['split']!='PILOT':continue
        for task in TASKS:
            ids=[family['family_id']+'-'+task+'-c'+str(c) for c in (0,1)]
            if not all(i in results for i in ids):continue
            a,b=[results[i]['initial_reset'] for i in ids]
            fields=('root_sha256','dof_sha256','body_rgb_sha256','wrist_rgb_sha256','tensor_rng_sha256','python_rng_state','numpy_rng_metadata','physics_seed')
            matches={k:a[k]==b[k] for k in fields}
            pairs.append(dict(family=family['family_id'],task=task,matched=all(matches.values()),fields=matches))
    totals={task:dict(valid=sum(r['task']==task for r in rows),successful=sum(r['task']==task and r['success'] for r in rows),required=10,
                     admitted=sum(r['task']==task for r in rows)==12 and sum(r['task']==task and r['success'] for r in rows)>=10) for task in TASKS}
    report=dict(schema='PI0255_PILOT_CPU_REPLAY_V1',status='PILOTS_COMPLETE' if len(results)==36 else 'PILOTS_IN_PROGRESS',
        source_commit=seal['pre_execution_commit'],canonical=seal['canonical_comment'],physics_starts=len(starts),valid_results=len(results),
        engineering_results=len(list((root/'episodes').glob('*/attempt-*/result.json')))-len(results),per_task=totals,
        failure_classes=dict(collections.Counter(r['failure'] for r in rows)),counterfactual_pairs=pairs,rows=rows,limitations=limitations,
        completed_task_episodes=sum(bool(r['task_ticks']) for r in rows),new_GR00T_loads=0,new_GR00T_queries=0,new_GR00T_backward=0,new_GR00T_optimizer=0,
        learned_policy_episodes=0,TRAIN_episodes=0,MT_VAL_episodes=0,TEST_executions=0,selected_checkpoint='NONE')
    write(root/'evidence/pilot_cpu_replay.json',report)
    print(json.dumps({k:report[k] for k in ('status','physics_starts','valid_results','per_task','failure_classes','completed_task_episodes')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();replay(args.root)
