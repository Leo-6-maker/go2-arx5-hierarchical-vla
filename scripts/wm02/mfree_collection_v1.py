"""Finite per-family production using the existing journal and native owner.

Two family workers may run independently. C4 is a view of their first two
qualified centre episodes, not another physical qualification panel.
"""
from contextlib import closing
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from scripts.pi0292.durable import OwnerLock, write
from scripts.pi0294.e1.journal import Journal
from .mobile_boundary_runner_v1 import registered
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.mobile_grasp_budget_v3 import qualification_record

def read(path): return json.loads(Path(path).read_bytes())


def capture_check(result):
    capture=(result.get('detail') or {}).get('capture')
    if not capture: raise ValueError('COMPLETE_CAPTURE_MANIFEST_REQUIRED')
    if digest(capture['path'])!=capture['sha256']: raise ValueError('SEALED_CAPTURE_MANIFEST')
    manifest=read(capture['path']); labels=manifest['labels']
    if manifest['writer_error'] or digest(labels['path'])!=labels['sha256']:
        raise ValueError('SUCCESSFUL_BOUNDED_WRITER_AND_LABEL_SEAL')
    rows=read(labels['path'])
    if not rows or len(rows)!=manifest['queries']: raise ValueError('NONEMPTY_LIVE_QUERIES')
    for shard in manifest['frame_shards']:
        if digest(shard['path'])!=shard['sha256']: raise ValueError('SEALED_LIVE_RGB_SHARD')
    if sum(s['count'] for s in manifest['frame_shards'])!=len(rows): raise ValueError('ONE_RGB_PAIR_PER_QUERY')
    for index,row in enumerate(rows):
        if (row['native_step']!=5703+250*index or abs(row['native_time_s']-.002*row['native_step'])>1e-8
                or row['supervision_mask']!=[True]*40 or row['consumed_mask']!=[True]*10+[False]*30
                or row['actual_elapsed_prefix_ticks']!=250
                or row['next_query_boundary']['native_step']!=row['native_step']+250
                or not row['recent_reference']['exact_receiver_knot']):
            raise ValueError('LIVE_QUERY_H40_P10_AND_ACCEPTED_050_BOUNDARY')
    return dict(manifest=capture, queries=len(rows), planned_points=40*len(rows),
        confirmed_consumed_points=10*len(rows), original_RGB=True, unconsumed_is_executed=False)


def run(spec_path):
    spec=read(spec_path); root=Path(spec['registered_root']); lease=registered(spec,spec_path)
    if (root/'SUPERVISOR.private.json').exists(): raise PermissionError('EXISTING_PRODUCTION_OWNER_NO_REPLAY')
    write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),epoch=time.time()))
    journal=Journal(root/'control/journal',root.name); outcomes=[]; counts={}; failure=None; unsafe=False; code=1
    campaign=Path(spec['collection_root']); stop=campaign/'control/TECHNICAL_STOP.private.json'
    try:
        dependency=spec.get('first_capture_dependency')
        while dependency and not Path(dependency).exists():
            if stop.exists(): raise RuntimeError('OTHER_FAMILY_CAPTURE_TECHNICAL_STOP')
            if time.time()>=spec['deadline_epoch']-900: raise TimeoutError('FINITE_FIRST_CAPTURE_WAIT')
            time.sleep(2)
        with closing(OwnerLock(spec['mobile_world_lock'])) as world:
            for path in spec['cases']:
                case=read(path); layout=case['collection_contract']['layout_id']
                if counts.get(layout,0)>=2: continue
                if unsafe or stop.exists() or (root/'STOP.json').exists() or time.time()>=spec['deadline_epoch']-900: break
                write(root/'PROGRESS.private.json',dict(current=case['collection_contract'],
                    terminal=len(outcomes),successful_by_layout=counts,epoch=time.time()))
                children=[]; log=root/'logs'/(Path(path).parent.name+'.log')
                def launch():
                    argv=[spec['python'],'-B',str(Path(spec['source_root'])/'scripts/wm02/mobile_boundary_bootstrap_v1.py'),
                        '--case',path,'--world-lock-fd',str(world.handle.fileno())]
                    with log.open('xb') as handle:
                        children.append(subprocess.Popen(argv,cwd=spec['source_root'],env=dict(os.environ,**spec['native_environment']),
                            stdout=handle,stderr=subprocess.STDOUT,pass_fds=(world.handle.fileno(),)))
                    return dict(pid=children[0].pid)
                journal.once(Path(path).parent.name,dict(spec_sha256=digest(path)),launch)
                if not children: raise PermissionError('PRIOR_CHILD_EFFECT_REQUIRES_RECONCILIATION')
                child=children[0]
                try: child.wait(timeout=case['wall_limit_s']+180)
                except BaseException:
                    child.terminate()
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired: child.kill();child.wait()
                    raise
                case_root=Path(path).parent; result=read(case_root/'RESULT.private.json')
                qualified=qualification_record(result)['qualified']; capture=None
                if qualified:
                    capture=capture_check(result); counts[layout]=counts.get(layout,0)+1
                value=dict(root=str(case_root),condition=case['mobile_condition'],
                    collection=case['collection_contract'],result_sha256=digest(case_root/'RESULT.private.json'),
                    qualified=qualified,capture=capture,physical_requests=1,
                    native_integrations=result['actual_native_integrations'],unsafe=result['unsafe'],
                    technical_failure=result['technical_failure'],detail=result.get('detail'))
                outcomes.append(value);write(root/'OUTCOMES.private.json',outcomes)
                if qualified and not(campaign/'FIRST_CAPTURE_READY.private.json').exists():
                    write(campaign/'FIRST_CAPTURE_READY.private.json',dict(source_sha=spec['source_sha'],
                        result=dict(path=str(case_root/'RESULT.private.json'),sha256=value['result_sha256']),capture=capture))
                if result['unsafe']: unsafe=True
                elif result['technical_failure']: raise RuntimeError('CAPTURE_TECHNICAL_FAILURE_RETAINED_NO_RETRY')
            code=0
    except BaseException:
        failure=traceback.format_exc()
        if not stop.exists(): write(stop,dict(group=str(root),reason=failure,epoch=time.time()))
    for path in spec['cases']:
        case_root=Path(path).parent
        if not(case_root/'SUPERVISOR.private.json').exists():
            old=read(case_root/'control/LEASE.json'); layout=read(path)['collection_contract']['layout_id']
            reason='LAYOUT_TWO_SUCCESSES_REACHED' if counts.get(layout,0)>=2 else 'FAMILY_NATIVE_SAFETY_ISOLATION' if unsafe else 'TECHNICAL_STOP_OR_DEADLINE'
            write(case_root/'control/LEASE.json',dict(old,status='CLOSED_UNSTARTED',reason=reason,
                physical_requests=0,native_integrations=0,closed_epoch=time.time()))
    result=dict(schema='MFREE_C20_FAMILY_PRODUCTION_ACTUAL_V1',source_sha=spec['source_sha'],
        outcomes=outcomes,successful_by_layout=counts,qualified_episodes=sum(counts.values()),
        physical_requests=len(outcomes),native_integrations=sum(v['native_integrations'] for v in outcomes),
        new_optimizer_updates=0,new_training_visits=0,new_model_queries=0,unsafe=unsafe,
        failure=failure,exit_code=code,automatic_retry=False,maximum_predeclared_attempts=20)
    write(root/'RESULT.private.json',result);write(root/'WORKER_EXIT.private.json',dict(exit_code=code))
    write(root/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
    return code


if __name__=='__main__':
    import sys
    raise SystemExit(run(sys.argv[sys.argv.index('--production')+1]))
