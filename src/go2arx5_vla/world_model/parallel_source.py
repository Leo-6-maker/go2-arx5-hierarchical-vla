"""Bounded S-domain CPU process profiling; never a target WBC throughput claim."""
from pathlib import Path
import os,time,json,resource


def sample_worker(job):
    # Spawned process owns one engine and one CPU. No global controller changes.
    os.sched_setaffinity(0,{job['cpu']})
    import numpy as np
    from .native_adapter import SourceAdapter
    from .action_compiler import compile_reference,SOURCE
    from .contracts import FRAME,TCP
    from .transitions import TransitionRecorder
    from .storage import dump_private,write_json
    folder=Path(job['folder']);folder.mkdir();spec=job['spec'];card=job['card']
    adapter=SourceAdapter(card,spec['asset_root'],executor_id=spec['executor_id'],seed=card['seed'])
    began,cpu=time.monotonic(),time.process_time();recorder=TransitionRecorder(native_dt=.002)
    def guard():
        if time.time()>=spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():raise TimeoutError('FINITE_SOURCE_PROFILE')
    try:
        ref=adapter.save_complete(folder/'start-restore.pkl.gz')
        for block in range(5):
            guard();state=adapter.snapshot();raw=np.zeros((8,7));raw[:,6]=state['gripper_fraction']
            plan=compile_reference(raw,state['tcp'],state['time_s'],backend=SOURCE,frame=FRAME,tcp=TCP)
            if adapter.gate(plan)['decision']=='REJECT'or adapter.preflight(plan,guard)['decision']!='PASS':
                recorder.reject(episode_id=card['id'],reason='SOURCE_S1_REJECT',state=state,action=raw);break
            for k in range(8):
                row=adapter.advance_recorded(plan,k,run_id=job['run_id'],actor_version='SOURCE_PROFILE_HOLD',restore_ref=ref)
                recorder.append(row)
                if row['terminal']or row['truncated']:break
            if recorder.rows and(recorder.rows[-1]['terminal']or recorder.rows[-1]['truncated']):break
        dump_private(folder/'RECORDS.pkl.gz',dict(rows=recorder.rows,events=recorder.events))
        result=dict(episode_id=card['id'],transitions=len(recorder.rows),live_ticks=sum(r['physics_ticks']for r in recorder.rows),
            shadow_ticks=adapter.shadow_ticks,hard_safety=list(adapter.engine.safety),S1_error=adapter.max_shadow_error,
            rejected=len(recorder.events),CPU_s=time.process_time()-cpu,worker_wall_s=time.monotonic()-began,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pid=os.getpid(),cpu=job['cpu'])
        write_json(folder/'RESULT.json',result);return result
    finally:adapter.close()


def profile(spec,output):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from .storage import write_json
    cards=json.loads(Path(spec['diagnostic_cards']).read_text());reports=[]
    cpus=spec['cpus']
    for envs in (1,4,16):
        available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
        # Source mesh footprint measured at the preceding level; protect1GiB.
        per_env=max([r['maximum_RSS_KiB']*1024 for p in reports for r in p.get('workers',[])]or[768*1024**2])
        if envs>len(cpus)or envs*per_env+1024**3>available:
            reports.append(dict(environments=envs,status='NOT_RUN_ACTUAL_CPU_OR_RAM_HEADROOM',
                available_RAM_bytes=available,conservative_per_env_RAM_bytes=per_env,available_CPUs=len(cpus),actual_transitions=0));continue
        jobs=[];level=output/f'envs-{envs}';level.mkdir()
        for index in range(envs):
            card=dict(cards[index%len(cards)]);card.update(id=f'WM02-S-PROFILE-{envs}-{index}',parent_id=f'WM02-S-PROFILE-PARENT-{index}',scene_group=f'WM02-S-PROFILE-SCENE-{index}')
            jobs.append(dict(spec=spec,card=card,cpu=cpus[index],folder=str(level/card['id']),run_id=output.name))
        began=time.monotonic()
        # Actual simultaneous processes, not threads serialized by a Python loop.
        with ProcessPoolExecutor(max_workers=envs,mp_context=multiprocessing.get_context('spawn'))as pool:
            workers=list(pool.map(sample_worker,jobs))
        elapsed=time.monotonic()-began;count=sum(r['transitions']for r in workers)
        row=dict(environments=envs,status='SOURCE_NATIVE_COMPLETE',process_workers=envs,workers=workers,
            actual_transitions=count,wall_s=elapsed,transitions_per_s=count/elapsed,
            CPU_s=sum(r['CPU_s']for r in workers),includes_spawn_and_engine_construction_and_recording=True,
            backend='S_FIXED_BASE_MUJOCO_NO_WBC',target_qualified=False)
        reports.append(row);write_json(level/'RESULT.json',row);print(json.dumps(row),flush=True)
    result=dict(status='SOURCE_PARALLEL_PROFILE_COMPLETE_TARGET_BLOCKED',levels=reports,
        actual_atomic_transitions=sum(r['actual_transitions']for r in reports),
        live_ticks=sum(w['live_ticks']for r in reports for w in r.get('workers',[])),
        shadow_ticks=sum(w['shadow_ticks']for r in reports for w in r.get('workers',[])),
        target_parallel_1_4_16='NOT_RUN_QUARANTINE',model_updates=0,actor_updates=0,
        native_dt_s=.002,atomic_dt_s=.05,solver_or_safety_changes=0)
    write_json(output/'RESULT.json',result);return result
