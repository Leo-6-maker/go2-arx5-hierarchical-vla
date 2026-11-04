"""Bounded DEV-only pre-close native qualification from actual complete saved states."""
from pathlib import Path
import json
import os
import time
from .contact_identification import candidates, execute, compare_repeats
from .paired_replay import read
from .native_adapter import SourceAdapter
from .storage import digest, load_private, dump_private, write_json
from .replay import tensor_state
import numpy as np


def sample(job):
    os.sched_setaffinity(0,{job['cpu']})
    spec,entry=job['spec'],job['entry'];out=Path(job['output']);out.mkdir()
    began=time.monotonic();cpu=time.process_time()
    for name,pin in entry['files'].items():
        if digest(name)!=pin:raise ValueError('Original saved prefix identity changed')
    card=json.loads(Path(entry['card']).read_text())
    if card['ident_index']not in range(16,20)or card['ident_partition']!='DIAGNOSTIC_DEV':
        raise ValueError('Only four declared diagnostic DEV parents are admitted')
    rows=read(entry['prefix'])['rows']
    start=next(i for i,r in enumerate(rows)if abs(r['t0']-1.6)<1e-6 and r['native_progress']['point']==0)
    history=rows[start-8:start];nominal=np.stack([r['action_raw']for r in rows[start:start+8]])
    if len(history)!=8:raise ValueError('Eight actual history atoms required')
    adapter=SourceAdapter(card,spec['asset_root'],executor_id=spec['executor_id'],seed=card['seed'])
    results={};runs={};live=0
    def guard():
        if time.time()>spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():
            raise TimeoutError('FINITE_DEV_ONSET_PROBE')
    try:
        saved=load_private(entry['restore']);adapter.restore(saved);initial=adapter.snapshot(force=True)
        if initial['native_grasp']or abs(initial['time_s']-1.6)>1e-6:
            raise ValueError('Pre-close observed native state differs')
        if not np.array_equal(tensor_state(initial),tensor_state(history[-1]['state_after'])):
            raise ValueError('Complete restore does not join actual common history')
        ref=adapter.save_complete(out/'common-restore.pkl.gz')
        dump_private(out/'COMMON_HISTORY.pkl.gz',dict(rows=history,state=initial,restore_ref=ref))
        interventions=candidates(nominal,float(adapter.last_command[-1]),initial)
        for name,raw in interventions.items():
            adapter.restore(saved);v=execute(adapter,raw,out/(name+'.pkl.gz'),ref,job['run_id'],guard)
            runs[name]=v;live+=v['physics_ticks'];rr=v['rows']
            results[name]=dict(transitions=len(rr),rejections=len(v['events']),
                terminal=bool(rr and rr[-1]['terminal']),
                grasp_establishment_transitions=sum(not r['state_before']['native_grasp']and r['state_after']['native_grasp']for r in rr),
                endpoint_native_grasp=bool(rr and rr[-1]['state_after']['native_grasp']),
                endpoint_object_xyz=rr[-1]['state_after']['object_pose'][:3,3].tolist()if rr else None)
        controls={}
        for name,tail in (('repeat',False),('tail-only',True)):
            adapter.restore(saved);v=execute(adapter,interventions['nominal'],out/(name+'.pkl.gz'),ref,job['run_id'],guard,tail=tail)
            live+=v['physics_ticks'];controls[name]=compare_repeats(runs['nominal']['rows'],v['rows'],runs['nominal']['final_restore'],v['final_restore'])
        report=dict(scene=card['ident_index'],partition=card['ident_partition'],restore_time_s=initial['time_s'],candidates=results,
            controls=controls,new_transitions=live//25,live_physics_ticks=live,shadow_ticks=adapter.shadow_ticks,
            CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-began)
        write_json(out/'RESULT.json',report);return report
    finally:adapter.close()


def run(spec,output):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    entries=spec['prefix_inputs']
    if sorted(e['scene']for e in entries)!=list(range(16,20)):raise ValueError('Exact four DEV parent identities required')
    began=time.monotonic();jobs=[dict(spec=spec,entry=e,cpu=spec['cpus'][i%len(spec['cpus'])],output=str(Path(output)/f"scene-{e['scene']:02d}"),run_id=Path(output).name)for i,e in enumerate(entries)]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))as pool:scenes=list(pool.map(sample,jobs))
    result=dict(protocol='WM02_S_DEV_ONSET_PROBE_V2',source_sha=spec['source_sha'],scenes=scenes,
        actual_transitions=sum(x['new_transitions']for x in scenes),main_transitions=sum(v['transitions']for x in scenes for v in x['candidates'].values()),
        live_physics_ticks=sum(x['live_physics_ticks']for x in scenes),shadow_ticks=sum(x['shadow_ticks']for x in scenes),
        CPU_s=sum(x['CPU_s']for x in scenes),wall_s=time.monotonic()-began,
        WM_updates=0,Actor_updates=0,target_transitions=0,complete_tasks=0,heldout_parents_used=0)
    write_json(Path(output)/'RESULT.json',result);return result
