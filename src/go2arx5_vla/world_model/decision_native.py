"""Observed-state source anchors and finite candidate qualification; original safety/driver."""
from pathlib import Path
import hashlib
import json
import os
import time
import traceback
import numpy as np
from .native_adapter import SourceAdapter
from .contact_identification import execute,compare_repeats,supported,LO,HI
from .paired_replay import read
from .replay import tensor_state,SLICES
from .storage import digest,load_private,dump_private,write_json

KINDS=('preclose','transport','supported_release')


def intent(value):
    if abs(value)<1e-12:return LO
    if abs(value-1)<1e-12:return HI
    if LO<=value<=HI:return float(value)
    raise ValueError('Last command outside working domain; no relabel or execution clipping')


def proposals(kind,state,context,last,scene,decision=0):
    """Only current numeric observation, task context and known command; no event/Teacher input."""
    old=intent(last);hold=np.zeros((8,7));hold[:,6]=old
    if kind=='preclose':
        close=hold.copy();close[:,6]=LO
        actions=dict(hold=hold,nominal=close,left=close.copy(),right=close.copy(),grip_timing=close.copy())
        actions['left'][:,0]-=.0005;actions['right'][:,0]+=.0005;actions['grip_timing'][:4,6]=old
    elif kind=='transport':
        direction=np.asarray(context[:2])-state[SLICES['object_position']][:2]
        direction=direction/max(1e-12,np.linalg.norm(direction));perp=np.array([-direction[1],direction[0]])
        actions={'hold':hold}
        for name,axis in (('toward_goal',direction),('away',-direction),('left',perp),('right',-perp)):
            raw=hold.copy();raw[:,:2]=.0005*axis;actions[name]=raw
    elif kind=='supported_release':
        release=hold.copy();release[:,6]=HI
        actions=dict(hold=hold,release=release,delay2=release.copy(),delay4=release.copy(),slow_open=release.copy())
        actions['delay2'][:2,6]=old;actions['delay4'][:4,6]=old;actions['slow_open'][:,6]=np.linspace(old,HI,8)
    else:raise ValueError('Unknown mechanism task')
    names=sorted(actions,key=lambda n:hashlib.sha256(f'WM02_NATIVE_DECISION_V3/{scene}/{kind}/{decision}/{n}'.encode()).digest())
    return names,np.stack([actions[n]for n in names])


def physical_anchor(kind,adapter,state,context):
    """Construction-only physical qualification, never a policy/model input."""
    obj=state['object_pose'][:3,3];tcp=state['tcp'][:3,3];gap=tcp-obj
    if kind=='preclose':
        return (not state['native_grasp']and state['gripper_fraction']>=.85 and np.linalg.norm(gap[:2])<=.005
                and -.003<=gap[2]<=.008 and np.linalg.norm(state['tool_linear_velocity'])<=.04)
    if kind=='transport':
        return (state['native_grasp']and obj[2]>=.49 and np.linalg.norm(gap)<=.015
                and np.linalg.norm(state['tool_linear_velocity'])<=.07)
    if kind=='supported_release':
        return state['native_grasp']and supported(adapter,state)and np.linalg.norm(obj[:2]-context[:2])<=.02
    raise ValueError('Unknown mechanism task')


def bank(adapter,entry):
    for path,pin in entry['files'].items():
        if digest(path)!=pin:raise ValueError('Saved construction/restore identity differs')
    prefix=read(entry['prefix'])['rows'];context=np.array([*adapter.card['goal_xy'],.46,.44,.04]);result={}
    for path in entry['restores']:
        saved=load_private(path);adapter.restore(saved);state=adapter.snapshot(force=True)
        history=[r for r in prefix if r['t1']<=state['time_s']+1e-8][-8:]
        if len(history)!=8:continue
        if not np.array_equal(tensor_state(history[-1]['state_after']),tensor_state(state)):
            raise ValueError('Saved anchor does not join original real history')
        for kind in KINDS:
            if kind not in result and physical_anchor(kind,adapter,state,context):
                result[kind]=dict(saved=saved,history=history,time_s=state['time_s'],context=context)
        if len(result)==3:break
    return result


def local_guard(spec):
    def guard():
        if time.time()>spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():
            raise TimeoutError('FINITE_SOURCE_DECISION_DIAGNOSTIC')
    return guard


def qualify_scene(job):
    os.sched_setaffinity(0,{job['cpu']});spec=job['spec'];entry=job['entry'];out=Path(job['out']);out.mkdir()
    card=json.loads(Path(entry['card']).read_text());start=time.monotonic();cpu=time.process_time();adapter=None;results={};live=0;failure=None
    try:
        if spec.get('domain')=='S_NEW_DECISION_DATA_V5':
            from .decision_collection import cards
            if card!=cards()[entry['scene']]:raise ValueError('New whole-parent partition/card differs')
        elif card['ident_index']!=entry['scene']or card['ident_partition']!='DIAGNOSTIC_DEV':raise ValueError('Exact original DEV parent required')
        adapter=SourceAdapter(card,spec['asset_root'],executor_id=spec['executor_id'],seed=card['seed']);anchors=bank(adapter,entry)
        guard=local_guard(spec)
        for kind in KINDS:
            if kind not in anchors:results[kind]=dict(status='ANCHOR_NOT_FOUND',unrun=5);continue
            anchor=anchors[kind];adapter.restore(anchor['saved']);initial=adapter.snapshot(force=True)
            names,raw=proposals(kind,tensor_state(initial),anchor['context'],float(adapter.last_command[-1]),entry['scene'])
            folder=out/kind;folder.mkdir();ref=adapter.save_complete(folder/'common-restore.pkl.gz')
            dump_private(folder/'COMMON_HISTORY.pkl.gz',dict(rows=anchor['history'],state=initial,restore_ref=ref))
            runs={};candidates={}
            for name,action in zip(names,raw):
                adapter.restore(anchor['saved']);run=execute(adapter,action,folder/(name+'.pkl.gz'),ref,out.name,guard)
                runs[name]=run;live+=run['physics_ticks'];rows=run['rows'];state=adapter.snapshot(force=True)
                candidates[name]=dict(transitions=len(rows),rejections=len(run['events']),terminal=bool(rows and rows[-1]['terminal']),
                    grasp=bool(state['native_grasp']),support=bool(supported(adapter,state)),
                    object_endpoint_xyz=state['object_pose'][:3,3].tolist(),
                    grasp_establishments=sum(not r['state_before']['native_grasp']and r['state_after']['native_grasp']for r in rows))
            control='nominal'if kind=='preclose'else'hold'if kind=='transport'else'release';action=raw[names.index(control)];controls={}
            for name,tail in (('repeat',False),('tail-only',True)):
                adapter.restore(anchor['saved']);r=execute(adapter,action,folder/(name+'.pkl.gz'),ref,out.name,guard,tail=tail)
                live+=r['physics_ticks'];controls[name]=compare_repeats(runs[control]['rows'],r['rows'],runs[control]['final_restore'],r['final_restore'])
            endpoints=np.array([x['object_endpoint_xyz']for x in candidates.values()]);spread=float(np.linalg.norm(endpoints[:,None]-endpoints[None],axis=-1).max())
            complete=all(x['transitions']==16 and not x['terminal']for x in candidates.values())
            contact=(any(x['grasp']for x in candidates.values())and any(not x['grasp']for x in candidates.values()))
            signal=(spread>.001 if kind=='transport'else contact)
            if kind=='supported_release':signal=signal and all(x['support']for x in candidates.values()if not x['grasp'])
            qualified=complete and signal and all(x['qualified']for x in controls.values())
            results[kind]=dict(status='QUALIFIED'if qualified else'SCIENTIFIC_ANCHOR_NOT_QUALIFIED',time_s=anchor['time_s'],
                candidates=candidates,controls=controls,object_pair_spread_m=spread,real_contact_contrast=contact,qualified=qualified)
            write_json(folder/'RESULT.json',results[kind])
    except BaseException:failure=traceback.format_exc()
    finally:
        shadow=adapter.shadow_ticks if adapter else 0
        if adapter:adapter.close()
    result=dict(scene=entry['scene'],anchors=results,failure=failure,live_physics_ticks=live,shadow_ticks=shadow,
        actual_transitions=live//25,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(out/'RESULT.json',result);return result


def qualify(spec,output):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    if sorted(x['scene']for x in spec['prefix_inputs'])!=list(range(16,20)):raise ValueError('Four DEV parents only')
    start=time.monotonic();jobs=[dict(spec=spec,entry=e,cpu=spec['cpus'][i%len(spec['cpus'])],out=str(Path(output)/f"scene-{e['scene']:02d}"))for i,e in enumerate(spec['prefix_inputs'])]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))as pool:scenes=list(pool.map(qualify_scene,jobs))
    result=dict(protocol='WM02_OBSERVED_ANCHOR_QUALIFICATION_V3',source_sha=spec['source_sha'],scenes=scenes,
        declared_parents=4,declared_anchors=12,declared_main_branches=60,
        all_scientific_anchors_qualified=all(len(s['anchors'])==3 and all(x.get('qualified')for x in s['anchors'].values())for s in scenes),
        actual_transitions=sum(s['actual_transitions']for s in scenes),live_physics_ticks=sum(s['live_physics_ticks']for s in scenes),
        shadow_ticks=sum(s['shadow_ticks']for s in scenes),CPU_s=sum(s['CPU_s']for s in scenes),wall_s=time.monotonic()-start,
        WM_updates=0,Actor_updates=0,target_integrations=0,new_independent_parents=0)
    write_json(Path(output)/'RESULT.json',result);return result
