"""Finite S-domain paired contact interventions using the original native/S1 seam."""
from pathlib import Path
from copy import deepcopy
import json,os,time,traceback
import numpy as np
from .storage import write_json,dump_private,load_private,digest
from .native_adapter import SourceAdapter
from .contracts import command_labels,FRAME,TCP
from .action_compiler import compile_reference,SOURCE

ANCHORS=('precontact','grasp_onset','stable_grasp','supported_release')
CANDIDATES=('hold','nominal','left','right','grip_timing')
LO,HI=.002/.044,.042/.044


def cards():
    from scripts.pi0294.scene_cards import scene_group,task_card
    out=[]
    for i in range(24):
        group=scene_group('TRAIN','F0',110302+i)
        group.update(group_id=f'WM01/TRAIN/{i+2:03d}',scene_group=f'WM02-ID-V1/{i:02d}',split='TRAIN')
        c=task_card(group,'T1');c.update(parent_id=group['group_id'],ident_index=i,
            ident_partition='FIT'if i<16 else'DIAGNOSTIC_DEV'if i<20 else'ONCE_HOLDOUT')
        out.append(c)
    keys=[tuple([*c['source_xy'],*c['goal_xy'],c['cube_yaw']])for c in out]
    if len(set(keys))!=24:raise ValueError('Independent physical layout identities required')
    return out


def physical_gripper(logical):return LO+(HI-LO)*np.asarray(logical)


def candidates(nominal_raw,last_opening,state):
    raw=np.asarray(nominal_raw,dtype=float).copy();raw[:,6]=physical_gripper(raw[:,6])
    old=float(physical_gripper(last_opening));hold=np.zeros((8,7));hold[:,6]=old
    out={'hold':hold,'nominal':raw.copy(),'left':raw.copy(),'right':raw.copy(),'grip_timing':raw.copy()}
    out['left'][:,0]-=.0005;out['right'][:,0]+=.0005
    if abs(raw[-1,6]-old)>.05:out['grip_timing'][:4,6]=old
    elif state['native_grasp']:out['grip_timing'][:,6]=.25
    for r in out.values():
        if (np.linalg.norm(r[:,:3],axis=-1)>.01+1e-12).any()or(np.linalg.norm(r[:,3:6],axis=-1)>.05+1e-12).any():raise ValueError('Candidate construction exceeds original primitive; no clipping')
    return out


def supported(adapter,state):
    table=adapter.engine.geoms['table'];cube=adapter.engine.geoms['cube_geom']
    return any({c['geom0'],c['geom1']}=={table,cube}and c['normal_force_N']>=.05 for c in state['contacts'])


def anchor_due(name,adapter,state):
    stage=adapter.teacher.stage;grasp=state['native_grasp'];height=state['object_pose'][2,3]
    if name=='precontact':return stage=='align'and not grasp and state['tcp'][2,3]-height<=.03
    if name=='grasp_onset':return grasp and stage in('close','verify_grasp')
    if name=='stable_grasp':return grasp and height>=.48 and stage in('lift','carry')
    if name=='supported_release':return stage=='release'and supported(adapter,state)
    raise ValueError('Unknown anchor')


def execute(adapter,raw,folder,restore,run_id,guard,*,tail=False):
    rows=[];events=[];shadow0=adapter.shadow_ticks
    for block in range(2):
        guard();state=adapter.snapshot(force=True)
        action=raw if block==0 else np.c_[np.zeros((8,6)),np.full(8,raw[-1,6])]
        plan=compile_reference(action,state['tcp'],state['time_s'],backend=SOURCE,frame=FRAME,tcp=TCP)
        if tail:
            # No consumed row is changed; preserve rotations and gripper.
            delta=.002*np.sin(np.linspace(0,np.pi,32))
            plan['poses'][8:,0,3]+=delta;plan['rows10'][8:,0]+=delta
        gate=adapter.gate(plan)
        prediction=adapter.preflight(plan,guard)if gate['decision']!='REJECT'else None
        if gate['decision']=='REJECT'or prediction['decision']!='PASS':
            events.append(dict(block=block,reason='ORIGINAL_GATE_OR_S1_REJECT',gate=gate,preflight=prediction,physics_ticks=0));break
        for k in range(8):
            row=adapter.advance_recorded(plan,k,run_id=run_id,actor_version='WM02_CONTACT_ID_V1',restore_ref=restore)
            rows.append(row)
            if row['terminal']or row['truncated']:break
        if rows and(rows[-1]['terminal']or rows[-1]['truncated']):break
    final=adapter.capture();dump_private(folder,dict(rows=rows,events=events,final_restore=final))
    return dict(rows=rows,events=events,shadow_ticks=adapter.shadow_ticks-shadow0,
        physics_ticks=sum(r['physics_ticks']for r in rows),final_restore=final)


def compare_repeats(a,b,first_restore=None,second_restore=None):
    if len(a)!=len(b):return dict(qualified=False,reason='LENGTH_DIFFERS')
    error=max([float(np.max(np.abs(x['native_after']['integration']-y['native_after']['integration'])))for x,y in zip(a,b)]or[0.])
    contacts=all(x['state_after']['contacts']==y['state_after']['contacts']for x,y in zip(a,b))
    terminal=all((x['terminal'],x['truncated'],x['safety_reason'])==(y['terminal'],y['truncated'],y['safety_reason'])for x,y in zip(a,b))
    from .restore_audit import compare
    controller=compare([r.get('controller_memory_after')for r in a],[r.get('controller_memory_after')for r in b])
    complete=compare(first_restore,second_restore)if first_restore is not None else {'passed':False,'mismatches':['NO_COMPLETE_RESTORE_PROOF'],'max_abs_error':0.}
    return dict(qualified=len(a)==16 and error<=1e-10 and contacts and terminal and controller['passed']and complete['passed'],
        max_integration_error=error,contacts_equal=contacts,terminal_equal=terminal,transitions_per_execution=len(a),
        controller_comparison=controller,complete_snapshot_comparison=complete)


def sample_scene(job):
    os.sched_setaffinity(0,{job['cpu']});spec=job['spec'];card=job['card'];folder=Path(job['folder']);folder.mkdir()
    began=time.monotonic();cpu=time.process_time();adapter=None;prefix=[];results={};live=shadow=0;failure=None
    def guard():
        if time.time()>spec['deadline_epoch']or(Path(spec['registered_root'])/'STOP.json').exists():raise TimeoutError('FINITE_CONTACT_IDENTIFICATION')
    try:
        adapter=SourceAdapter(card,spec['asset_root'],executor_id=spec['executor_id'],seed=card['seed'])
        write_json(folder/'CARD.json',card)
        for macro in range(50):
            guard();plan=adapter.teacher_plan();state=adapter.snapshot()
            # Teacher remains construction-only; every candidate shares this exact state/history.
            if len(prefix)>=8:
                for name in ANCHORS:
                    if name in results or not anchor_due(name,adapter,state):continue
                    parent=folder/name;parent.mkdir();ref=adapter.save_complete(parent/'common-restore.pkl.gz');saved=load_private(ref['path'])
                    raw,ok=command_labels(plan['poses'][:8],plan['opening'][:8],state['tcp'])
                    if not ok:raise ValueError('Original nominal not representable')
                    initial_opening=float(adapter.last_command[-1])if adapter.last_command is not None else state['gripper_fraction']
                    interventions=candidates(raw,initial_opening,state);runs={}
                    dump_private(parent/'COMMON_HISTORY.pkl.gz',dict(rows=prefix[-8:],state=state,restore_ref=ref))
                    for candidate in CANDIDATES:
                        adapter.restore(saved)
                        v=execute(adapter,interventions[candidate],parent/(candidate+'.pkl.gz'),ref,job['run_id'],guard)
                        live+=v['physics_ticks'];shadow+=v['shadow_ticks'];runs[candidate]=v
                    adapter.restore(saved);repeat=execute(adapter,interventions['nominal'],parent/'repeat.pkl.gz',ref,job['run_id'],guard)
                    live+=repeat['physics_ticks'];shadow+=repeat['shadow_ticks'];repeatability=compare_repeats(runs['nominal']['rows'],repeat['rows'],runs['nominal']['final_restore'],repeat['final_restore'])
                    adapter.restore(saved);negative=execute(adapter,interventions['nominal'],parent/'tail-only.pkl.gz',ref,job['run_id'],guard,tail=True)
                    live+=negative['physics_ticks'];shadow+=negative['shadow_ticks'];tail_equal=compare_repeats(runs['nominal']['rows'],negative['rows'],runs['nominal']['final_restore'],negative['final_restore'])
                    result=dict(anchor=name,time_s=state['time_s'],native_grasp=state['native_grasp'],supported=supported(adapter,state),
                        main_transitions=sum(len(v['rows'])for v in runs.values()),repeat_transitions=len(repeat['rows']),tail_transitions=len(negative['rows']),
                        candidates={k:dict(transitions=len(v['rows']),rejections=len(v['events']),terminal=bool(v['rows']and v['rows'][-1]['terminal']))for k,v in runs.items()},
                        repeatability=repeatability,tail_physical_equivalence=tail_equal,full_task_success_claimed=False)
                    write_json(parent/'RESULT.json',result);results[name]=result
                    adapter.restore(saved)
                    if not repeatability['qualified']and len(runs['nominal']['rows'])==16:raise RuntimeError('ACTUAL_COMPLETE_RESTORE_REPEAT_MISMATCH')
                    if not tail_equal['qualified']and len(runs['nominal']['rows'])==16 and len(negative['rows'])==16:raise RuntimeError('SOURCE_UNCONSUMED_TAIL_CHANGED_PHYSICS')
                if len(results)==4:break
            # Restore original Teacher plan after each complete branch; never feed branch outcomes to it.
            plan['action_raw'],ok=command_labels(plan['poses'][:8],plan['opening'][:8],adapter.snapshot()['tcp'])
            if not ok:raise ValueError('Teacher construction primitive exceeded')
            ref=adapter.save_complete(folder/f'prefix-{macro:02d}-restore.pkl.gz')
            if adapter.gate(plan)['decision']=='REJECT'or adapter.preflight(plan,guard)['decision']!='PASS':
                failure='ORIGINAL_PREFIX_GATE_OR_S1_REJECT';break
            for k in range(8):
                row=adapter.advance_recorded(plan,k,run_id=job['run_id'],actor_version='ORIGINAL_SOURCE_TEACHER_CONSTRUCTION',restore_ref=ref)
                prefix.append(row);live+=row['physics_ticks']
                if row['terminal']or row['truncated']:failure='NATIVE_PREFIX_TERMINAL';break
            if failure:break
    except BaseException:
        failure=traceback.format_exc()
    finally:
        if adapter is not None:
            total_shadow=adapter.shadow_ticks;adapter.close()
        else:total_shadow=0
        dump_private(folder/'PREFIX.pkl.gz',dict(rows=prefix))
    report=dict(scene_index=card['ident_index'],original_parent=card['parent_id'],partition=card['ident_partition'],anchors=results,
        missing_anchors=[n for n in ANCHORS if n not in results],prefix_transitions=len(prefix),
        main_transitions=sum(v['main_transitions']for v in results.values()),
        repeat_transitions=sum(v['repeat_transitions']for v in results.values()),tail_transitions=sum(v['tail_transitions']for v in results.values()),
        live_physics_ticks=live,shadow_ticks=total_shadow,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-began,failure=failure)
    write_json(folder/'RESULT.json',report);return report


def collect(spec,output):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    output=Path(output);began=time.monotonic();jobs=[]
    definitions=cards();write_json(output/'CARDS.json',definitions)
    indices=spec.get('scene_indices',list(range(24)))
    if not indices or len(set(indices))!=len(indices)or any(type(i)is not int or i not in range(24)for i in indices):raise ValueError('Exact predeclared scene indices required')
    definitions=[definitions[i]for i in indices]
    for i,card in enumerate(definitions):jobs.append(dict(spec=spec,card=card,cpu=spec['cpus'][i%len(spec['cpus'])],folder=str(output/f"scene-{card['ident_index']:02d}"),run_id=output.name))
    # Each batch has four independently owned engines; there is one registered parent owner.
    with ProcessPoolExecutor(max_workers=min(4,len(jobs)),mp_context=multiprocessing.get_context('spawn'))as pool:
        reports=[]
        for report in pool.map(sample_scene,jobs):
            reports.append(report);print(json.dumps(report),flush=True)
    report=dict(version='WM02_S_CONTACT_IDENTIFICATION_V1',source_sha=spec['source_sha'],scenes=reports,scene_attempts=len(indices),scene_indices=indices,
        source_native_transitions=sum(r['prefix_transitions']+r['main_transitions']+r['repeat_transitions']+r['tail_transitions']for r in reports),
        main_branch_transitions=sum(r['main_transitions']for r in reports),prefix_transitions=sum(r['prefix_transitions']for r in reports),
        repeat_transitions=sum(r['repeat_transitions']for r in reports),tail_transitions=sum(r['tail_transitions']for r in reports),
        live_physics_ticks=sum(r['live_physics_ticks']for r in reports),shadow_ticks=sum(r['shadow_ticks']for r in reports),
        CPU_s=sum(r['CPU_s']for r in reports),wall_s=time.monotonic()-began,target_transitions=0,Actor_updates=0)
    write_json(output/'RESULT.json',report);return report
