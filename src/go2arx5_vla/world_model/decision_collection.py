"""New bounded40-parent decision data; frozen physical predicates and original source driver."""
from pathlib import Path
import json,os,time,traceback
import numpy as np
from .decision_native import physical_anchor,local_guard,KINDS,qualify_scene
from .contact_identification import cards as old_cards
from .native_adapter import SourceAdapter
from .contracts import command_labels
from .storage import write_json,dump_private,digest


def cards():
    from scripts.pi0294.scene_cards import scene_group,task_card
    result=[]
    for i in range(40):
        group=scene_group('TRAIN','F0',110420+i)
        group.update(group_id=f'WM01/TRAIN/{100+i:03d}',scene_group=f'WM02-DECISION-V5/{i:02d}',split='TRAIN')
        card=task_card(group,'T1');card.update(parent_id=group['group_id'],ident_index=i,
            ident_partition='FIT'if i<24 else'DIAGNOSTIC_DEV'if i<32 else'ONCE_HOLDOUT',decision_data_version='V5')
        result.append(card)
    key=lambda c:tuple([*c['source_xy'],*c['goal_xy'],c['cube_yaw']])
    if len({key(c)for c in result})!=40 or {key(c)for c in result}&{key(c)for c in old_cards()}:
        raise ValueError('New physical parents must be unique and disjoint from old data')
    return result


def construct(job):
    spec=job['spec'];card=job['card'];folder=Path(job['out']);folder.mkdir();prefix=[];found={};adapter=None
    start=time.monotonic();cpu=time.process_time();failure=None;guard=local_guard(spec);live=0
    os.sched_setaffinity(0,{job['cpu']});write_json(folder/'CARD.json',card)
    try:
        adapter=SourceAdapter(card,spec['asset_root'],executor_id=spec['executor_id'],seed=card['seed'])
        context=np.array([*card['goal_xy'],.46,.44,.04])
        for macro in range(50):
            guard();plan=adapter.teacher_plan();state=adapter.snapshot(force=True)
            if len(prefix)>=8:
                for kind in KINDS:
                    if kind not in found and physical_anchor(kind,adapter,state,context):
                        path=folder/(kind+'-restore.pkl.gz');adapter.save_complete(path);found[kind]=str(path)
            if len(found)==3:break
            plan['action_raw'],ok=command_labels(plan['poses'][:8],plan['opening'][:8],state['tcp'])
            if not ok:raise ValueError('Original Teacher construction primitive exceeded')
            ref=adapter.save_complete(folder/f'prefix-{macro:02d}-restore.pkl.gz')
            if adapter.gate(plan)['decision']=='REJECT'or adapter.preflight(plan,guard)['decision']!='PASS':
                failure='ORIGINAL_PREFIX_GATE_OR_S1_REJECT';break
            for k in range(8):
                row=adapter.advance_recorded(plan,k,run_id=folder.name,actor_version='ORIGINAL_SOURCE_TEACHER_CONSTRUCTION',restore_ref=ref)
                prefix.append(row);live+=row['physics_ticks']
                if row['terminal']or row['truncated']:failure='NATIVE_PREFIX_TERMINAL';break
            if failure:break
    except BaseException:failure=traceback.format_exc()
    finally:
        shadow=adapter.shadow_ticks if adapter else 0
        if adapter:adapter.close()
        dump_private(folder/'PREFIX.pkl.gz',dict(rows=prefix))
    paths=[folder/'CARD.json',folder/'PREFIX.pkl.gz',*[Path(found[k])for k in KINDS if k in found]]
    entry=dict(scene=card['ident_index'],card=str(folder/'CARD.json'),prefix=str(folder/'PREFIX.pkl.gz'),
        restores=[found[k]for k in KINDS if k in found],files={str(p):digest(p)for p in paths})
    result=dict(entry=entry,prefix_transitions=len(prefix),live_physics_ticks=live,shadow_ticks=shadow,
        found_anchors=list(found),failure=failure,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(folder/'RESULT.json',result);return result


def scene(job):
    folder=Path(job['out']);folder.mkdir();prefix=construct(dict(job,out=str(folder/'construction')))
    branch=qualify_scene(dict(spec=job['spec'],entry=prefix['entry'],cpu=job['cpu'],out=str(folder/'branches')))
    anchors=branch['anchors'];main=sum(c['transitions']for a in anchors.values()for c in a.get('candidates',{}).values())
    # Controls are executed once per observed anchor. Their saved streams are the
    # authoritative count, including partial/rejected outcomes.
    from .paired_replay import read
    controls={n:sum(len(read(p)['rows'])for p in(folder/'branches').glob('*/'+n+'.pkl.gz'))for n in ('repeat','tail-only')}
    result=dict(scene=job['card']['ident_index'],partition=job['card']['ident_partition'],construction=prefix,branches=branch,
        main_transitions=main,repeat_transitions=controls['repeat'],tail_transitions=controls['tail-only'],
        prefix_transitions=prefix['prefix_transitions'],actual_transitions=prefix['prefix_transitions']+branch['actual_transitions'],
        live_physics_ticks=prefix['live_physics_ticks']+branch['live_physics_ticks'],shadow_ticks=prefix['shadow_ticks']+branch['shadow_ticks'],
        CPU_s=prefix['CPU_s']+branch['CPU_s'])
    write_json(folder/'RESULT.json',result);return result


def collect(spec,output):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    if spec.get('scene_indices')!=list(range(40)):raise ValueError('Freeze all40 intended parents including UNRUN')
    qualification=json.loads(Path(spec['B1_qualification']).read_text())
    if digest(spec['B1_qualification'])!=spec['B1_sha256']or qualification.get('all_scientific_anchors_qualified')is not True:
        raise ValueError('Actual previous B1 scientific qualification required')
    output=Path(output);start=time.monotonic();definitions=cards();write_json(output/'CARDS.json',definitions)
    jobs=[dict(spec=spec,card=c,cpu=spec['cpus'][i%len(spec['cpus'])],out=str(output/f'scene-{i:02d}'))for i,c in enumerate(definitions)]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'))as pool:
        scenes=[]
        for r in pool.map(scene,jobs):
            scenes.append(r)
            # Holdout outcomes stay sealed until a separately declared model endpoint.
            print(json.dumps(dict(scene=r['scene'],partition=r['partition'],actual_transitions=r['actual_transitions'])),flush=True)
    keys=('main_transitions','repeat_transitions','tail_transitions','prefix_transitions','actual_transitions','live_physics_ticks','shadow_ticks','CPU_s')
    result=dict(protocol='WM02_NEW_DECISION_DATA_V5',source_sha=spec['source_sha'],parents=40,partition_counts=[24,8,8],
        main_branch_denominator=600,main_transition_limit=9600,control_transition_limit=3840,
        **{k:sum(s[k]for s in scenes)for k in keys},wall_s=time.monotonic()-start,
        WM_updates=0,Actor_updates=0,target_integrations=0,full_tasks=0,holdout_outcomes='SEALED_NOT_EVALUATED',
        results=[dict(scene=s['scene'],partition=s['partition'],result=str(output/f"scene-{s['scene']:02d}/RESULT.json"))for s in scenes])
    write_json(output/'RESULT.json',result);return result
