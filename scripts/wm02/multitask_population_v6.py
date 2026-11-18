"""Freeze instruction-crossed scene populations before Teacher or learning."""
import copy
from itertools import product

from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS


# Fixed placement-body translations in metres. Every cluster retains its pair
# of destination regions. Each clock/reset identity is fresh and shared by the
# six instructions within a cluster.
OFFSETS = {
    'TEACHER': ((0.,0.),(-.010,-.006),(.010,-.006),(0.,.011)),
    'SCREEN': ((-.007,.005),(.009,.004)),
    'CONFIRM': ((.003,-.010),(-.004,.014)),
    'DEV': ((-.014,-.010),(-.013,.008),(.008,-.012),(.014,.011),(.005,.016)),
    'TEST': ((-.016,.003),(-.016,-.014),(.014,-.014),(.016,.002),(-.008,.017)),
}
STAGE_START = {'TEACHER':661000,'SCREEN':662000,'CONFIRM':663000,'DEV':664000,'TEST':665000}


def population(cards):
    if set(cards)!={'F1','F2'}:
        raise ValueError('BOTH_VISUAL_SCENE_PROTOTYPES_REQUIRED')
    matrix={stage:[] for stage in OFFSETS}
    clusters=[]
    for stage,offsets in OFFSETS.items():
        for family_index,family in enumerate(('F1','F2')):
            for index,(dx,dy) in enumerate(offsets):
                card=copy.deepcopy(cards[family])
                for xyz in card['object_positions_world'].values():
                    xyz[0]+=dx
                    xyz[1]+=dy
                cluster_id=f'M-{stage}-{family}-{index}'
                reset=STAGE_START[stage]+10*family_index+index
                clusters.append(dict(id=cluster_id,stage=stage,family=family,card=card,reset=reset))
                for cell,instruction in enumerate(INSTRUCTIONS):
                    matrix[stage].append(dict(id=f'{cluster_id}-{instruction}',cluster=cluster_id,
                        instruction=instruction,family=family,reset=reset,
                        noise_seed=2026111800+reset*10+cell,
                        causal_approach_perturbation_m=None))
    # One initial feasibility attempt per instruction; balance families.
    teachers=matrix['TEACHER']
    first=[next(r for r in teachers if r['instruction']==task and
                r['cluster']==f'M-TEACHER-{("F1","F2")[index%2]}-0')
           for index,task in enumerate(INSTRUCTIONS)]
    taken={r['id'] for r in first}
    matrix['TEACHER']=first+[r for r in teachers if r['id'] not in taken]
    for number,row in enumerate(matrix['SCREEN']):
        row['arm_order']=['M-BASE','M-SRC'] if number%2==0 else ['M-SRC','M-BASE']
    sizes={stage:len(rows) for stage,rows in matrix.items()}
    if sizes!={'TEACHER':48,'SCREEN':24,'CONFIRM':24,'DEV':60,'TEST':60}:
        raise ValueError('EXACT_V6_MULTITASK_MATRIX_REQUIRED')
    identities={(c['family'],tuple(tuple(xyz) for xyz in c['card']['object_positions_world'].values()))
                for c in clusters}
    if len(identities)!=len(clusters):
        raise ValueError('MULTITASK_LAYOUT_CROSSES_SPLITS_OR_REPEATS')
    return dict(schema='SLA_V6_MULTITASK_POPULATION_PROPOSED_GEOMETRY',clusters=clusters,
        conditions=matrix,first_feasibility_ids=[r['id'] for r in first],
        scene_instruction_independent=True,perturbed_initial_approaches=0,
        independent_DEV_TEST_scene_reset_clusters=10,selection_before_TEST=True,
        old_TEST_reused=False,native_dispatch_qualified=False)


def compile_population(spec):
    import json
    from pathlib import Path
    import time
    import mujoco

    from scripts.pi0292.durable import write
    from go2arx5_vla.world_model.multitask_scene_v6 import overlay_xml, robot_contract
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from go2arx5_vla.world_model.storage import digest
    from .multitask_binding_audit_v6 import pinned
    from .multitask_scene_audit_v6 import asset_binding

    if (spec['physics'] or spec['new_physical_requests_budget'] or spec['new_model_generations_budget']
            or mujoco.__version__!='3.2.7' or mujoco.get_mjcb_control() is not None):
        raise PermissionError('MULTITASK_POPULATION_NO_CONTROLLER_OR_MODEL_CALLS')
    root=Path(spec['registered_root'])
    original={}
    for family,ref in spec['templates'].items():
        template=json.loads(pinned(ref).read_bytes())
        template['asset_root'],_=asset_binding(template)
        old=template['model_path']
        if digest(old)!=template['frozen_files'][old]:
            raise ValueError('ORIGINAL_TRAIN_TEMPLATE_CHANGED')
        original[family]=(template,mujoco.MjModel.from_binary_path(old))
    result=population(spec['prototype_cards'])
    write(root/'GEOMETRY_POPULATION.private.json',result)
    seen=set()
    for index,cluster in enumerate(result['clusters']):
        if time.time()>=spec['deadline_epoch']-60 or (root/'STOP.json').exists():
            raise TimeoutError('FINITE_MULTITASK_COLD_POPULATION_LIMIT')
        template,base=original[cluster['family']]
        xml,binding=overlay_xml(template,base,cluster['card'])
        folder=root/cluster['id']
        folder.mkdir()
        with forbid_integration(mujoco) as calls:
            model=mujoco.MjModel.from_xml_string(xml)
            protected=robot_contract(base,model)
            path=folder/'SCENE.private.mjb'
            mujoco.mj_saveModel(model,str(path),None)
            if calls:
                raise RuntimeError('COLD_MULTITASK_POPULATION_ATTEMPTED_INTEGRATION')
        model_sha=digest(path)
        if model_sha in seen:
            raise ValueError('GEOMETRIC_SPLIT_SCENES_MUST_DIFFER')
        seen.add(model_sha)
        cluster.update(scene=dict(path=str(path),sha256=model_sha),scoring_geometry=binding,
                       robot_fingerprint=protected['fingerprint'])
        write(root/'PROGRESS.private.json',dict(compiled=index+1,total=len(result['clusters']),
            new_native_requests=0,new_integrations=0,new_generations=0))
    write(root/'COMPILED_POPULATION.private.json',result)
    return dict(status='PREREGISTERED_COLD_36_SCENE_POPULATION_READY',exit_code=0,
        compiled_clusters=len(result['clusters']),actual_scene_bytes=sum(
            Path(c['scene']['path']).stat().st_size for c in result['clusters']),
        registered_condition_counts={k:len(v) for k,v in result['conditions'].items()},
        first_teacher_feasibility=len(result['first_feasibility_ids']),
        protection_fingerprints=sorted({c['robot_fingerprint'] for c in result['clusters']}),
        population_sha256=digest(root/'COMPILED_POPULATION.private.json'),
        native_dispatch_qualified=False)


def main(spec):
    from pathlib import Path
    import time
    import traceback
    from scripts.pi0292.durable import write
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close

    start,cpu=time.monotonic(),time.process_time()
    try:
        result=compile_population(spec)
    except BaseException:
        result=dict(exit_code=1,status='PRESERVED_MULTITASK_POPULATION_TECHNICAL_FAILURE',
                    first_failure=traceback.format_exc())
    result.update(source_sha=spec['source_sha'],new_updates=0,new_native_requests=0,
                  new_integrations=0,new_generations=0,wall_s=time.monotonic()-start,
                  CPU_s=time.process_time()-cpu)
    write(Path(spec['registered_root'])/'RESULT.private.json',result)
    close(spec,result['exit_code'])
    return result['exit_code']
