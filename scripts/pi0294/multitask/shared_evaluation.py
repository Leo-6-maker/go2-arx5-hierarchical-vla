"""Original four-task DEV routing through the existing source capture loop.

This is source simulation. It neither submits work nor changes the frozen
WBC, target quarantine, scoring thresholds or original exposed DEV records.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .book_policy import Student,dev_card,episode
from .book_reset import digest,pristine_warmstart
from .shared_task_sampling import TASKS
from .raw_parent_delivery import PARENT_ID

PROTOCOL='SHARED4_RAW_RESET_DEV68_V1'
KIND='V7_SHARED4_FROZEN_DEV_CONDITION'
INFERENCE_KIND='V7_SHARED4_TRAIN16_INFERENCE_QUALIFICATION'
LIMITS=dict(model_loads=1,optimizer_updates=0,physics_integrations=18500,generation_queries=80,RGB_pairs=642,source_constructors=1)
INFERENCE_LIMITS=dict(model_loads=1,optimizer_updates=0,physics_integrations=0,generation_queries=16,RGB_pairs=0,source_constructors=0)
COUNTS=dict(T1=24,T2=24,BOOK=10,DOOR=10)
SOURCES=tuple('scripts/pi0294/multitask/'+n for n in (
    'shared_evaluation.py','book_policy.py','book_reset.py','capture_core.py',
    'source_engine.py','night_book_audit.py','cube_raw_score.py','cabinet_raw_checks.py'))+(
    'scripts/pi0294/scene.py','scripts/pi0294/score.py','scripts/pi0291_arx5_data/engine.py',
    'scripts/pi0291_arx5_data/config.py','scripts/pi0291_arx5_data/scene.py',
    'scripts/pi0294/multitask/libero_book_scene.py','scripts/pi0294/multitask/libero_book_assets.py',
    'scripts/pi0294/multitask/robocasa_cabinet_scene.py','scripts/pi0294/multitask/robocasa_cabinet_assets.py',
    'src/go2arx5_vla/target_sim/scorer.py')
NATIVE_CHECKS=('all_four_fresh_source_constructors','controller_and_IK_clear',
    'original_task_geometry_and_home','original_dual_cameras_all_four',
    'original_T1_T2_BOOK_noise_and_fixed_new_DOOR_noise','live_predecessor_rejected',
    'residual_writer_rejected','unknown_close_rejected','dirty_initial_state_rejected')


def checked_population(value):
    tasks=value.get('tasks',{})
    if (value.get('schema')!='SHARED4_COMPLETE_ORIGINAL_TASK_DEV_REFERENCE_V2'
            or set(tasks)!=set(TASKS)or {t:len(v)for t,v in tasks.items()}!=COUNTS
            or value.get('blind_TEST')is not False or value.get('DEV_used_for_statistics_or_optimizer')is not False):
        raise PermissionError('ALL_ORIGINAL68_EXPOSED_DEV_DEFINITIONS_WITHOUT_DOWNSAMPLING')
    for task,rows in tasks.items():
        if len({r['id']for r in rows})!=len(rows):raise PermissionError('ONE_TASK_CONDITION_ONCE')
        for r in rows:
            if r.get('task')!=task or r.get('study_split'if task=='DOOR'else'split')!='DEV':
                raise PermissionError('NO_TRAIN_OR_TEST_IN_SHARED_DEV')
            if task in('T1','T2')and(r.get('noise_bank')not in(294101,294102)
                    or r.get('noise_namespace')!='PI0294/PV1/DEV/N'+str(r['noise_bank'])):
                raise PermissionError('ORIGINAL_BOTH_T1_T2_NOISE_BANKS')
    if any({bank:sum(r['noise_bank']==bank for r in tasks[t])for bank in(294101,294102)}!={294101:12,294102:12}for t in('T1','T2')):
        raise PermissionError('PRESERVE_TWELVE_GEOMETRIES_AND_BOTH_NOISE_BANKS')
    return value


def plan(population,*,population_sha256,checkpoint_manifest_sha256,raw_zero_step=False):
    checked_population(population)
    if any(not isinstance(v,str)or len(v)!=64 or any(c not in'0123456789abcdef'for c in v)
           for v in(population_sha256,checkpoint_manifest_sha256)):
        raise ValueError('EXACT_POPULATION_AND_MODEL_MANIFESTS')
    if type(raw_zero_step)is not bool or raw_zero_step and checkpoint_manifest_sha256!=PARENT_ID:
        raise PermissionError('RAW_ZERO_STEP_IS_THE_ORIGINAL_SEVEN_FILE_PARENT')
    return dict(schema=PROTOCOL,task_DEV_population_sha256=population_sha256,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,raw_zero_step=raw_zero_step,
        conditions=[deepcopy(r)for t in TASKS for r in population['tasks'][t]],task_counts=COUNTS,
        warmup_s=5,task_duration_s=32,H40=40,execution_prefix=8,denoise_steps=4,
        one_registered_process_per_condition=True,same_condition_retry=False,
        original_results_and_STOP_preserved=True,exposed_DEV_not_blind_TEST=True,
        continuous_gripper_decode=True,privileged_student_inputs=False,
        target_WBC_or_hardware=False,safety_or_scoring_changes=False)


def condition_run_name(panel,index):
    return f"v7-shared4-{digest(panel)[:16]}-{index:03}"


def source_files():
    root=Path(__file__).resolve().parents[3]
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in SOURCES}


def validate_manifest(m,panel,population,qualification):
    if (panel!=plan(population,population_sha256=panel.get('task_DEV_population_sha256'),
                   checkpoint_manifest_sha256=panel.get('checkpoint_manifest_sha256'),raw_zero_step=panel.get('raw_zero_step'))
            or m.get('kind')!=KIND or m.get('limits')!=LIMITS
            or type(m.get('condition_index'))is not int or not 0<=m['condition_index']<68
            or m.get('run_uuid')!=condition_run_name(panel,m['condition_index'])
            or m.get('evaluation_checkpoint_manifest_sha256')!=panel['checkpoint_manifest_sha256']
            or m.get('raw_zero_step')!=panel['raw_zero_step']
            or m.get('sealed_inputs',{}).get('SHARED_DEV_PLAN.json')!=digest(panel)
            or m.get('sealed_inputs',{}).get('TASK_DEV_POPULATION.json')!=panel['task_DEV_population_sha256']
            or m.get('sealed_inputs',{}).get('TASK_NATIVE_QUALIFICATION.json')!=digest(qualification)):
        raise PermissionError('FIXED_RAW_MODEL_BY_ALL68_ORIGINAL_DEV_SINGLE_CONDITION')
    pin=m.get('sealed_inputs',{}).get('RESET_PREDECESSOR.json')
    templates=m.get('task_templates',{})
    if (not isinstance(pin,str)or len(pin)!=64 or any(c not in'0123456789abcdef'for c in pin)
            or set(templates)!=set(TASKS)or any(not isinstance(v.get('root'),str)
                or not isinstance(v.get('manifest_sha256'),str)or len(v['manifest_sha256'])!=64 for v in templates.values())):
        raise PermissionError('SEALED_PREDECESSOR_AND_ALL_FOUR_EXISTING_TASK_TEMPLATES')
    if (qualification.get('schema')!='SHARED4_SOURCE_RESET_NATIVE_QUALIFICATION_V1'
            or qualification.get('passed')is not True or qualification.get('uid')!=os.getuid()
            or qualification.get('host')!=os.uname().nodename or qualification.get('numpy_version')!=np.__version__
            or qualification.get('MuJoCo_version')!='3.2.7'
            or qualification.get('checks')!=dict.fromkeys(NATIVE_CHECKS,True)
            or qualification.get('source_files')!=source_files()
            or qualification.get('task_templates')!=m.get('task_templates')):
        raise PermissionError('ACTUAL_FOUR_TASK_CAMERA_RESET_AND_FAULT_QUALIFICATION')
    return panel['conditions'][m['condition_index']]


class SharedStudent(Student):
    def __init__(self,instance,condition,record,guard):
        self.condition=condition
        namespace=(condition['noise_namespace']if condition['task']in('T1','T2')else
                   'BOOK5_DEV10'if condition['task']=='BOOK'else'SHARED4_CUSTOM_PORT08_DEV_V1')
        super().__init__(instance,condition['id'],record,guard,seed_namespace=namespace)

    def generation_seed(self,sequence):
        c=self.condition
        key=c['scene_group']+'/'+c['task']if c['task']in('T1','T2')else c['id']
        return int.from_bytes(hashlib.sha256(f'{self.seed_namespace}/{key}/{sequence}'.encode()).digest()[:8],'big')%(2**63)


def card_for_condition(condition):
    return dev_card(condition)if condition['task']=='BOOK'else deepcopy(condition)


def make_engine(card,template):
    """Route to the original constructors; add only recorder measurements."""
    assets=json.loads((Path(template)/'ASSET_BINDINGS.private.json').read_bytes())
    if card['task']in('T1','T2'):
        import mujoco
        from scripts.pi0294.scene import make_engine as original
        engine=original(card,assets['robot_root'])
        base=engine._id(mujoco.mjtObj.mjOBJ_BODY,'base_link');engine.robot_bodies=set()
        for i in range(engine.model.nbody):
            parent=i
            while parent and parent!=base:parent=int(engine.model.body_parentid[parent])
            if parent==base:engine.robot_bodies.add(i)
        engine.obstacles=set()
        snapshot,render=engine.snapshot,engine.render
        def measured_snapshot():
            s=snapshot();s['robot_object_contact']=any(engine.geoms['cube_geom']in(c['geom0'],c['geom1'])
                and any(int(engine.model.geom_bodyid[g])in engine.robot_bodies for g in(c['geom0'],c['geom1']))for c in s['contacts'])
            return s
        def measured_render(state):
            value=render(state);engine.render_pairs=getattr(engine,'render_pairs',0)+1;return value
        engine.snapshot,engine.render=measured_snapshot,measured_render
        return engine
    from .source_engine import make_engine as source
    if card['task']=='BOOK':
        definitions={n:(Path(template)/'inputs'/n/(n+'.xml')).read_bytes()for n in('black_book','desk_caddy')}
        return source(card,assets['robot_root'],pinned_libero=dict(definitions=definitions,
            asset_paths={n:assets['files'][n]['path']for n in assets['libero_binary_names']}))
    if card['task']=='DOOR':
        definitions={n:Path(assets['files'][key]['path']).read_bytes()for n,key in assets['robocasa_definition_members'].items()}
        textures={n:assets['files'][key]['path']for n,key in assets['robocasa_texture_members'].items()}
        return source(card,assets['robot_root'],pinned_robocasa=dict(definitions=definitions,texture_paths=textures))
    raise PermissionError('ONLY_FOUR_DECLARED_SOURCE_TASKS')


def qualification_extra(engine,folder,images,calibration):
    if engine.card['task']in('T1','T2'):
        from scripts.pi0294.scene import rendered_field_of_view
        view=rendered_field_of_view(images,calibration,engine.card)
        if not view['passed']:raise PermissionError('ORIGINAL_CUBE_BOTH_CAMERAS_AND_FULL_TARGET_VIEW')
        return dict(original_cube_camera_qualification=view)
    if engine.card['task']!='DOOR':return {}
    import mujoco
    j=engine._id(mujoco.mjtObj.mjOBJ_JOINT,engine.task_binding['passive_joint'])
    return dict(source_port=engine.card['source_port'],
        card_sha256=hashlib.sha256((folder/'CARD.json').read_bytes()).hexdigest(),
        native_scene_sha256=hashlib.sha256((folder/'NATIVE_SCENE.private.xml').read_bytes()).hexdigest(),
        passive_joint_name=engine.task_binding['passive_joint'],passive_joint_qposadr=int(engine.model.jnt_qposadr[j]),
        passive_joint_dofadr=int(engine.model.jnt_dofadr[j]),passive_joint_range=engine.model.jnt_range[j].tolist())


def initial_witness(engine,student,card):
    if card['task']=='BOOK':
        from .book_reset import initial_witness as book_witness
        return dict(book_witness(engine,student,card),shared_protocol=PROTOCOL)
    from scripts.pi0291_arx5_data.config import PARAMETERS as P
    from scripts.pi0291_arx5_data.scene import DOWNWARD_ROTATION
    d,ik=engine.data,engine.ik_data;s=engine.snapshot();actual=engine.pose('tool')
    expected=np.eye(4);expected[:3,:3]=DOWNWARD_ROTATION;expected[:3,3]=[*card.get('source_xy',[0,0]),P['table_top']+.10]
    if 'initial_tcp'in card:expected=np.asarray(card['initial_tcp'])
    opening=P['finger_travel']if card['task']in('T1','T2')else .042
    angle=np.arccos(np.clip((np.trace(actual[:3,:3]@expected[:3,:3].T)-1)/2,-1.,1.))
    if (d is ik or d.time!=0 or ik.time!=0 or engine.steps!=0 or engine.renderer is not None or engine.safety
            or any(np.any(np.asarray(getattr(d,n))!=0)for n in('qvel','ctrl','act','qfrc_applied','xfrc_applied'))
            or np.any(ik.qvel!=0)or engine.ik_solve_calls!=0 or engine.ik_cache_hits!=0
            or any(getattr(engine,n)is not None for n in('_ik_target','_cached_ik_solution','_cached_ik_error'))
            or not np.array_equal(engine.arm_targets,d.qpos[engine.arm_q])
            or not np.array_equal(engine.finger_targets,np.full(2,opening))
            or not np.array_equal(d.qpos[engine.finger_q],np.full(2,opening))
            or s['safety_valid']is not True or s['native_grasp']
            or student.identifier!=card['id']or student.index!=-1 or student.last_plan!=-1
            or student.queries!=0 or student.completed_generations!=0 or student.state is not None or student.images is not None
            or np.linalg.norm(actual[:3,3]-expected[:3,3])>.0002 or angle>.002
            or not np.array_equal(d.qacc_warmstart,pristine_warmstart(engine))):
        raise PermissionError('SHARED_NEW_SOURCE_HOME_CONTROLLER_IK_AND_STUDENT_STATE')
    if card['task']in('T1','T2'):
        if not np.allclose(np.asarray(s['object_pose'])[:3,3],[*card['source_xy'],P['table_top']+.02],atol=1e-12,rtol=0):
            raise PermissionError('ORIGINAL_CUBE_GEOMETRY_INITIAL_POSITION')
    elif s['native_robocasa_door_status']['native_is_closed']is not True:
        raise PermissionError('ORIGINAL_CABINET_MUST_BEGIN_CLOSED')
    return dict(protocol=PROTOCOL,task=card['task'],steps=0,simulation_time_s=0.,
        controller_and_IK_clear=True,new_data_and_IK_data=True,initial_home_qualified=True,
        original_scene_sha256=hashlib.sha256(engine.xml.encode()).hexdigest(),
        first_generation_seed=student.generation_seed(0),seed_namespace=student.seed_namespace,
        qpos=np.asarray(d.qpos).tolist(),extra_physics_integrations=0,acceleration_cache_matches_pristine_native_forwards=True)


def run_condition(root,condition,instance,template,guard):
    from .night_book_audit import audit
    task=condition['task'];kind='ROBOCASA_CABINET'if task=='DOOR'else task
    def policy_audit(folder,**kwargs):
        value=audit(folder,task_kind=kind,**kwargs)
        return dict(value,schema='SHARED4_POLICY_CONDITION_RAW_AUDIT_V1',
            source_raw_integrity_status=value['status'],
            status='COMPLETE_SOURCE_POLICY_CONDITION'if value.get('complete_episode',False)else'SOURCE_POLICY_CONDITION_INCOMPLETE',
            qualified_source_pilot=False,training_admitted=False,expert_supervision_eligible=False,
            student_actions_are_not_expert_labels=True)
    row=episode(root,condition,instance,template,guard,
        student_factory=lambda inst,identifier,record,check:SharedStudent(inst,condition,record,check),
        card_factory=card_for_condition,engine_factory=make_engine,qualification_extra=qualification_extra,
        initial_witness=initial_witness,reset_protocol={'schema':PROTOCOL},
        warmup_opening=1. if task in('T1','T2')else .042/.044,
        raw_auditor=policy_audit)
    return dict(row,task=task,protocol=PROTOCOL)


def summarize(panel,rows):
    expected={(r['task'],r['id'])for r in panel['conditions']};seen=set();tasks={}
    for row in rows:
        key=(row['task'],row['id'])
        if key not in expected or key in seen:raise PermissionError('ONE_NEW_MODEL_CONDITION_RESULT_WITHOUT_RETRIES')
        seen.add(key)
        if row['success']and not row['valid_model_condition']:raise ValueError('SUCCESS_REQUIRES_VALID_CONDITION')
    for task in TASKS:
        subset=[r for r in rows if r['task']==task];valid=sum(bool(r['valid_model_condition'])for r in subset)
        successes=sum(bool(r['success'])for r in subset)
        tasks[task]=dict(planned=COUNTS[task],valid=valid,successes=successes,
            engineering_failures=len(subset)-valid,unrun=COUNTS[task]-len(subset),
            hard_safety_stops=sum(bool(r['hard_native_safety'])for r in subset),
            success_rate_among_valid=None if not valid else successes/valid)
    complete=all(t['unrun']==0 and t['engineering_failures']==0 for t in tasks.values())
    rates=[t['success_rate_among_valid']for t in tasks.values()]
    return dict(protocol=PROTOCOL,complete_all68=complete,tasks=tasks,
        macro_success_rate=None if not complete else sum(rates)/4,
        worst_task_success_rate=None if not complete else min(rates),
        exposed_DEV_not_independent_TEST=True,scientific_success_not_inferred_from_incomplete_panel=True)
