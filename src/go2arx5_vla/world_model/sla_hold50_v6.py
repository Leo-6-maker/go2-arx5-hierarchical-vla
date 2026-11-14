"""Prospective fixed50 PICK_HOLD strata;not a completed two-task D100 dataset."""
import json
from pathlib import Path
from . import task_learning_grasp_v1 as task,sla_target_coverage_v5 as coverage
from .storage import digest
from .stable_task_scene_v1 import pinned

VERSION='SLA_D100_PROSPECTIVE_PICK_HOLD50_V6'
PANEL=VERSION+'_PANEL'
DECLARATION='experiments/sla_go2_posttrain/prospective_hold50_v6.json'
RESTART=13
REPAIRED_OBJECTS=('BLUE_BOX50_30_40','GREEN_CUBE35','PURPLE_BOX45_35_50')


def configuration(spec):
    v=spec['sla_five_object_hold_v5']
    task.need(v==dict(version=VERSION,object_id=v.get('object_id'),pose_id=v.get('pose_id'))
        and v['object_id']in coverage.OBJECTS and v['pose_id']in coverage.POSES
        and spec.get('task_learning_role')=='CORELAB_EXPERT_TRAIN4_SEMANTICS_NOT_BC_V3'
        and spec.get('task_learning_teacher_mode')=='NORMAL_TEACHER_V7'
        and spec.get('task_learning_control')is None and not spec.get('target_gr00t_policy')
        and spec.get('sla_pick_place_v5')is None,'PROSPECTIVE_HOLD50_EXACT_OBJECT_POSE_EXPERT_ONLY')
    return True


def command(object_id):
    return 'GRASP_'+object_id+'_LIFT20MM_HOLD1S_PROSPECTIVE_V6'


def declaration():
    p=Path(__file__).resolve().parents[3]/DECLARATION;d=json.loads(p.read_bytes())
    expected=[dict(object_id=o,pose_id=p,restart=RESTART)for p in coverage.POSES for o in coverage.OBJECTS]
    task.need(d['schema']==VERSION and d['fixed_plan']=='f61305f9c1f847e460851fd067892160f71ca432'
        and d['conditions']==expected and d['requested']==50 and d['maximum_actors']==4
        and d['old_episodes_relabelled']is False and d['completed_D100_claim']is False
        and d['failed_condition_retries']is False and d['source_geometry_version']==coverage.VERSION
        and d['unchanged_task_score']==task.score_contract(),
        'ALL_FIFTY_PREDECLARED_STRATA_WITH_ORIGINAL_SCORE_NOT_RENAMED_OLD_DATA')
    return d,dict(path=str(p),sha256=digest(p))


def protocol(spec):
    configuration(spec)
    from . import sla_five_object_hold_v5 as five
    v=spec['sla_five_object_hold_v5'];repair=v['object_id']in REPAIRED_OBJECTS
    return dict(task.protocol(task.CONSTRUCTION_ROLE),schema=VERSION,stage=five.ROLE,
        task_revision=VERSION,task_command=command(v['object_id']),language=five.language(spec),
        scene_family='PROSPECTIVE_FIXED_TEN_POSE_FIVE_OBJECT_PICK_HOLD_COMPONENT',
        object_id=v['object_id'],pose_id=v['pose_id'],prefix_atoms=10,policy_queries=60,final_hold_atoms=40,
        expert_consumer='EEF_STREAM_V2',reference_TTL_s=.4,reference_renewal_max_s=.35,
        expert_recorded_success_tail='CORELAB_ACTUAL_TWO_SECOND_EXPERT_SUCCESS_TAIL_V3',
        prospective_D100_component='PICK_HOLD50',D100_new=False,PICK_PLACE_admitted=False,
        original_score_unchanged=True,original_grasp_Teacher_and_score_unchanged=not repair,
        grasp_goal_delta_world_m=five.REPAIRED_GOAL_DELTA_WORLD_M if repair else[0.,0.,0.],
        original_three_contact_rejections_retained=True)


def make_teacher(spec,initial):
    configuration(spec)
    from .sla_five_object_hold_v5 import GeometryTeacher
    return (GeometryTeacher if spec['sla_five_object_hold_v5']['object_id']in REPAIRED_OBJECTS else task.Teacher)(initial)


def validate_panel(panel):
    from .task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER
    from .task_learning_collection_v1 import qualified_base
    from .sla_five_object_hold_v5 import scene_id
    d,pin=declaration();rows=panel.get('assignments')or[]
    task.need(panel.get('schema')==PANEL and panel['expert_version']==VERSION
        and panel['source_sha']and panel['tasks']==['PICK_HOLD']and panel['D100_new']is False
        and panel['prospective_D100_component']=='PICK_HOLD50'
        and panel['old_episodes_relabelled']is False
        and panel['base_grasp_Teacher_sha256']==teacher_fingerprint()==FROZEN_TEACHER
        and panel['preregistration']==pin and pinned(pin)==d
        and len(rows)==len({r['native_run']for r in rows})==50,
        'HOLD50_FIXED_PROSPECTIVE_POPULATION_AND_UNCHANGED_QUALIFIED_TEACHER')
    for r,e in zip(rows,d['conditions'],strict=True):
        task.need(all(r[k]==e[k]for k in('object_id','pose_id','restart'))
            and r['parent']==scene_id(r['object_id'],r['pose_id'])and r['split']=='TRAIN'
            and r['layout']==r['object_id']+'-'+r['pose_id']and r['prefix_atoms']==10
            and r['command']==command(r['object_id'])and len(r['model_sha256'])==64
            and Path(r['native_run']).name==r['native_run']and not r['native_run'].startswith('.'),
            'HOLD50_NO_STRATUM_SUBSTITUTION_OR_FAILED_ROW_REPLAY')
    qualified_base(panel['base_Teacher_qualification'])
    return rows
