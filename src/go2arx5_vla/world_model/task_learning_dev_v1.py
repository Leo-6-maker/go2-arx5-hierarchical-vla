"""New DEV layouts and four Teacher references; neither is TRAIN or native VLA DEV."""
from pathlib import Path
from . import task_learning_grasp_v1 as task

DEV_OFFSETS={f'DEV{i+1:02d}':(x,y)for i,(x,y)in enumerate(
    (x,y)for x in(-.003,-.001,.001,.003)for y in(-.003,-.0015,.0005,.0015,.003))}
REFERENCE_LAYOUTS=('DEV01','DEV05','DEV16','DEV20')
SCHEMA='TASK_LEARNING_FIXED_DEV_REFERENCE_PANEL_V1'


def purpose(role):
    task.need(role==task.DEV_REFERENCE_ROLE,'EXPLICIT_DEV_REFERENCE_ROLE')
    return 'DEV_REFERENCE_NOT_BC'


def experiment_role(role):
    purpose(role)
    return 'TASK_LEARNING_FIXED_DEV_TEACHER_REFERENCES_NO_BC'


def native_schema(role):
    purpose(role)
    return 'TASK_LEARNING_FIXED_DEV_REFERENCE_NATIVE_ACTUAL_V1'


def validate(panel):
    from .task_learning_scene_v1 import scene_id
    from .task_learning_collection_v1 import qualified_base
    from .task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER
    role=task.DEV_REFERENCE_ROLE;rows=panel.get('assignments')or[]
    task.need(panel.get('schema')==SCHEMA and panel.get('task_learning_role')==role
        and panel.get('protocol')==task.protocol(role)
        and len(panel.get('source_sha',''))==40
        and panel.get('Teacher_implementation_source_sections_sha256')==teacher_fingerprint()==FROZEN_TEACHER
        and len(rows)==4 and [r['layout']for r in rows]==list(REFERENCE_LAYOUTS)
        and len({r['native_run']for r in rows})==4,
        'FOUR_FIXED_NEW_DEV_REFERENCES_NOT_THE_FORTY_NATIVE_VLA_CONDITIONS')
    for row in rows:
        task.need(row['parent']==scene_id(row['layout'])and row['split']=='DEV'
            and row['restart']==1 and row['Teacher_mode']=='NORMAL_TEACHER_V7'
            and row['command']==task.COMMAND and row['purpose']==purpose(role)
            and Path(row['native_run']).name==row['native_run']and not row['native_run'].startswith('.')
            and len(row['model_sha256'])==64 and row.get('diagnostic_control')is None,
            'NEW_DEV_REFERENCE_CLOCK_TASK_AND_SPLIT_NO_TRAIN_RELABEL')
    qualified_base(panel['base_Teacher_qualification'])
    return rows


def classify(row,**kwargs):
    from .task_learning_teacher_panel_v1 import classify as original
    return original(row,expected_role=task.DEV_REFERENCE_ROLE,
        expected_native_schema=native_schema(task.DEV_REFERENCE_ROLE),**kwargs)


def summarize(panel,outcomes):
    rows=validate(panel);wanted={r['native_run']for r in rows}
    task.need(len({v['native_run']for v in outcomes})==len(outcomes)
        and all(v['native_run']in wanted for v in outcomes),'NO_DEV_REFERENCE_RETRY_OR_FOREIGN_RESULT')
    attempted=sum(v['attempted']is True for v in outcomes)
    ready=sum(v['reset_ready']is True for v in outcomes)
    success=sum(v['task_success']and v['reset_ready']and v['technical_valid']for v in outcomes)
    hard=sum(v['hard_safety_stop']is True for v in outcomes)
    technical=sum(v['technical_valid']is not True for v in outcomes)
    complete=len(outcomes)==attempted==4
    return dict(schema='TASK_LEARNING_FIXED_DEV_REFERENCE_SUMMARY_V1',requested=4,
        attempted=attempted,reset_ready=ready,task_successes=success,
        hard_safety_stop_conditions=hard,technical_failure_conditions=technical,
        complete=complete,passes=complete and ready==success==4 and hard==technical==0,
        qualified_training_demonstrations=0,GR00T_queries=0,model_updates=0,
        forty_condition_native_VLA_DEV='NOT_RUN_BY_TEACHER_REFERENCES')
