"""Frozen twenty-request Teacher gate; construction success is not qualification."""
import ast
import hashlib
from pathlib import Path
from . import task_learning_grasp_v1 as task
from .task_learning_scene_v1 import QUALIFICATION_OFFSETS,scene_id

SCHEMA='TASK_LEARNING_TEACHER_QUALIFICATION_PANEL_V1'
FROZEN_TEACHER='1bc18176e3c97d33058309c7ff341398c50d097bbe7e42b91b52ab2db9984081'


def teacher_fingerprint():
    raw=Path(task.__file__).read_text();lines=raw.splitlines(keepends=True);tree=ast.parse(raw)
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name in('Teacher','FiftyMsProgram','need')]
    # Hash exact frozen source sections rather than Python-version-specific
    # AST fields (the Core native and local test interpreters differ).
    sections='\n'.join(''.join(lines[n.lineno-1:n.end_lineno]).rstrip('\n')for n in nodes)
    return hashlib.sha256(sections.encode()).hexdigest()


def validate(panel):
    rows=panel.get('assignments',[])
    task.need(panel.get('schema')==SCHEMA and panel.get('protocol')==task.protocol(task.QUALIFICATION_ROLE)
        and panel.get('Teacher_implementation_source_sections_sha256')==teacher_fingerprint()==FROZEN_TEACHER
        and len(rows)==20 and len({r['native_run']for r in rows})==20,
        'FROZEN_V7_TEACHER_AND_TWENTY_DISTINCT_REQUESTS')
    for layout in QUALIFICATION_OFFSETS:
        matches=[r for r in rows if r['layout']==layout]
        task.need(len(matches)==2 and {r['restart']for r in matches}=={1,2},'TEN_LAYOUTS_TWO_INDEPENDENT_RESTARTS')
    for row in rows:
        task.need(row['parent']==scene_id(row['layout'])and row['split']=='TRAIN'
            and row['command']==task.COMMAND and row['purpose']=='TEACHER_QUALIFICATION_NOT_BC'
            and Path(row['native_run']).name==row['native_run']and not row['native_run'].startswith('.')
            and row.get('diagnostic_control')is None and len(row['model_sha256'])==64,
            'QUALIFICATION_POPULATION_NEVER_CONSTRUCTION_OR_BC')
    return rows


def classify(row,*,native,worker,lease,readiness,score,spec,expected_source,attempted,
             expected_role=task.QUALIFICATION_ROLE,
             expected_native_schema='TASK_LEARNING_TEACHER_QUALIFICATION_NATIVE_ACTUAL_V1'):
    identity=(spec.get('task_learning_role')==expected_role
        and spec.get('source_sha')==expected_source and spec.get('phase')==task.PHASE)
    children=native.get('owned_process_exits')or[]
    clean=(native.get('shutdown_clean')is True and len(children)==4
        and {p.get('name')for p in children}=={'sim','solver','controller','planner'}
        and all(p.get('exit_code')==0 for p in children))
    failure=native.get('failure')or''
    known=any(v in failure for v in('POST_HOLD_START_READINESS_REJECTED','ORIGINAL_STEADY_PREFIX_FAILED_NO_NEW_ACTION',
        'NEW_TASK_STANDING_SCORE_BOUND_EXCEEDED','ORIGINAL_FULL_NATIVE_CONTACT_REJECT','ORIGINAL_NATIVE_JOINT_LIMIT_REJECT',
        'ORIGINAL_FIVE_ITERATION_NATIVE_IK_REJECT','ORIGINAL_JOINT_SPEED_REJECT','ORIGINAL_JOINT_ACCELERATION_REJECT',
        'NATIVE_WBC_FEEDBACK_CLOCK_NOT_READY_AT_PAUSE'))
    completed=(native.get('status')=='NATIVE_STAGE_PASSED'and native.get('actual_target_integrations')==task.E
        and native.get('reference_qualification',{}).get('schema')==expected_native_schema)
    technical=bool(identity and clean and worker and lease.get('status')=='CLOSED'
        and(completed or known or native.get('physics_stops')))
    ready=readiness.get('passes')is True and readiness.get('native_step')==task.B
    success=bool(technical and completed and ready and not native.get('physics_stops')
        and score.get('task_success')is True and worker.get('exit_code')==0)
    return dict(row,attempted=attempted,reset_ready=ready,task_success=success,technical_valid=technical,
        hard_safety_stop=bool(native.get('physics_stops')),native_status=native.get('status'),
        actual_native_integrations=native.get('actual_target_integrations',0),worker_exit_code=worker.get('exit_code'),
        first_failure=failure.splitlines()[-1]if failure else None,wall_s=worker.get('wall_s'),
        CPU_s=worker.get('CPU_s',0)+worker.get('child_CPU_s',0),score=score)


def summarize(panel,outcomes):
    rows=validate(panel);wanted={r['native_run']for r in rows}
    task.need(len({v['native_run']for v in outcomes})==len(outcomes)
        and all(v['native_run']in wanted for v in outcomes),'NO_RETRIED_OR_UNREGISTERED_QUALIFICATION_RESULT')
    attempted=sum(v['attempted']is True for v in outcomes)
    ready=sum(v['reset_ready']is True for v in outcomes)
    successes=sum(v['task_success']is True and v['reset_ready']is True and v['technical_valid']is True for v in outcomes)
    hard=sum(v['hard_safety_stop']is True for v in outcomes)
    technical=sum(v['technical_valid']is not True for v in outcomes)
    complete=len(outcomes)==attempted==20
    return dict(schema='TASK_LEARNING_TEACHER_QUALIFICATION_SUMMARY_V1',requested=20,attempted=attempted,
        reset_ready=ready,task_successes=successes,hard_safety_stop_conditions=hard,technical_failure_conditions=technical,
        P_reset_ready=ready/20,P_success_given_reset_ready=successes/ready if ready else None,
        P_success_per_requested_episode=successes/20,complete=complete,
        thresholds=dict(minimum_reset_ready=18,minimum_conditional_Teacher_success=.9,maximum_hard_safety_stops=0),
        passes=complete and technical==hard==0 and ready>=18 and successes*10>=ready*9,
        qualified_training_demonstrations=0,GR00T_queries=0,model_updates=0,
        independent_restarts_not_bit_exact_forks=True)
