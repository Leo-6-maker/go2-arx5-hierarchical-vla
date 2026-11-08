"""Fixed new TRAIN population and recovery checks; no diagnostic relabeling."""
from pathlib import Path
from . import task_learning_grasp_v1 as task
from . import task_learning_teacher_panel_v1 as teacher
from . import task_learning_scene_v1 as scenes
from . import task_learning_recovery_v1 as recovery
from .stable_task_scene_v1 import pinned
from .storage import digest

SCHEMA='TASK_LEARNING_CONTINUOUS_TEACHER_PANEL_V1'


def purpose(role):
    return {task.RECOVERY_QUALIFICATION_ROLE:'UNDER_CLOSE_RECOVERY_QUALIFICATION_NOT_BC',
        task.COLLECTION_ROLE:'CONTINUOUS_TEACHER_DEMONSTRATION'}[role]


def experiment_role(role):
    return {task.RECOVERY_QUALIFICATION_ROLE:'TASK_LEARNING_RECOVERY_QUALIFICATION_NO_BC',
        task.COLLECTION_ROLE:'TASK_LEARNING_CONTINUOUS_COLLECTION_PENDING_EXPORT'}[role]


def native_schema(role):
    return {task.RECOVERY_QUALIFICATION_ROLE:'TASK_LEARNING_RECOVERY_QUALIFICATION_NATIVE_ACTUAL_V1',
        task.COLLECTION_ROLE:'TASK_LEARNING_CONTINUOUS_TEACHER_NATIVE_ACTUAL_V1'}[role]


def expected_assignments(role):
    if role==task.RECOVERY_QUALIFICATION_ROLE:
        return [(layout,1,recovery.UNDER_CLOSE)for layout in scenes.RECOVERY_OFFSETS]
    task.need(role==task.COLLECTION_ROLE,'EXPLICIT_NEW_CONTINUOUS_TRAIN_POPULATION')
    return [(layout,restart,recovery.NORMAL if restart<=2 else recovery.UNDER_CLOSE)
        for restart in range(1,5)for layout in scenes.TRAIN_OFFSETS]


def qualified_base(descriptor):
    result=pinned(descriptor['result']);lease=pinned(descriptor['lease']);panel=pinned(descriptor['panel'])
    teacher.validate(panel);s=result.get('summary')or{}
    task.need(Path(descriptor['result']['path']).parent==Path(descriptor['panel']['path']).parent
        ==Path(descriptor['lease']['path']).parent.parent
        and lease.get('status')=='CLOSED'and lease.get('worker_exit_code')==0
        and result.get('status')=='TEACHER_QUALIFICATION_COMPLETE'and result.get('exit_code')==0
        and s.get('schema')=='TASK_LEARNING_TEACHER_QUALIFICATION_SUMMARY_V1'
        and s.get('complete')is True and s.get('passes')is True
        and s.get('requested')==s.get('attempted')==20 and s.get('reset_ready',0)>=18
        and s.get('task_successes',0)*10>=s.get('reset_ready',0)*9
        and s.get('hard_safety_stop_conditions')==s.get('technical_failure_conditions')==0,
        'ACTUAL_CLOSED_TWENTY_REQUEST_TEACHER_GATE_BEFORE_RECOVERY_OR_DATA')


def validate(panel):
    role=panel.get('task_learning_role');expected=expected_assignments(role);rows=panel.get('assignments')or[]
    task.need(panel.get('schema')==SCHEMA and panel.get('protocol')==task.protocol(role)
        and len(panel.get('source_sha',''))==40
        and panel.get('Teacher_implementation_source_sections_sha256')==teacher.teacher_fingerprint()==teacher.FROZEN_TEACHER
        and panel.get('recovery_implementation_sha256')==digest(recovery.__file__)
        and len(rows)==len(expected)and len({r['native_run']for r in rows})==len(rows)
        and [(r['layout'],r['restart'],r['Teacher_mode'])for r in rows]==expected,
        'FROZEN_NEW_LAYOUTS_ORDER_MODES_AND_SEPARATE_SOURCE_ROLES')
    for row in rows:
        task.need(row['parent']==scenes.scene_id(row['layout'])and row['split']=='TRAIN'
            and row['command']==task.COMMAND and row['purpose']==purpose(role)
            and Path(row['native_run']).name==row['native_run']and not row['native_run'].startswith('.')
            and len(row['model_sha256'])==64 and row.get('diagnostic_control')is None,
            'ONE_NEW_TRAIN_OR_RECOVERY_REQUEST_NEVER_OLD_DIAGNOSTIC')
    qualified_base(panel['base_Teacher_qualification'])
    if role==task.COLLECTION_ROLE:
        proof=panel['recovery_qualification'];result=pinned(proof['result']);lease=pinned(proof['lease'])
        recovery_panel=pinned(proof['panel'])
        task.need(recovery_panel.get('task_learning_role')==task.RECOVERY_QUALIFICATION_ROLE,
            'RECOVERY_PROOF_CANNOT_POINT_TO_COLLECTION_OR_ITSELF')
        validate(recovery_panel)
        s=result.get('summary')or{}
        task.need(recovery_panel['task_learning_role']==task.RECOVERY_QUALIFICATION_ROLE
            and result.get('panel_sha256')==proof['panel']['sha256']
            and result.get('source_sha')==recovery_panel['source_sha']
            and Path(proof['result']['path']).parent==Path(proof['panel']['path']).parent
            ==Path(proof['lease']['path']).parent.parent
            and lease.get('status')=='CLOSED'and lease.get('worker_exit_code')==0
            and result.get('exit_code')==0 and s.get('schema')=='TASK_LEARNING_RECOVERY_QUALIFICATION_SUMMARY_V1'
            and s.get('complete')is True and s.get('passes')is True
            and s.get('requested')==s.get('attempted')==s.get('reset_ready')==s.get('genuine_recovery_successes')==4
            and s.get('hard_safety_stop_conditions')==s.get('technical_failure_conditions')==0,
            'ACTUAL_FOUR_FRESH_UNDER_CLOSE_RECOVERIES_BEFORE_FORMAL_DATA')
    return rows


def classify(row,*,native,worker,lease,readiness,score,spec,expected_source,attempted):
    role=spec.get('task_learning_role')
    value=teacher.classify(row,native=native,worker=worker,lease=lease,readiness=readiness,
        score=score,spec=spec,expected_source=expected_source,attempted=attempted,
        expected_role=role,expected_native_schema=native_schema(role))
    q=native.get('reference_qualification')or{};e=q.get('recovery_evidence')or{}
    genuine=bool(value['task_success']and row['Teacher_mode']==recovery.UNDER_CLOSE
        and e.get('intervention_issued')is True and e.get('actual_under_close_failed')is True
        and e.get('native_grasp_after_correction')is True
        and q.get('excluded_intervention_decisions')==1 and q.get('expert_policy_decisions')==14)
    value.update(native_task_success=value['task_success'],genuine_recovery_success=genuine,
        recovery_evidence=e,excluded_intervention_decisions=q.get('excluded_intervention_decisions',0),
        expert_policy_decisions=q.get('expert_policy_decisions',0))
    if role==task.RECOVERY_QUALIFICATION_ROLE:value['task_success']=genuine
    return value


def summarize(panel,outcomes):
    rows=validate(panel);wanted={r['native_run']for r in rows};role=panel['task_learning_role']
    task.need(len({r['native_run']for r in outcomes})==len(outcomes)
        and all(r['native_run']in wanted for r in outcomes),'NO_RETRIED_OR_FOREIGN_COLLECTION_OUTCOME')
    attempted=sum(r['attempted']is True for r in outcomes);ready=sum(r['reset_ready']is True for r in outcomes)
    success=[r for r in outcomes if r['task_success']and r['technical_valid']and r['reset_ready']]
    recovered=[r for r in outcomes if r['genuine_recovery_success']]
    hard=sum(r['hard_safety_stop']is True for r in outcomes);technical=sum(r['technical_valid']is not True for r in outcomes)
    complete=len(outcomes)==attempted==len(rows)
    if role==task.RECOVERY_QUALIFICATION_ROLE:
        passes=complete and ready==len(recovered)==4 and technical==hard==0
    else:
        passes=complete and len(success)>=80 and len({r['layout']for r in success})>=20
        passes=passes and len(recovered)>=20 and len({r['layout']for r in recovered})>=10 and technical==hard==0
    return dict(schema='TASK_LEARNING_RECOVERY_QUALIFICATION_SUMMARY_V1'if role==task.RECOVERY_QUALIFICATION_ROLE
        else'TASK_LEARNING_CONTINUOUS_TEACHER_COLLECTION_SUMMARY_V1',requested=len(rows),attempted=attempted,
        reset_ready=ready,task_successes=len(success),genuine_recovery_successes=len(recovered),
        successful_layouts=len({r['layout']for r in success}),recovery_layouts=len({r['layout']for r in recovered}),
        hard_safety_stop_conditions=hard,technical_failure_conditions=technical,complete=complete,passes=passes,
        P_reset_ready=ready/len(rows),P_success_given_reset_ready=len(success)/ready if ready else None,
        P_success_per_requested_episode=len(success)/len(rows),
        eligible_native_full_or_recovery_episodes=len(success)if role==task.COLLECTION_ROLE else 0,
        qualified_training_demonstrations=0,Processor_admitted=False,GR00T_queries=0,model_updates=0)
