"""Separate24-request V8 recovery qualification, never BC or student DAgger."""
import copy
from pathlib import Path

from . import task_learning_grasp_v1 as task
from . import task_learning_teacher_v8 as v8
from . import task_learning_teacher_panel_v1 as legacy
from .task_learning_scene_v1 import RECOVERY_OFFSETS,scene_id
from .stable_task_scene_v1 import pinned
from .storage import digest

SCHEMA='TEACHER_V8_RECOVERY24_PANEL_V1'
EXPERIMENT='TEACHER_V8_INDEPENDENT_RECOVERY_QUALIFICATION_NO_BC'
NATIVE_SCHEMA='TEACHER_V8_RECOVERY_QUALIFICATION_NATIVE_ACTUAL_V1'
PURPOSE='TEACHER_V8_QUALIFICATION_NOT_BC_NOT_STUDENT_DAGGER'


def assignments():
    return [(layout,restart,kind)for restart in (1,2)for layout in RECOVERY_OFFSETS for kind in v8.TYPES]


def build_panel(source_sha,scene_descriptors,base_Teacher_qualification,run_prefix):
    task.need(len(source_sha)==40 and Path(run_prefix).name==run_prefix and not run_prefix.startswith('.'),
        'V8_FRESH_PANEL_SOURCE_AND_PREFIX')
    scenes={layout:pinned(scene_descriptors[layout])for layout in RECOVERY_OFFSETS};rows=[]
    for layout,restart,kind in assignments():
        scene=scenes[layout]
        task.need(scene['parent']==scene_id(layout)and scene['layout']==layout
            and scene['physics_integrations']==0 and scene['compiled']['fresh_XML_compilation']is True,
            'V8_ONLY_SEALED_ORIGINAL_STATIC_SCENE_CUSTODY')
        rows.append(dict(native_run=f'{run_prefix}-{layout.lower()}-r{restart}-{kind.lower()}',
            layout=layout,restart=restart,recovery_condition=kind,offset_sign=1 if restart==1 else -1,
            Teacher_mode=v8.VERSION,parent=scene_id(layout),split='TRAIN',command=task.COMMAND,
            purpose=PURPOSE,model_sha256=scene['compiled']['scene_sha256']))
    panel=dict(schema=SCHEMA,task_learning_role=v8.ROLE,protocol=v8.contract(),source_sha=source_sha,
        V8_implementation_SHA256=digest(v8.__file__),preserved_V7_fingerprint=legacy.teacher_fingerprint(),
        base_Teacher_qualification=base_Teacher_qualification,assignments=rows,
        native_qualification_executed=False,expert_training_eligible=False)
    validate(panel);return panel


def validate(panel):
    from .task_learning_collection_v1 import qualified_base
    rows=panel.get('assignments')or[]
    task.need(panel.get('schema')==SCHEMA and panel.get('task_learning_role')==v8.ROLE
        and panel.get('protocol')==v8.contract()and panel.get('V8_implementation_SHA256')==digest(v8.__file__)
        and panel.get('preserved_V7_fingerprint')==legacy.teacher_fingerprint()==legacy.FROZEN_TEACHER
        and len(panel.get('source_sha',''))==40 and len(rows)==24 and len({r['native_run']for r in rows})==24
        and [(r['layout'],r['restart'],r['recovery_condition'])for r in rows]==assignments(),
        'V8_COMPLETE_FIXED24_RECOVERY_REQUESTS_AND_UNCHANGED_V7')
    for row in rows:
        task.need(row['Teacher_mode']==v8.VERSION and row['offset_sign']==(1 if row['restart']==1 else -1)
            and row['purpose']==PURPOSE and row['split']=='TRAIN'and row['command']==task.COMMAND
            and row['parent']==scene_id(row['layout'])and len(row['model_sha256'])==64
            and Path(row['native_run']).name==row['native_run']and not row['native_run'].startswith('.')
            and row.get('diagnostic_control')is None,'V8_UNIQUE_PREREGISTERED_MODE_SCENE_AND_PURPOSE')
    qualified_base(panel['base_Teacher_qualification'])
    return rows


def native_scene_admission(spec):
    """Reuse only original static scene custody, not an old recovery outcome."""
    from .task_learning_scene_v1 import admission
    original=copy.deepcopy(spec);original['task_learning_role']=task.RECOVERY_QUALIFICATION_ROLE
    original['task_learning_grasp']=task.protocol(task.RECOVERY_QUALIFICATION_ROLE)
    proof=admission(original)
    panel=pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    rows=validate(panel);matches=[r for r in rows if r['native_run']==Path(spec['registered_root']).name]
    task.need(len(matches)==1 and spec['task_learning_role']==v8.ROLE,'V8_OWN_REQUEST_AND_TYPED_ROLE')
    row=matches[0]
    task.need(spec.get('task_learning_teacher_mode')==row['Teacher_mode']==v8.VERSION
        and spec.get('teacher_v8_recovery_condition')==row['recovery_condition']
        and spec.get('teacher_v8_offset_sign')==row['offset_sign']
        and row['parent']==proof['parent']==spec['scene_id']==spec['data_parent']
        and row['model_sha256']==digest(spec['model_path'])and proof['split']==spec['data_role']=='TRAIN',
        'V8_ORIGINAL_STATIC_MODEL_AND_NEW_RECOVERY_POPULATION')
    return dict(proof,static_custody_reused_without_relabelling_old_native_results=True,
        new_native_TeacherV8_qualification_required=True)


def classify(row,*,native,worker,lease,readiness,score,spec,expected_source,attempted,recovery_exit=None):
    value=legacy.classify(row,native=native,worker=worker,lease=lease,readiness=readiness,score=score,
        spec=spec,expected_source=expected_source,attempted=attempted,expected_role=v8.ROLE,expected_native_schema=NATIVE_SCHEMA)
    event=recovery_exit or{}
    owned=native.get('owned_process_exits')or[]
    clean=native.get('shutdown_clean')is True and len(owned)==4 and{p.get('name')for p in owned}=={'sim','solver','controller','planner'}and all(p.get('exit_code')==0 for p in owned)
    unavailable=('TEACHER_V8_RECOVERY_UNAVAILABLE:'in(native.get('failure')or'')and bool(event)
        and event.get('Teacher_version')==v8.VERSION and event.get('recovery_condition')==row['recovery_condition'])
    identity=(spec.get('task_learning_role')==v8.ROLE and spec.get('phase')==task.PHASE
        and spec.get('source_sha')==expected_source and attempted is True)
    if unavailable and identity and clean and worker and lease.get('status')=='CLOSED'and value['reset_ready']:
        value['technical_valid']=True
    result=native.get('reference_qualification')or{};e=result.get('teacher_v8_recovery_evidence')or{}
    recovered=bool(value['task_success']and e.get('Teacher_version')==v8.VERSION
        and e.get('recovery_condition')==row['recovery_condition']and e.get('intervention_issued')is True
        and e.get('actual_recovery_entry_observed')is True and e.get('native_grasp_after_correction')is True
        and result.get('excluded_intervention_decisions')==1 and result.get('expert_policy_decisions')==14)
    value.update(original_task_success=value['task_success'],task_success=recovered,genuine_recovery_success=recovered,
        recovery_evidence=e,recovery_exit=event,qualification_condition_not_established=event.get('reason')=='DECLARED_RECOVERY_CONDITION_NOT_ESTABLISHED',
        new_BC_demonstrations=0,new_student_rollin_states=0,new_DAgger_labels=0)
    return value


def summarize(panel,outcomes):
    rows=validate(panel);wanted={r['native_run']for r in rows}
    task.need(len({r['native_run']for r in outcomes})==len(outcomes)and all(r['native_run']in wanted for r in outcomes),
        'V8_NO_FOREIGN_OR_RETRIED_OUTCOME')
    per_type={kind:dict(requested=8,attempted=sum(r['attempted']is True for r in outcomes if r['recovery_condition']==kind),
        genuine_recovery_successes=sum(r['genuine_recovery_success']and r['technical_valid']for r in outcomes if r['recovery_condition']==kind))for kind in v8.TYPES}
    attempted=sum(r['attempted']is True for r in outcomes);hard=sum(r['hard_safety_stop']is True for r in outcomes)
    technical=sum(r['technical_valid']is not True for r in outcomes)
    successes=sum(r['genuine_recovery_success']and r['technical_valid']for r in outcomes);complete=len(outcomes)==attempted==24
    return dict(schema='TEACHER_V8_RECOVERY24_SUMMARY_V1',requested=24,attempted=attempted,completed=len(outcomes),complete=complete,
        reset_ready=sum(r['reset_ready']is True for r in outcomes),genuine_recovery_successes=successes,
        hard_safety_stop_conditions=hard,technical_failure_conditions=technical,per_type=per_type,
        condition_not_established=sum(r['qualification_condition_not_established']for r in outcomes),
        passes=complete and technical==hard==0 and successes>=22 and all(v['genuine_recovery_successes']>=7 for v in per_type.values()),
        P_success_per_requested=successes/24,qualified_training_demonstrations=0,GR00T_queries=0,model_updates=0,
        threshold=dict(minimum_total=22,minimum_per_type=7,maximum_hard_safety_stops=0),DAgger_labels=0)
