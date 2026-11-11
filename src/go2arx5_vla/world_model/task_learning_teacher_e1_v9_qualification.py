"""Fixed four-request E1/E2/E3 study, separate from original V8 recovery24."""
from pathlib import Path
from . import task_learning_grasp_v1 as task
from . import task_learning_teacher_e1_v9 as v9
from . import task_learning_teacher_v8 as v8
from . import task_learning_teacher_panel_v1 as legacy
from .task_learning_teacher_v8_qualification import archived_scene_spec
from .task_learning_scene_v1 import scene_id
from .stable_task_scene_v1 import pinned
from .storage import digest

SCHEMA='TEACHER_V9_E1_SMALL4_PANEL_V1'
EXPERIMENT='TEACHER_V9_E1_E2_E3_INDEPENDENT_SMALL4_NO_BC'
NATIVE_SCHEMA='TEACHER_V9_E1_E2_E3_NATIVE_ACTUAL_V1'
PURPOSE='TEACHER_V9_E1_SMALL4_NOT_BC_NOT_STUDENT_DAGGER'
CONTRACT_SHA='c054a1df9fe9d587267035ec860f847b284097e69ae188b5d300ac2b3aa792c0'
FROZEN_V8_SHA='b0ab7732ea7f12469a6690cf795516b056ab3611ead59f66a4b31168aa0636e7'
ORDER=(('RECOVERY01',1,'SUPPORTED_SIDE_OFFSET',1),('RECOVERY01',1,'SUPPORTED_SINGLE_PAD',1),
       ('RECOVERY04',2,'SUPPORTED_SIDE_OFFSET',-1),('RECOVERY04',2,'SUPPORTED_SINGLE_PAD',-1))
LAYOUTS=('RECOVERY01','RECOVERY04')


def build_panel(source_sha,scene_descriptors,base_Teacher_qualification,run_prefix,*,scene_producers,preregistration):
    task.need(len(source_sha)==40 and Path(run_prefix).name==run_prefix and not run_prefix.startswith('.'),
        'V9_FRESH_SOURCE_AND_PREFIX')
    rows=[]
    for layout,restart,kind,sign in ORDER:
        scene=pinned(scene_descriptors[layout])
        task.need(scene['parent']==scene_id(layout)and scene['layout']==layout
            and scene['physics_integrations']==0 and scene['compiled']['fresh_XML_compilation']is True,
            'V9_ORIGINAL_SEALED_STATIC_RECOVERY_SCENE')
        rows.append(dict(native_run=f'{run_prefix}-{layout.lower()}-r{restart}-{kind.lower()}',
            layout=layout,restart=restart,recovery_condition=kind,offset_sign=sign,Teacher_mode=v9.VERSION,
            parent=scene_id(layout),split='TRAIN',command=task.COMMAND,purpose=PURPOSE,
            model_sha256=scene['compiled']['scene_sha256']))
    panel=dict(schema=SCHEMA,task_learning_role=v9.ROLE,protocol=v9.contract(),source_sha=source_sha,
        V9_implementation_SHA256=digest(v9.__file__),unchanged_V8_Oracle_SHA256=digest(v8.__file__),
        preserved_V7_fingerprint=legacy.teacher_fingerprint(),preregistration=preregistration,
        base_Teacher_qualification=base_Teacher_qualification,scene_producers=scene_producers,assignments=rows,
        native_qualification_executed=False,expert_training_eligible=False)
    validate(panel);return panel


def validate(panel):
    from .task_learning_collection_v1 import qualified_base
    rows=panel.get('assignments')or[];contract=pinned(panel['preregistration'])
    task.need(panel.get('schema')==SCHEMA and panel.get('task_learning_role')==v9.ROLE
        and panel.get('protocol')==v9.contract()and panel.get('V9_implementation_SHA256')==digest(v9.__file__)
        and panel.get('unchanged_V8_Oracle_SHA256')==digest(v8.__file__)==FROZEN_V8_SHA
        and panel.get('preserved_V7_fingerprint')==legacy.teacher_fingerprint()==legacy.FROZEN_TEACHER
        and len(panel.get('source_sha',''))==40 and set(panel.get('scene_producers',{}))==set(LAYOUTS)
        and panel['preregistration']['sha256']==CONTRACT_SHA and contract['requested']==4
        and [(r['layout'],r['restart'],r['condition'],r['offset_sign'])for r in contract['planned_small_E1_population']]==list(ORDER)
        and len(rows)==4 and len({r['native_run']for r in rows})==4
        and [(r['layout'],r['restart'],r['recovery_condition'],r['offset_sign'])for r in rows]==list(ORDER),
        'V9_FOUR_FIXED_REQUESTS_UNCHANGED_V7_V8_AND_ORIGINAL_PREREGISTRATION')
    for kind,(offset,grip)in v9.INTERVENTIONS.items():
        task.need(contract['candidate_interventions'][kind]['tool_Y_offset_m']==offset
            and contract['candidate_interventions'][kind]['ending_nominal_grip']==grip,
            'V9_INTERVENTION_MATCHES_PREDECLARED_ACTUAL_GEOMETRY_HYPOTHESIS')
    for row in rows:
        task.need(row['Teacher_mode']==v9.VERSION and row['purpose']==PURPOSE and row['split']=='TRAIN'
            and row['command']==task.COMMAND and row['parent']==scene_id(row['layout'])
            and len(row['model_sha256'])==64 and Path(row['native_run']).name==row['native_run']
            and not row['native_run'].startswith('.')and row.get('diagnostic_control')is None,
            'V9_ORIGINAL_SCENE_AND_DISTINCT_NOT_BC_PURPOSE')
    qualified_base(panel['base_Teacher_qualification']);return rows


def native_scene_admission(spec):
    from .task_learning_scene_v1 import admission
    panel=pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    matches=[r for r in validate(panel)if r['native_run']==Path(spec['registered_root']).name]
    task.need(len(matches)==1 and spec['task_learning_role']==v9.ROLE,'V9_EXACT_NEW_REQUEST')
    row=matches[0]
    original=archived_scene_spec(panel['scene_producers'][row['layout']],spec['task_learning_scene_reference'],spec['model_path'])
    proof=admission(original)
    task.need(spec.get('task_learning_teacher_mode')==row['Teacher_mode']==v9.VERSION
        and spec.get('teacher_v9_recovery_condition')==row['recovery_condition']
        and spec.get('teacher_v9_offset_sign')==row['offset_sign']
        and row['parent']==proof['parent']==spec['scene_id']==spec['data_parent']
        and row['model_sha256']==digest(spec['model_path'])and proof['split']==spec['data_role']=='TRAIN',
        'V9_ARCHIVED_STATIC_SCENE_NOT_OLD_RECOVERY_OUTCOME')
    return dict(proof,static_custody_reused_without_relabelling_old_native_results=True,
        new_native_TeacherV9_E1_E2_E3_qualification_required=True)


def classify(row,*,native,worker,lease,readiness,score,spec,expected_source,attempted,
             recovery_exit=None,entry_witness=None,recovered_witness=None):
    value=legacy.classify(row,native=native,worker=worker,lease=lease,readiness=readiness,score=score,
        spec=spec,expected_source=expected_source,attempted=attempted,
        expected_role=v9.ROLE,expected_native_schema=NATIVE_SCHEMA)
    event=recovery_exit or{};entry=entry_witness or{};recovered=recovered_witness or{}
    owned=native.get('owned_process_exits')or[]
    clean=native.get('shutdown_clean')is True and len(owned)==4 and {p.get('name')for p in owned}=={
        'sim','solver','controller','planner'}and all(p.get('exit_code')==0 for p in owned)
    identity=(spec.get('task_learning_role')==v9.ROLE and spec.get('phase')==task.PHASE
        and spec.get('source_sha')==expected_source and attempted is True)
    unavailable=('TEACHER_V9_RECOVERY_UNAVAILABLE:'in(native.get('failure')or'')
        and event.get('Teacher_version')==v9.VERSION and event.get('recovery_condition')==row['recovery_condition'])
    if unavailable and identity and clean and worker and lease.get('status')=='CLOSED'and value['reset_ready']:
        value['technical_valid']=True
    E1=bool(entry.get('Teacher_version')==v9.VERSION and entry.get('recovery_condition')==row['recovery_condition']
        and entry.get('E1_condition_established')is True and entry.get('previous_complete_H40_solver_gripper_consumption')is True
        and entry.get('native_step')==entry.get('intervention_native_start_step',-1000)+1000
        and v9.entry_condition(row['recovery_condition'],entry))
    E2=bool(E1 and recovered.get('Teacher_version')==v9.VERSION and recovered.get('recovery_condition')==row['recovery_condition']
        and recovered.get('E2_actual_recovered')is True and recovered.get('native_grasp')is True
        and recovered.get('opposing_contact')is True and recovered.get('relative_motion_valid')is True
        and recovered.get('left_object_contact')is True and recovered.get('right_object_contact')is True
        and recovered['native_step']>entry['native_step']and recovered.get('E1_entry_native_step')==entry['native_step'])
    E3=value['task_success'];result=native.get('reference_qualification')or{}
    actual=bool(E1 and E2 and E3 and result.get('excluded_intervention_decisions')==1
        and result.get('expert_policy_decisions')==14)
    value.update(original_task_success=E3,E1_condition_established=E1,E2_recovery_evaluable=E1,
        E2_actual_recovered=E2 if E1 else None,E3_original_task_success=E3,
        task_success=actual,genuine_recovery_success=actual,entry_witness=entry,recovered_witness=recovered,
        recovery_exit=event,qualification_condition_not_established=event.get('reason')=='DECLARED_RECOVERY_CONDITION_NOT_ESTABLISHED',
        new_BC_demonstrations=0,new_student_rollin_states=0,new_DAgger_labels=0)
    return value


def summarize(panel,outcomes):
    rows=validate(panel);wanted={r['native_run']:r for r in rows}
    task.need(len({r['native_run']for r in outcomes})==len(outcomes)
        and all(r['native_run']in wanted and all(r[k]==wanted[r['native_run']][k]for k in(
            'layout','restart','recovery_condition','offset_sign','parent','split'))for r in outcomes),
        'V9_NO_FOREIGN_RETRIED_OR_RELABELLED_OUTCOME')
    def count(values):
        valid=[v for v in values if v['technical_valid']]
        E1=sum(v['E1_condition_established']for v in valid);E2=sum(v['E2_actual_recovered']is True for v in valid)
        return dict(requested=2,completed=len(values),E1_established=E1,E2_evaluable=E1,E2_recovered=E2,
            P_E2_given_E1=E2/E1 if E1 else None,E3_original_task_successes=sum(v['E3_original_task_success']for v in valid),
            genuine_complete_recovery_successes=sum(v['genuine_recovery_success']for v in valid))
    per_type={kind:count([v for v in outcomes if v['recovery_condition']==kind])for kind in v9.TYPES}
    technical=sum(not v['technical_valid']for v in outcomes);hard=sum(v['hard_safety_stop']for v in outcomes)
    complete=len(outcomes)==4 and all(v['attempted']for v in outcomes)
    established=sum(v['E1_established']for v in per_type.values());success=sum(v['genuine_complete_recovery_successes']for v in per_type.values())
    return dict(schema='TEACHER_V9_E1_E2_E3_SMALL4_SUMMARY_V1',requested=4,completed=len(outcomes),complete=complete,
        per_type=per_type,E1_passes=complete and technical==hard==0 and established==4,
        passes=complete and technical==hard==0 and success==4,genuine_recovery_successes=success,
        hard_safety_stop_conditions=hard,technical_failure_conditions=technical,
        qualified_training_demonstrations=0,GR00T_queries=0,model_updates=0,DAgger_labels=0,
        new24_registered=False,original_V8_population_unchanged=True)
