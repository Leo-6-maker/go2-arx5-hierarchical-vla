"""Fresh full24 qualification; reuse audited small4, never relabel old V8."""
import json
from pathlib import Path
from . import task_learning_grasp_v1 as task
from . import task_learning_teacher_v8 as v8
from . import task_learning_teacher_e1_v9 as small
from . import task_learning_teacher_v9_full24 as teacher
from . import task_learning_teacher_panel_v1 as legacy
from .task_learning_scene_v1 import RECOVERY_OFFSETS,scene_id
from .task_learning_teacher_v8_qualification import archived_scene_spec
from .stable_task_scene_v1 import pinned
from .storage import digest

SCHEMA='TEACHER_V9_REPAIRED_E1_FULL24_PANEL_V1'
EXPERIMENT='TEACHER_V9_REPAIRED_E1_FULL24_INDEPENDENT_QUALIFICATION_NO_BC'
NATIVE_SCHEMA='TEACHER_V9_REPAIRED_E1_FULL24_NATIVE_ACTUAL_V1'
PURPOSE='TEACHER_V9_FULL24_NOT_BC_NOT_STUDENT_DAGGER'
CONTRACT_SHA='767bd29c27782c1bf0599a768c007be84a4b4fb4c425851a9383ffa2e44003e8'
SMALL_RESULT_SHA='ff0b73999187b3ae5bbbed6f88c5a32a1f9b718825130b9f78b853c1404a22b7'
SMALL_OUTCOMES_SHA='b351cb9c84bb30ca3e7d6f76e4b077448717d7b115c79719d957d3af269ac8ab'
V8_SOURCE_SHA='b0ab7732ea7f12469a6690cf795516b056ab3611ead59f66a4b31168aa0636e7'
SMALL_SOURCE_SHA='09491e7813cb36eef6482043c8f3f38a7476e716c929b65e77529058e196152c'
_CLOSED_CUSTODY={}


def assignments():
    return [(layout,restart,kind)for restart in(1,2)for layout in RECOVERY_OFFSETS for kind in teacher.TYPES]


def qualified_small4(reference):
    """Check the closed actual producers and saved audit pins once per process."""
    key=(reference['path'],reference['sha256'])
    if key in _CLOSED_CUSTODY:return _CLOSED_CUSTODY[key]
    from .task_learning_teacher_e1_v9_qualification import validate as validate_small
    from .task_learning_grasp_native_v1 import complete_consumption
    proof=pinned(reference);refs=proof['original_references']
    result=pinned(refs['result']);inp=pinned(refs['input']);outcomes=pinned(refs['outcomes']);lease=pinned(refs['lease']);panel=pinned(refs['panel'])
    rows=validate_small(panel);root=Path(refs['result']['path']).parent
    task.need(proof['schema']=='TEACHER_V9_SMALL4_CLOSED_FORCE_SCORE_CUSTODY_V1'
        and refs['result']['sha256']==SMALL_RESULT_SHA and refs['outcomes']['sha256']==SMALL_OUTCOMES_SHA
        and root.name=='task-learning-teacher-v9-e1-small4-20261111-01'
        and Path(refs['input']['path'])==root/'INPUT.private.json'
        and Path(refs['outcomes']['path'])==root/'OUTCOMES.private.json'
        and Path(refs['lease']['path'])==root/'control/LEASE.json'
        and lease['status']=='CLOSED'and lease['worker_exit_code']==result['exit_code']==0
        and lease['input_sha256']==refs['input']['sha256']
        and result['source_sha']==inp['source_sha']==panel['source_sha']=='586420c63fe04759fa2c88b3be4ca35c2fc68308'
        and result['panel_sha256']==inp['panel_sha256']==refs['panel']['sha256']
        and result['summary']['complete']and result['summary']['passes']and result['summary']['genuine_recovery_successes']==4
        and len(outcomes)==len(rows)==len(proof['saved_audits'])==4,
        'FULL24_REUSE_REAL_CLOSED_AND_AUDITED_SMALL4_NOT_STATIC_SUCCESS')
    audits={a['native_run']:a for a in proof['saved_audits']}
    task.need([r['native_run']for r in outcomes]==[r['native_run']for r in rows],
        'FULL24_EXACT_SMALL4_POPULATION_CUSTODY')
    for row in outcomes:
        r=root/'runs'/row['native_run'];s=pinned(dict(path=str(r/'EXECUTION_SPEC.json'),sha256=json.loads((r/'control/LEASE.json').read_bytes())['spec_sha256']))
        # Native results include the full consumption receipts and exceed the
        # small-metadata reader's16MiB limit. Keep canonical custody and the
        # closed producer's exact hash, without changing the shared reader.
        npath=r/'native-01/RESULT.json'
        task.need(npath.is_absolute()and npath.resolve()==npath and not npath.is_symlink()
            and digest(npath)==row['result_sha256'],'FULL24_EXACT_ORIGINAL_NATIVE_RESULT')
        n=json.loads(npath.read_bytes());q=n['reference_qualification'];audit=audits[row['native_run']]
        child=json.loads((r/'control/LEASE.json').read_bytes());worker=json.loads((r/'WORKER_EXIT.private.json').read_bytes())
        task.need(child['status']=='CLOSED'and child['worker_exit_code']==worker['exit_code']==0
            and s['task_learning_role']==small.ROLE and s['source_sha']==result['source_sha']
            and row['technical_valid']and row['genuine_recovery_success']and not row['hard_safety_stop']
            and row['E1_condition_established']and row['E2_actual_recovered']and row['E3_original_task_success']
            and n['actual_target_integrations']==audit['original_native_integrations_audited']==21203
            and n['status']=='NATIVE_STAGE_PASSED'and n['shutdown_clean']and not n['physics_stops']
            and len(n['owned_process_exits'])==4 and all(c['exit_code']==0 for c in n['owned_process_exits'])
            and complete_consumption(s,q)and q['excluded_intervention_decisions']==1 and q['expert_policy_decisions']==14
            and audit['native_result_sha256']==row['result_sha256']and audit['source_sha']==s['source_sha']
            and audit['score_exactly_recomputed_621']and audit['score']==row['score']==q['new_task_score']
            and audit['confirmed_commands']==600 and audit['real_current_RGB']==30 and audit['trace_dropped']==0
            and len(audit['E1_E2_exact_saved_packet_checks'])==2
            and all(w['all_current_force_gap_support_motion_and_consumed_fields_exact']for w in audit['E1_E2_exact_saved_packet_checks'])
            and audit['new_GR00T_model_calls']==audit['optimizer_updates']==audit['physics_integrations']==0,
            'FULL24_ORIGINAL_FORCE_CONSUMPTION_SCORE_AND_INDEPENDENT_SAVED_AUDIT')
    _CLOSED_CUSTODY[key]=proof;return proof


def build_panel(source_sha,scenes,base_Teacher_qualification,run_prefix,*,scene_producers,preregistration,small4_qualification):
    task.need(len(source_sha)==40 and Path(run_prefix).name==run_prefix and not run_prefix.startswith('.'),
        'FULL24_NEW_SOURCE_AND_PANEL_PREFIX')
    rows=[]
    for layout,restart,kind in assignments():
        scene=pinned(scenes[layout])
        rows.append(dict(native_run=f'{run_prefix}-{layout.lower()}-r{restart}-{kind.lower()}',layout=layout,
            restart=restart,recovery_condition=kind,offset_sign=1 if restart==1 else -1,Teacher_mode=teacher.VERSION,
            parent=scene_id(layout),split='TRAIN',command=task.COMMAND,purpose=PURPOSE,
            model_sha256=scene['compiled']['scene_sha256']))
    panel=dict(schema=SCHEMA,task_learning_role=teacher.ROLE,protocol=teacher.contract(),source_sha=source_sha,
        full24_controller_sha256=digest(teacher.__file__),unchanged_V8_source_sha256=digest(v8.__file__),
        unchanged_small4_source_sha256=digest(small.__file__),preserved_V7_fingerprint=legacy.teacher_fingerprint(),
        preregistration=preregistration,small4_qualification=small4_qualification,
        base_Teacher_qualification=base_Teacher_qualification,scene_producers=scene_producers,assignments=rows,
        expert_training_eligible=False,native_qualification_executed=False)
    validate(panel);return panel


def validate(panel):
    from .task_learning_collection_v1 import qualified_base
    rows=panel.get('assignments')or[];contract=pinned(panel['preregistration'])
    task.need(panel.get('schema')==SCHEMA and panel.get('task_learning_role')==teacher.ROLE
        and panel.get('protocol')==teacher.contract()and len(panel.get('source_sha',''))==40
        and panel.get('full24_controller_sha256')==digest(teacher.__file__)
        and panel.get('unchanged_V8_source_sha256')==digest(v8.__file__)==V8_SOURCE_SHA
        and panel.get('unchanged_small4_source_sha256')==digest(small.__file__)==SMALL_SOURCE_SHA
        and panel.get('preserved_V7_fingerprint')==legacy.teacher_fingerprint()==legacy.FROZEN_TEACHER
        and panel['preregistration']['sha256']==CONTRACT_SHA and contract['requested']==24
        and contract['layouts']==list(RECOVERY_OFFSETS)and contract['conditions']==list(teacher.TYPES)
        and contract['predecessor_small4_result_sha256']==SMALL_RESULT_SHA
        and contract['predecessor_small4_outcomes_sha256']==SMALL_OUTCOMES_SHA
        and contract['interventions']==teacher.contract()['interventions']
        and set(panel['scene_producers'])==set(RECOVERY_OFFSETS)
        and len(rows)==24 and len({r['native_run']for r in rows})==24
        and [(r['layout'],r['restart'],r['recovery_condition'])for r in rows]==assignments(),
        'FULL24_PREREGISTERED_NEW_POPULATION_AND_UNCHANGED_QUALIFIED_CONTROLLERS')
    for row in rows:
        task.need(row['Teacher_mode']==teacher.VERSION and row['offset_sign']==(1 if row['restart']==1 else -1)
            and row['purpose']==PURPOSE and row['split']=='TRAIN'and row['command']==task.COMMAND
            and row['parent']==scene_id(row['layout'])and len(row['model_sha256'])==64
            and Path(row['native_run']).name==row['native_run']and not row['native_run'].startswith('.')
            and row.get('diagnostic_control')is None,'FULL24_NO_FOREIGN_OR_OLD_V8_REQUEST')
    qualified_base(panel['base_Teacher_qualification']);qualified_small4(panel['small4_qualification'])
    return rows


def native_scene_admission(spec):
    from .task_learning_scene_v1 import admission
    panel=pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    matches=[r for r in validate(panel)if r['native_run']==Path(spec['registered_root']).name]
    task.need(len(matches)==1 and spec['task_learning_role']==teacher.ROLE,'FULL24_OWN_NEW_NATIVE_REQUEST')
    row=matches[0];original=archived_scene_spec(panel['scene_producers'][row['layout']],spec['task_learning_scene_reference'],spec['model_path']);proof=admission(original)
    task.need(spec.get('task_learning_teacher_mode')==row['Teacher_mode']==teacher.VERSION
        and spec.get('teacher_full24_recovery_condition')==spec.get('teacher_v8_recovery_condition')==spec.get('teacher_v9_recovery_condition')==row['recovery_condition']
        and spec.get('teacher_full24_offset_sign')==spec.get('teacher_v8_offset_sign')==spec.get('teacher_v9_offset_sign')==row['offset_sign']
        and row['parent']==proof['parent']==spec['scene_id']==spec['data_parent']
        and row['model_sha256']==digest(spec['model_path'])and proof['split']==spec['data_role']=='TRAIN',
        'FULL24_EXACT_DELEGATE_CONFIGURATION_AND_ARCHIVED_STATIC_SCENE')
    return dict(proof,new_full24_native_qualification_required=True,old_V8_and_small4_outcomes_unchanged=True)


def classify(row,*,native,worker,lease,readiness,score,spec,expected_source,attempted,
             recovery_exit=None,entry_witness=None,recovered_witness=None):
    value=legacy.classify(row,native=native,worker=worker,lease=lease,readiness=readiness,score=score,
        spec=spec,expected_source=expected_source,attempted=attempted,
        expected_role=teacher.ROLE,expected_native_schema=NATIVE_SCHEMA)
    event=recovery_exit or{};entry=entry_witness or{};recovered=recovered_witness or{}
    owned=native.get('owned_process_exits')or[]
    clean=native.get('shutdown_clean')is True and len(owned)==4 and {p.get('name')for p in owned}=={'sim','solver','controller','planner'}and all(p.get('exit_code')==0 for p in owned)
    if('TEACHER_V9_FULL24_RECOVERY_UNAVAILABLE:'in(native.get('failure')or'')
        and event.get('Teacher_version')==teacher.VERSION and event.get('recovery_condition')==row['recovery_condition']
        and spec.get('task_learning_role')==teacher.ROLE and spec.get('source_sha')==expected_source
        and spec.get('phase')==task.PHASE and clean and worker and lease.get('status')=='CLOSED'and value['reset_ready']):
        value['technical_valid']=True
    E1=bool(entry.get('Teacher_version')==teacher.VERSION and entry.get('recovery_condition')==row['recovery_condition']
        and entry.get('E1_condition_established')is True and entry.get('previous_complete_H40_solver_gripper_consumption')is True
        and teacher.entry_condition(row['recovery_condition'],entry))
    E2=bool(E1 and recovered.get('Teacher_version')==teacher.VERSION and recovered.get('recovery_condition')==row['recovery_condition']
        and recovered.get('E2_actual_recovered')is True and recovered.get('native_grasp')is True
        and recovered.get('opposing_contact')is True and recovered.get('relative_motion_valid')is True
        and recovered.get('left_object_contact')is True and recovered.get('right_object_contact')is True
        and recovered.get('E1_entry_native_step')==entry['native_step']<recovered.get('native_step',0))
    E3=value['task_success'];q=native.get('reference_qualification')or{}
    success=bool(E1 and E2 and E3 and q.get('excluded_intervention_decisions')==1 and q.get('expert_policy_decisions')==14)
    value.update(original_task_success=E3,E1_condition_established=E1,E2_recovery_evaluable=E1,
        E2_actual_recovered=E2 if E1 else None,E3_original_task_success=E3,
        task_success=success,genuine_recovery_success=success,entry_witness=entry,recovered_witness=recovered,
        recovery_exit=event,qualification_condition_not_established=event.get('reason')=='DECLARED_RECOVERY_CONDITION_NOT_ESTABLISHED',
        new_BC_demonstrations=0,new_student_rollin_states=0,new_DAgger_labels=0)
    return value


def summarize(panel,outcomes):
    rows=validate(panel);wanted={r['native_run']:r for r in rows}
    task.need(len({r['native_run']for r in outcomes})==len(outcomes)
        and all(r['native_run']in wanted and all(r[k]==wanted[r['native_run']][k]for k in('layout','restart','recovery_condition','offset_sign','parent','split'))for r in outcomes),
        'FULL24_NO_RETRIED_FOREIGN_OR_RELABELLED_OUTCOME')
    per_type={}
    for kind in teacher.TYPES:
        values=[r for r in outcomes if r['recovery_condition']==kind and r['technical_valid']]
        E1=sum(r['E1_condition_established']for r in values);E2=sum(r['E2_actual_recovered']is True for r in values)
        per_type[kind]=dict(requested=8,completed=sum(r['recovery_condition']==kind for r in outcomes),
            E1_established=E1,E2_evaluable=E1,E2_recovered=E2,P_E2_given_E1=E2/E1 if E1 else None,
            E3_original_task_successes=sum(r['E3_original_task_success']for r in values),
            genuine_recovery_successes=sum(r['genuine_recovery_success']for r in values))
    complete=len(outcomes)==24 and all(r['attempted']for r in outcomes);technical=sum(not r['technical_valid']for r in outcomes);hard=sum(r['hard_safety_stop']for r in outcomes)
    total=sum(v['genuine_recovery_successes']for v in per_type.values())
    return dict(schema='TEACHER_V9_REPAIRED_E1_FULL24_SUMMARY_V1',requested=24,completed=len(outcomes),complete=complete,
        per_type=per_type,genuine_recovery_successes=total,technical_failure_conditions=technical,hard_safety_stop_conditions=hard,
        passes=complete and technical==hard==0 and total>=22 and all(v['genuine_recovery_successes']>=7 for v in per_type.values()),
        thresholds=dict(minimum_total=22,minimum_per_condition=7,maximum_hard_safety_stops=0,maximum_technical_failures=0),
        qualified_training_demonstrations=0,GR00T_queries=0,model_updates=0,DAgger_labels=0,
        original_V8_and_small4_populations_unchanged=True)
