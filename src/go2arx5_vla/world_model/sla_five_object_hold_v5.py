"""Five fresh native HOLD construction requests over the checked V5.1 assets.

Reuses the exact qualified Teacher, score, RGB recorder and P10 consumer. This
is object feasibility, not formal D100 admission or a student evaluation.
"""
from pathlib import Path

from . import task_learning_grasp_v1 as task
from . import sla_target_coverage_v5 as coverage
from .stable_task_scene_v1 import pinned
from .storage import digest

VERSION = 'SLA_FIVE_OBJECT_PICK_HOLD_CONSTRUCTION_V5_1'
PANEL = VERSION + '_PANEL'
ROLE = 'CORELAB_EXPERT_TRAIN4_SEMANTICS_NOT_BC_V3'
RESTART = 11
ENDPOINT = 21703
NAMES = {
    'RED_CUBE40': 'red cube', 'GREEN_CUBE35': 'green cube',
    'BLUE_BOX50_30_40': 'blue rectangular block',
    'YELLOW_BOX35_45_40': 'yellow rectangular block',
    'PURPLE_BOX45_35_50': 'purple rectangular block',
}


def configuration(spec):
    value = spec.get('sla_five_object_hold_v5')
    if value is None:
        return False
    task.need(value == dict(version=VERSION, object_id=value.get('object_id'), pose_id='POSE01')
        and value.get('object_id') in NAMES and spec.get('task_learning_role') == ROLE
        and spec.get('task_learning_teacher_mode') == 'NORMAL_TEACHER_V7'
        and spec.get('task_learning_control') is None and not spec.get('target_gr00t_policy')
        and spec.get('sla_pick_place_v5') is None,
        'EXPLICIT_FIVE_OBJECT_HOLD_EXPERT_CONFIGURATION')
    return True


def command(object_id):
    return 'GRASP_' + object_id + '_LIFT20MM_HOLD1S_V5_1'


def language(spec):
    configuration(spec)
    return 'Grasp the ' + NAMES[spec['sla_five_object_hold_v5']['object_id']] + ', lift it at least 20 millimeters, and hold it steady for one second.'


def scene_id(object_id):
    return 'SLA_TARGET_V5_1_' + object_id + '_POSE01'


def protocol(spec):
    configuration(spec)
    obj = spec['sla_five_object_hold_v5']['object_id']
    return dict(task.protocol(task.CONSTRUCTION_ROLE), schema=VERSION, stage=ROLE,
        task_revision=VERSION, task_command=command(obj), language=language(spec),
        scene_family='FIVE_DECLARED_OBJECTS_POSE01_NATIVE_FEASIBILITY_NOT_D100',
        object_id=obj, pose_id='POSE01', prefix_atoms=10, policy_queries=60,
        final_hold_atoms=40, expert_consumer='EEF_STREAM_V2', reference_TTL_s=.4,
        reference_renewal_max_s=.35, expert_recorded_success_tail='CORELAB_ACTUAL_TWO_SECOND_EXPERT_SUCCESS_TAIL_V3',
        D100_new=False, original_grasp_Teacher_and_score_unchanged=True)


def declared(spec):
    task.need(configuration(spec), 'FIVE_OBJECT_HOLD_CONFIGURATION_REQUIRED')
    task.need(spec.get('phase') == task.PHASE and spec.get('kind') == task.KIND
        and spec.get('task_learning_grasp') == protocol(spec)
        and spec.get('task_command') == command(spec['sla_five_object_hold_v5']['object_id'])
        and spec.get('live_skill_start_step') == task.CALIBRATION_B
        and spec.get('max_ticks') == ENDPOINT and spec.get('physics_dt_s') == .002
        and spec.get('WBC_changes') is False, 'FIVE_OBJECT_HOLD_ORIGINAL_NATIVE_CLOCK')


def validate_panel(panel):
    from .task_learning_teacher_panel_v1 import teacher_fingerprint, FROZEN_TEACHER
    from .task_learning_collection_v1 import qualified_base
    rows = panel.get('assignments') or []
    task.need(panel.get('schema') == PANEL and panel.get('source_sha')
        and panel.get('tasks') == ['PICK_HOLD'] and panel.get('D100_new') is False
        and panel.get('base_grasp_Teacher_sha256') == teacher_fingerprint() == FROZEN_TEACHER
        and [r['object_id'] for r in rows] == list(NAMES)
        and len({r['native_run'] for r in rows}) == 5,
        'FIVE_FIXED_OBJECTS_NO_EASIER_SUBSTITUTION_OR_FORMAL_RELABEL')
    for row in rows:
        task.need(row['parent'] == scene_id(row['object_id']) and row['split'] == 'TRAIN'
            and row['pose_id'] == 'POSE01' and row['restart'] == RESTART
            and row['prefix_atoms'] == 10 and row['command'] == command(row['object_id'])
            and len(row['model_sha256']) == 64
            and Path(row['native_run']).name == row['native_run'] and not row['native_run'].startswith('.'),
            'EXACT_FIVE_OBJECT_CONSTRUCTION_INPUTS')
    qualified_base(panel['base_Teacher_qualification'])
    return rows


def native_scene_admission(spec):
    from .native_body_camera_capture_v1 import session_configuration
    declared(spec)
    task.need(session_configuration(spec)
        and spec.get('expert_eef_stream_v2') == dict(version='EEF_STREAM_V2', prefix_atoms=10),
        'FIVE_OBJECT_REUSES_ORIGINAL_RGB_AND_FULL_PREVIEW_P10')
    proof = pinned(spec['task_learning_scene_reference'])
    producer_pin = spec['sla_five_object_static_result_v5']
    producer = pinned(producer_pin)
    root = Path(producer_pin['path']).parent
    lease = pinned(dict(path=str(root/'control/LEASE.json'), sha256=digest(root/'control/LEASE.json')))
    worker = pinned(dict(path=str(root/'WORKER_EXIT.private.json'), sha256=digest(root/'WORKER_EXIT.private.json')))
    value = spec['sla_five_object_hold_v5']; obj = value['object_id']
    matching = [r for r in producer['rows'] if r['declaration']['object_id'] == obj and r['declaration']['pose_id'] == 'POSE01']
    task.need(lease['status'] == 'CLOSED' and lease['worker_exit_code'] == worker['exit_code'] == 0
        and lease['spec_sha256'] == digest(root/'EXECUTION_SPEC.json')
        and producer['schema'] == coverage.VERSION and producer['passed'] == producer['completed'] == 50
        and len(matching) == 1 and matching[0]['result_sha256'] == spec['task_learning_scene_reference']['sha256']
        and proof['passes'] and proof['declaration'] == coverage.declaration(obj, 'POSE01')
        and proof['builder_sha256'] == spec['source_files']['src/go2arx5_vla/world_model/sla_target_coverage_v5.py']
        and proof['saved_startup_checks'] == 5703 and proof['actual_integrations'] == 0
        and proof['native_execution_authorized'] is False
        and proof['robot_contract']['status'] == 'ROBOT_AND_NATIVE_SAFETY_EXACT_WITH_EXPLICIT_OBJECT_VARIATION'
        and proof['scene_sha256'] == digest(spec['model_path'])
        and spec['data_role'] == 'TRAIN' and spec['scene_id'] == spec['data_parent'] == scene_id(obj),
        'CLOSED_FIFTY_CASE_STATIC_CUSTODY_AND_EXACT_OBJECT_MODEL')
    panel = pinned(dict(path=spec['stable_task_panel'], sha256=spec['stable_task_panel_sha256']))
    rows = validate_panel(panel)
    matches = [r for r in rows if r['native_run'] == Path(spec['registered_root']).name]
    task.need(panel['source_sha'] == spec['source_sha'] and len(matches) == 1
        and matches[0]['object_id'] == obj and matches[0]['parent'] == spec['data_parent']
        and matches[0]['model_sha256'] == proof['scene_sha256']
        and spec['expert_cadence_restart'] == RESTART,
        'FIVE_OBJECT_FRESH_FIXED_REQUEST_NO_FAILED_CONDITION_REPLAY')
    return proof
