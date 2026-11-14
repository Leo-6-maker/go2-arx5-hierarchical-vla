"""Four TRAIN camera/event semantic checks, separate from student evaluation.

The completed Teacher qualification is reused. These engineering episodes do
not grant BC/export eligibility or alter the old hundred-episode population.
"""
from pathlib import Path

from .storage import digest
from .stable_task_scene_v1 import pinned

ROLE = 'CORELAB_EXPERT_TRAIN4_SEMANTICS_NOT_BC_V3'
SCHEMA = 'CORELAB_EXPERT_TRAIN4_SEMANTICS_PANEL_V3'
NATIVE_SCHEMA = 'CORELAB_EXPERT_TRAIN4_SEMANTICS_NATIVE_V3'
LAYOUTS = ('TRAIN01', 'TRAIN07', 'TRAIN13', 'TRAIN25')
PURPOSE = 'EXPERT_CAMERA_EVENT_SEMANTICS_NOT_BC_OR_STUDENT_EVALUATION'
RECORDED_TAIL = 'CORELAB_ACTUAL_TWO_SECOND_EXPERT_SUCCESS_TAIL_V3'
CADENCE_PANEL = 'CORELAB_EXPERT_FIXED_CADENCE_PANEL_V3'
CADENCE_PURPOSE = 'MATCHED_EXPERT_CADENCE_QUALIFICATION_NOT_BC'
CADENCES = (40, 10)


def protocol():
    from . import task_learning_grasp_v1 as task
    return dict(task.protocol(task.CONSTRUCTION_ROLE), stage=ROLE,
                scene_family='FOUR_PREREGISTERED_TRAIN_LAYOUTS_ENGINEERING_ONLY',
                observation='BODY_BASE_FIXED_V1_20261111_PLUS_ORIGINAL_NATIVE_WRIST')


def recorded_tail(spec):
    from .task_learning_grasp_v1 import need
    value = spec.get('expert_recorded_success_tail_v3')
    if value is None:
        return False
    need(value == RECORDED_TAIL and spec.get('task_learning_role') == ROLE
         and spec.get('expert_dense_observation_v3') is not None and spec.get('sla_expert_epsilon_v3') is None,
         'TWO_SECOND_TAIL_REQUIRES_FRESH_DENSE_TRAIN_EXPERT')
    return True


def native_protocol(spec):
    if spec.get('sla_five_object_hold_v5')is not None:
        from .sla_five_object_hold_v5 import protocol as five_object_protocol
        return five_object_protocol(spec)
    if spec.get('sla_pick_place_v5')is not None:
        from .sla_pick_place_v5 import configuration,protocol
        configuration(spec)
        return protocol(spec)
    from . import task_learning_grasp_v1 as task
    from .eef_stream_v2 import PREFIX_ATOMS, VERSION, configuration
    original = task.protocol(spec.get('task_learning_role', task.CONSTRUCTION_ROLE))
    stream = configuration(spec)
    if not recorded_tail(spec):
        return original
    updated = dict(original, final_hold_atoms=40, expert_recorded_success_tail=RECORDED_TAIL)
    if stream:
        updated.update(prefix_atoms=PREFIX_ATOMS, policy_queries=task.QUERIES * task.PREFIX_ATOMS // PREFIX_ATOMS,
                       expert_consumer=VERSION, reference_TTL_s=.4, reference_renewal_max_s=.35)
    return updated


def native_endpoint(spec):
    if spec.get('sla_five_object_hold_v5')is not None:
        from .sla_five_object_hold_v5 import configuration,ENDPOINT
        configuration(spec)
        return ENDPOINT
    if spec.get('sla_pick_place_v5')is not None:
        from .sla_pick_place_v5 import configuration,endpoint
        configuration(spec)
        return endpoint(spec)
    from . import task_learning_grasp_v1 as task
    if spec.get('task_learning_role') == task.SLA_DENSE_ROLE:
        from .sla_dense_native_v5 import ENDPOINT
        return ENDPOINT
    return task.E + 500 if recorded_tail(spec) else task.E


def validate_panel(panel):
    from . import task_learning_grasp_v1 as task
    from .task_learning_scene_v1 import scene_id
    from .task_learning_teacher_panel_v1 import teacher_fingerprint, FROZEN_TEACHER
    rows = panel.get('assignments') or []
    task.need(panel.get('schema') == SCHEMA and panel.get('protocol') == protocol()
              and panel.get('task_learning_role') == ROLE
              and panel.get('Teacher_implementation_source_sections_sha256') == teacher_fingerprint() == FROZEN_TEACHER
              and tuple(r.get('layout') for r in rows) == LAYOUTS
              and len({r.get('native_run') for r in rows}) == 4,
              'V3_FOUR_FIXED_TRAIN_EXPERT_SEMANTIC_REQUESTS')
    for row in rows:
        task.need(row.get('parent') == scene_id(row['layout']) and row.get('split') == 'TRAIN'
                  and row.get('command') == task.COMMAND and row.get('purpose') == PURPOSE
                  and row.get('restart') == 1 and row.get('Teacher_mode') == 'NORMAL_TEACHER_V7'
                  and len(row.get('model_sha256', '')) == 64
                  and Path(row['native_run']).name == row['native_run'] and not row['native_run'].startswith('.'),
                  'V3_UNCHANGED_EXPERT_AND_EXPLICIT_TRAIN_SCENE')
    from .task_learning_collection_v1 import qualified_base
    qualified_base(panel['base_Teacher_qualification'])
    return rows


def validate_cadence_panel(panel):
    from . import task_learning_grasp_v1 as task
    from .task_learning_scene_v1 import scene_id
    from .task_learning_teacher_panel_v1 import teacher_fingerprint, FROZEN_TEACHER
    from .task_learning_collection_v1 import qualified_base

    rows = panel['assignments']
    expected = {(layout, restart, atoms) for layout in LAYOUTS for restart in (1, 2) for atoms in CADENCES}
    actual = {(r['layout'], r['restart'], r['prefix_atoms']) for r in rows}
    task.need(panel['schema'] == CADENCE_PANEL and panel['task_learning_role'] == ROLE
              and panel['Teacher_implementation_source_sections_sha256'] == teacher_fingerprint() == FROZEN_TEACHER
              and len(rows) == len(expected) == len({r['native_run'] for r in rows})
              and actual == expected, 'FIXED_SIXTEEN_MATCHED_CADENCE_REQUESTS')
    model_by_layout = {}
    for row in rows:
        sha = row['model_sha256']
        task.need(row['parent'] == scene_id(row['layout']) and row['split'] == 'TRAIN'
                  and row['command'] == task.COMMAND and row['purpose'] == CADENCE_PURPOSE
                  and row['Teacher_mode'] == 'NORMAL_TEACHER_V7'
                  and len(sha) == 64 and sha == model_by_layout.setdefault(row['layout'], sha)
                  and Path(row['native_run']).name == row['native_run'] and not row['native_run'].startswith('.'),
                  'CADENCE_SAME_LAYOUT_MODEL_AND_FROZEN_TEACHER')
    qualified_base(panel['base_Teacher_qualification'])
    return rows


def native_scene_admission(spec):
    if spec.get('sla_five_object_hold_v5')is not None:
        from .sla_five_object_hold_v5 import native_scene_admission as five_object_admission
        return five_object_admission(spec)
    if spec.get('sla_pick_place_v5')is not None:
        from .sla_pick_place_v5 import native_scene_admission as place_admission
        return place_admission(spec)
    from . import task_learning_grasp_v1 as task
    from .task_learning_scene_v1 import validate_static_provenance
    from .native_body_camera_capture_v1 import session_configuration
    task.need(spec.get('task_learning_role') == ROLE and session_configuration(spec)
              and spec.get('task_learning_teacher_mode') == 'NORMAL_TEACHER_V7'
              and spec.get('task_learning_control') is None,
              'V3_FRESH_EXPERT_BODY_SESSION_ONLY')
    proof = pinned(spec['task_learning_scene_reference'])
    panel = pinned(dict(path=spec['stable_task_panel'], sha256=spec['stable_task_panel_sha256']))
    from .sla_named_expert_v5 import (SCHEMA as NAMED, PRODUCTION_SCHEMA, REPAIRED_BENCHMARK_SCHEMA,
                                      LAYOUTS as NAMED_LAYOUTS, validate as named_rows)
    named = panel.get('schema') in (NAMED,PRODUCTION_SCHEMA,REPAIRED_BENCHMARK_SCHEMA)
    task.need(proof.get('layout') in (NAMED_LAYOUTS if named else LAYOUTS), 'V3_TRAIN_ONLY_NEVER_DEV_TEST')
    validate_static_provenance(spec, proof, expected_role=ROLE)
    cadence = panel.get('schema') == CADENCE_PANEL
    rows = named_rows(panel) if named else validate_cadence_panel(panel) if cadence else validate_panel(panel)
    matches = [r for r in rows if r['native_run'] == Path(spec['registered_root']).name]
    task.need(len(matches) == 1 and matches[0]['parent'] == spec['data_parent']
              and matches[0]['model_sha256'] == digest(spec['model_path']),
              'V3_ONE_PREREGISTERED_EXPERT_REQUEST_NO_RETRY')
    if cadence or named:
        task.need(panel['source_sha'] == spec['source_sha'] and recorded_tail(spec)
                  and spec['task_learning_grasp']['prefix_atoms'] == matches[0]['prefix_atoms']
                  and spec['expert_cadence_restart'] == matches[0]['restart'],
                  'FRESH_MATCHED_CADENCE_AND_INDEPENDENT_RESTART_IDENTITY')
    return proof
