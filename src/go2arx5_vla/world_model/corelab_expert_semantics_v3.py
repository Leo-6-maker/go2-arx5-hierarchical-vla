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


def protocol():
    from . import task_learning_grasp_v1 as task
    return dict(task.protocol(task.CONSTRUCTION_ROLE), stage=ROLE,
                scene_family='FOUR_PREREGISTERED_TRAIN_LAYOUTS_ENGINEERING_ONLY',
                observation='BODY_BASE_FIXED_V1_20261111_PLUS_ORIGINAL_NATIVE_WRIST')


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


def native_scene_admission(spec):
    from . import task_learning_grasp_v1 as task
    from .task_learning_scene_v1 import validate_static_provenance
    from .native_body_camera_capture_v1 import session_configuration
    task.need(spec.get('task_learning_role') == ROLE and session_configuration(spec)
              and spec.get('task_learning_teacher_mode') == 'NORMAL_TEACHER_V7'
              and spec.get('task_learning_control') is None,
              'V3_FRESH_EXPERT_BODY_SESSION_ONLY')
    proof = pinned(spec['task_learning_scene_reference'])
    task.need(proof.get('layout') in LAYOUTS, 'V3_TRAIN_ONLY_NEVER_DEV_TEST')
    validate_static_provenance(spec, proof, expected_role=ROLE)
    panel = pinned(dict(path=spec['stable_task_panel'], sha256=spec['stable_task_panel_sha256']))
    rows = validate_panel(panel)
    matches = [r for r in rows if r['native_run'] == Path(spec['registered_root']).name]
    task.need(len(matches) == 1 and matches[0]['parent'] == spec['data_parent']
              and matches[0]['model_sha256'] == digest(spec['model_path']),
              'V3_ONE_PREREGISTERED_EXPERT_REQUEST_NO_RETRY')
    return proof
