"""Fresh PICK_HOLD collection and matched 1/2/4-world throughput experiment.

These are new resets with complete command/RGB records. Neither old
engineering episodes nor the two-episode fitting set are relabelled D100.
"""
from pathlib import Path

from . import task_learning_grasp_v1 as task
from .task_learning_scene_v1 import scene_id

SCHEMA = 'SLA_NAMED_PICK_HOLD48_AND_ACTOR_BENCHMARK_V5'
LAYOUTS = ('TRAIN01','TRAIN04','TRAIN07','TRAIN10','TRAIN13','TRAIN16','TRAIN21','TRAIN25')
ACTORS = (1,2,4)
RESTARTS = (3,4)
PURPOSE = 'FRESH_NAMED_PICK_HOLD_COMPLETE_TARGET_AND_MATCHED_ACTOR_THROUGHPUT'


def validate(panel):
    from .corelab_expert_semantics_v3 import ROLE
    from .task_learning_teacher_panel_v1 import teacher_fingerprint, FROZEN_TEACHER
    from .task_learning_collection_v1 import qualified_base
    rows = panel['assignments']
    expected = [(a,l,r) for a in ACTORS for l in LAYOUTS for r in RESTARTS]
    task.need(panel.get('schema') == SCHEMA and panel.get('task_learning_role') == ROLE
              and panel['Teacher_implementation_source_sections_sha256'] == teacher_fingerprint() == FROZEN_TEACHER
              and panel['tasks'] == ['PICK_HOLD'] and panel['D100_new'] is False
              and panel['PICK_PLACE_admitted'] is False and panel['old_engineering_episodes_reused'] == 0
              and len(rows) == 48 and len({r['native_run'] for r in rows}) == 48,
              'NEW_NAMED_PICK_HOLD_NOT_FORMAL_D100_OR_OLD_ENGINEERING_DATA')
    models = {}
    for row,(actors,layout,restart) in zip(rows,expected,strict=True):
        task.need(row['actor_count'] == actors and row['layout'] == layout and row['restart'] == restart
                  and row['prefix_atoms'] == 10 and row['parent'] == scene_id(layout)
                  and row['split'] == 'TRAIN' and row['command'] == task.COMMAND
                  and row['purpose'] == PURPOSE and row['Teacher_mode'] == 'NORMAL_TEACHER_V7'
                  and len(row['model_sha256']) == 64
                  and row['model_sha256'] == models.setdefault(layout,row['model_sha256'])
                  and Path(row['native_run']).name == row['native_run'],
                  'SAME_SIXTEEN_CONDITIONS_AND_RECORDING_CONTRACT_AT_EACH_ACTOR_COUNT')
    qualified_base(panel['base_Teacher_qualification'])
    return rows
