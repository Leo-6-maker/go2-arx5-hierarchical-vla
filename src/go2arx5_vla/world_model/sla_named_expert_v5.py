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
PRODUCTION_SCHEMA = 'SLA_NAMED_PICK_HOLD24_PARALLEL_PRODUCTION_V5'
PRODUCTION_RESTARTS = (5,6,7)
PRODUCTION_ACTORS = 3
PRODUCTION_PURPOSE = 'FRESH_NAMED_PICK_HOLD_PARALLEL_PRODUCTION_NOT_MATCHED_ACTOR_BENCHMARK'
REPAIRED_BENCHMARK_SCHEMA = 'SLA_NAMED_PICK_HOLD48_STARTUP_REPAIRED_ACTOR_BENCHMARK_V5'
REPAIRED_BENCHMARK_RESTARTS = (8,9)
REPAIRED_BENCHMARK_PURPOSE = 'FRESH_MATCHED_ACTOR_BENCHMARK_AFTER_EVIDENCED_STARTUP_REPAIR'


def validate(panel):
    from .corelab_expert_semantics_v3 import ROLE
    from .task_learning_teacher_panel_v1 import teacher_fingerprint, FROZEN_TEACHER
    from .task_learning_collection_v1 import qualified_base
    production = panel.get('schema') == PRODUCTION_SCHEMA
    repaired_benchmark = panel.get('schema') == REPAIRED_BENCHMARK_SCHEMA
    rows = panel['assignments']
    expected = ([(PRODUCTION_ACTORS,l,r) for r in PRODUCTION_RESTARTS for l in LAYOUTS]
                if production else [(a,l,r) for a in ACTORS for l in LAYOUTS
                                     for r in (REPAIRED_BENCHMARK_RESTARTS if repaired_benchmark else RESTARTS)])
    task.need(panel.get('schema') in (SCHEMA,PRODUCTION_SCHEMA,REPAIRED_BENCHMARK_SCHEMA) and panel.get('task_learning_role') == ROLE
              and panel['Teacher_implementation_source_sections_sha256'] == teacher_fingerprint() == FROZEN_TEACHER
              and panel['tasks'] == ['PICK_HOLD'] and panel['D100_new'] is False
              and panel['PICK_PLACE_admitted'] is False and panel['old_engineering_episodes_reused'] == 0
              and len(rows) == len(expected) and len({r['native_run'] for r in rows}) == len(expected),
              'NEW_NAMED_PICK_HOLD_NOT_FORMAL_D100_OR_OLD_ENGINEERING_DATA')
    if production:
        task.need(panel.get('matched_actor_benchmark') is False
                  and panel.get('other_live_collection_preserved') is True,
                  'PARALLEL_PRODUCTION_IS_NOT_AN_ISOLATED_ACTOR_THROUGHPUT_COMPARISON')
    if repaired_benchmark:
        task.need(panel.get('matched_actor_benchmark') is True
                  and panel.get('other_live_collection_preserved') is True
                  and panel.get('startup_repair') == dict(paused_readiness_s=600,total_wall_s=3600,
                      original_native_markers_required=True,physics_or_Teacher_changed=False)
                  and panel.get('prior_partial_benchmark_relabelled') is False,
                  'FRESH_MATCHED_POPULATION_SAME_REPAIRED_STARTUP_AT_ALL_ACTOR_COUNTS')
    models = {}
    for row,(actors,layout,restart) in zip(rows,expected,strict=True):
        task.need(row['actor_count'] == actors and row['layout'] == layout and row['restart'] == restart
                  and row['prefix_atoms'] == 10 and row['parent'] == scene_id(layout)
                  and row['split'] == 'TRAIN' and row['command'] == task.COMMAND
                  and row['purpose'] == (PRODUCTION_PURPOSE if production else
                      REPAIRED_BENCHMARK_PURPOSE if repaired_benchmark else PURPOSE)
                  and row['Teacher_mode'] == 'NORMAL_TEACHER_V7'
                  and len(row['model_sha256']) == 64
                  and row['model_sha256'] == models.setdefault(layout,row['model_sha256'])
                  and Path(row['native_run']).name == row['native_run'],
                  'SAME_SIXTEEN_CONDITIONS_AND_RECORDING_CONTRACT_AT_EACH_ACTOR_COUNT')
    qualified_base(panel['base_Teacher_qualification'])
    return rows
