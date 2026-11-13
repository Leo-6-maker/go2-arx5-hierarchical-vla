"""V4.1 limited target pilot. It is neither D100 nor the D300 T100 arm."""
import math
import numpy as np

from .config_v1 import SEED
from .mixed_visits_v4 import Catalog, MixedVisits

RECIPE = 'SLA_NAMED_DENSE_PICK_HOLD_FM32_3K_V4_1'
CORE_QUALIFICATION_RECIPE = 'SLA_NAMED_DENSE_PICK_HOLD_CORE_MICRO1_QUAL4_V4_1'
UPDATES = 3000
WARMUP = 150


def microbatch_contract(spec):
    expected = 1 if spec.get('operation') == 'DENSE_CORE_MICRO1_QUAL4' else 8
    if spec.get('microbatch') != expected or spec.get('training_matmul_precision') != 'highest':
        raise ValueError('ONLY_DISTINCT_CORE_QUAL4_MAY_USE_MICRO1_MAIN_RECIPE_REMAINS_MICRO8')


def learning_rates():
    return [5e-5*j/WARMUP if j <= WARMUP else
            1e-5+.5*(5e-5-1e-5)*(1+math.cos(math.pi*(j-WARMUP)/(UPDATES-WARMUP)))
            for j in range(1, UPDATES+1)]


def visit_map(dataset):
    catalog = Catalog('T', dataset.manifest_sha, dataset.catalog_rows())
    # Reuse the target-only task/episode/phase/anchor bank. This does not
    # relabel the two-episode pilot as the formal D300 comparison arm.
    planner = MixedVisits('T100_3K', {'T': catalog}, seed=SEED)
    values = np.fromiter((v.row_index for step in range(UPDATES)
                         for v in planner.batch(step)), dtype='<i8', count=UPDATES*32)
    return values, dict(catalog_sha256=catalog.sha256, seed=SEED,
        sampling='TASK_THEN_EPISODE_THEN_PHASE_THEN_ANCHOR',
        student_phase_input=False, planned_target_visits=UPDATES*32,
        planned_source_visits=0, formal_D300_comparison_arm=False)
