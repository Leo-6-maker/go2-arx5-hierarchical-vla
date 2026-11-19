"""Fixed first six TRAIN-only PLACE conditions for the SCALE adaptation review."""

PLAN = 'cc96f1709170e47298463337d050c79a24b7c7d9'
SCOPE = 'SLA_SCALE_TARGET_ADAPTATION_TEACHER'
AUTHORITY_SOURCE = 'CURRENT_USER_SCALE_TARGET_ADAPTATION_REVIEW'
STAGE = 'ADAPTATION_TRAIN_PLACE_FIRST6'
SECOND_PLAN = '5734a73cbda5c333cc3d7d7e0f032e679a4c08b6'
SECOND_SCOPE = 'SLA_SCALE_TARGET_ADAPTATION_EXPERT_BATCH2'
SECOND_AUTHORITY = 'CURRENT_USER_MULTISOURCE_EMBODIMENT_AUTONOMOUS_V1'
RELEASE_SCOPE = 'SLA_SCALE_TARGET_ADAPTATION_RELEASE_OFFSET'
RELEASE_STAGE = 'ADAPTATION_TRAIN_PLACE_RELEASE_OFFSET'
RELEASE_CELL = ('F2', 'PLACE-blue-left')
RELEASE_OFFSET_XY_M = (0., .008)
SECOND_STAGE = 'ADAPTATION_TRAIN_PLACE_SECOND6'
CELLS = (
    ('F1', 'PLACE-red-left'), ('F1', 'PLACE-blue-right'),
    ('F2', 'PLACE-red-left'), ('F2', 'PLACE-blue-right'),
    ('F2', 'PLACE-blue-left'), ('F2', 'PLACE-red-right'),
)
NATIVE_TICK_CAP = 50703*len(CELLS)
SCOPES = {SCOPE: (PLAN, AUTHORITY_SOURCE, STAGE), SECOND_SCOPE: (SECOND_PLAN, SECOND_AUTHORITY, SECOND_STAGE),
          RELEASE_SCOPE: (SECOND_PLAN, SECOND_AUTHORITY, RELEASE_STAGE)}


def batch_cells(scope):
    return (RELEASE_CELL,) if scope == RELEASE_SCOPE else CELLS


def tick_cap(scope):
    return 50703*len(batch_cells(scope))


def conditions(population, scope=SCOPE):
    plan, _, stage = SCOPES[scope]
    expected = batch_cells(scope)
    rows = population['conditions'][stage]
    clusters = {row['id']: row for row in population['clusters']}
    if (population['schema'] != 'SLA_SCALE_TARGET_ADAPTATION_POPULATION_V1'
            or population['plan'] != plan or population['old_TEST_reused'] is not False
            or population['assignment_uses_outcomes'] is not False
            or tuple((row['family'],row['instruction']) for row in rows) != expected
            or len(clusters) != len(expected)
            or set(clusters) != {row['cluster'] for row in rows}
            or len({row['id'] for row in rows}) != len(expected)
            or len({row['reset'] for row in rows}) != len(expected)
            or len({row['scene']['sha256'] for row in clusters.values()}) != len(expected)
            or any(row['split'] != 'TRAIN' or row['stage'] != stage for row in clusters.values())):
        raise ValueError('EXPECTED_THE_FROZEN_DISJOINT_TARGET_TRAIN_POPULATION')
    return rows
