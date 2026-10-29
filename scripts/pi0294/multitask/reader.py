"""Bind the existing actual-state/committed-EEF10 reader to new source data.

The reader still hashes NPZ/RGB, uses the existing relative EEF processor and
returns only processed student input. This adds sampling, not a second codec.
No current twelve-demo E1/E2 reader or R0 identity is changed.
"""
from functools import partial

from .sampling import SharedTaskSampler, TASKS, sha


def bind_source_reader(reader, *, seed, recovery_round=False):
    """Prepare an already-audited fixed-base source reader for the shared model.

    New statistics are fitted elsewhere from the initial normal TRAIN bank.
    A later recovery manifest keeps that common version, rather than silently
    refitting normalization when new student visits arrive.
    This function never fits them or initializes the neural processor.
    """
    fit = reader.manifest.get('shared_task_statistics_fit', {})
    train = [r for r in reader.manifest['episodes'] if r['split'] == 'TRAIN']
    normal = [r for r in train if r.get('learning_lane') == 'normal']
    expected = {r['id']: r['sha256'] for r in normal}
    if (reader.split != 'TRAIN' or reader.for_optimization is not True
            or reader.manifest.get('learning_scope') != 'SOURCE_SHARED_FOUR_TASK_EEF10_V1'
            or set(r['task'] for r in normal) != set(TASKS)
            or len({r['id'] for r in train}) != len(train) or not sha(reader.statistics_sha)
            or fit != dict(split='TRAIN', episode_sha256=expected,
                           tasks=list(TASKS), statistics_sha256=reader.statistics_sha)):
        raise ValueError('ONE_NEW_STATISTICS_VERSION_FROM_ALL_FROZEN_SOURCE_TRAIN_TASKS')
    # Validate the entire support now, before the Trainer can request a loader.
    SharedTaskSampler(reader, seed=seed, recovery_round=recovery_round)

    class View:
        def __init__(self):
            self.make_sampler = partial(SharedTaskSampler, seed=seed, recovery_round=recovery_round)

        def __len__(self):
            return len(reader)

        def __getitem__(self, index):
            # The original implementation builds VLAStepData with metadata={}.
            # Teacher events/geometry/recovery provenance never enter it.
            return reader[index]

        def __getattr__(self, name):
            return getattr(reader, name)

    return View()
