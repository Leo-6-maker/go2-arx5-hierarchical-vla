"""Twelve-parent view of the original reader, with one arm-independent cursor.

No new data/processor, native admission, copying or statistical refit is done.
The existing reader still checks numeric/RGB identities and actual EEF labels.
"""
from . import minimal_study as study
from scripts.pi0294.native_dataset import NativeDataset

PANEL_SHA = 'd603324fee6cc69f5b488b47cfaf13dd4bc1feb8d8dc6411a1a739822cfff74d'
ORIGINAL = 'b8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df'
CURSOR_SHA = 'cb73066ac016eca8aec659b66c87059c7edd203f3af8992379267f4a6c454658'


class Dataset(NativeDataset):
    def __init__(self, *args, panel, **kwargs):
        study.need(study.digest(panel) == PANEL_SHA, 'ONLY_FIXED_SIX_PAIRED_GEOMETRIES')
        super().__init__(*args, **kwargs)
        study.need(self.split == 'TRAIN' and self.for_optimization
                   and self.manifest_sha == panel['original_manifest_sha256'] == ORIGINAL
                   and self.statistics_sha == study.R0, 'MINIMAL_ORIGINAL_TRAIN_R0_ONLY')
        originals = {r['id']: r for r in self.manifest['episodes']}
        for selected in panel['parents']:
            actual = originals[selected['id']]
            study.need(all(actual[k] == v for k, v in selected.items()) and actual['split'] == 'TRAIN',
                       'SELECTED_PARENT_BYTES_AND_LINEAGE_UNCHANGED')
        ids = {r['id'] for r in panel['parents']}
        self.rows = [(row, anchor) for row, anchor in self.rows if row['id'] in ids]
        study.need(len(self.rows) == 7212 and {r['id'] for r, _ in self.rows} == ids,
                   'EXACT_TWELVE_PARENTS_AND_7212_WINDOWS')
        self.original_manifest_sha, self.panel = self.manifest_sha, panel
        self.manifest_sha = study.digest(dict(schema=study.SCHEMA, original=ORIGINAL, panel=PANEL_SHA))
        self.seed = panel['seed']
        self.make_sampler = Sampler


class Sampler:
    def __init__(self, dataset, start=0, stop=32768, seed=20261019, positioned=True):
        study.need(type(start) is int and type(stop) is int and 0 <= start <= stop <= 32768
                   and seed == 20261019 and type(positioned) is bool,
                   'MATCHED_SEED_AND_EXACT_COMMITTED_CURSOR_RANGE')
        study.need(study.digest(dataset.panel) == PANEL_SHA, 'FROZEN_PAIRED_PANEL_REQUIRED')
        self.dataset, self.start, self.stop = dataset, start, stop
        self.seed, self.positioned = seed, positioned
        self.lookup = {(r['id'], a): i for i, (r, a) in enumerate(dataset.rows)}
        study.need(len(self.lookup) == len(dataset.rows) == 7212, 'NO_DUPLICATE_OR_EXTRA_READER_WINDOW')
        stream = list(study.paired_cursors(dataset.panel))
        study.need(study.digest(stream) == CURSOR_SHA, 'PREFROZEN_FULL_CURSOR_SEQUENCE_REQUIRED')
        try:
            self.indices = tuple(self.lookup[(r['parent'], r['anchor'])] for r in stream)
        except KeyError as exc:
            raise ValueError('PREFROZEN_WINDOW_MISSING_FROM_ACTUAL_READER') from exc

    def __len__(self):
        return self.stop - self.start

    def index_at(self, cursor):
        study.need(type(cursor) is int and self.start <= cursor < self.stop, 'CURSOR_OUTSIDE_CURRENT_SESSION')
        return self.indices[cursor]

    def __iter__(self):
        for cursor in range(self.start, self.stop):
            index = self.index_at(cursor)
            yield (cursor, index) if self.positioned else index

    def state_dict(self, consumed):
        study.need(type(consumed) is int and self.start <= consumed <= self.stop and consumed % 32 == 0,
                   'ONLY_COMMITTED_32_SAMPLE_BOUNDARIES')
        # Stop is the scientific horizon, not the first512 session boundary.
        # Thus first512 and own-state continuation share the same sampler ID.
        return dict(algorithm='ARX5_MATCHED_TWELVE_PARENT_ABSOLUTE_CURSOR_V1', seed=self.seed,
                    cursor=consumed, stop=32768, panel_sha256=PANEL_SHA,
                    full_cursor_sha256=CURSOR_SHA, manifest_sha256=self.dataset.manifest_sha,
                    statistics_sha256=study.R0, metadata_not_model_input=True)
