"""Read a frozen SCALE20K target snapshot with the original G7 contract."""
from pathlib import Path

from .checkpoint_v1 import sha
from .dense_g7_v6 import VERSION
from .multisource_d4 import C20G7Dataset
from .multitask_data_v6 import C20_MANIFEST, C20_STATISTICS, MultitaskDataset, TeacherDataset
from .runtime_v1 import read_pin

SCHEMA = 'SLA_SCALE20K_TARGET_C20_AND_ADMITTED_PLACE_V1'


class HistoricalPlace(TeacherDataset):
    def __init__(self, pin, legacy, *, resize_with_pad, cache_bytes):
        self.manifest = manifest = read_pin(pin)
        self.manifest_sha = pin['sha256']
        self.root = Path(pin['path']).parent
        if (manifest['schema'] != SCHEMA or manifest['split'] != 'TRAIN'
                or manifest['state'] != 'committed' or manifest['representation'] != VERSION
                or manifest['C20_manifest_sha256'] != C20_MANIFEST
                or manifest['C20_statistics_sha256'] != C20_STATISTICS
                or any(manifest[key] is not False for key in (
                    'phase_input_to_student', 'teacher_memory_input', 'unconsumed_suffix_is_executed'))
                or len(manifest['episodes']) > 3
                or any(not ep['task'].startswith('PLACE-') or ep['individual_pass'] is not True
                       for ep in manifest['episodes'])):
            raise ValueError('SCALE20K_FROZEN_PLACE_TRAIN_ADMISSION_REQUIRED')
        split = read_pin(manifest['split_inventory'])
        audit = read_pin(manifest['audit'])
        cases = {r['id']: r for r in audit['cases'] if r['admitted']}
        if (split['old_TEST_rollouts_read'] or len(cases) != len(manifest['episodes'])
                or len(audit['cases']) != 3
                or any(cases.get(ep['id'], {}).get('result_sha256') != ep['native_result']['sha256']
                       for ep in manifest['episodes'])):
            raise ValueError('SCALE20K_EXACT_PER_CAPTURE_AND_SPLIT_RECEIPT_REQUIRED')
        self.statistics = legacy.statistics
        self._load_episodes(resize_with_pad, cache_bytes)


class ScaleTargetDataset(MultitaskDataset):
    def __init__(self, spec, *, resize_with_pad, cache_bytes=512*1024**2):
        legacy_manifest = Path(spec['root'])/'MANIFEST.private.json'
        pin = spec['manifest']
        if (Path(pin['path']).resolve(strict=True) != legacy_manifest.resolve(strict=True)
                or pin['sha256'] != C20_MANIFEST or sha(legacy_manifest) != C20_MANIFEST):
            raise ValueError('SCALE20K_EXACT_C20_ANCESTOR_REQUIRED')
        self.legacy = C20G7Dataset(spec['root'], time_audit=spec['time_audit'],
            resize_with_pad=resize_with_pad, max_cache_bytes=cache_bytes//2)
        self.new = HistoricalPlace(spec['admission'], self.legacy,
            resize_with_pad=resize_with_pad, cache_bytes=cache_bytes//2)
        self.rows = self.legacy.rows + self.new.rows
        self.statistics = self.legacy.statistics
        self.labels = self.legacy.labels | self.new.labels
        self.codecs = self.legacy.codecs | self.new.codecs
        self.manifest_sha = self.new.manifest_sha
        self.manifest = dict(episodes=self.legacy.manifest['episodes'] + self.new.manifest['episodes'])
        if len(self.codecs) != len(self.manifest['episodes']) or len(self) != self.new.manifest['total_windows']:
            raise ValueError('SCALE20K_TARGET_EPISODE_AND_WINDOW_DENOMINATOR_CHANGED')
