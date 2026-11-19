"""Build the three SCALE20K dataset readers and a reproducible common bank."""
from collections import Counter
from pathlib import Path

from .checkpoint_v1 import sha
from .libero_scale20k import LiberoDataset
from .runtime_v1 import read_pin
from .scale20k_contract import DOMAINS, bank_sha256, visits
from .scale20k_target import ScaleTargetDataset
from .source_native_v6 import SourceDataset


class RobodojoDataset(SourceDataset):
    def __init__(self, root, manifest, *, raw_binding=None, **kwargs):
        if Path(manifest['path']).resolve(strict=True) != (Path(root)/'MANIFEST.private.json').resolve(strict=True):
            raise ValueError('SCALE20K_ROBODOJO_MANIFEST_PATH_INCONSISTENT')
        super().__init__(root,manifest['sha256'],**kwargs)
        if self.manifest['recipe'] != 'SLA_SCALE20K_ROBODOJO_TRAIN48_DEV12_V1' or self.manifest['TEST_labels_read']:
            raise ValueError('SCALE20K_ROBODOJO_TYPED_TRAIN_DEV_ONLY_REQUIRED')
        self.raw_binding = None
        if raw_binding is not None:
            binding = read_pin(raw_binding)
            if (kwargs['split'] != 'SOURCE_DEV' or binding['manifest_sha256'] != manifest['sha256']
                    or set(binding['raw']) != set(self.episodes)
                    or any(binding['raw'][episode]['sha256'] != row['raw_sha256']
                           for episode,row in self.episodes.items())):
                raise ValueError('SCALE20K_RELOCATED_RAW_SOURCE_REQUIRES_THE_ORIGINAL_DEV_BYTES')
            self.raw_binding = binding['raw']
        self.verified_raw = set()

    def raw_path(self,episode):
        path = super().raw_path(episode)
        if self.raw_binding is not None:
            site_root = self.root.resolve(strict=True)
            path = (site_root/self.raw_binding[episode]['file']).resolve(strict=True)
            if not path.is_relative_to(site_root):
                raise ValueError('SCALE20K_RELOCATED_RAW_SOURCE_MUST_STAY_INSIDE_SITE_ROOT')
        if episode not in self.verified_raw:
            if sha(path) != self.episodes[episode]['raw_sha256']:
                raise ValueError('SCALE20K_ROBODOJO_PINNED_RGB_FILE_BYTES_CHANGED')
            self.verified_raw.add(episode)
        return path


def readers(plan, resize_with_pad, *, source_split='TRAIN'):
    if set(plan) != set(DOMAINS):
        raise ValueError('SCALE20K_REQUIRES_REAL_DATA_IN_ALL_THREE_DOMAINS')
    target,rd,libero = (plan[domain] for domain in DOMAINS)
    d4 = ScaleTargetDataset(target, resize_with_pad=resize_with_pad)
    source = RobodojoDataset(rd['root'],rd['manifest'],split=source_split,resize_with_pad=resize_with_pad,
                             raw_binding=rd.get('raw_binding'))
    library = LiberoDataset(libero['root'],libero['manifest'],split=source_split,resize_with_pad=resize_with_pad)
    return dict(zip(DOMAINS,(d4,source,library),strict=True))


def bank_for(datasets):
    target,source,libero = (datasets[domain] for domain in DOMAINS)
    counts = Counter(row['id'] for row,_ in target.rows)
    # Balance HOLD and PLACE as skills. With only three PLACE recordings,
    # treating each PLACE wording as a separate sampling task would make
    # these three recordings the majority of target updates.
    target_rows = [dict(episode=row['id'],task='PLACE' if row['task'].startswith('PLACE-') else row['task'],
                        split=row['split'],admitted=True,window_count=counts[row['id']])
                   for row in target.manifest['episodes']]
    source_rows = [dict(episode=row['id'],task=row['task'],split=row['split'],admitted=row['valid'],window_count=row['windows'])
                   for row in source.episodes.values()]
    each = dict(zip(DOMAINS,(target_rows,source_rows,list(libero.episodes.values())),strict=True))
    if any(sum(r['window_count'] for r in each[d]) != len(datasets[d]) for d in DOMAINS):
        raise ValueError('SCALE20K_VISIT_BANK_AND_READER_INDEX_MUST_MATCH')
    banks = {domain:visits(each[domain],domain) for domain in DOMAINS}
    return banks,{domain:bank_sha256(bank) for domain,bank in banks.items()}
