"""Separate logit-label TRAIN100 reader; original RGB/state/visits are reused."""
from pathlib import Path

from .codec_v1 import Quantiles
from .dataset_v1 import TargetDataset
from .gripper_v2 import EPSILON, VERSION, GripLogitCodec
from .runtime_v1 import data as original_data, read_pin


class LogitTargetDataset(TargetDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codecs = {key: GripLogitCodec(value['T_policy_from_mount'])
                       for key, value in self.frames.items()}


def data(spec):
    base = original_data(spec)
    pin = spec['action_representation_audit']
    audit = read_pin(pin)
    if (audit.get('schema') != 'SLA_GRIP_LOGIT_TRAIN100_DERIVATION_V2'
            or not audit.get('passes') or audit['codec_version'] != VERSION
            or audit['epsilon'] != EPSILON
            or audit['original_target_audit']['sha256'] != spec['target_audit']['sha256']
            or audit['source_dataset_sha256'] != base.original.manifest_sha
            or audit['expert_windows'] != 1450 or not audit['state_statistics_unchanged']
            or not audit['pose_action_statistics_unchanged']):
        raise ValueError('EXACT_TRAIN_ONLY_LOGIT_DERIVATION_REQUIRED')
    stats = read_pin(dict(path=str(Path(pin['path']).parent / 'STATISTICS.private.json'),
                         sha256=audit['statistics_sha256']))
    return LogitTargetDataset(base.original.root, spec['camera_audit'], base.frames,
                             resize_with_pad=base.resize,
                             statistics={key: Quantiles(stats[key]['q01'], stats[key]['q99'])
                                         for key in ('state', 'actions')})
