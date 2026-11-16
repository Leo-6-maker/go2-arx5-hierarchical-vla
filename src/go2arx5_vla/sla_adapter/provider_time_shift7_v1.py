"""Distinct offline-only ordinary RGB/reference package for the slot intervention."""
import json
import shutil
from pathlib import Path
from .checkpoint_v1 import sha
from .provider_time_v1 import Provider as OriginalTimeProvider, ORDER
from .gripper_time_shift7_v1 import VERSION, SHIFT, SLOTS
from .gripper_latent_v2 import VERSION as STATS_CODEC, EPSILON
from .time_shift7_training_v1 import RECIPE, physical

SCHEMA = 'SLA_REFERENCE_CONTINUED_GRIP_TIME_SHIFT7_RGB_PACKAGE_V1'


def contract(m):
    b = m['bindings']
    if (m['schema'] != SCHEMA or m['codec'] != VERSION or m['decode_order'] != ORDER
            or m['parameter_slot_shift'] != SHIFT or m['parameter_slots'] != [[int(i) for i in g] for g in SLOTS]
            or m['step'] != 500 or m['cursor'] != 16000 or b['parent_step'] != 1000
            or b['parent_visits'] != 32000 or b['new_planned_updates'] != 500
            or b['recipe'] != RECIPE or b['codec_version'] != VERSION
            or b['parameter_slot_shift'] != SHIFT or b['statistics_codec_version'] != STATS_CODEC
            or b['gripper_lambda'] != 0 or m['previous_reference_required'] is not True
            or any(m[k] is not False for k in ('teacher_required', 'optimizer_required',
                'training_cache_required', 'physical_execution_admitted'))):
        raise ValueError('DISTINCT_MATCHED_SHIFT7_PACKAGE_REQUIRED')


def export(package, checkpoint, *, complete, active_template, statistics, tokenizer, parent_manifest):
    from .provider_v1 import active_rows
    b = complete['bindings']; checkpoint = Path(checkpoint); package = Path(package)
    if (complete['step'] != 500 or complete['cursor'] != 16000 or b['recipe'] != RECIPE
            or b['parameter_slot_shift'] != SHIFT or b['gripper_lambda'] != 0
            or b['parent_manifest'] != parent_manifest or b['derived_statistics_sha256'] != sha(statistics)
            or b['tokenizer'] != sha(tokenizer)):
        raise ValueError('ACTUAL_SHIFT500_AND_UNCHANGED_ASSETS_REQUIRED')
    package.mkdir(exist_ok=False); (package / 'adapters').mkdir(); rows, _ = active_rows(complete, active_template)
    copied = []
    for row in rows:
        path = checkpoint / row['file']
        if not path.resolve().is_relative_to(checkpoint.resolve()) or sha(path) != row['sha256']:
            raise ValueError('SHIFT500_ARRAY_CHANGED')
        target = package / 'adapters' / row['file']; shutil.copyfile(path, target)
        copied.append(dict(row, file='adapters/' + row['file']))
    shutil.copyfile(statistics, package / 'STATISTICS.json')
    shutil.copyfile(tokenizer, package / 'paligemma_tokenizer.model')
    result = dict(schema=SCHEMA, codec=VERSION, decode_order=ORDER,
        parameter_slots=[[int(i) for i in g] for g in SLOTS], parameter_slot_shift=SHIFT,
        label_epsilon=EPSILON, state_gripper_transformed=False, step=500, cursor=16000, bindings=b,
        parent_manifest=parent_manifest, original_complete_sha256=sha(checkpoint / 'COMPLETE.json'),
        adapter_arrays=copied, statistics_sha256=sha(package / 'STATISTICS.json'),
        tokenizer_sha256=sha(package / 'paligemma_tokenizer.model'),
        camera_keys=['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'], image_masks=[True, True, False],
        image_size=[224, 224], sampler_steps=10, action_horizon=40, label_interval_s=.05,
        previous_reference_required=True, teacher_required=False, optimizer_required=False,
        training_cache_required=False, physical_execution_admitted=False, endpoint_basis_applied=False)
    contract(result); (package / 'MANIFEST.private.json').write_text(json.dumps(result, indent=2)); return result


class Provider(OriginalTimeProvider):
    physical = staticmethod(physical)

    def validate_package_contract(self, manifest):
        contract(manifest)
