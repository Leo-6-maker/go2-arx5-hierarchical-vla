"""D4 A/B target adaptation with a single controlled initialization change."""
import hashlib
from pathlib import Path

import numpy as np

from .checkpoint_v1 import sha
from .close_temporal_v1 import CloseEngine
from .dense_g7_v6 import VERSION, label_margin_report
from .gripper_latent_v2 import EPSILON
from .mfree_training_v1 import MobileDataset
from .multitask_data_v6 import C20_MANIFEST, C20_STATISTICS, legacy_contexts, reference
from .multitask_initialization_v6 import sealed_source
from .multitask_training_v6 import MultitaskEngine
from .provider_multitask_v6 import Provider as TargetProvider, export_payload
from .provider_v3 import DECODE_ORDER
from .runtime_v1 import read_pin

ARMS = ('D4-A', 'D4-B')
UPDATES = 500
VISITS = UPDATES * 32
RECIPE = 'SLA_D4_C20_G7_MATCHED_TRANSFER_500'
SCHEMA = 'SLA_D4_C20_G7_RGB_PACKAGE_V1'


class C20G7Dataset(MobileDataset):
    def __init__(self, root, *, time_audit, portable_inputs=None, **kwargs):
        super().__init__(root, C20_MANIFEST, **kwargs)
        self.contexts = legacy_contexts(self, time_audit,
            read_pinned=lambda pin: read_pin(locate_input(portable_inputs, pin)))
        if self.manifest['statistics']['sha256'] != C20_STATISTICS:
            raise ValueError('D4_FROZEN_C20_TRAIN_STATISTICS_REQUIRED')

    def reference(self, index):
        return reference(self.contexts[index])


def locate_input(portable_inputs, pin):
    """Resolve an archived input without changing its original digest or JSON bytes."""
    if portable_inputs is None:
        return pin
    manifest = read_pin(portable_inputs)
    root = Path(portable_inputs['path']).parent.resolve(strict=True)
    member = manifest['members'][pin['path']]
    path = (root/member['file']).resolve(strict=True)
    if member['sha256'] != pin['sha256'] or not path.is_relative_to(root):
        raise ValueError('D4_PORTABLE_INPUT_DOES_NOT_MATCH_ORIGINAL_PIN')
    return dict(path=str(path),sha256=pin['sha256'])


class TransferEngine(MultitaskEngine):
    recipe = RECIPE
    source_arm = ARMS[1]

    def dataset_for_spec(self, spec):
        from openpi_client.image_tools import resize_with_pad

        audit = read_pin(spec['target_audit'])
        if (spec['operation'] != RECIPE or spec['arm'] not in ARMS
                or spec['microbatch'] != 4 or spec['training_matmul_precision'] != 'highest'
                or spec['planned_candidate_updates'] != UPDATES
                or spec['new_optimizer_updates_budget'] != UPDATES
                or spec['new_training_visits_budget'] != VISITS
                or spec['physics'] or spec['new_physical_requests_budget']
                or spec['dense_manifest_sha256'] != C20_MANIFEST
                or spec['inference_route'] != 'cached_actual'
                or spec['aux_samples_per_update'] != 4 or spec['aux_sampling_steps'] != 10
                or bool(spec.get('S1000_transfer_result')) != (spec['arm'] == self.source_arm)
                or any(spec.get(key) for key in ('B1_parent_package', 'C20_parent_package',
                    'action_representation_audit', 'resume', 'continuation'))
                or audit['schema'] != 'MFREE_TRAIN_FP32_DEPLOYMENT_AUDIT_V1'
                or not audit['passes'] or audit['manifest_sha256'] != C20_MANIFEST
                or sha(spec['tokenizer']['path']) != spec['tokenizer']['sha256']):
            raise ValueError('D4_EXACT_MATCHED_C20_TRAIN_CONTRACT_REQUIRED')
        return C20G7Dataset(spec['dataset'], time_audit=spec['time_codec_audit'],
            portable_inputs=spec.get('portable_inputs'),
            resize_with_pad=resize_with_pad, max_cache_bytes=spec['frame_cache_bytes'])

    def source_initialization(self, spec):
        return sealed_source(spec['S1000_transfer_result'], spec['parent_manifest'],
            locate_pin=lambda pin: locate_input(spec.get('portable_inputs'), pin))

    def visits_for_dataset(self, dataset):
        values = CloseEngine.visits_for_dataset(self, dataset)
        self.visit_bank_binding = dict(target_visits=VISITS, source_visits=0,
            sampling='ORIGINAL_C20_TASK_EPISODE_PHASE_ANCHOR_BANK',
            sha256=hashlib.sha256(values.tobytes()).hexdigest())
        return values

    def learning_rate_values_for_spec(self, spec):
        rates = CloseEngine.learning_rate_values_for_spec(self, spec)
        digest = hashlib.sha256(''.join(rate.hex()+'\n' for rate in rates).encode()).hexdigest()
        if digest != spec['learning_rates_sha256']:
            raise ValueError('D4_MATCHED_C20_500_ENTRY_LR_REQUIRED')
        return rates

    def population_binding(self):
        return dict(C20_only=True, C20_time_audit=self.spec['time_codec_audit'],
                    Teacher_episodes=0, source_optimization_visits=0)


def package_contract(manifest):
    bindings = manifest['bindings']
    if (manifest['schema'] != SCHEMA or manifest['codec'] != VERSION
            or manifest['decode_order'] != DECODE_ORDER or manifest['label_epsilon'] != EPSILON
            or (manifest['step'], manifest['cursor']) != (UPDATES, VISITS)
            or len(manifest['adapter_arrays']) != 28 or bindings['recipe'] != RECIPE
            or bindings['arm'] not in ARMS or bindings['codec_version'] != VERSION
            or bindings['inference_route'] != 'cached_actual'
            or bindings['dataset_manifest_sha256'] != C20_MANIFEST
            or manifest['statistics_sha256'] != C20_STATISTICS
            or bindings['derived_statistics_sha256'] != C20_STATISTICS
            or not bindings['C20_only'] or bindings['Teacher_episodes']
            or bindings['source_optimization_visits']
            or bool(bindings['source_initialization']) != (bindings['arm'] == ARMS[1])
            or manifest['previous_reference_required'] is not True
            or any(manifest[key] is not False for key in ('teacher_required', 'training_cache_required',
                'optimizer_required', 'state_gripper_transformed', 'physical_execution_admitted',
                'endpoint_basis_applied'))):
        raise ValueError('D4_EXACT_FIXED_C20_RGB_PACKAGE_REQUIRED')


def export(package, checkpoint, *, engine):
    statistics = engine.dataset.root/engine.dataset.manifest['statistics']['file']
    return export_payload(package, checkpoint, engine=engine, statistics=statistics,
                          schema=SCHEMA, validate=package_contract, step=UPDATES, cursor=VISITS)


class Provider(TargetProvider):
    def validate_package_contract(self, manifest):
        package_contract(manifest)

    def validate_statistics_metadata(self, statistics):
        from .gripper_v2 import VERSION as LOGIT_VERSION

        if (statistics['codec_version'] != LOGIT_VERSION or statistics['split'] != 'TRAIN'
                or statistics['state_gripper_transformed'] is not False
                or statistics['epsilon'] != EPSILON or statistics['fit_episodes'] != 20
                or statistics['fit_unique_windows'] != 1376):
            raise ValueError('D4_C20_TRAIN_ONLY_STATISTICS_REQUIRED')
