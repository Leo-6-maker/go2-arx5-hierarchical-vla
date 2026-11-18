"""Ordinary RGB G7 with the original one-sigmoid codec and cached sampler."""
from pathlib import Path
import shutil

from scripts.pi0292.durable import write
from .checkpoint_v1 import sha
from .dense_g7_v6 import VERSION, label_margin_report, previous_command_context
from .gripper_latent_v2 import EPSILON
from .multitask_data_v6 import SCHEMA as DATA_SCHEMA
from .multitask_contract_v6 import SCHEMA, contract
from .provider_v1 import active_rows
from .provider_v3 import DECODE_ORDER, Provider as LogitProvider
from .runtime_v1 import read_pin

def export(package, checkpoint, *, engine):
    checkpoint_root = Path(checkpoint['path'])
    complete = read_pin(dict(path=str(checkpoint_root/'COMPLETE.json'), sha256=checkpoint['sha256']))
    bindings = complete['bindings']
    statistics = engine.dataset.new.root/engine.dataset.manifest['statistics']['file']
    tokenizer = engine.spec['tokenizer']['path']
    if (complete['step'] != 1000 or complete['cursor'] != 32000 or bindings != engine.bindings
            or sha(statistics) != bindings['derived_statistics_sha256'] or sha(tokenizer) != bindings['tokenizer']):
        raise ValueError('ACTUAL_MATCHED_G7_ENDPOINT_AND_FROZEN_RGB_ASSETS_REQUIRED')
    package = Path(package)
    rows, _ = active_rows(complete, engine.active)
    package.mkdir(); (package/'adapters').mkdir()
    copied = []
    for row in rows:
        source = checkpoint_root/row['file']
        if not source.resolve().is_relative_to(checkpoint_root.resolve()) or sha(source) != row['sha256']:
            raise ValueError('BIND_ALL_G7_ENDPOINT_ADAPTER_BYTES')
        file = 'adapters/'+row['file']
        shutil.copyfile(source, package/file)
        copied.append(dict(row,file=file))
    shutil.copyfile(statistics,package/'STATISTICS.json')
    shutil.copyfile(tokenizer,package/'paligemma_tokenizer.model')
    manifest = dict(schema=SCHEMA, codec=VERSION, decode_order=DECODE_ORDER, label_epsilon=EPSILON,
        state_gripper_transformed=False, step=1000, cursor=32000, bindings=bindings, parent_manifest=bindings['parent_manifest'],
        original_complete_sha256=checkpoint['sha256'], adapter_arrays=copied, statistics_sha256=sha(package/'STATISTICS.json'),
        tokenizer_sha256=sha(package/'paligemma_tokenizer.model'), camera_keys=['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'],
        image_masks=[True,True,False], image_size=[224,224], sampler_steps=10, action_horizon=40, label_interval_s=.05,
        previous_reference_required=True, teacher_required=False, training_cache_required=False, optimizer_required=False,
        physical_execution_admitted=False, endpoint_basis_applied=False)
    contract(manifest)
    path = package/'MANIFEST.private.json'
    write(path,manifest)
    return dict(path=str(package),sha256=sha(path))


class Provider(LogitProvider):
    def validate_package_contract(self, manifest):
        contract(manifest)

    def validate_statistics_metadata(self, stats):
        if stats['schema'] != DATA_SCHEMA or stats['representation'] != VERSION or stats['split'] != 'TRAIN':
            raise ValueError('COMMON_G7_TRAIN_STATISTICS_REQUIRED')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from openpi.models import pi0
        from scripts.sla.close_goal_g1_v1 import make_sampler
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields
        from . import batch_v2

        fields, _ = extracted_fields(batch_v2, pi0)
        sample = make_sampler(self.graph, fields, cached=True, precision=None)
        self.infer = lambda active, frozen, obs, noise: sample(active, frozen, obs, noise)[0]

    def predict(self, *args, previous_reference, **kwargs):
        previous_command_context(previous_reference)
        result = super().predict(*args, **kwargs)
        result['codec'] = VERSION
        result['G7_gripper_speed'] = label_margin_report(result['physical_H40'], previous_reference)
        return result
