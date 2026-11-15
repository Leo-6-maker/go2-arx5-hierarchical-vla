"""New dense TRAIN seam over the qualified V3 model/optimizer implementation."""
from .checkpoint_v1 import sha
from .dense_dataset_v4 import DensePilotDataset
from .dense_recipe_v4 import RECIPE, CORE_QUALIFICATION_RECIPE, learning_rates, visit_map, microbatch_contract
from .gripper_v2 import VERSION
from .runtime_v1 import read_pin
from .runtime_v3 import DeviceResidentEngine


class DensePilotEngine(DeviceResidentEngine):
    def dataset_for_spec(self, spec):
        from openpi_client.image_tools import resize_with_pad
        microbatch_contract(spec)
        audit = read_pin(spec['target_audit'])
        if (audit.get('schema') != 'SLA_NAMED_DENSE_PILOT_ALL_WINDOW_CPU_READBACK_V4_1'
                or not audit.get('passes') or audit['read_windows'] != 122
                or audit['manifest_sha256'] != spec['dense_manifest_sha256']
                or spec.get('action_representation_audit')
                or sha(spec['tokenizer']['path']) != spec['tokenizer']['sha256']):
            raise ValueError('EXACT_NAMED_DENSE_TARGET_READER_AND_FIXED_V4_1_RECIPE_REQUIRED')
        return DensePilotDataset(spec['dataset'], spec['dense_manifest_sha256'], resize_with_pad=resize_with_pad)

    def visits_for_dataset(self, dataset):
        visits, self.visit_bank_binding = visit_map(dataset)
        return visits

    def learning_rate_values_for_spec(self, spec):
        return learning_rates()

    def __init__(self, spec, output):
        super().__init__(spec, output)
        recipe = CORE_QUALIFICATION_RECIPE if spec.get('operation') == 'DENSE_CORE_MICRO1_QUAL4' else RECIPE
        self.bindings.update(recipe=recipe, codec_version=VERSION,
            derived_statistics_sha256=self.dataset.manifest['statistics']['sha256'],
            dense_manifest_sha256=self.dataset.manifest_sha,
            dataset_id=self.dataset.manifest['dataset_id'], limited_PICK_HOLD_pilot=True,
            D100_new=False, PICK_PLACE_admitted=False, visit_bank=self.visit_bank_binding)
