"""Serve one exact-qualified Go2 endpoint using the same Core model as RGB16."""
from pathlib import Path
import os


class Pool:
    def __init__(self, spec):
        import jax
        from openpi_client.image_tools import resize_with_pad
        from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
        from scripts.sla.core_hopper_native_scope import accepted
        from scripts.sla.hopper_reference import core_provider
        from scripts.sla.core_canonical_proof import VERIFIED
        from scripts.sla.core_main3800_native_proof import EXPLORATORY, load_manifest
        from scripts.sla.hopper_site import pinned
        from scripts.sla.hopper_parent_custody import digest

        _,gate,site = accepted(spec)
        arm = spec['arm']
        if (spec['physics'] or spec['new_physical_requests_budget'] or spec['new_optimizer_updates_budget']
                or spec['new_training_visits_budget'] or spec['GPU_uuid'] != site['gpu']
                or spec['neural_owner_lock'] != site['gpu_lock'] or os.environ['CUDA_VISIBLE_DEVICES'] != site['gpu']
                or spec['JAX_memory_fraction'] != site['memory_fraction'] or jax.default_backend() != 'gpu'
                or len(jax.local_devices()) != 1 or 'RTX' not in jax.local_devices()[0].device_kind):
            raise PermissionError('Expected one qualified, leased Core inference process')
        source = pinned(site['input'])['target_D1']
        if (source['statistics']['sha256'] != site['model']['state_sha256']
                or digest(Path(site['action']['path'])) != site['action']['sha256']):
            raise ValueError('Frozen state or action statistics changed')
        data = AdaptTargetDataset(source['legacy'],source['captures'],resize_with_pad=resize_with_pad)
        data.bind_state(source['statistics'])
        path = Path(spec['packages'][arm]['path'])
        if gate['status'] == EXPLORATORY:
            from scripts.sla.hopper_reference import provider_from_manifest
            manifest = load_manifest(gate['package'], gate['original_package'])
            model = provider_from_manifest(path,manifest,site['model'],data,site['tokenizer']['path'])
        else:
            model, manifest = core_provider(path,gate['manifest'],site['model'],data,site['tokenizer']['path'],
                                            canonical=gate['status']==VERIFIED)
        if manifest['checkpoint_sha256'] != gate['checkpoint'] or manifest['action_sha256'] != site['action']['sha256']:
            raise ValueError('Student endpoint differs from its exact replay')
        self.providers = {arm:model}
        self.selected = None

    def select(self, arm):
        self.selected = self.providers[arm]

    def predict(self, *args, **kwargs):
        if self.selected is None:
            raise PermissionError('No endpoint selected for this request')
        return self.selected.predict(*args, **kwargs)
