"""Fresh named N20 data over the qualified dense model/optimizer implementation."""
from .checkpoint_v1 import sha
from .dense_recipe_v4 import microbatch_contract
from .named_dataset_v5 import AUDIT_SCHEMA, DATASET_ID, RECIPE, NamedTargetDataset, frame_cache_limit
from .runtime_v1 import read_pin
from .runtime_v4 import DensePilotEngine


class NamedTargetEngine(DensePilotEngine):
    def dataset_for_spec(self, spec):
        from openpi_client.image_tools import resize_with_pad
        from pathlib import Path
        microbatch_contract(spec)
        audit = read_pin(spec['target_audit'])
        if (audit.get('schema') != AUDIT_SCHEMA or not audit.get('passes')
                or audit['read_windows'] != 1220 or audit['episodes'] != 20
                or audit['dataset_id'] != DATASET_ID
                or audit['manifest_sha256'] != spec['dense_manifest_sha256']
                or spec.get('action_representation_audit')
                or sha(spec['tokenizer']['path']) != spec['tokenizer']['sha256']):
            raise ValueError('ALL_FRESH_N20_READBACK_BEFORE_MODEL_LOAD_REQUIRED')
        cache=frame_cache_limit(spec)
        if 'frame_cache_bytes'in spec:
            available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines()
                           if v.startswith('MemAvailable:'))
            if available<cache+24*1024**3:
                raise ValueError('HOST_FRAME_CACHE_REQUIRES_MODEL_LOAD_AND_EXISTING_WORKER_HEADROOM')
        return NamedTargetDataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad,
                                  max_cache_bytes=cache)

    def __init__(self, spec, output):
        super().__init__(spec,output)
        self.bindings.update(recipe=RECIPE, limited_PICK_HOLD_pilot=False,
            limited_fresh_named_N20_fitting=True, D100_new=False, D300_new=False,
            old_2EP_reused=0, old_engineering16_reused=0)
        if 'frame_cache_bytes'in spec:
            self.bindings['input_frame_cache_limit_bytes']=frame_cache_limit(spec)
