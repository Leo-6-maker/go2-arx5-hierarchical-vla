"""Same resident dense learner;separately bound current-anchor target labels."""
from .named_dataset_v5 import frame_cache_limit
from .dense_recipe_v4 import microbatch_contract
from .runtime_named_v5 import NamedTargetEngine
from .runtime_v1 import read_pin
from .checkpoint_v1 import sha
from .reference_labels_v6 import VERSION
from .reference_dataset_v6 import ReferenceTargetDataset,AUDIT_SCHEMA,DATASET_ID,RECIPE,ORIGINAL_MANIFEST_SHA


class ReferenceTargetEngine(NamedTargetEngine):
    def dataset_for_spec(self,spec):
        from pathlib import Path
        from openpi_client.image_tools import resize_with_pad
        microbatch_contract(spec);audit=read_pin(spec['target_audit'])
        if (audit.get('schema')!=AUDIT_SCHEMA or not audit.get('passes')
                or (audit['full_H40_envelope_pass'],audit['FP32_decoded_envelope_pass'])!=(1220,1220)
                or audit['read_windows']!=1220 or audit['original_manifest_sha256']!=ORIGINAL_MANIFEST_SHA
                or audit['label_semantics']!=VERSION or spec.get('action_representation_audit')
                or spec['reference_label_semantics']!=VERSION
                or sha(spec['tokenizer']['path'])!=spec['tokenizer']['sha256']):
            raise ValueError('ALL1220_CURRENT_ANCHOR_COMMAND_IDENTITIES_BEFORE_MODEL_LOAD')
        cache=frame_cache_limit(spec)
        available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:'))
        if available<cache+24*1024**3:raise ValueError('REFERENCE_INPUT_CACHE_REQUIRES_EXISTING_MODEL_HEADROOM')
        return ReferenceTargetDataset(spec['dataset'],spec['dense_manifest_sha256'],
            resize_with_pad=resize_with_pad,max_cache_bytes=cache)

    def __init__(self,spec,output):
        super().__init__(spec,output)
        self.bindings.update(recipe=RECIPE,dataset_id=DATASET_ID,reference_label_semantics=VERSION,
            original_observation_manifest_sha256=ORIGINAL_MANIFEST_SHA,
            reference_conversion_audit=spec['target_audit']['sha256'],
            current_anchor_equivalent_consumed_reference_labels=True)
