"""Dense pilot export; ordinary RGB uses the unchanged V3 logit provider."""
import json
from pathlib import Path

from .checkpoint_v1 import sha
from .dense_dataset_v4 import manifest_contract
from .dense_recipe_v4 import RECIPE
from .gripper_v2 import VERSION
from .provider_v3 import write_verified_package


def export(package, checkpoint, *, active_template, dataset, tokenizer, parent_manifest):
    checkpoint, dataset = Path(checkpoint), Path(dataset)
    complete = json.loads((checkpoint/'COMPLETE.json').read_bytes())
    binding = complete['bindings']
    manifest = json.loads((dataset/'MANIFEST.private.json').read_bytes())
    manifest_contract(manifest)
    if (complete['schema'] != 'SLA_COMPLETE_PEFT_STATE_V1' or complete['step'] not in (4, 3000)
            or complete['cursor'] != 32*complete['step'] or binding.get('recipe') != RECIPE
            or binding.get('codec_version') != VERSION or binding.get('D100_new') is not False
            or binding.get('PICK_PLACE_admitted') is not False
            or binding.get('dense_manifest_sha256') != sha(dataset/'MANIFEST.private.json')
            or binding.get('derived_statistics_sha256') != manifest['statistics']['sha256']
            or binding.get('parent_manifest') != parent_manifest or sha(tokenizer) != binding['tokenizer']):
        raise ValueError('ACTUAL_LIMITED_DENSE_PILOT_ENDPOINT_AND_DATA_BINDINGS_REQUIRED')
    from .dense_dataset_v4 import member
    stats = member(dataset, manifest['statistics']['file'])
    if sha(stats) != binding['derived_statistics_sha256']:
        raise ValueError('FROZEN_DENSE_PILOT_STATISTICS_CHANGED')
    return write_verified_package(package, checkpoint, complete=complete, active_template=active_template,
                                  statistics=stats, tokenizer=tokenizer, parent_manifest=parent_manifest)
