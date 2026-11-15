"""Export the spatial pilot's common policy; reuse the exact RGB logit provider."""
import json
from pathlib import Path

from .checkpoint_v1 import sha
from .dense_dataset_v4 import manifest_contract, member
from .gripper_v2 import VERSION
from .provider_v3 import write_verified_package
from .spatial_runtime_v5 import RECIPE


def export(package,checkpoint,*,active_template,dataset,tokenizer,parent_manifest):
    checkpoint,dataset = Path(checkpoint),Path(dataset)
    complete = json.loads((checkpoint/'COMPLETE.json').read_bytes());b = complete['bindings']
    m = json.loads((dataset/'MANIFEST.private.json').read_bytes());manifest_contract(m)
    if (complete['schema'] != 'SLA_COMPLETE_PEFT_STATE_V1' or complete['step'] != 3000
            or complete['cursor'] != 96000 or b['recipe'] != RECIPE or b['alpha'] != .5
            or b['codec_version'] != VERSION or b['dense_manifest_sha256'] != sha(dataset/'MANIFEST.private.json')
            or b['parent_manifest'] != parent_manifest or sha(tokenizer) != b['tokenizer']
            or b['derived_statistics_sha256'] != m['statistics']['sha256']
            or b['D100_new'] is not False or b['PICK_PLACE_admitted'] is not False):
        raise ValueError('COMPLETE_CLEAN_BASE_MATCHED_SPATIAL_3K_TARGET_ONLY_ENDPOINT_REQUIRED')
    stats = member(dataset,m['statistics']['file'])
    if sha(stats) != b['derived_statistics_sha256']:raise ValueError('EXACT_SPATIAL_PILOT_TRAIN_STATISTICS')
    return write_verified_package(package,checkpoint,complete=complete,active_template=active_template,
        statistics=stats,tokenizer=tokenizer,parent_manifest=parent_manifest)
