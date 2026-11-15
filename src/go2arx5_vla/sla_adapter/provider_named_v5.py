"""Ordinary-RGB export for the separately named fresh N20 endpoint."""
import json
from pathlib import Path

from .checkpoint_v1 import sha
from .dense_dataset_v4 import member
from .gripper_v2 import VERSION
from .named_dataset_v5 import DATASET_ID, RECIPE, manifest_contract
from .provider_v3 import write_verified_package


def export(package, checkpoint, *, active_template, dataset, tokenizer, parent_manifest):
    checkpoint, dataset = Path(checkpoint), Path(dataset)
    complete = json.loads((checkpoint/'COMPLETE.json').read_bytes())
    b = complete['bindings']
    m = json.loads((dataset/'MANIFEST.private.json').read_bytes())
    manifest_contract(m)
    if (complete['schema'] != 'SLA_COMPLETE_PEFT_STATE_V1' or complete['step'] != 3000
            or complete['cursor'] != 96000 or len(complete['arrays']) != 87
            or b['recipe'] != RECIPE or b['dataset_id'] != DATASET_ID or b['codec_version'] != VERSION
            or b['dense_manifest_sha256'] != sha(dataset/'MANIFEST.private.json')
            or b['derived_statistics_sha256'] != m['statistics']['sha256']
            or b['parent_manifest'] != parent_manifest or sha(tokenizer) != b['tokenizer']
            or b['D100_new'] is not False or b['D300_new'] is not False
            or b['PICK_PLACE_admitted'] is not False or b['old_2EP_reused'] != 0):
        raise ValueError('COMPLETE_FRESH_N20_3K_NOT_OLD_PILOT_OR_D100_REQUIRED')
    statistics = member(dataset,m['statistics']['file'])
    if sha(statistics) != b['derived_statistics_sha256']:
        raise ValueError('FRESH_N20_TRAIN_NORMALIZER_CHANGED')
    return write_verified_package(package,checkpoint,complete=complete,active_template=active_template,
        statistics=statistics,tokenizer=tokenizer,parent_manifest=parent_manifest)
