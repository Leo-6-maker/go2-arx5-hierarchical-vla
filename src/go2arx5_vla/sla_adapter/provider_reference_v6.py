"""Ordinary RGB provider is unchanged;export enforces the new label identity."""
import json
from pathlib import Path
from .checkpoint_v1 import sha
from .dense_dataset_v4 import member
from .reference_dataset_v6 import DATASET_ID,RECIPE,manifest_contract
from .reference_labels_v6 import VERSION
from .provider_v3 import write_verified_package


def export(package,checkpoint,*,active_template,dataset,tokenizer,parent_manifest):
    checkpoint,dataset=Path(checkpoint),Path(dataset)
    complete=json.loads((checkpoint/'COMPLETE.json').read_bytes());b=complete['bindings']
    m=json.loads((dataset/'MANIFEST.private.json').read_bytes());manifest_contract(m)
    if (complete['schema']!='SLA_COMPLETE_PEFT_STATE_V1' or complete['step']!=3000
            or complete['cursor']!=96000 or len(complete['arrays'])!=87
            or b['recipe']!=RECIPE or b['dataset_id']!=DATASET_ID or b['reference_label_semantics']!=VERSION
            or b['dense_manifest_sha256']!=sha(dataset/'MANIFEST.private.json')
            or b['derived_statistics_sha256']!=m['statistics']['sha256']
            or b['reference_conversion_audit']!=m['reference_conversion']['sha256']
            or b['parent_manifest']!=parent_manifest or sha(tokenizer)!=b['tokenizer']):
        raise ValueError('FULL87_REFERENCE_LABEL3K_AND_ORIGINAL_ORDINARY_RGB_PROVIDER_REQUIRED')
    stats=member(dataset,m['statistics']['file'])
    if sha(stats)!=b['derived_statistics_sha256']:raise ValueError('REFERENCE_TRAIN_ONLY_NORMALIZER_BYTES')
    return write_verified_package(package,checkpoint,complete=complete,active_template=active_template,
        statistics=stats,tokenizer=tokenizer,parent_manifest=parent_manifest)
