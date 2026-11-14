"""New label identity over the unchanged qualified N20 observation cohort."""
import json
from pathlib import Path

from .dense_dataset_v4 import DensePilotDataset,sha,member
from .named_dataset_v5 import manifest_contract as original_population_contract
from .named_dataset_v5 import SCHEMA as OLD_SCHEMA,DATASET_ID as OLD_ID,SCOPE as OLD_SCOPE
from .reference_labels_v6 import VERSION

SCHEMA='SLA_FRESH_N20_ANCHOR_EQUIVALENT_TARGET_DATASET_V6'
DATASET_ID='T_PICK_HOLD_N20_ANCHOR_EQUIVALENT_CONSUMED_REFERENCE_V6'
SCOPE='SAME_N20_OBSERVATIONS_NEW_CURRENT_ANCHOR_COMMAND_LABELS_NOT_D100'
RECIPE='SLA_N20_ANCHOR_EQUIVALENT_FM32_3K_V6'
OPERATION='N20_ANCHOR_EQUIVALENT_FM3K'
AUDIT_SCHEMA='SLA_N20_REFERENCE_LABEL_CONVERSION_ACTUAL_V6'
ORIGINAL_MANIFEST_SHA='ec1e69592a7825ff2e7cc008ff46d1478cd8fe9a0451c17b6a89834878332cc1'


def manifest_contract(m):
    if (m.get('schema')!=SCHEMA or m.get('dataset_id')!=DATASET_ID or m.get('scope')!=SCOPE
            or m.get('label_semantics')!=VERSION
            or m.get('original_target_manifest',{}).get('sha256')!=ORIGINAL_MANIFEST_SHA
            or m.get('original_observations_and_states_unchanged')is not True
            or m.get('all40_target_knots_actually_consumed')is not True
            or m.get('native_reference_transport_changed')is not False):
        raise ValueError('EXPLICIT_NEW_REFERENCE_LABEL_IDENTITY_WITH_UNCHANGED_N20_OBSERVATIONS')
    # Reuse only the original population/clock/coverage contract. This local
    # projection grants no original V5 label identity or training admission.
    original_population_contract(dict(m,schema=OLD_SCHEMA,dataset_id=OLD_ID,scope=OLD_SCOPE))


class ReferenceTargetDataset(DensePilotDataset):
    validate_manifest=staticmethod(manifest_contract)

    def __init__(self,root,expected_manifest_sha256,**kwargs):
        root=Path(root);m=json.loads((root/'MANIFEST.private.json').read_bytes());manifest_contract(m)
        original=m['original_target_manifest'];parent=json.loads(member(root,original['file']).read_bytes())
        if sha(member(root,original['file']))!=ORIGINAL_MANIFEST_SHA:
            raise ValueError('EXACT_ORIGINAL_N20_MANIFEST_REQUIRED')
        audit=m['reference_conversion'];p=member(root,audit['file'])
        if sha(p)!=audit['sha256']:
            raise ValueError('REFERENCE_CONVERSION_AUDIT_BYTES_CHANGED')
        a=json.loads(p.read_bytes())
        if (a.get('schema')!=AUDIT_SCHEMA or not a.get('passes')
                or (a['full_H40_envelope_pass'],a['FP32_decoded_envelope_pass'])!=(1220,1220)
                or a['read_windows']!=1220 or a['original_manifest_sha256']!=ORIGINAL_MANIFEST_SHA
                or any(a[k]for k in('RGB_requalified','model_loads','optimizer_updates','physics_integrations'))):
            raise ValueError('ALL1220_CURRENT_ANCHOR_IDENTITY_AND_FP32_ENVELOPES_REQUIRED')
        for before,after in zip(parent['episodes'],m['episodes'],strict=True):
            if any(before[k]!=after[k]for k in('id','layout','restart','mount','frame_shards',
                    'expert_stage_audit_only','supervised_anchors','source_spec_sha256','source_run')):
                raise ValueError('SAME_CURRENT_STATES_RGB_MOUNTS_AND_FIXED_TRAIN_POPULATION')
        super().__init__(root,expected_manifest_sha256,**kwargs)
