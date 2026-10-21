"""Cold B metadata from the same qualified source inputs, not GPU admission."""
import base64
from copy import deepcopy
import hashlib
import re

ARM_INPUTS={'plans/PRIMARY_TRAIN_BLOCK.json','plans/SOURCE_EVALUATION_PLAN.json'}
RELOCATED={'plans/REUSED_NEW_BANK_BINDING.private.json','plans/DEC1_CHECKPOINT_STORAGE_POLICY.json'}


def checked_inputs(row):
    encoded=row['small_inputs_base64'];digests=row['small_input_sha256']
    if set(encoded)!=set(digests):raise PermissionError('SOURCE_PAIR_EXACT_TEMPLATE_INPUT_SET')
    for name,value in encoded.items():
        payload=base64.b64decode(value,validate=True)
        if hashlib.sha256(payload).hexdigest()!=digests[name]:
            raise PermissionError('SOURCE_PAIR_CANONICAL_TEMPLATE_BYTES')
    return encoded


def prepare_B(qualified_A,original_pair,*,source_sha):
    if (set(original_pair)!={'A_CONTROL','B_COVERAGE'} or qualified_A.get('arm')!='A_CONTROL'
            or re.fullmatch('[0-9a-f]{40}',source_sha) is None or qualified_A.get('source')!=source_sha):
        raise PermissionError('SOURCE_PAIR_SAME_PUBLISHED_SOURCE_AND_ORIGINAL_ARMS')
    a,b=(checked_inputs(original_pair[name]) for name in ('A_CONTROL','B_COVERAGE'))
    if set(a)!=set(b) or {name for name in a if a[name]!=b[name]}!=ARM_INPUTS:
        raise PermissionError('SOURCE_PAIR_ONLY_ORIGINAL_TWO_ARM_SPECIFIC_INPUTS')
    for name,value in a.items():
        if name not in ARM_INPUTS|RELOCATED and qualified_A['small'].get(name)!=value:
            raise PermissionError('SOURCE_PAIR_UNCHANGED_COMMON_RECIPE_AND_DATA_INPUTS:'+name)
    for name in ('metadata_windows','metadata_composite_manifest_sha256','metadata_sampler_support_sha256'):
        if original_pair['A_CONTROL'][name]!=original_pair['B_COVERAGE'][name]:
            raise PermissionError('SOURCE_PAIR_MATCHED_METADATA_SUPPORT')
    prepared=deepcopy(qualified_A);prepared['arm']='B_COVERAGE'
    for name in ARM_INPUTS:prepared['small'][name]=b[name]
    prepared['cold_pair_binding']=dict(status='COLD_B_METADATA_ONLY_NOT_ADMISSION_OR_SUBMISSION',
        same_source=True,same_qualified_closed_data_references=True,canonical_arm_overrides=sorted(ARM_INPUTS),
        previous_A_terminal_audit_and_settlement_required=True,fresh_native_remaining_work_admission_required=True,
        new_GPU=0,new_PBS=0,new_queries=0,new_optimizer_updates=0,new_physics=0)
    return prepared
