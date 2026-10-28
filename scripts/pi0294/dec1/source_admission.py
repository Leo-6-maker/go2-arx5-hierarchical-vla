"""DEC1-only source training stage amendment, not site/resource admission."""
import hashlib
import json

from scripts.pi0294.pv1.future_admission import _file
from scripts.pi0294.pv1.recipe import validate_primary
from .authority import AUTHORITY, BANK, PARENT_AUTHORITY, PARENT, R0, HARD_CLOSE

VERSION = 'DEC1_SOURCE_TRAIN_STAGE_V1'
BINDING = 'plans/DEC1_SOURCE_TRAIN_ADMISSION.json'
COPY = 'plans/DEC1_NATIVE_ADOPTION_AND_BUDGET.json'


def admitted(root, manifest, request, stage, sealed_input):
    """Only the named new authority bypasses the historical B0 ordering gate.

    Native scope/lease, recipe, full resume, fresh forecast and all allocator
    checks still run through the existing launcher. No successful B0 claim is
    manufactured here and this entry cannot admit DEV selection or LOCKBOX.
    """
    authority = manifest.get('authority', {})
    version = authority.get('training_admission_version')
    requested = request.get('scientific_admission_version')
    if version is None and requested is None:
        return False
    if (version != VERSION or requested != VERSION
            or authority.get('protocol_ref') != PARENT_AUTHORITY
            or authority.get('amendments') != [AUTHORITY]
            or request.get('stage') != 'P2_PRIMARY'
            or stage.get('route') != 'COVERAGE_FIRST'
            or stage.get('new_bank_manifest_sha256') != BANK):
        raise PermissionError('DEC1_EXPLICIT_VERSION_AUTHORITY_FIXED_SOURCE_PRIMARY_ONLY')
    block = request.get('training_block', {})
    if block.get('arm') not in ('A_CONTROL', 'B_COVERAGE') or block.get('replica') is not False:
        raise PermissionError('DEC1_ONLY_ORIGINAL_TWO_SOURCE_ARMS_NOT_REPLICA')
    original=sealed_input('plans/FROZEN_SOURCE_PAIR_RECIPE.json')
    recipes=sealed_input('plans/PRIMARY_LEARNING_RECIPES.json')['recipes']
    for arm in ('A_CONTROL','B_COVERAGE'):
        recipe=recipes[arm]
        validate_primary(recipe,original)
        if (recipe['arm']!=arm or recipe['route']!='COVERAGE_FIRST'
                or recipe['replication_of_local_recipe'] is not False):
            raise PermissionError('DEC1_BOTH_ORIGINAL_SOURCE_RECIPES_BEFORE_MODEL_LOAD')
    binding = sealed_input(BINDING)
    adopted = sealed_input(COPY)
    raw = _file(root, COPY)
    native = _file(root.parent.parent, 'authority/DEC1/DEC1_NATIVE_ADOPTION_AND_BUDGET.json')
    digest = hashlib.sha256(raw).hexdigest()
    if (raw != native or json.loads(raw) != adopted
            or binding.get('schema') != 'DEC1_SOURCE_TRAIN_STAGE_ADMISSION_V1'
            or binding.get('authority_release') != AUTHORITY
            or binding.get('adoption_sha256') != digest
            or binding.get('parent_checkpoint_manifest_sha256') != PARENT
            or binding.get('statistics_sha256') != R0
            or binding.get('new_source_bank_manifest_sha256') != BANK
            or binding.get('source_only_original_recipe') is not True
            or binding.get('use_target_task_data') is not False
            or binding.get('baseline_relative_selection_authorized') is not False
            or binding.get('LOCKBOX_authorized') is not False):
        raise PermissionError('DEC1_EXACT_NATIVE_ADOPTION_SEALED_COPY_AND_FIXED_TRAIN_IDENTITY')
    if (adopted.get('schema') != 'DEC1_NATIVE_ADOPTION_AND_BUDGET_V1'
            or adopted.get('authority_release') != AUTHORITY
            or adopted.get('original_authority_release') != PARENT_AUTHORITY
            or adopted.get('status') != 'DEC1_AUTHORITY_ADOPTED_CONTROL_ONLY'
            or adopted.get('campaign_run_uuid') != stage.get('campaign_run_uuid')
            or adopted.get('original_T0_epoch') != binding.get('original_T0_epoch')
            or adopted.get('original_hard_close_epoch') != HARD_CLOSE
            or adopted.get('clock_reset') is not False or adopted.get('additive_budget') is not False
            or adopted.get('source_training_requires_complete_B0') is not False
            or adopted.get('legacy_authority_and_baseline_relative_release_require_complete_B0') is not True):
        raise PermissionError('DEC1_NATIVE_AMENDMENT_NO_CLOCK_RESET_OR_OLD_RELEASE_GATE_CHANGE')
    return True
