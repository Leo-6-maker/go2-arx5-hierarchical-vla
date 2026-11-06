"""Language-only policy views of pinned saved inputs, without a live grant.

The view is a new diagnostic input, never a rewritten original observation.
Only ``language`` is replaced. Original pixels, state, frames, task, calibration
and historical producer evidence remain bound to their original artifacts.
This module performs no query, decode, rendering, FK, physics or admission.
"""
import copy
import hashlib
import json
from pathlib import Path

from scripts.pi0292.relay_protocol import digest as document_hash, unseal, validate_observation
from .fusion_short_skill import LANGUAGE as SHORT_LANGUAGE

MODE = 'FROZEN_PROMPT_CONTROL_VIEW_V1'
SCHEMA = 'WM02_FUSION_FROZEN_PROMPT_CONTROL_VIEW_V1'
STUDY = 'SAME_SAVED_INPUT_SOURCE_TARGET_ORIGINAL_SHORT_LANGUAGE_2X2_V1'
SOURCE_OBSERVATION = '71558ff5bcbc03e83eef9fbe79c868e75a2fece5c712f708682c8f23a090a594'
CHECKPOINT = '94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'
INVENTORY = 'f4c9486006de1a58721a1fc723361264e406ac4910ba46f14a31233c7a28fe23'
SEED = 366104
BASE_KINDS = ('SOURCE_71558', 'TARGET_FULL1_V3')
PROMPT_VARIANTS = ('ORIGINAL_71558', 'CURRENT_SHORT')
VIEW_KEYS = {'schema', 'study', 'base_kind', 'prompt_variant', 'condition',
             'source_observation', 'target_archive', 'base_payload_sha256',
             'policy_payload_sha256', 'non_language_payload_sha256',
             'original_language_sha256', 'selected_language_sha256',
             'checkpoint_manifest_sha256', 'checkpoint_file_inventory_sha256', 'seed'}


def _need(ok, reason):
    if not ok:
        raise ValueError('FUSION_PROMPT_CONTROL_' + reason)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _text_hash(text):
    return _hash(text.encode('utf-8'))


def _path(value):
    _need(isinstance(value, str) and Path(value).is_absolute(), 'ABSOLUTE_SAVED_PATH_REQUIRED')
    path = Path(value)
    _need(not path.is_symlink(), 'SAVED_ARTIFACT_MUST_NOT_BE_SYMLINK')
    return path


def _source(descriptor):
    _need(isinstance(descriptor, dict) and set(descriptor) == {'path', 'sha256'}
          and descriptor['sha256'] == SOURCE_OBSERVATION, 'EXACT_71558_SOURCE_OBSERVATION_REQUIRED')
    # Pin and parse the same read; no digest/open/parse race.
    raw = _path(descriptor['path']).read_bytes()
    _need(_hash(raw) == SOURCE_OBSERVATION, 'ORIGINAL_SOURCE_OBSERVATION_BYTES_CHANGED')
    record = json.loads(raw)
    body = unseal(record)
    payload = body['payload']
    validate_observation(payload)
    return payload, dict(observation_sha256=SOURCE_OBSERVATION,
                        packet_body_sha256=document_hash(body),
                        payload_sha256=document_hash(payload))


def _base(source_observation, base_kind, target_archive):
    original, source_provenance = _source(source_observation)
    _need(base_kind in BASE_KINDS, 'REGISTERED_BASE_KIND_REQUIRED')
    if base_kind == 'SOURCE_71558':
        _need(target_archive is None, 'SOURCE_VIEW_CANNOT_CLAIM_TARGET_ARCHIVE')
        return original, original, dict(source=source_provenance)
    _need(isinstance(target_archive, dict)
          and set(target_archive) == {'native_spec', 'gpu_spec', 'pins'}, 'COMPLETE_TARGET_ARCHIVE_DESCRIPTOR_REQUIRED')
    from .fusion_source_posture_query_archive_v3 import read_archive
    archive = read_archive(_path(target_archive['native_spec']), _path(target_archive['gpu_spec']),
                           pins=target_archive['pins'])
    _need(archive['status'] == 'PAIR_COMPLETED_AND_PAUSE_REVALIDATED'
          and archive['external_pins']['seed'] == SEED
          and archive['external_pins']['checkpoint_manifest_sha256'] == CHECKPOINT
          and archive['external_pins']['checkpoint_file_inventory_sha256'] == INVENTORY,
          'ACTUAL_MATCHED_SEED_FULL1_TARGET_ARCHIVE_REQUIRED')
    payload = archive['payload']
    validate_observation(payload)
    _need(payload['language'] == SHORT_LANGUAGE, 'BASE_TARGET_MUST_BE_COMPLETED_SHORT_PROMPT_CONDITION')
    body = unseal(archive['observation'])
    shadow = unseal(body['original_shadow_ABI'])
    # Retain original sidecar provenance outside the policy ABI. In particular,
    # a changed prompt never changes the saved native task/score declaration.
    target = dict(archive_schema=archive['schema'], external_pins=copy.deepcopy(archive['external_pins']),
        historical_read=True, current_native_pause_checked=False,
        native_execution_authorized=False, actual_capture=copy.deepcopy(body['capture']),
        native_step=body['native_step'], native_time_s=body['native_time_s'],
        original_task=copy.deepcopy(body['task']), original_task_sha256=body['task_sha256'],
        world=copy.deepcopy(body['world']), renderer=copy.deepcopy(body['renderer']),
        capture_provenance=copy.deepcopy(body['provenance']),
        original_shadow_provenance=copy.deepcopy(shadow['provenance']),
        target_observation=copy.deepcopy(shadow['target_observation']),
        source_policy_mapping={k: copy.deepcopy(v) for k,v in shadow['source_policy_view'].items() if k != 'payload'})
    return payload, original, dict(source=source_provenance, target=target)


def _derive(base, original, prompt_variant):
    _need(prompt_variant in PROMPT_VARIANTS, 'REGISTERED_PROMPT_VARIANT_REQUIRED')
    result = copy.deepcopy(base)
    result['language'] = original['language'] if prompt_variant == 'ORIGINAL_71558' else SHORT_LANGUAGE
    validate_observation(result)
    before = {k:v for k,v in base.items() if k != 'language'}
    after = {k:v for k,v in result.items() if k != 'language'}
    _need(document_hash(before) == document_hash(after), 'NON_LANGUAGE_POLICY_INPUT_CHANGED')
    return result, document_hash(before)


def _manifest(source_observation, base_kind, prompt_variant, target_archive, base, original, payload, non_language):
    return dict(schema=SCHEMA, study=STUDY, base_kind=base_kind, prompt_variant=prompt_variant,
        condition=('SOURCE' if base_kind == 'SOURCE_71558' else 'TARGET') +
                  ('_ORIGINAL' if prompt_variant == 'ORIGINAL_71558' else '_SHORT'),
        source_observation=copy.deepcopy(source_observation), target_archive=copy.deepcopy(target_archive),
        base_payload_sha256=document_hash(base), policy_payload_sha256=document_hash(payload),
        non_language_payload_sha256=non_language, original_language_sha256=_text_hash(original['language']),
        selected_language_sha256=_text_hash(payload['language']), checkpoint_manifest_sha256=CHECKPOINT,
        checkpoint_file_inventory_sha256=INVENTORY, seed=SEED)


def make_view(*, source_observation, base_kind, prompt_variant, target_archive=None):
    """Build a JSON-safe manifest by reading the actual saved inputs.

    ``source_observation`` is exactly ``{path, sha256}`` for the 71558 file.
    Target views also require ``{native_spec, gpu_spec, pins}``, with the V3
    historical reader's exact 19 external pins. No payload/GT is accepted in
    this manifest. Writing the returned manifest and registering a new finite
    query remain the original caller's responsibility.
    """
    base, original, _ = _base(source_observation, base_kind, target_archive)
    payload, non_language = _derive(base, original, prompt_variant)
    return _manifest(source_observation, base_kind, prompt_variant, target_archive,
                     base, original, payload, non_language)


def prompt_control_payload(view_bytes, *, view_sha256, query_spec):
    """Resolve an explicitly pinned view for one separately admitted query.

    Historical validation grants no current pause, action installation, native
    execution or task success. The inference caller still owns all original
    runtime/admission/model/Processor checks and its single real generation.
    """
    _need(isinstance(view_bytes, bytes) and _hash(view_bytes) == view_sha256, 'EXACT_VIEW_BYTES_REQUIRED')
    view = json.loads(view_bytes)
    _need(isinstance(view, dict) and set(view) == VIEW_KEYS, 'EXACT_VIEW_SCHEMA_REQUIRED')
    _need(type(query_spec.get('seed')) is int and query_spec['seed'] == SEED
          and query_spec.get('checkpoint_manifest_sha256') == CHECKPOINT
          and document_hash(query_spec.get('checkpoint_files')) == INVENTORY
          and query_spec.get('numeric_mode') == 'ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST'
          and type(query_spec.get('max_queries')) is int and query_spec['max_queries'] == 1
          and query_spec.get('inference_mode') == 'OFFLINE_REAL_VLA_INFERENCE',
          'MATCHED_FROZEN_SINGLE_QUERY_CONTRACT_REQUIRED')
    base, original, provenance = _base(view['source_observation'], view['base_kind'], view['target_archive'])
    payload, non_language = _derive(base, original, view['prompt_variant'])
    expected = _manifest(view['source_observation'], view['base_kind'], view['prompt_variant'], view['target_archive'],
                         base, original, payload, non_language)
    _need(document_hash(view) == document_hash(expected), 'DERIVED_VIEW_DECLARATION_CHANGED')
    target = view['base_kind'] == 'TARGET_FULL1_V3'
    return payload, dict(observation_mode=MODE, input_mode='SAVED_INPUT_LANGUAGE_ONLY_DIAGNOSTIC_NOT_ONLINE',
        study=STUDY, condition=view['condition'], prompt_variant=view['prompt_variant'],
        view_sha256=view_sha256, base_payload_sha256=view['base_payload_sha256'],
        policy_payload_sha256=view['policy_payload_sha256'], non_language_payload_sha256=non_language,
        base_language_sha256=_text_hash(base['language']), original_language_sha256=view['original_language_sha256'],
        selected_language_sha256=view['selected_language_sha256'], language_changed=payload['language'] != base['language'],
        only_language_field_may_differ=True, original_observation_unmodified=True,
        original_task_and_score_unmodified=True, base_provenance=provenance,
        image_sha256=[dict(camera=x['camera'], sha256=x['sha256']) for x in payload['images']],
        state_eef10_sha256=document_hash(payload['state_eef10']), frame=payload['frame'], tcp=payload['tcp'],
        actual_saved_target_images=target, target_online_images=False, historical_read=True,
        current_native_pause_checked=False, native_execution_authorized=False,
        target_embodiment_qualified=False, target_task_native_qualified=False,
        source_target_input_equivalence_established=False, between_base_domain_effect_identified=False,
        language_effect_established_by_this_reader=False,
        source_outcomes_supplied=False, new_queries=0, new_integrations=0, processor_decodes=0,
        task_success_established=False, action_installations_authorized=False)
