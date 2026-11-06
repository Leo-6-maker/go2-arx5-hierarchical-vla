"""One saved graded V9 input, original draw reuse versus one new short prompt.

This is a separate study from the completed source/target 2x2. Only language
may change; original task/score and the native duration FAIL remain sidecars.
"""
import copy
import hashlib
import json

from scripts.pi0292.relay_protocol import digest as document_hash, validate_observation
from . import fusion_graded_query_archive_v9 as archive
from .fusion_short_skill import LANGUAGE as SHORT_LANGUAGE

MODE = 'FROZEN_GRADED_PROMPT_CONTROL_VIEW_V2'
SCHEMA = 'WM02_FUSION_GRADED_PROMPT_CONTROL_VIEW_V2'
STUDY = 'SAME_SAVED_GRADED_V9_INPUT_ORIGINAL_REUSE_SHORT_LANGUAGE_V2'
BASE_KIND = 'TARGET_SLOW_GRADED_V9'
BASE_KINDS = (BASE_KIND,)
PROMPT_VARIANTS = ('ORIGINAL_71558', 'CURRENT_SHORT')
CHECKPOINT, INVENTORY, SEED = archive.CHECKPOINT, archive.INVENTORY, archive.SEED
VIEW_KEYS = {'schema','study','base_kind','prompt_variant','condition','target_archive',
             'base_payload_sha256','policy_payload_sha256','non_language_payload_sha256',
             'original_language_sha256','selected_language_sha256','checkpoint_manifest_sha256',
             'checkpoint_file_inventory_sha256','seed','expected_noise_sha256','reuse_existing_original_query'}


def _need(ok, reason):
    if not ok:
        raise ValueError('GRADED_PROMPT_CONTROL_V2_' + reason)


def _text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _base(target_archive, base_kind):
    _need(base_kind == BASE_KIND and isinstance(target_archive, dict)
          and set(target_archive) == {'native_spec','gpu_spec','pins'}, 'EXACT_GRADED_V9_ARCHIVE_DESCRIPTOR')
    value = archive.read_archive(target_archive['native_spec'], target_archive['gpu_spec'], pins=target_archive['pins'])
    _need(value['schema'] == archive.SCHEMA
          and value['status'] == 'CLOSED_V9_DURATION_FAIL_WITH_VALID_SAVED_QUERY_AND_PAUSE_RETURN', 'VALID_HISTORICAL_GRADED_QUERY')
    return value['payload'], value['provenance']


def _derive(base, prompt_variant):
    _need(prompt_variant in PROMPT_VARIANTS, 'EXACT_TWO_LANGUAGE_VARIANTS')
    payload = copy.deepcopy(base)
    payload['language'] = base['language'] if prompt_variant == 'ORIGINAL_71558' else SHORT_LANGUAGE
    validate_observation(payload)
    before, after = ({k:v for k,v in value.items() if k != 'language'} for value in (base,payload))
    _need(document_hash(before) == document_hash(after), 'ONLY_LANGUAGE_MAY_CHANGE')
    return payload, document_hash(before)


def _manifest(target_archive, prompt_variant, base, payload, non_language):
    return dict(schema=SCHEMA, study=STUDY, base_kind=BASE_KIND, prompt_variant=prompt_variant,
        condition='TARGET_GRADED_V9_'+('ORIGINAL_REUSE' if prompt_variant == 'ORIGINAL_71558' else 'CURRENT_SHORT'),
        target_archive=copy.deepcopy(target_archive), base_payload_sha256=document_hash(base),
        policy_payload_sha256=document_hash(payload), non_language_payload_sha256=non_language,
        original_language_sha256=_text_hash(base['language']), selected_language_sha256=_text_hash(payload['language']),
        checkpoint_manifest_sha256=CHECKPOINT, checkpoint_file_inventory_sha256=INVENTORY, seed=SEED,
        expected_noise_sha256=target_archive['pins']['noise_sha256'], reuse_existing_original_query=prompt_variant == 'ORIGINAL_71558')


def make_view(*, target_archive, prompt_variant, base_kind=BASE_KIND):
    base, _ = _base(target_archive, base_kind)
    payload, non_language = _derive(base, prompt_variant)
    return _manifest(target_archive, prompt_variant, base, payload, non_language)


def prompt_control_payload(view_bytes, *, view_sha256, query_spec):
    _need(isinstance(view_bytes, bytes) and hashlib.sha256(view_bytes).hexdigest() == view_sha256, 'EXACT_VIEW_BYTES')
    view = json.loads(view_bytes)
    _need(isinstance(view, dict) and set(view) == VIEW_KEYS and view['prompt_variant'] == 'CURRENT_SHORT',
          'ONLY_CURRENT_SHORT_CAN_REQUEST_NEW_QUERY_ORIGINAL_REUSES_SAVED_DRAW')
    _need(query_spec.get('observation_mode') == MODE and query_spec.get('recorded_observation_sha256') == view_sha256
          and type(query_spec.get('seed')) is int and query_spec['seed'] == SEED
          and query_spec.get('checkpoint_manifest_sha256') == CHECKPOINT
          and document_hash(query_spec.get('checkpoint_files')) == INVENTORY
          and query_spec.get('numeric_mode') == 'ORIGINAL_FP32_WEIGHTS_BF16_AUTOCAST'
          and type(query_spec.get('max_queries')) is int and query_spec['max_queries'] == 1
          and query_spec.get('inference_mode') == 'OFFLINE_REAL_VLA_INFERENCE', 'EXACT_MATCHED_NEW_SHORT_QUERY_CONTRACT')
    base, provenance = _base(view['target_archive'], view['base_kind'])
    payload, non_language = _derive(base, view['prompt_variant'])
    expected = _manifest(view['target_archive'], view['prompt_variant'], base, payload, non_language)
    _need(document_hash(view) == document_hash(expected), 'EXACT_RECOMPUTED_LANGUAGE_ONLY_VIEW')
    _need(query_spec.get('expected_noise_sha256', view['expected_noise_sha256']) == view['expected_noise_sha256'],
          'MATCHED_EXPECTED_NOISE_IDENTITY')
    return payload, dict(observation_mode=MODE, study=STUDY, base_kind=BASE_KIND, condition=view['condition'],
        prompt_variant=view['prompt_variant'], view_sha256=view_sha256,
        base_payload_sha256=view['base_payload_sha256'], policy_payload_sha256=view['policy_payload_sha256'],
        non_language_payload_sha256=non_language, original_language_sha256=view['original_language_sha256'],
        selected_language_sha256=view['selected_language_sha256'], expected_noise_sha256=view['expected_noise_sha256'],
        image_sha256=[dict(camera=item['camera'], sha256=item['sha256']) for item in payload['images']],
        state_eef10_sha256=document_hash(payload['state_eef10']), frame=payload['frame'], tcp=payload['tcp'],
        noise_match_requires_new_result=True, only_language_field_may_differ=True,
        original_task_and_score_unmodified=True, original_observation_unmodified=True,
        actual_saved_target_images=True, target_online_images=False, historical_read=True,
        base_provenance=provenance, current_native_pause_checked=False, native_execution_authorized=False,
        task_phase_equivalence_established=False, task_success_established=False,
        language_effect_established_by_this_reader=False, new_queries=0, new_integrations=0, processor_decodes=0)
