"""A small fixed TRAIN-only language panel; not a collector or query runner."""
import base64
import hashlib
import math
from pathlib import Path
import uuid

from scripts.pi0292.eval_contract import FRAME, TCP
from . import minimal_study as study
from .minimal_dataset import PANEL_SHA

SCHEMA = 'ARX5_MINIMAL_TRAIN_PRECONTACT16_LANGUAGE24_V1'
SEEDS = (2026101901, 2026101902)


def semantic_input(state, images, language):
    study.need(len(state) == 10 and all(type(v) in (int, float) and math.isfinite(v) for v in state)
               and isinstance(language, str) and bool(language.strip()), 'FINITE_ACTUAL_EEF10_AND_LANGUAGE')
    study.need([r['camera'] for r in images] == ['external_rgb', 'wrist_rgb']
               and all(isinstance(r.get('sha256'), str) and len(r['sha256']) == 64 for r in images),
               'TWO_ORDERED_ACTUAL_RGB_HASHES')
    return dict(state_eef10=list(state), images=[dict(camera=r['camera'], sha256=r['sha256']) for r in images],
                language=language, frame=FRAME, tcp=TCP)


def wire_input_fingerprint(payload):
    study.need(payload['frame'] == FRAME and payload['tcp'] == TCP, 'ORIGINAL_FRAME_AND_TCP')
    for row in payload['images']:
        image = base64.b64decode(row['png_base64'], validate=True)
        study.need(hashlib.sha256(image).hexdigest() == row['sha256'], 'ACTUAL_WIRE_PNG_BYTES_CHANGED')
    return study.digest(semantic_input(payload['state_eef10'], payload['images'], payload['language']))


def prepare(capture, panel):
    study.need(study.digest(panel) == PANEL_SHA
               and capture['status'] == 'ACTUAL_EXISTING_SIX_TRAIN_APPROACH_OBSERVATIONS_NO_MODEL_OR_RENDER'
               and capture['new_scientific_queries'] == capture['new_physics'] == capture['new_capture'] == 0
               and len(capture['observations']) == 6, 'ACTUAL_SAVED_TRAIN_INPUT_READBACK_ONLY')
    pairs = {panel['parents'][i]['group']: panel['parents'][i:i+2] for i in range(0, 12, 2)}
    study.need({r['group'] for r in capture['observations']} == set(pairs), 'EXACT_SIX_FROZEN_TRAIN_GEOMETRIES')
    rows = []
    for geometry in [panel['parents'][i]['group'] for i in range(0, 12, 2)]:
        obs = next(o for o in capture['observations'] if o['group'] == geometry)
        left, right = pairs[geometry]
        study.need(obs['split'] == 'TRAIN' and obs['anchor'] == 16 and obs['stage'] == 'approach'
                   and obs['first_contact_anchor'] > 16 and obs['new_labels'] is False
                   and obs['parent_T1'] == left['id'] and obs['parent_T2'] == right['id']
                   and obs['source_numeric_sha256'] == {'T1': left['sha256'], 'T2': right['sha256']}
                   and set(obs['instructions']) == {'T1', 'T2'}
                   and obs['instructions']['T1'] != obs['instructions']['T2'],
                   'PRECONTACT_NOT_GRASPED_STATE_AND_ORIGINAL_PAIRED_TASK_TEXT')
        signature = study.digest(semantic_input(obs['state_eef10'], obs['images'], 'NO_LANGUAGE_COMPARISON'))
        for task in ('T1', 'T2'):
            semantic = semantic_input(obs['state_eef10'], obs['images'], obs['instructions'][task])
            for seed in SEEDS:
                rows.append(dict(id=f'language24-{len(rows):02d}', group=geometry, source_parent=left['id'],
                    instruction_source_parent=left['id'] if task == 'T1' else right['id'], anchor=16,
                    instruction_task=task, noise_seed=seed, semantic_input_sha256=study.digest(semantic),
                    same_RGB_state_without_language_sha256=signature,
                    instruction_sha256=hashlib.sha256(obs['instructions'][task].encode()).hexdigest(),
                    counterfactual_instruction=task == 'T2', teacher_action_target_present=False,
                    usable_as_BC_training_label=False, split='TRAIN'))
    return dict(schema=SCHEMA, source_panel_sha256=PANEL_SHA, cases=rows,
                per_model_generation_upper_bound=24, three_model_scientific_generation_upper_bound=72,
                models=['E0_RAW', 'E1_ADAPTER_ONLY', 'E2_SHARED_ACTION_HEAD'],
                qualification_and_zero_step_queries_included=False,
                environment='EXISTING_LEGACY_MATCHED_ONLY', common_home=False,
                visual_counterfactuals=False, language_response_is_not_task_success=True,
                scientific_queries_executed=0, native_adopted=False,
                first_original24_eval_waits_for_full_diagnostic_panel=False)


def payload_for_case(case, capture, reader):
    """Build the existing wire payload, using original PNG bytes, no re-render.

    Actual neural dispatch/query accounting stays in the original Journal and
    resident model. The original reader revalidates current numeric/RGB input;
    counterfactual instructions never acquire a Teacher action label.
    """
    obs = next(o for o in capture['observations'] if o['group'] == case['group'])
    study.need(obs['parent_T1'] == case['source_parent'] and obs['anchor'] == case['anchor'] == 16,
               'EXACT_PRECONTACT_SOURCE_OBSERVATION')
    indices = [i for i, (r, anchor) in enumerate(reader.rows)
               if r['id'] == case['source_parent'] and anchor == 16]
    study.need(len(indices) == 1, 'ONE_ACTUAL_ORIGINAL_READER_WINDOW')
    state, _, _, original_text = reader.raw(indices[0])
    state = [float(v) for v in state]
    study.need(state == obs['state_eef10'] and original_text == obs['instructions']['T1'],
               'CURRENT_ACTUAL_STATE_AND_ORIGINAL_TASK_TEXT_UNCHANGED')
    images=bound_reader_images(obs,reader,indices[0])
    payload = dict(snapshot_uuid=str(uuid.uuid4()), state_eef10=state, images=images,
                   language=obs['instructions'][case['instruction_task']], frame=FRAME, tcp=TCP)
    study.need(wire_input_fingerprint(payload) == case['semantic_input_sha256'],
               'PREFROZEN_SEMANTIC_MODEL_INPUT_CHANGED')
    return payload


def bound_reader_images(observation,reader,index):
    """Use actual current-host raw bindings, never a capture's old host path.

    The original numeric member selects the two exact PNG files at anchor16.
    Preserve their encoded bytes; no render, PNG recompression or copied raw
    data is introduced. The original reader has already checked actual state
    and image format before this bounded wire-payload read.
    """
    import numpy as np
    from scripts.pi0294.native_dataset import sha
    row,anchor=reader.rows[index];numeric=Path(reader.root)/row['file']
    study.need(anchor==observation['anchor']==16 and row['id']==observation['parent_T1']
        and numeric.resolve().is_relative_to(Path(reader.root).resolve())
        and sha(numeric)==row['sha256']==observation['source_numeric_sha256']['T1'],
        'ACTUAL_BOUND_READER_SAME_FROZEN_NUMERIC_MEMBER')
    rawroot=Path(reader.raw_bindings.get(row['id'],row['raw_root'])).resolve();images=[]
    with np.load(numeric,allow_pickle=False)as packet:
        for camera,old in enumerate(observation['images']):
            rel=Path(str(packet['rgb_paths'][anchor,camera]));path=(rawroot/rel).resolve()
            study.need(not rel.is_absolute()and '..'not in rel.parts and path.is_relative_to(rawroot)
                and str(packet['rgb_sha256'][anchor,camera])==old['sha256'],
                'ACTUAL_BOUND_READER_ORIGINAL_TWO_PNG_PATHS_AND_HASHES')
            image=path.read_bytes()
            study.need(hashlib.sha256(image).hexdigest()==old['sha256'],'ORIGINAL_PNG_BYTES_CHANGED')
            images.append(dict(camera=old['camera'],sha256=old['sha256'],
                png_base64=base64.b64encode(image).decode()))
    return images
