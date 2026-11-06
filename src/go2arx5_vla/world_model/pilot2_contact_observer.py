"""Read-only forecasts at actual pauses; predictions are not physical labels."""
from pathlib import Path
import hashlib
import io
import json
import os
import time
import numpy as np
from .pilot2_contact_model import HORIZONS, EVENTS, planned_openings, observed_events
from .contact_features import state_feature
from .pilot2_contact_numpy import predict


def validate(spec, episode):
    config = spec.get('contact_forecast_observer')
    if config is None: return None
    if (config.get('schema') != 'WM02_PILOT2_PAUSED_CONTACT_FORECAST_V1'
            or episode.get('schema') != 'WM02_PILOT2_SUPPORTED_EPISODE_V1'
            or episode.get('purpose') != 'SUPPORTED_CONTACT_COLLECTION'):
        raise PermissionError('PREDECLARED_STATIC_CONTACT_PROGRAM_ONLY_NOT_UNKNOWN_POLICY_FUTURE')
    anchors = config.get('anchors', [])
    models = config.get('models', [])
    if (not anchors or any(type(i) is not int or not 8 <= i < episode['task_atoms']-1 for i in anchors)
            or anchors != sorted(set(anchors)) or not models
            or any(not m.get('name') or len(m.get('sha256', '')) != 64 for m in models)
            or any(m.get('arm', 'ACTION_CONDITIONED') not in ('ACTION_CONDITIONED', 'NO_FUTURE_PROGRAM') for m in models)
            or len({m['name'] for m in models}) != len(models)):
        raise ValueError('FROZEN_FORECAST_MODELS_AND_VALID_UNIQUE_PAUSE_ANCHORS_REQUIRED')
    return config


def _load(config):
    arrays = []
    for model in config['models']:
        contents = Path(model['path']).read_bytes()
        if hashlib.sha256(contents).hexdigest() != model['sha256']:
            raise PermissionError('FROZEN_CONTACT_FORECAST_MODEL_CHANGED')
        with np.load(io.BytesIO(contents), allow_pickle=False) as archive:
            value = {k: archive[k].copy() for k in archive.files}
        if 'forecast_horizons' in value and tuple(value['forecast_horizons'].tolist()) != HORIZONS:
            raise ValueError('EXACT_LOGGED_FORECAST_HORIZONS_REQUIRED')
        if predict(value, value['mean'][None]).shape != (1, len(HORIZONS), 10):
            raise ValueError('SIX_DECLARED_CONTACT_FORECAST_HORIZONS_REQUIRED')
        arrays.append(value)
    return (tuple((m['name'], m['path'], m['sha256'], m.get('arm', 'ACTION_CONDITIONED')) for m in config['models']), arrays)


def prepare(adapter, episode):
    config = validate(adapter.spec, episode)
    if config is None: return None
    # Load/verify weights before any target integration, not in a PD pause.
    adapter._contact_forecast_models = _load(config)
    return record


def record(adapter, episode, rows):
    config = adapter.spec['contact_forecast_observer']
    anchor = len(rows)-1
    if anchor not in config['anchors']: return None
    step = rows[-1]['native_end_step']
    if step != 2528+25*anchor or adapter.observe()['step'] != step:
        raise RuntimeError('ACTUAL_CURRENT_CONTACT_PAUSE_REQUIRED_FOR_FORECAST')
    seen = getattr(adapter, '_contact_forecast_steps', set())
    if step in seen: raise FileExistsError('CONTACT_PAUSE_FORECAST_ALREADY_RECORDED')
    # This observer deliberately supports only programs fixed before execution.
    # A policy episode must not reveal its as-yet-unselected future program.
    if episode.get('schema') != 'WM02_PILOT2_SUPPORTED_EPISODE_V1' or episode.get('purpose') != 'SUPPORTED_CONTACT_COLLECTION':
        raise PermissionError('STATIC_KNOWN_PROGRAM_REQUIRED_FOR_PAUSED_FORECAST')
    start = time.monotonic()
    history = np.concatenate([np.r_[state_feature(row), observed_events(row)] for row in rows[-9:]]).astype(np.float32)
    plan = planned_openings(episode); future = plan[anchor+1:anchor+65]
    future = np.pad(future, (0, 64-len(future)), constant_values=plan[-1]).astype(np.float32)
    x = np.r_[history, future][None]
    key = tuple((m['name'], m['path'], m['sha256'], m.get('arm', 'ACTION_CONDITIONED')) for m in config['models'])
    cached = getattr(adapter, '_contact_forecast_models', None)
    if cached is None:
        raise RuntimeError('CONTACT_FORECAST_MODELS_MUST_BE_PREPARED_BEFORE_PHYSICS')
    if cached[0] != key: raise PermissionError('FORECAST_OBSERVER_CONFIG_CHANGED_DURING_EPISODE')
    predictions = {}
    fallback = any(row['stage'] == 'EARLY_RELEASE' for row in rows)
    if not fallback:
        for model, arrays in zip(config['models'], cached[1]):
            model_x = x
            arm = model.get('arm', 'ACTION_CONDITIONED')
            if arm == 'NO_FUTURE_PROGRAM':
                model_x = np.r_[history, np.full(64, rows[-1]['executed_reference_EEF10'][-1], dtype=np.float32)][None]
            raw = predict(arrays, model_x)
            if raw.shape != (1, len(HORIZONS), 10): raise ValueError('SIX_DECLARED_CONTACT_FORECAST_HORIZONS_REQUIRED')
            predictions[model['name']] = dict(portable_sha256=model['sha256'], arm=arm,
                input_sha256=hashlib.sha256(model_x.astype('<f4').tobytes()).hexdigest(), delta_mm=raw[0, :, :6].tolist(),
                event_probability=(1/(1+np.exp(-np.clip(raw[0, :, 6:], -50, 50)))).tolist())
    after = adapter.observe()['step']
    if after != step: raise RuntimeError('PHYSICS_ADVANCED_DURING_READ_ONLY_CONTACT_FORECAST')
    point = dict(schema='WM02_PILOT2_PAUSED_CONTACT_FORECAST_POINT_V1', source_sha=adapter.spec['source_sha'],
        episode_manifest_sha256=adapter.spec.get('episode_manifest_sha256'),
        command_program=episode['command_program'], close_start_atom=episode['close_start_atom'],
        anchor_index=anchor, native_step_before=step, native_step_after=after, forecast_horizons_atoms=list(HORIZONS),
        event_names=list(EVENTS), state_delta_order=['TCP_x_mm','TCP_y_mm','TCP_z_mm','object_x_mm','object_y_mm','object_z_mm'],
        horizon_within_declared_task=[anchor+h < len(plan) for h in HORIZONS],
        history_sha256=hashlib.sha256(history.astype('<f4').tobytes()).hexdigest(),
        input_sha256=hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),
        future_program_known_before_episode=True, future_measured_response_observed=False,
        prefix_past_safe_fallback=fallback, skipped_reason='PAST_SAFE_EARLY_RELEASE' if fallback else None,
        predictions=predictions, prediction_wall_s=time.monotonic()-start, recorded_epoch=time.time())
    with (Path(adapter.output)/'CONTACT_FORECASTS.private.jsonl').open('a') as stream:
        stream.write(json.dumps(point, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    seen.add(step); adapter._contact_forecast_steps = seen
    return point
