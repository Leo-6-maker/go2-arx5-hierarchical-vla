"""Freeze matching SCALE20K offline rows before any endpoint prediction."""
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np

from .checkpoint_v1 import sha
from .codec_v1 import pad32, rigid, roundtrip_error
from .dense_g7_v6 import label_margin_report
from .runtime_v1 import read_pin
from .scale20k_contract import DOMAINS, OFFLINE_PER_DOMAIN as PER_DOMAIN, PLAN, RESTORE_ORDINALS as RESTORE
from .scale20k_data import readers


SCHEMA = 'SLA_SCALE20K_OFFLINE_PANEL_V1'
SPLITS = dict(target='TRAIN', robodojo='SOURCE_DEV', libero='SOURCE_DEV')


def data_sha(spec):
    return dict(manifests={d: spec[d]['manifest']['sha256'] for d in DOMAINS},
                target_admission=spec['target']['admission']['sha256'],
                target_clock=spec['target']['time_audit']['sha256'])


def balanced_rows(candidates, count, rng):
    """Round-robin tasks and episodes, without returning a window twice."""
    pools = defaultdict(lambda: defaultdict(list))
    for index, (task, episode) in enumerate(candidates):
        pools[task][episode].append(index)
    tasks = sorted(pools)
    if not tasks or count <= 0:
        raise ValueError('SCALE20K_FIXED_OFFLINE_WINDOWS_REQUIRED')
    for task in tasks:
        episodes = pools[task]
        pools[task] = [list(rng.permutation(episodes[ep])) for ep in sorted(episodes)]
    offsets = dict.fromkeys(tasks, 0)
    while count:
        for task in tasks:
            episodes = pools[task]
            while not episodes[offsets[task] % len(episodes)]:
                if not any(episodes):
                    raise ValueError('SCALE20K_FIXED_TASK_PANEL_HAS_TOO_FEW_UNIQUE_WINDOWS')
                offsets[task] += 1
            pool = episodes[offsets[task] % len(episodes)]
            offsets[task] += 1
            yield int(pool.pop())
            count -= 1
            if not count:
                break


def panel_rows(datasets):
    result = {}
    for number, domain in enumerate(DOMAINS):
        reader = datasets[domain]
        if domain == 'target':
            candidates = [('PLACE' if ep['task'].startswith('PLACE-') else 'HOLD', ep['id'])
                          for ep, _ in reader.rows]
        else:
            candidates = [(reader.episodes[episode]['task'], episode) for episode, _ in reader.rows]
        rng = np.random.default_rng(2026111971 + number)
        result[domain] = [dict(index=index, identity=reader.identity(index),
                               group=candidates[index][0], noise_id=int(rng.integers(2**32)))
                          for index in balanced_rows(candidates, PER_DOMAIN, rng)]
    return result


def target_contexts(reader, rows):
    from scripts.pi0292.durable import read
    from scripts.sla.export_mfree_c20_v1 import _reference

    archived, result = {}, {}
    for row in rows:
        index = row['index']
        episode, anchor = reader.rows[index]
        eid = episode['id']
        if eid not in archived:
            native = read_pin(episode['native_result'])
            path = Path(episode['native_result']['path']).with_name('EXECUTION_SPEC.json')
            lease = read(path.parent/'control/LEASE.json')
            spec_sha = sha(path)
            spec = read_pin(dict(path=str(path), sha256=lease['spec_sha256']))
            if lease['status'] != 'CLOSED' or spec['source_sha'] != native['source_sha']:
                raise ValueError('SCALE20K_OFFLINE_TARGET_REQUIRES_CLOSED_CAPTURE_PROVENANCE')
            cap_pin = (episode['original_live_capture'] if 'original_live_capture' in episode
                       else native['detail']['capture'])
            capture = read_pin(cap_pin)
            archived[eid] = (rigid(spec['policy_from_world']), read_pin(capture['labels']),
                             spec_sha, capture['labels']['sha256'])
        frame, captured, spec_sha, labels_sha = archived[eid]
        item = captured[anchor]
        tick = row['identity']['native_step']
        if (item['query'] != anchor or item['native_step'] != tick
                or item['recent_reference']['native_step'] != tick):
            raise ValueError('SCALE20K_OFFLINE_TARGET_CONTEXT_MUST_MATCH_PAUSED_FRAME')
        result[str(index)] = dict(policy_from_world=frame.tolist(), reference=rigid(_reference(item)).tolist(),
                                  previous_command=reader.reference(index), capture_labels_sha256=labels_sha,
                                  native_spec_sha256=spec_sha)
    return result


def prepare(spec, resize_with_pad):
    datasets = readers(spec, resize_with_pad, source_split='SOURCE_DEV')
    rows = panel_rows(datasets)
    return dict(schema=SCHEMA, plan=PLAN, data=data_sha(spec), splits=SPLITS, rows=rows,
                restore_ordinals=RESTORE, target_contexts=target_contexts(datasets['target'], rows['target']),
                new_updates=0, new_model_generations=0, new_native_requests=0, old_TEST_access=False)


def validate(panel, spec, datasets):
    if (panel['schema'] != SCHEMA or panel['plan'] != PLAN or panel['data'] != data_sha(spec)
            or panel['splits'] != SPLITS or panel['rows'] != panel_rows(datasets)
            or tuple(panel['restore_ordinals']) != RESTORE or panel['old_TEST_access'] is not False
            or set(panel['target_contexts']) != {str(r['index']) for r in panel['rows']['target']}):
        raise ValueError('SCALE20K_EXACT_FROZEN_OFFLINE_PANEL_REQUIRED')


def measure(domain, dataset, row, prediction, target_actions, panel):
    from .scale20k_training import ACTION_CHANNELS

    index = row['index']
    channels = ACTION_CHANNELS[domain]
    active = prediction[:, :channels]
    expected = target_actions[:, :channels]
    metrics = dict(normalized_active_mae=float(np.abs(active-expected).mean()),
                   normalized_padding_rms=float(np.sqrt(np.square(prediction[:, channels:]).mean())),
                   normalized_action_motion=float(np.abs(np.diff(active, axis=0)).mean()),
                   label_action_motion=float(np.abs(np.diff(expected, axis=0)).mean()))
    if domain == 'target':
        from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
        from scripts.sla.close_goal_g1_v1 import width_metrics

        state, target, _, _ = dataset.raw(index)
        physical = dataset.codecs[row['identity']['episode_id']].decode(
            state, pad32(dataset.statistics['actions'].unnormalize(active)))
        context = panel['target_contexts'][str(index)]
        if context['previous_command'] != dataset.reference(index):
            raise ValueError('SCALE20K_OFFLINE_TARGET_ACTUAL_PREVIOUS_COMMAND_CHANGED')
        envelope = analyze(physical, state, np.asarray(context['policy_from_world']), np.asarray(context['reference']))
        seam = label_margin_report(physical, context['previous_command'])
        metrics.update(pose_error=roundtrip_error(target, physical), width=width_metrics(physical, target),
            envelope=envelope, gripper_seam=seam,
            H40_and_seam_pass=envelope['original_full_H40_pass'] and not seam['original_execution_speed_exceeded'])
    elif domain == 'robodojo':
        from .source_evaluation_v6 import errors

        physical = dataset.statistics['actions'].unnormalize(active)
        arrays = dataset.arrays(row['identity']['episode'])
        anchor = row['identity']['anchor']
        metrics.update(recorded_reference=errors(physical, arrays['actions'][anchor], arrays['state'][anchor],
                                                 dataset.statistics), dynamic=bool(arrays['dynamic'][anchor]))
    else:
        lo, hi = (np.asarray(dataset.statistics['action'][k], np.float32) for k in ('q01', 'q99'))
        span = hi-lo+1e-6
        physical = (active+1)*.5*span+lo
        truth = (expected+1)*.5*span+lo
        metrics.update(native_command_channel_mae=np.abs(physical-truth).mean(axis=0).tolist(),
                       normalized_zero_command_mae=float(np.abs(expected-(-2*lo/span-1)).mean()))
    return physical, metrics


def bytes_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
