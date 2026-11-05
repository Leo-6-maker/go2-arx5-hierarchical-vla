"""Direct multi-horizon supported-contact prediction, never a full simulator.

Future inputs come from the predeclared program, not the achieved reference or
receiver memory. A horizon crossing a safety fallback is masked, not relabelled
as if the proposed program had executed. All past information is shared by the
action-conditioned and action-blind comparisons.
"""
import numpy as np
from .contact_features import state_feature
from .pilot2_supported import contract, stage_for

HORIZONS = (1, 4, 8, 16, 32, 64)
EVENTS = ('PAD_CONTACT_ANY', 'NATIVE_GRASP_END', 'TABLE_SUPPORT_END', 'OPEN_SUPPORTED_END')


def planned_openings(episode):
    from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
    contract(episode)
    return np.array([float(teacher_physical_fraction(
        .4 if stage_for(i, episode) in ('CLOSE', 'SUPPORTED_HOLD') else 1.))
        for i in range(episode['task_atoms'])])


def observed_events(row):
    ticks = row['native_contact_events']
    start, end = row['native_start_step'], row['native_end_step']
    if [e['native_step'] for e in ticks] != list(range(start+1, end+1)) or end-start != 25:
        raise ValueError('ALL25_ACTUAL_NATIVE_CONTACT_LABELS_REQUIRED')
    if any(bool(e['contact_positive']) != (e['max_pad_cube_normal_force_N'] >= .05) for e in ticks):
        raise ValueError('ORIGINAL_CONTACT_FORCE_THRESHOLD_REQUIRED')
    last = ticks[-1]
    if bool(row['contact_any']) != any(e['contact_positive'] for e in ticks):
        raise ValueError('ACTUAL_INTERVAL_CONTACT_LABEL_REQUIRED')
    return np.array([row['contact_any'], last['native_grasp'], last['table_support'],
        not last['contact_positive'] and min(last['finger_q']) >= .040 and last['table_support']], np.float32)


def replay(episodes, horizons=HORIZONS):
    if not horizons or any(type(h) is not int or h not in HORIZONS for h in horizons):
        raise ValueError('DECLARED_DIRECT_FORECAST_HORIZONS_REQUIRED')
    parents, runs = {}, set()
    out = {k: [] for k in ('history', 'plan', 'blind_plan', 'delta_mm', 'events', 'valid',
                           'transition', 'current_events', 'cv_mm', 'split', 'identities')}
    for ep in episodes:
        parent, split = ep['parent'], ep['split']
        if split not in ('TRAIN', 'DEV') or parent in parents and parents[parent] != split:
            raise ValueError('WHOLE_PARENT_TRAIN_DEV_ONLY_NO_TEST_REQUIRED')
        if ep['run_id'] in runs:
            raise ValueError('DUPLICATE_PHYSICAL_CONTACT_EPISODE')
        parents[parent] = split; runs.add(ep['run_id'])
        rows = ep['samples']; episode = contract(ep['episode']); plan = planned_openings(episode)
        if len(rows) != episode['task_atoms'] or ep['parent'] != episode['parent'] or split != episode['split']:
            raise ValueError('COMPLETE_REAL_CONTACT_EPISODE_AND_SAME_PARTITION_REQUIRED')
        observations, poses, events, actual = [], [], [], []
        for i, row in enumerate(rows):
            if (row['native_start_step'], row['native_end_step']) != (2503+25*i, 2528+25*i):
                raise ValueError('ORIGINAL_CONTACT_ATOM_CLOCK_REQUIRED')
            memory = row['actual_gripper_internal_memory']
            if memory['status'] != 'ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY' or memory['values']['steps'] != row['native_end_step']:
                raise ValueError('ACTUAL_SAME_PAUSE_GRIPPER_MEMORY_REQUIRED')
            events.append(observed_events(row)); observations.append(state_feature(row))
            poses.append(np.r_[row['TCP_m'], np.asarray(row['object_pose'])[:3, 3]])
            actual.append(row['executed_reference_EEF10'][-1])
        observations, poses, events, actual = map(np.asarray, (observations, poses, events, actual))
        for i in range(8, len(rows)-1):
            # Only past observations/receiver memory. Planned openings are known
            # at this boundary; no future stage or measured timestamp is read.
            future = plan[i+1:i+1+max(horizons)]
            future = np.pad(future, (0, max(horizons)-len(future)), constant_values=plan[-1])
            history = np.concatenate([np.r_[observations[j], events[j]] for j in range(i-8, i+1)])
            delta, target, valid, transitions = [], [], [], []
            for h in horizons:
                inside = i+h < len(rows)
                executed = inside and np.allclose(actual[i+1:i+h+1], plan[i+1:i+h+1], atol=1e-7, rtol=0)
                delta.append((poses[i+h]-poses[i])/.001 if inside else np.zeros(6))
                target.append(events[i+h] if inside else np.zeros(4))
                valid.append(executed)
                transitions.append(bool(inside and np.any(events[i+1:i+h+1] != events[i:i+h])))
            values = dict(history=history, plan=future, blind_plan=np.full_like(future, actual[i]),
                delta_mm=delta, events=target, valid=valid, transition=transitions,
                current_events=events[i], cv_mm=(poses[i]-poses[i-1])/.001,
                split=split, identities=dict(run_id=ep['run_id'], parent=parent,
                    anchor_native_step=rows[i]['native_end_step']))
            for key, value in values.items(): out[key].append(value)
    if not out['history']:
        raise ValueError('NONEMPTY_ACTUAL_CONTACT_HISTORY_REQUIRED')
    for key in out:
        if key not in ('split', 'identities'):
            out[key] = np.asarray(out[key], dtype=bool if key in ('valid', 'transition') else np.float32)
            if not np.isfinite(out[key]).all(): raise ValueError('FINITE_ACTUAL_CONTACT_DATA_REQUIRED')
    out['split'] = np.asarray(out['split']); out['horizons'] = tuple(horizons)
    return out




def metrics(delta, probability, data, ids):
    result = {}
    for j, horizon in enumerate(data['horizons']):
        result[str(horizon)] = {}
        for label, stratum in [('NATURAL', np.ones(len(data['valid']), bool)),
                                ('EVENT_TRANSITION', data['transition'][:, j]),
                                ('STEADY', ~data['transition'][:, j])]:
            selected = np.asarray(ids)[data['valid'][ids, j] & stratum[ids]]
            value = dict(valid_forecasts=len(selected))
            if len(selected):
                error = delta[selected, j]-data['delta_mm'][selected, j]
                truth, pred = data['events'][selected, j], probability[selected, j]
                value.update(TCP_delta_RMSE_mm=float(np.sqrt(np.mean(error[:, :3]**2))),
                    object_delta_RMSE_mm=float(np.sqrt(np.mean(error[:, 3:]**2))), events={})
                for k, name in enumerate(EVENTS):
                    t, p = truth[:, k] > .5, pred[:, k] >= .5
                    tp, fp, fn = int(np.sum(t&p)), int(np.sum(~t&p)), int(np.sum(t&~p))
                    value['events'][name] = dict(positive_labels=int(t.sum()),
                        Brier=float(np.mean((pred[:, k]-truth[:, k])**2)), TP=tp, FP=fp, FN=fn,
                        precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None)
            result[str(horizon)][label] = value
    return result
