"""Core-local analysis of sealed predictions and recorded contacts; no policy calls."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .storage import digest, write_json
from . import fusion_trajectory_adapter as limits
from .task_learning_grasp_v1 import FiftyMsProgram, B

LIMITS = {
    'maximum_translation_speed_m_s': limits.MAX_TRANSLATION_SPEED_M_S,
    'maximum_angular_speed_rad_s': limits.MAX_ANGULAR_SPEED_RAD_S,
    'maximum_finger_speed_m_s': limits.MAX_FINGER_SPEED_M_S,
    'maximum_discrete_translation_acceleration_m_s2': limits.MAX_TRANSLATION_ACCELERATION_M_S2,
    'maximum_discrete_angular_acceleration_rad_s2': limits.MAX_ANGULAR_ACCELERATION_RAD_S2,
}


def checked(pin):
    path = Path(pin['path'])
    if path.is_symlink() or digest(path) != pin['sha256']:
        raise ValueError('Changed sealed analysis input')
    return json.loads(path.read_bytes())


def geometry(raw, state):
    from scripts.pi0292.eval_contract import matrix10
    anchor = matrix10(state)
    desired = raw.astype(float).copy()
    desired[:, 9] = (limits.WORKING_CLOSED_M +
                    (limits.WORKING_OPEN_M-limits.WORKING_CLOSED_M)*raw[:, 9])/limits.MECHANICAL_TRAVEL_M
    metrics = limits._geometry(desired, anchor)
    reasons = [k for k, maximum in LIMITS.items() if metrics[k] > maximum*(1+1e-10)]
    if not np.all((raw[:, 9] >= 0) & (raw[:, 9] <= 1)):
        reasons.append('decoded_gripper_outside_0_1')
    error = None
    try:
        FiftyMsProgram(raw, anchor, anchor, np.array([0., 0., 0., 1., 0., 0., 0.]))
    except ValueError as exc:
        error = str(exc)
    return dict(metrics=metrics, failed_components=reasons, original_envelope_exception=error,
                original_envelope_pass=error is None,
                first_position_jump_m=float(np.linalg.norm(raw[0, :3]-state[:3])),
                first_gripper_jump=float(raw[0, 9]-state[9]),
                decoded_gripper_min=float(raw[:, 9].min()), decoded_gripper_max=float(raw[:, 9].max()),
                exact_closed_knots=int(np.count_nonzero(raw[:, 9] == 0)),
                exact_open_knots=int(np.count_nonzero(raw[:, 9] == 1)),
                first_point_not_included_in_original_discrete_rate_gate=True)


def prediction_audit(result_pin, samples, dataset_pin, output):
    result, manifest = checked(result_pin), checked(dataset_pin)
    if result['actual_VLA_queries'] != 40 or len(result['records']) != 40:
        raise ValueError('Original complete Core40 required')
    episodes = {e['id']: e for e in manifest['episodes']}
    output = Path(output); output.mkdir(exist_ok=False)
    rows = []; predictions = {}
    for record in result['records']:
        file = Path(samples)/record['sample_file']
        if file.name != record['sample_file'] or digest(file) != record['sha256']:
            raise ValueError('Changed prediction member')
        with np.load(file, allow_pickle=False) as z:
            raw, expert, state = [z[k].copy() for k in ('physical_H40', 'expert_policy_H40', 'actual_state10')]
        if raw.shape != expert.shape or raw.shape != (40, 10) or not np.isfinite(raw).all():
            raise ValueError('Full finite unchanged H40 required')
        episode = episodes[record['episode']]
        if record['anchor'] not in episode['supervised_anchors']:
            raise ValueError('Excluded intervention in sealed probes')
        row = {k:record[k] for k in ('episode','anchor','seed','phase_audit_only','genuine_recovery_audit_only')}
        row.update(prediction_sha256=record['sha256'], student=geometry(raw,state), expert=geometry(expert,state),
                   event_errors=record['errors'].get('close_events'), cross_chunk=None)
        predictions[record['episode'], record['anchor']] = raw
        rows.append(row)
    for row in rows:
        previous = predictions.get((row['episode'], row['anchor']-1))
        if previous is not None:
            current = predictions[row['episode'], row['anchor']]
            row['cross_chunk'] = dict(previous_anchor=row['anchor']-1,
                position_gap_m=float(np.linalg.norm(current[0,:3]-previous[-1,:3])),
                gripper_gap=float(current[0,9]-previous[-1,9]),
                interpretation='Counterfactual join of predictions at recorded Teacher states; neither prediction was executed')
    reasons = Counter(k for row in rows for k in row['student']['failed_components'])
    summary = dict(schema='TASK_LEARNING_CORE40_LOCAL_ENVELOPE_AUDIT_V1',
        input_result=result_pin, dataset=dataset_pin, requested=40, audited=len(rows),
        original_envelope_pass=sum(r['student']['original_envelope_pass'] for r in rows),
        original_envelope_reject=sum(not r['student']['original_envelope_pass'] for r in rows),
        expert_envelope_pass=sum(r['expert']['original_envelope_pass'] for r in rows),
        failures_by_component=dict(reasons), adjacent_prediction_pairs=sum(r['cross_chunk'] is not None for r in rows),
        missing_adjacent_prediction_pairs=sum(r['anchor']>0 and r['cross_chunk'] is None for r in rows),
        rows=rows, original_limits=LIMITS, new_model_loads=0, new_queries=0, optimizer_updates=0,
        native_integrations=0, policy_execution_authorized=False, native_contact_gate_not_evaluated=True,
        clipping_applied=False, Teacher_fallback=False, cross_hardware_reject_replaced=False)
    write_json(output/'RESULT.private.json', summary)
    return summary


def contact_events(dataset_pin, output, guard=lambda: None):
    """Stream actual native trace contacts; phase names never stand in for contacts."""
    from scripts.pi0292.durable import write
    manifest = checked(dataset_pin); output = Path(output); output.mkdir(exist_ok=False)
    episodes = [r for r in manifest['episodes'] if r['split']=='TRAIN']
    rows=[]; windows=[]
    for episode in episodes:
        guard(); checked(episode['proof']['spec'])
        root = Path(episode['proof']['native']['path']).parent
        trace = root/'physics.jsonl'; hasher = __import__('hashlib').sha256()
        first={}; last={}; counts=Counter(); samples=0; previous=None; contact_transitions=[]
        with trace.open('rb') as stream:
            for raw in stream:
                hasher.update(raw); value=json.loads(raw)
                if value['kind']=='model':
                    geoms=value['geoms']; continue
                if value['kind']!='sample': continue
                step=int(value['step']); samples+=1
                if (previous is None and step!=1) or (previous is not None and step!=previous+3):
                    raise ValueError('Changed recorded 6ms contact trace cadence')
                previous=step
                if step<B: continue
                active=set()
                for c in value['contacts']:
                    pair=tuple(sorted((int(c['geom1']),int(c['geom2']))))
                    names=[geoms[i]['name'] for i in pair]
                    if 'task_cube_geom' not in names: continue
                    if float(c['force'][0]) < .05: continue
                    label=' / '.join(n or '<unnamed>' for n in names)
                    active.add(label); counts[label]+=1; first.setdefault(label,step); last[label]=step
                # Preserve transitions, rather than conflating contact duration and contact onset.
                if not contact_transitions or set(contact_transitions[-1]['pairs'])!=active:
                    contact_transitions.append(dict(native_step=step,pairs=sorted(active)))
        if previous!=21202 or samples!=7068: raise ValueError('Incomplete sealed 6ms TRAIN contact trace')
        onsets=[t['native_step'] for i,t in enumerate(contact_transitions) if i>0 and set(t['pairs'])-set(contact_transitions[i-1]['pairs'])]
        for anchor in episode['supervised_anchors']:
            start=B+1000*anchor; end=start+1000
            events=[s for s in onsets if start<=s<end]
            windows.append(dict(episode=episode['id'],anchor=anchor,phase_audit_only=episode['expert_stage_audit_only'][anchor],
                                contact_onset_native_steps=events,actual_contact_onset=bool(events),
                                onset_intervals_native_steps=[[s-3,s]for s in events]))
        rows.append(dict(episode=episode['id'],samples=samples,trace_sha256=hasher.hexdigest(),
                         first_contact_native_step=first,last_contact_native_step=last,contact_sample_counts=dict(counts),
                         transitions=contact_transitions, native_grasp_not_inferred_from_contact_alone=True))
        write(output/'PROGRESS.private.json',dict(completed_episodes=len(rows),requested=len(episodes),windows=len(windows)))
    if len(episodes)!=100 or len(windows)!=1450: raise ValueError('Original TRAIN100 denominator changed')
    result=dict(schema='TASK_LEARNING_TRAIN100_RECORDED_CONTACT_EVENTS_V1',dataset=dataset_pin,
                requested_episodes=100,audited_episodes=len(rows),distinct_supervised_windows=len(windows),
                contact_onset_windows=sum(r['actual_contact_onset'] for r in windows),rows=rows,windows=windows,
                model_loads=0,queries=0,optimizer_updates=0,native_integrations=0,TEST_read=False,
                event_condition='Recorded task_cube_geom pair normal force >=0.05N; not a native-grasp certificate',
                recorded_contact_period_s=.006,onset_time_resolution_s=.006,
                unrecorded_between_sample_contacts_not_excluded=True,
                event_metadata_is_model_input=False, original_data_modified=False)
    write_json(output/'RESULT.private.json', result)
    return result
