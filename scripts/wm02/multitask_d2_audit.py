"""Join saved mobile references to native packets and time archived evidence reads."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from go2arx5_vla.world_model.mobile_boundary_clock_v1 import accepted_boundary
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.target_reference import installed_target_identities


def read(path):
    return json.loads(Path(path).read_bytes())


def lines(path):
    with path.open('rb') as stream:
        yield from map(json.loads, stream)


def checked(path, sha):
    if digest(path) != sha:
        raise ValueError(f'ARCHIVE_BYTES_CHANGED: {path.name}')
    return path


def replay_history(native, labels):
    planner = list(lines(native/'planner.jsonl'))
    policies = [row for row in lines(native/'ROS_CONSUMPTION.private.jsonl')
                if row['topic_key'] == 'policy']
    confirmed = [i['boundary'] for q in labels for i in q['intervals'] if i['confirmed_ticks'] == 25]
    timings = {}
    cache = {}
    for fraction in (.125, .25, .5, 1.):
        n_policy = int(fraction*len(policies))
        n_event = int(fraction*len(planner))
        subset, policy_subset = planner[:n_event], policies[:n_policy]
        identities = {event['ref_id'] for event in subset if event['kind'] == 'installed'}
        sample = [w for w in confirmed if w['accepted_wire_sha256'] in identities][-24:]
        totals = Counter()
        for w in sample:
            wire, clock = w['accepted_wire_sha256'], w['native_time_s']
            started = time.perf_counter_ns()
            installed_target_identities(subset, wire)
            totals['identity_lookup_ns'] += time.perf_counter_ns()-started
            started = time.perf_counter_ns()
            try:
                _, evidence = accepted_boundary(subset, wire, clock, policy_subset,
                                                policy_identity_cache=cache)
            except ValueError:
                totals['witness_missing_in_truncated_archive'] += 1
            else:
                exact = (evidence['accepted_pose7'] == w['accepted_pose7']
                         and evidence['solver_target_identity'] == w['solver_target_identity'])
                totals['exact_pose_and_solver_id'] += exact
                totals['exact_pose'] += evidence['accepted_pose7'] == w['accepted_pose7']
            totals['accepted_boundary_with_cache_ns'] += time.perf_counter_ns()-started
            started = time.perf_counter_ns()
            encoded = json.dumps(w, allow_nan=False).encode()
            totals['boundary_JSON_encode_ns'] += time.perf_counter_ns()-started
            totals['encoded_bytes'] += len(encoded)
        timings[str(fraction)] = dict(planner_events=n_event, policy_events=n_policy,
                                      atom_lookups=len(sample), **totals)
    return dict(task_confirmed_atoms=len(confirmed), archive_cuts=timings,
                timing_scope='archive memory lookup and serialization, no live I/O or scheduling')


def audit_case(case, result_sha, output, *, history=False):
    import mujoco

    if mujoco.__version__ != '3.2.7' or mujoco.get_mjcb_control() is not None:
        raise ValueError('ARCHIVED_GEOMETRY_REQUIRES_PINNED_MUJOCO_WITHOUT_CONTROLLER')
    if read(case/'control/LEASE.json')['status'] != 'CLOSED':
        raise PermissionError('ARCHIVE_ONLY_REQUIRES_COLD_CASE')
    started = time.perf_counter()
    spec = read(case/'EXECUTION_SPEC.json')
    result = read(checked(case/'RESULT.private.json', result_sha))
    native = case/'native-01'
    scene = Path(spec['model_path'])
    model = mujoco.MjModel.from_binary_path(str(checked(scene, spec['frozen_files'][str(scene)])))
    data = mujoco.MjData(model)
    sites = model.site('x5_ee').id
    bodies = [model.body(name).id for name in ('x5_base_link', 'task_cube', 'task_blue_cube')]
    manifest = read(native/'LIVE_DATA/MANIFEST.private.json')
    label_pin = manifest['labels']
    labels = read(checked(Path(label_pin['path']), label_pin['sha256']))
    steps = result['actual_native_integrations']
    actual = np.empty((steps, 4, 4, 4))
    qpos = np.empty((steps, model.nq))
    qvel = np.empty((steps, model.nv))
    planned_query = np.full(steps, -1, np.int32)
    planned_slot = np.full(steps, -1, np.int8)
    consumed = np.zeros(steps, bool)
    accepted_knot = np.full((steps, 7), np.nan)
    accepted_ids = {}
    phases = Counter()
    complete_atoms = 0
    tick_rows = 0
    state_file = native/'state.bin.journal'
    with forbid_integration(mujoco) as calls:
        for tick_rows, packet in enumerate(packets(state_file), 1):
            if packet['step'] != tick_rows:
                raise ValueError('ARCHIVE_EXPECTS_UNBROKEN_NATIVE_TICK_CHAIN')
            data.qpos[:] = packet['qpos']
            data.qvel[:] = packet['qvel']
            mujoco.mj_kinematics(model, data)
            i = tick_rows-1
            qpos[i], qvel[i] = packet['qpos'], packet['qvel']
            actual[i, :, :, :] = np.eye(4)
            actual[i, 0, :3, :3] = data.site_xmat[sites].reshape(3,3)
            actual[i, 0, :3, 3] = data.site_xpos[sites]
            actual[i, 1:, :3, :3] = data.xmat[bodies].reshape(-1,3,3)
            actual[i, 1:, :3, 3] = data.xpos[bodies]
        if calls or tick_rows != steps:
            raise ValueError('ORIGINAL_NATIVE_DENOMINATOR_CHANGED')
    for row in labels:
        phases[row['phase']] += 1
        if any(row['consumed_mask'][10:]):
            raise ValueError('ARCHIVED_P10_SUFFIX_IS_NOT_EXECUTED')
        for interval in row['intervals']:
            start, atom = interval['start_native_step'], interval['atom']
            # Join native post-integration ticks to the command interval that
            # covers them. Accepted poses are only filled at actual knots.
            end = min(start+25, steps)
            planned_query[start:end] = row['query']
            planned_slot[start:end] = atom
            if interval['confirmed_ticks'] == 25:
                witness = interval['boundary']
                if (start+25 != interval['end_native_step'] or witness['native_step'] != end
                        or not row['consumed_mask'][atom]):
                    raise ValueError('EXACT_NATIVE_ATOM_AND_ACCEPTED_KNOT_REQUIRED')
                accepted_knot[end-1] = witness['accepted_pose7']
                accepted_ids[end] = [witness['accepted_wire_sha256'], witness['solver_target_identity']]
                consumed[start:end] = True
                complete_atoms += 1
    output.mkdir(mode=0o700)
    state_output = output/'TICK_REFERENCE_CHAIN.private.npz'
    np.savez_compressed(state_output, tick=np.arange(1, steps+1), actual_TCP_mount_red_blue=actual,
                        qpos=qpos, qvel=qvel, planned_query=planned_query, planned_atom=planned_slot,
                        fully_acknowledged_atom=consumed, accepted_pose7_at_native_knot=accepted_knot,
                        nominal_H40=np.asarray([q['nominal_world_H40'] for q in labels]),
                        submitted_H40=np.asarray([q['installed_reference_H40'] for q in labels]))
    id_output = output/'ACCEPTED_IDENTITIES.private.json'
    id_output.write_text(json.dumps(dict(accepted_ids_by_tick=accepted_ids,
        controller_selected_reference='UNKNOWN', accepted_pose7_empty_in_between=True)))
    score_file = native/'MULTITASK_SCORE_PROGRESS.private.json'
    profile = replay_history(native, labels) if history else None
    return dict(condition=case.name, result_sha256=result_sha, source_sha=spec['source_sha'],
                scene_sha256=spec['frozen_files'][str(scene)], labels_sha256=label_pin['sha256'],
                original_journal_sha256=digest(state_file), trace_sha256=digest(native/'physics.jsonl'),
                private_tick_chain_sha256=digest(state_output), private_accepted_ids_sha256=digest(id_output),
                native_packets=steps, actual_scientific_requests=result['physical_requests'],
                native_step_size_s=.002, complete_atoms=complete_atoms, query_counts_by_phase=dict(phases),
                partial_ticks=sum(int(q.get('partial_interval_ticks',0) or 0) for q in labels),
                actual_step_count_reconstructed=tick_rows, nominal_submitted_rows=len(labels)*40,
                first_native_failure=(result['physics_stops'][0]['reason'] if result['physics_stops']
                                      else (result['detail'] or {}).get('terminal_reason')),
                integrations_after_STOP=result['trace_audit']['integrations_after_first_STOP'],
                last_score=read(score_file) if score_file.exists() else None, controller_selected='UNKNOWN',
                profile=profile, CPU_scope='OFFLINE_PYTHON_ONLY', wall_s=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    cases = read(args.manifest)
    args.output.mkdir(mode=0o700)
    rows = {}
    for item in cases:
        ident = str(item['number'])
        rows[ident] = audit_case(Path(item['case']), item['result_sha256'], args.output/ident,
                                 history=item['number']==18)
    print(json.dumps(dict(cases=rows, new_native_requests=0, new_integrations=0,
                          new_generations=0, new_updates=0), allow_nan=False))


if __name__ == '__main__':
    main()
