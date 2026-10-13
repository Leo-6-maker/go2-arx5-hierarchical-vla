"""Saved-state audit and READY-last admission; never integrates or renders."""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from scripts.iteration.queue import digest, value_digest
from scripts.pi0292.durable import read, write


def independent_push_score(data, card, complete, table_height):
    import numpy as np
    def sustained(flags):
        count = 0
        for flag in flags:
            count = count + 1 if flag else 0
            if count >= 7:
                return True
        return False
    xyz = data['object_pose'][:, :3, 3]
    grasp = np.logical_and.reduce([data[k] for k in ('object_present', 'contact_valid',
        'gripper_state_valid', 'native_grasp', 'left_object_contact', 'right_object_contact')])
    contact = data['robot_object_contact']
    micro = data['contact_audit']
    times = data['contact_time_s']
    high_valid = bool(micro.shape == (16000, 5) and times.shape == (16000,)
        and np.isfinite(micro).all() and np.max(np.abs(times - np.arange(1, 16001) * .002)) <= 1e-8)
    longest, current = 0, 0
    for flag in micro[:, 1] if high_valid else []:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    corners = xyz[:, None, :] + np.einsum('nij,kj->nki', data['object_pose'][:, :3, :3],
        np.asarray(list(itertools.product((-.02, .02), repeat=3))))
    hold = dict(zone_corners=(np.abs(corners[..., :2] - card['goal_xy']) <= .07).all(axis=(1, 2)),
        table_height=np.abs(xyz[:, 2] - (table_height + .02)) <= .005,
        linear_speed=np.linalg.norm(data['object_linear_velocity'], axis=1) <= .02,
        angular_speed=np.linalg.norm(data['object_angular_velocity'], axis=1) <= .10,
        released=~grasp & ~data['native_grasp'], robot_separated=~contact,
        tcp_clear=np.linalg.norm(data['tcp_world'][:, :3, 3] - xyz, axis=1) >= .08,
        object_and_contacts_valid=data['object_present'] & data['contact_valid'] & data['gripper_state_valid'])
    indices = np.flatnonzero((data['time_s'] >= 28 - 1e-8) & (data['time_s'] <= 30 + 1e-8))
    rise = max(float((xyz[:, 2] - xyz[0, 2]).max()),
               float(micro[:, 3].max() - xyz[0, 2]) if high_valid else float('inf'))
    history = dict(actual_robot_contact=bool(contact.any() or high_valid and micro[:, 0].any()),
        horizontal_push=bool(np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1).max() >= .10),
        no_lift=bool(rise <= .005 + 1e-8),
        no_sustained_bilateral_grasp=bool(longest * .002 < .3 - 1e-8 and not sustained(grasp)),
        highrate_contact_complete=high_valid)
    safe = bool(data['safety_valid'].all() and not any(data[k].any() for k in
        ('native_abort', 'illegal_contact', 'joint_limit_penetration', 'reset', 'takeover'))
        and high_valid and not micro[:, 4].any())
    passed = int(np.logical_and.reduce(list(hold.values()))[indices].sum())
    return dict(history=history, hold_observed=len(indices), hold_pass=passed,
        hold_subpredicates={k: int(v[indices].sum()) for k, v in hold.items()}, safety_valid=safe,
        maximum_center_height_rise_m=rise, maximum_bilateral_grasp_s=longest * .002,
        persistent_success=bool(complete and safe and all(history.values()) and len(indices) == 41 and passed == 41))


def reconstruct_micro(folder, card, asset_root):
    import mujoco
    import numpy as np
    from .scene import make_engine
    with np.load(folder / 'physics.npz', allow_pickle=False) as f:
        micro, clock, initial = f['contact_audit'].copy(), f['contact_time_s'].copy(), f['object_pose'][0].copy()
    with np.load(folder / 'highrate.npz', allow_pickle=False) as f:
        high, nq, nv = f['rows'].copy(), int(f['nq']), int(f['nv'])
    if len(high) != len(micro) or len(clock) != len(high):
        raise ValueError('HIGH_RATE_CONTACT_ROWS_MISMATCH')
    engine = make_engine(card, asset_root)
    maximum = 0.
    try:
        for index, row in enumerate(high):
            engine.data.qpos[:] = row[1:1 + nq]
            engine.data.qvel[:] = row[1 + nq:1 + nq + nv]
            engine.data.ctrl[:] = row[1 + nq + nv:1 + nq + nv + 8]
            engine.data.time = row[0]
            engine.data.xfrc_applied[:] = 0
            engine.data.qfrc_applied[:] = 0
            mujoco.mj_forward(engine.model, engine.data)
            state = engine.snapshot()
            from scripts.pi0291_arx5_data.config import PARAMETERS
            robot_contact = False
            for contact in state['contacts']:
                geoms = (int(contact['geom0']), int(contact['geom1']))
                if engine.geoms['cube_geom'] not in geoms:
                    continue
                if contact['normal_force_N'] < PARAMETERS['minimum_contact_force_N']:
                    continue
                other = geoms[0] if geoms[1] == engine.geoms['cube_geom'] else geoms[1]
                name = mujoco.mj_id2name(engine.model, mujoco.mjtObj.mjOBJ_BODY,
                                       int(engine.model.geom_bodyid[other])) or ''
                robot_contact |= name.startswith('x5_')
            obj = engine.pose('cube')
            expected = np.asarray([float(robot_contact), float(state['native_grasp']),
                np.linalg.norm(obj[:2, 3] - initial[:2, 3]), obj[2, 3], row[-1]])
            error = float(np.max(np.abs(expected - micro[index])))
            if error > 1e-8 or abs(clock[index] - row[0]) > 1e-8:
                raise ValueError('RECONSTRUCTED_HIGH_RATE_CONTACT_OR_HEIGHT_DIFFERS')
            maximum = max(maximum, error)
    finally:
        engine.close()
    return dict(reconstructed_rows=len(high), max_abs_error=maximum, integrations=0, render_calls=0)


def audit_episode(root, identifier, asset_root):
    root = Path(root)
    folder = root / 'episodes' / identifier
    if (folder / 'READY.json').exists() or (folder / 'AUDIT.json').exists():
        raise FileExistsError('PRESERVE_FIRST_AUDIT_AND_READY')
    ticket = read(root / 'reservations' / (identifier + '.json'))
    worker = read(folder / 'WORKER_RECEIPT.json')
    card = read(folder / 'CARD.json')
    if ticket['fixture'] is not False or worker['fixture'] is not False:
        raise ValueError('ACTUAL_PHYSICS_RECEIPT_REQUIRED')
    reserved_card = {key: card[key] for key in ticket['card']}
    if (reserved_card != ticket['card'] or value_digest(reserved_card) != ticket['card_sha256']
            or worker['card_sha256'] != ticket['card_sha256']):
        raise ValueError('FIRST_ATTEMPT_CARD_CHANGED')
    if card['id'] != identifier or Path(ticket['output']).resolve() != folder.resolve():
        raise ValueError('RESERVATION_EPISODE_BINDING_CHANGED')
    if worker.get('source_sha') != ticket['source_sha']:
        raise ValueError('WORKER_SOURCE_DIFFERS_FROM_RESERVATION')
    before = digest(folder / 'FILE_MANIFEST.json')
    started = time.monotonic()
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    result = dict(id=identifier, fixture=False, integrations=0, render_calls=0,
                  model_loads=0, admitted=False, status='AUDIT_FAIL_PRESERVED')
    try:
        import mujoco
        import numpy as np
        # Reuse the exact saved-state sensor/FK/contact/calibration verifier.
        native_path = Path(__file__).resolve().parents[1] / 'pi0291_arx5_data'
        sys.path.insert(0, str(native_path))
        base = importlib.import_module('scripts.pi0292.audit_data')
        from scripts.pi0291_arx5_data.config import PARAMETERS
        from .scene import make_engine
        with np.load(folder / 'highrate.npz', allow_pickle=False) as archive:
            external = archive['external_force_max']
            if external.shape != (worker['physics_steps'], 2) or not np.isfinite(external).all() or np.any(external != 0):
                raise ValueError('NONZERO_OR_INCOMPLETE_ALL_BODY_EXTERNAL_FORCE_EVIDENCE')
        def forbidden(*args, **kwargs):
            raise PermissionError('AUDITOR_FORBIDS_PHYSICS_INTEGRATION')
        originals = [mujoco.mj_step, mujoco.mj_step1, mujoco.mj_step2]
        previous_score = base.independent_score
        if card['task'] == 'T2':
            base.independent_score = independent_push_score
        mujoco.mj_step = mujoco.mj_step1 = mujoco.mj_step2 = forbidden
        try:
            detail = base.audit_episode(root, identifier,
                lambda bound, mode: make_engine(bound, asset_root), PARAMETERS)
            micro = reconstruct_micro(folder, card, asset_root) if card['task'] == 'T2' else None
        finally:
            base.independent_score = previous_score
            mujoco.mj_step, mujoco.mj_step1, mujoco.mj_step2 = originals
        if before != digest(folder / 'FILE_MANIFEST.json'):
            raise ValueError('RAW_CHANGED_DURING_AUDIT')
        if worker.get('samples_sha256') != detail.get('samples_sha256'):
            raise ValueError('WORKER_AND_AUDITOR_SAMPLE_IDENTITY_DIFFERS')
        admitted = bool(detail['status'] == 'PASS_FULL_RAW_AUDIT'
            and detail['complete'] and detail['independent_score']['persistent_success']
            and detail['all_action_supervision_valid'] and detail['paired_samples'] == 641
            and detail['physics_steps'] == 16000 and detail['actual_committed_H40_windows'] == 601)
        result.update(status='PASS_INDEPENDENT_RAW_AUDIT', admitted=admitted,
            quality_terminal='ADMITTED' if admitted else 'REJECTED_PRESERVED', detail=detail,
            micro_reconstruction=micro, card_sha256=ticket['card_sha256'],
            samples_sha256=detail.get('samples_sha256'), manifest_sha256=before,
            source_sha=worker.get('source_sha'), group=card['group_id'], task=card['task'],
            split=card['split'], family=card['family'], first_attempt=True)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    own = resource.getrusage(resource.RUSAGE_SELF)
    result.update(CPU_s=own.ru_utime + own.ru_stime - cpu0.ru_utime - cpu0.ru_stime,
                  wall_s=time.monotonic() - started)
    write(folder / 'AUDIT.json', result)
    if result['status'] == 'PASS_INDEPENDENT_RAW_AUDIT':
        write(folder / 'READY.json', dict(schema='PI0294_EPISODE_READY_V1',
            id=identifier, fixture=False, terminal=result['quality_terminal'], admitted=result['admitted'],
            card_sha256=result['card_sha256'], samples_sha256=result['samples_sha256'],
            raw_manifest_sha256=before, audit_sha256=digest(folder / 'AUDIT.json'),
            worker_receipt_sha256=digest(folder / 'WORKER_RECEIPT.json'),
            first_attempt=True, status='SEALED_AUDITED_EPISODE'))
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'id', 'asset-root'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--cpus')
    p.add_argument('--reservation', required=True)
    p.add_argument('--manifest')
    args = p.parse_args()
    if args.cpus:
        os.sched_setaffinity(0, {int(c) for c in args.cpus.split(',')})
    from .bank import validate_cpu_ticket
    validate_cpu_ticket(args.root, args.reservation, args.manifest)
    value = audit_episode(args.root, args.id, args.asset_root)
    print(json.dumps(dict(id=args.id, status=value['status'], admitted=value['admitted'], integrations=0)))
    return 0 if value['status'] == 'PASS_INDEPENDENT_RAW_AUDIT' else 2


if __name__ == '__main__':
    raise SystemExit(main())
