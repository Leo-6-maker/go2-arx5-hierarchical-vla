"""Read-only PI0258 A0 over the six sealed PI0256 Bfix recordings."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0258')
OLD = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0256')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def decompose(q, soft_margin, hard):
    assert q.shape == soft_margin.shape and hard.shape == (q.shape[1], 2)
    hard_margin = np.minimum(q - hard[:, 0], hard[:, 1] - q)
    return hard_margin, soft_margin < 0, hard_margin < 0


def check():
    h, s, b = decompose(np.array([[0., .5], [-.1, .9]]),
                        np.array([[-.1, .4], [-.2, .0]]), np.array([[0., 1.], [0., 1.]]))
    assert s.tolist() == [[True, False], [True, False]]
    assert b.tolist() == [[False, False], [True, False]]
    assert h[1, 0] == -.1


def run():
    check()
    source = json.loads((OLD / 'evidence/runtime_readback.json').read_text())
    files = {x['name']: x for x in source['files']}
    readback = []
    for row in source['files']:
        actual = digest(row['path'])
        assert actual == row['sha256'], row['name']
        readback.append(dict(name=row['name'], path=row['path'], sha256=actual))
    urdf = ET.parse(files['native_urdf']['path']).getroot()
    joints = [j for j in urdf.findall('joint') if j.attrib['type'] != 'fixed']
    names = [j.attrib['name'] for j in joints]
    assert names[12:18] == ['x5_joint%d' % i for i in range(1, 7)]
    hard = np.array([[float(j.find('limit').attrib[k]) for k in ('lower', 'upper')] for j in joints])
    cases, joint_rows, input_index = [], [], []
    for i in range(6):
        folder = OLD / 'episodes' / ('pi0256-prep-Bfix%02d' % i)
        result = json.loads((folder / 'result.json').read_text())
        assert result['validity'] == 'VALID'
        raw = folder / 'raw_ticks.jsonl'
        assert digest(raw) == result['files']['raw_ticks.jsonl']['sha256']
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        assert len(rows) == 250 and all(r['clean'] and r['pre_autoreset_event'] is None for r in rows)
        q = np.array([r['post_step']['telemetry']['dof_pos'] for r in rows])
        soft_margin = np.array([r['post_step']['telemetry']['joint_limit_margin'] for r in rows])
        hm, soft, hard_bad = decompose(q, soft_margin, hard)
        dt = np.array([r['dt_s'] for r in rows])
        # Confirm the stored native predicate exactly, without changing its semantics.
        assert np.array_equal(soft.any(axis=1), [r['joint_limit_violation'] for r in rows])
        expected_soft = np.column_stack((hard.mean(axis=1) - .45 * np.diff(hard).ravel(),
                                         hard.mean(axis=1) + .45 * np.diff(hard).ravel()))
        sm_recomputed = np.minimum(q - expected_soft[:, 0], expected_soft[:, 1] - q)
        assert np.max(abs(sm_recomputed - soft_margin)) < 3e-7
        for j, name in enumerate(names):
            role = 'ACTOR_COMMANDED_LEG' if j < 12 else 'ACTOR_COMMANDED_ARM' if j < 18 else 'NO_ACTOR_CHANNEL_FIXED_ZERO_POSITION_DRIVE'
            joint_rows.append(dict(case=folder.name, joint=name, role=role, hard_lower=hard[j, 0],
                hard_upper=hard[j, 1], soft_lower=expected_soft[j, 0], soft_upper=expected_soft[j, 1],
                soft_ticks=int(soft[:, j].sum()), soft_duration_s=float(dt[soft[:, j]].sum()),
                soft_max_penetration=float(max(0., -soft_margin[:, j].min())),
                hard_ticks=int(hard_bad[:, j].sum()), hard_duration_s=float(dt[hard_bad[:, j]].sum()),
                hard_max_penetration=float(max(0., -hm[:, j].min())), native_termination_ticks=0))
        errors = []; history_errors = []; targets = []; residuals = []; base_attitudes = []
        for k, row in enumerate(rows):
            state = row['pre_step']; base = np.array(state['root'][0]); rb = Rotation.from_quat(base[3:7])
            actual = Rotation.from_quat(state['quaternion_xyzw']); desired = Rotation.from_quat(state['desired_quaternion'])
            position_error = rb.inv().apply(np.array(state['desired_world']) - state['TCP_world'])
            relative = desired * actual.inv()
            rotation_error = rb.inv().apply(relative.as_rotvec())
            qe = relative.as_quat()
            native_rotation_proxy = rb.inv().apply(qe[:3] * np.sign(qe[3]))
            target_position = rb.inv().apply(np.array(state['desired_world']) - base[:3])
            body_matrix = (rb.inv() * desired).as_matrix()
            target_rotation = body_matrix[:2].reshape(-1)
            shim_columns = body_matrix[:, :2].reshape(-1)
            f = folder / 'actor_inputs' / ('%04d-arm-observation.npz' % k)
            assert digest(f) == result['files'][str(f.relative_to(folder))]['sha256']
            with np.load(f, allow_pickle=False) as a:
                obs = a['obs'][0]; history = a['obs_history'][0]
            with np.load(folder / 'actor_inputs' / ('%04d-arm.npz' % k), allow_pickle=False) as a:
                assert np.array_equal(a['history'][0], history)
            history_errors.append(float(np.max(abs(history[-len(obs):] - obs))))
            errors.append([float(np.max(abs(obs[:3] - position_error))),
                float(np.max(abs(obs[3:6] - native_rotation_proxy))),
                float(np.max(abs(obs[12:15] - target_position))),
                float(np.max(abs(obs[15:21] - target_rotation))),
                float(np.max(abs(obs[15:21] - shim_columns)))])
            targets.append(target_rotation.tolist()); residuals.append(rotation_error.tolist())
            base_attitudes.append(rb.as_euler('xyz').tolist())
        phases = {}
        for name, select in [('moving', [r['time_s'] <= 1.95 for r in rows]),
                             ('terminal_hold', [r['time_s'] > 1.95 for r in rows])]:
            selected = [r for r, use in zip(rows, select) if use]
            phases[name] = dict(ticks=len(selected), **{
                key: dict(rms=float(np.sqrt(np.mean([r[key]**2 for r in selected]))),
                          maximum=float(max(r[key] for r in selected)))
                for key in ('position_error_m', 'rotation_error_rad', 'speed_mps', 'angular_speed_radps')})
        arms = np.array([r['policy']['arm_action_6d'] for r in rows])
        plans = np.array([r['policy']['internal_body_plan_9d'] for r in rows])
        commanded = np.array([r['post_step']['telemetry']['joint_pos_target'][12:18] for r in rows])
        cases.append(dict(case=folder.name, source_sha256=digest(raw), phases=phases,
            soft_any_ticks=int(soft.any(axis=1).sum()), soft_arm_ticks=int(soft[:, 12:18].any(axis=1).sum()),
            soft_leg_ticks=int(soft[:, :12].any(axis=1).sum()), soft_gripper_ticks=int(soft[:, 18:].any(axis=1).sum()),
            gripper_only_ticks=int((soft[:, 18:].any(axis=1) & ~soft[:, :18].any(axis=1)).sum()),
            hard_any_ticks=int(hard_bad.any(axis=1).sum()),
            actor_input_max_abs_errors=dict(zip(['ee_position_body','ee_native_sine_half_angle_proxy_body','target_position_body','target_rot6d_rows_body','target_rot6d_shim_columns_body'],np.max(errors,axis=0).tolist())),
            actor_history_tail_max_error=max(history_errors), arm_actor_abs_max=float(abs(arms).max()),
            body_plan_abs_ge1_counts=(abs(plans)>=1).sum(axis=0).tolist(),
            recorded_joint_target_vs_quarter_actor_max_abs=float(np.max(abs(commanded - .25 * arms))),
            recorded_joint_target_vs_quarter_clipped10_actor_max_abs=float(np.max(abs(commanded - .25 * np.clip(arms,-10,10)))),
            base_roll_pitch_yaw_range=np.column_stack((np.min(base_attitudes,axis=0),np.max(base_attitudes,axis=0))).tolist(),
            new_physics=0, new_actor_calls=0))
        np.savez_compressed(ROOT / 'evidence' / (folder.name + '-A0.npz'),
            actor_input_component_errors=np.asarray(errors),rotation_log_residual_base=np.asarray(residuals),
            hard_margin=hm,soft_margin=soft_margin,synthetic=False)
        input_index.append(dict(case=folder.name,result_sha256=digest(folder/'result.json'),raw_sha256=digest(raw)))
    path=ROOT/'evidence/per_joint_margins.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(joint_rows[0]));writer.writeheader();writer.writerows(joint_rows)
    write(ROOT/'evidence/control_audit.json',dict(schema='PI0258_CONTROLLER_A0_V1',cases=cases,
        source_readback=readback,input_index=input_index,new_physics=0,new_GR00T=0,
        gripper_role='No actor output, but native position drives remain enabled; not a freely passive DOF',
        limits='Native soft factor0.9; original external admission all-DOF flag retained. Native actual termination0 in these Bfix traces.',
        training_support='UNKNOWN until original training-distribution artifacts are evidenced',
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print(json.dumps(cases))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:check();print('PASS_LIMIT_DECOMPOSITION_NEGATIVES')
    else:
        (ROOT/'evidence').mkdir(parents=True,exist_ok=True)
        run()
