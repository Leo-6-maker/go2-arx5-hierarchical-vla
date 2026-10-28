"""Two budgeted native fixtures. No hidden warm-up simulation or model calls."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

# IsaacGym must be loaded before libraries that might import Torch.
from environment import Environment
import numpy as np
from scipy.spatial.transform import Rotation
from contract import AUTH, DT, REPO, measured_opening


def plain(value):
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return [plain({k: row[k] for k in value.dtype.names}) for row in value]
        return value.tolist()
    if isinstance(value, np.void):
        return {k: plain(value[k]) for k in value.dtype.names}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def append(stream, row):
    stream.write(json.dumps(plain(row), sort_keys=True) + '\n'); stream.flush(); os.fsync(stream.fileno())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--authorization', required=True); p.add_argument('--fixture', choices=['freefall', 'tool-drive'], required=True)
    p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    if a.authorization != AUTH:
        raise ValueError('PI0275_AUTHORITY_REQUIRED')
    old = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0274/evidence/QUALITY_QUERY_LANES_COMPLETE.json')
    if not old.exists() or json.loads(old.read_text())['status'] != 'ALL_FIXED_NEW_QUALITY_QUERY_LANES_COMPLETE':
        raise RuntimeError('HEALTHY_PI0274_WORKERS_MUST_FINISH_FIRST')
    a.root.mkdir(parents=True, exist_ok=True)
    with (a.root / 'native.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        card = json.loads((a.root / 'EXECUTION_MANIFEST.json').read_text())
        for row in card['files']:
            if sha(REPO / row['path']) != row['sha256']:
                raise RuntimeError('PI0275_EXECUTION_SOURCE_DRIFT')
        out = a.root / 'native' / a.fixture
        if out.exists():
            raise RuntimeError('EXISTING_FIXTURE_ATTEMPT_DO_NOT_DUPLICATE')
        out.mkdir(parents=True); (out / 'frames').mkdir()
        family = card['scene_card']['scene_families'][0]
        # Persist a start before native create_sim/load/graphics, including failed loading.
        with (a.root / 'starts.jsonl').open('a') as ledger:
            if sum(1 for _ in (a.root / 'starts.jsonl').open()) >= card['budget']['total_starts']:
                raise RuntimeError('TOTAL_START_HARD_CAP')
            append(ledger, dict(kind='NATIVE_FIXTURE_START', fixture=a.fixture, authorization=AUTH,
                                execution_manifest_sha256=sha(a.root / 'EXECUTION_MANIFEST.json'),
                                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), maximum_ticks=1200, maximum_RGB_pairs=200))
        env = None; complete_ticks = 0; frames = 0; failure = None; observed = []
        result = dict(status='INTERRUPTED_NATIVE_ATTEMPT', physics_ticks=0, raw_consumption='SEE_JOURNAL')
        prior = [json.loads(line) for line in (a.root / 'consumption.jsonl').read_text().splitlines()] if (a.root / 'consumption.jsonl').exists() else []
        reserved_ticks = sum(x['kind'] == 'PHYSICS_TICK_RESERVED' for x in prior)
        reserved_frames = sum(x['kind'] == 'RGB_PAIR_RESERVED' for x in prior)
        try:
            env = Environment(family, freefall=a.fixture == 'freefall')
            engine = sys.modules['isaacgym']; api = sys.modules['isaacgym.gymapi']
            inventory = dict(isaacgym_file=engine.__file__, isaacgym_sha256=sha(engine.__file__),
                             gymapi_file=api.__file__, gymapi_sha256=sha(api.__file__),
                             body_mapping={str(k): list(v) for k, v in env.bodies.items()},
                             tool_body_mass_kg={name: props.mass for name, props in zip(env.tool_names, env.masses)},
                             effective_dof_properties=plain(env.props), native_pipeline='CPU PhysX, CPU state/pair contacts, GPU graphics',
                             contact_units_status='INSTALLED_DOCUMENTATION_BOUND; NATIVE_WEIGHT_CHECK_PENDING',
                             logical_initial_placement_time_s=env.tick * DT)
            (out / 'native_inventory.json').write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
            with (out / 'trace.jsonl').open('a') as trace, (a.root / 'consumption.jsonl').open('a') as journal:
                for k in range(1201):
                    s = env.state(); observed.append(s)
                    contacts = env.contact_flags(s, 'red')
                    command_p = np.array(family['initial_tcp_position_m']); command_r = np.diag([1., -1., -1.]); width = .08
                    if a.fixture == 'tool-drive':
                        # Fixed clear-space rate-limited schedule, not teacher tuning.
                        t = k * DT
                        if t < 1: width = .08
                        elif t < 3: width = .08 - .04 * (t - 1)
                        elif t < 4: width = 0.
                        elif t < 6: width = .04 * (t - 4)
                        command_p[0] += min(.020, max(0., .10 * (t - 6)))
                        angle = min(.1, max(0., .5 * (t - 7)))
                        command_r = Rotation.from_rotvec([0, 0, angle]).as_matrix() @ command_r
                    label = np.r_[command_p, command_r[:2].ravel(), width / .08]
                    append(trace, dict(tick=k, measurement=s, contact=contacts, commanded_label=label))
                    if k == 1200:
                        break
                    if k % 6 == 0:
                        if reserved_frames >= card['budget']['RGB_pairs']:
                            raise RuntimeError('RGB_PAIR_HARD_CAP')
                        append(journal, dict(kind='RGB_PAIR_RESERVED', fixture=a.fixture, tick=k, ordinal=frames + 1))
                        reserved_frames += 1
                        image, matrices = env.render(s)
                        np.savez_compressed(out / 'frames' / ('%04d.npz' % frames), **image, timestamp=s['time'],
                                            tcp_position=s['p'], tcp_rotation=s['r'], measured_width_m=s['width'])
                        append(trace, dict(kind='CAMERA', tick=k, camera=matrices)); frames += 1
                    if reserved_ticks >= card['budget']['physics_ticks']:
                        raise RuntimeError('PHYSICS_TICK_HARD_CAP')
                    append(journal, dict(kind='PHYSICS_TICK_RESERVED', fixture=a.fixture, tick=k, substeps=2)); reserved_ticks += 1
                    applied = env.step(label, s); complete_ticks += 1
                    append(trace, dict(kind='APPLIED_EFFORTS', tick=k, **applied))
            if a.fixture == 'freefall':
                z = np.array([s['object_position'][0, 2] for s in observed])
                speed = np.linalg.norm([s['object_velocity'][0] for s in observed[-120:]], axis=-1)
                supports = [env.contact_flags(s, 'red')['supported'] for s in observed[-120:]]
                body_ids = {name: bid for bid, name in env.bodies.items()}
                support_force = []
                for state in observed[-120:]:
                    support_force.append(sum(float(c['lambda']) * abs(float(c['normal']['z'])) for c in state['contacts']
                                             if {int(c['body0']), int(c['body1'])} == {body_ids[('table', 'table')], body_ids[('red', 'red')]}))
                force_ratio = float(np.median(support_force) / (.05 * 9.81))
                if not (z[0] - z[12] > .015 and all(supports) and speed.max() < .02 and abs(z[-1] - .665) < .002 and .75 <= force_ratio <= 1.25):
                    failure = 'FREEFALL_CONTACT_OR_SETTLING_FAIL'
            else:
                widths = np.array([s['width'] for s in observed])
                s = observed[-1]
                angle = Rotation.from_matrix(s['r'] @ np.diag([1., -1., -1.])).magnitude()
                if not (widths.min() <= .001 and abs(widths[-1] - .08) <= .001 and abs(s['p'][0] - .020) <= .005 and abs(angle - .1) <= .05):
                    failure = 'FINITE_DRIVE_OPENING_OR_ORIENTATION_FAIL'
            result = dict(status='NATIVE_FIXTURE_PASS' if failure is None else 'VALID_NATIVE_CHECK_FAIL', failure=failure,
                          physics_ticks=complete_ticks, RGB_pairs=frames, source_manifest_sha256=sha(a.root / 'EXECUTION_MANIFEST.json'),
                          contact_dtype=env.contact_dtype, contact_force_binding='installed SDK RigidContact.lambda: contact force magnitude, N; last substep, no dt conversion',
                          contact_floor_N=.001, setter_attempts=env.setter_attempts, tool_body_names=env.tool_names, tool_dof_names=env.dof_names,
                          learned_calls=0, teacher_updates=0, WBC_runs=0, selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
            if a.fixture == 'freefall':
                result['median_support_force_N'] = float(np.median(support_force)); result['support_force_to_weight_ratio'] = force_ratio
        except Exception as exc:
            result = dict(status='NATIVE_ENGINEERING_OR_EVIDENCE_INCOMPLETE', exception=repr(exc), physics_ticks=complete_ticks,
                          RGB_pairs=frames, learned_calls=0, WBC_runs=0, raw_reserved_consumption='SEE_APPEND_ONLY_JOURNAL')
            raise
        finally:
            if env is not None:
                env.close()
            (out / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
