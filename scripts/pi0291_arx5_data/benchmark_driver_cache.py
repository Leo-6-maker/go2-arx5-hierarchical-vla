"""Saved-state diagnostic only: no mj_step, renderer, or physics episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import mujoco
from .engine import Engine, DRIVER_VERSION, rotation_error
from .config import PARAMETERS as P
from .static_probe import independent_fk
from go2arx5_vla.task_data.io import file_hash, write


def main(episode, asset_root, output):
    began = time.process_time()
    episode = Path(episode)
    card = json.loads((episode/'CARD.json').read_text())
    samples = json.loads((episode/'samples.json').read_text())
    with np.load(episode/'highrate.npz') as source:
        rows = source['rows'].copy()
        nq, nv = int(source['nq']), int(source['nv'])
    engine = Engine(card, asset_root=asset_root)
    indices = sorted(set(range(0, 640, 16)) | {29, 46, 53, 87, 120, 148, 181, 190, 203, 256, 560, 639})
    old_cpu, cached_cpu, results = 0., 0., []
    max_snapshot_error, max_contact_error = 0., 0.
    for index in indices:
        if index:
            state = rows[index*25-1]
            assert abs(state[0]-index/20) < 1e-8
            engine.data.qpos[:] = state[1:1+nq]
            engine.data.qvel[:] = state[1+nq:1+nq+nv]
            engine.data.ctrl[:] = state[1+nq+nv:1+nq+nv+engine.model.nu]
        else:
            engine.data.qpos[:] = engine.model.qpos0
            engine.data.qpos[engine.arm_q] = samples[0]['execution']['actual_joint_q']
            engine.data.qpos[engine.finger_q] = samples[0]['execution']['actual_finger_q']
            engine.data.qvel[:] = 0.
            engine.data.ctrl[:] = 0.
        engine.data.time = index/20
        engine.data.xfrc_applied[:] = 0.
        engine.data.qfrc_applied[:] = 0.
        one = engine.snapshot()
        mujoco.mj_forward(engine.model, engine.data)
        two = engine.snapshot()
        for name in ('tcp', 'object_pose', 'object_linear_velocity', 'object_angular_velocity', 'actuator_force'):
            max_snapshot_error = max(max_snapshot_error, float(np.max(np.abs(one[name]-two[name]))))
        assert len(one['contacts']) == len(two['contacts'])
        for left, right in zip(one['contacts'], two['contacts']):
            assert (left['geom0'], left['geom1']) == (right['geom0'], right['geom1'])
            max_contact_error = max(max_contact_error, abs(left['normal_force_N']-right['normal_force_N']))
        target = np.asarray(samples[index]['supervision']['committed'])
        initial = engine.data.qpos[engine.arm_q].copy()
        engine.arm_targets = initial.copy()
        tick = time.process_time()
        for _ in range(25):
            old_solution, old_error = engine.solve_ik(target)
            engine.arm_targets += np.clip(old_solution-engine.arm_targets,
                                           -P['arm_target_speed_rad_s']*P['dt'], P['arm_target_speed_rad_s']*P['dt'])
        old_cpu += time.process_time()-tick
        old_targets = engine.arm_targets.copy()
        engine.arm_targets = initial.copy()
        engine.reset_ik_cache()
        tick = time.process_time()
        for _ in range(25):
            solution, error = engine.command_ik(target)
            engine.arm_targets += np.clip(solution-engine.arm_targets,
                                           -P['arm_target_speed_rad_s']*P['dt'], P['arm_target_speed_rad_s']*P['dt'])
        cached_cpu += time.process_time()-tick
        assert engine.ik_solve_calls == 1 and engine.ik_cache_hits == 24
        residual_pose = independent_fk(solution)
        independent_error = [float(np.linalg.norm(residual_pose[:3, 3]-target[:3, 3])),
                             float(np.linalg.norm(rotation_error(target[:3, :3], residual_pose[:3, :3])))]
        results.append(dict(index=index, old_ik_error=old_error, cached_ik_error=error,
                            independently_composed_cached_residual=independent_error,
                            max_slewed_joint_target_difference_rad=float(np.max(np.abs(old_targets-engine.arm_targets))),
                            cached_solve_calls=engine.ik_solve_calls, cached_hits=engine.ik_cache_hits))
        changed = target.copy()
        changed[0, 3] += .0001
        engine.command_ik(changed)
        assert engine.ik_solve_calls == 2
        engine.reset_ik_cache()
        engine.command_ik(target)
        assert engine.ik_solve_calls == 1 and engine.ik_cache_hits == 0
    assert engine.steps == 0
    assert max_snapshot_error < 1e-12 and max_contact_error < 1e-10
    result = dict(status='ZERO_INTEGRATION_DRIVER_CACHE_DIAGNOSTIC', driver_version=DRIVER_VERSION,
                  episode=card['id'], actual_input_source_sha=card['source_sha'], physics_steps=0, new_render_pairs=0,
                  actual_source_hashes={name:file_hash(episode/name) for name in ('samples.json', 'highrate.npz', 'physics.npz')},
                  input_command_count=len(indices), legacy_IK_cpu_s=old_cpu, cached_IK_cpu_s=cached_cpu,
                  measured_IK_speedup=old_cpu/max(cached_cpu, 1e-12),
                  max_snapshot_abs_error=max_snapshot_error, max_contact_force_error_N=max_contact_error,
                  max_cached_independent_position_error_m=max(r['independently_composed_cached_residual'][0] for r in results),
                  max_cached_independent_rotation_error_rad=max(r['independently_composed_cached_residual'][1] for r in results),
                  max_slewed_joint_target_difference_rad=max(r['max_slewed_joint_target_difference_rad'] for r in results),
                  cache_change_and_explicit_reset_tests='PASS', rows=results,
                  equivalence_scope='Snapshot duplicate-forward removal checked; IK numerical trajectories explicitly not claimed bitwise equivalent',
                  physical_qualification='NOT_RUN_BY_THIS_DIAGNOSTIC; owner must run distinct pre-pilot commissioning',
                  CPU_s=time.process_time()-began)
    write(Path(output), result)
    engine.close()
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', required=True)
    parser.add_argument('--asset-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.episode, args.asset_root, args.output)
