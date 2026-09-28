#!/usr/bin/env python3
"""Original C0 native reference control; no model, labels, or training."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    for name in ('source-root', 'trajectory', 'arm', 'dog', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--target-position-world-m', nargs=3, type=float, required=True)
    p.add_argument('--task', choices=tuple('ABCD'), required=True)
    p.add_argument('--seed', type=int, default=22301)
    p.add_argument('--sim-device', default='cuda:0')
    args = p.parse_args()
    assert args.seed == 22301
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert not any(args.output_dir.iterdir()), 'first-valid output must not be overwritten'
    code = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/reference-317a')
    sys.path[:0] = [str(code / 'scripts'), str(code / 'src')]
    import stage21_native_receding_horizon as native_api
    from go2arx5_vla.benchmark.lifecycle import reference_is_complete
    from go2arx5_vla.safety import TrajectorySafetyGateV1, GateDecision
    args.mode = 'gt_receding'
    target = np.asarray(args.target_position_world_m)
    native = env = None
    rows, frames = [], []
    result = {'schema': 'PI0252_NATIVE_C0_ENVIRONMENT_CONTROL_V1', 'task': args.task,
              'reset_seed': args.seed, 'target_world_m': target.tolist(),
              'model_queries': 0, 'optimizer_updates': 0, 'reference_updates': 0,
              'selected_checkpoint': 'NONE', 'status': 'ENGINEERING_INCOMPLETE'}
    try:
        trajectory = native_api._load_pose_file(args.trajectory)
        native, env, actors, loads, cfg = native_api._build_native(args, trajectory)
        native_api._set_target(native, target)
        anchor = native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
        from pi0227_expert_reference_binding import _remap_trajectory_to_target
        active = _remap_trajectory_to_target(trajectory, anchor, target)
        expert_chunk = native_api._make_chunk(active, chunk_id='expert-C0', observation_timestamp=0.0, start_time_s=0.0)
        assert np.min(np.linalg.norm(expert_chunk.eef_position + anchor - target, axis=1)) <= 1e-8
        result['initial_bind'] = native.initial_bind_reference(expert_chunk)
        result['reference_target_binding'] = 'ORIGINAL_PI0227_INTERIOR_WAYPOINT_TO_TARGET'
        np.savez_compressed(args.output_dir / 'actual_expert_reference.npz',
            timestamp=expert_chunk.point_timestamps, position=expert_chunk.eef_position,
            quaternion=expert_chunk.eef_orientation, anchor_world_m=anchor, target_world_m=target)
        pos, quat = native_api._current_eef(native)
        gate = TrajectorySafetyGateV1().evaluate(native.stage21_current_chunk,
            now=0.0, achieved_position=pos-anchor, achieved_orientation=quat)
        result.update(gate=gate.as_dict(), actor_loads=loads, anchor_world_m=anchor.tolist(),
                      native_control_dt_s=float(native.dt))
        assert gate.decision is not GateDecision.REJECT, 'original expert reference gate rejected'
        frames.append(native_api._capture_scene(native))
        termination = 'TICK_CAP'
        for step in range(500):
            done, policy = native_api._policy_step(native, env, actors)
            event = native.pop_stage16_termination()
            pos, quat = native_api._current_eef(native)
            t = float(native.traj_sim_time[0].detach().cpu().item())
            s = native.traj_batch.s_ref(native.traj_sim_time)
            desired = native.traj_batch.p_at(s)[0].detach().cpu().numpy()
            row = dict(step=step, reference_time_s=t, achieved_world_m=pos.tolist(),
                       desired_world_m=desired.tolist(), target_distance_m=float(np.linalg.norm(pos-target)),
                       tracking_error_m=float(np.linalg.norm(pos-desired)), done=bool(done),
                       reset_buf=bool(native.reset_buf[0].item()), policy=policy,
                       termination_event=event)
            row.update(native_api._native_runtime_telemetry(native))
            rows.append(row)
            frames.append(native_api._capture_scene(native))
            if done or event:
                termination = event.get('termination_code', 'NATIVE_TASK_TERMINATION') if event else 'NATIVE_TASK_TERMINATION'
                break
            if reference_is_complete(sim_time=t, reference_duration=float(native.traj_batch.T[0].item()),
                    s_ref=float(s[0].item()), reference_length=float(native.traj_batch.L[0].item()), dt=float(native.dt)):
                termination = 'REFERENCE_COMPLETE'
                break
        clean = bool(rows) and all(not r['done'] and not r['reset_buf'] and not r['termination_event']
                    and not r.get('invalid_state', False) and not r.get('fall_flag', False)
                    and not r.get('joint_limit_termination', False)
                    and np.isfinite(r['achieved_world_m']).all() for r in rows)
        distances = [r['target_distance_m'] for r in rows]
        hit = [i for i, d in enumerate(distances) if d <= .035]
        result.update(status='PASS' if clean and hit and termination == 'REFERENCE_COMPLETE' else 'CONTROL_FAIL',
                      physical_clean=clean, complete_duration=termination == 'REFERENCE_COMPLETE',
                      termination=termination, rows=len(rows), min_world_tcp_distance_m=min(distances),
                      final_world_tcp_distance_m=distances[-1], first_hit_tick=hit[0] if hit else None,
                      tracking_error_max_m=max(r['tracking_error_m'] for r in rows))
    except Exception as e:
        result['engineering_error'] = {'type': type(e).__name__, 'message': str(e)}
    finally:
        if native is not None:
            result['native_load_events'] = native._stage21_load_events
        if env is not None:
            env.close()
        if frames:
            np.savez_compressed(args.output_dir / 'actual_camera_frames.npz',
                                body_rgb=np.stack([f[0] for f in frames]), wrist_rgb=np.stack([f[1] for f in frames]))
        (args.output_dir / 'rows.jsonl').write_text(''.join(json.dumps(native_api._jsonable(r))+'\n' for r in rows))
        (args.output_dir / 'control.json').write_text(json.dumps(native_api._jsonable(result), indent=2)+'\n')
    print(json.dumps(result, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x), flush=True)
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
