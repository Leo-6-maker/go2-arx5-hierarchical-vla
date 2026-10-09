"""Finite PI0287 native simulation entry, one lease, no GR00T or optimizer."""
import argparse
import fcntl
import json
from pathlib import Path
import resource
import time


def pose(state):
    import numpy as np
    from go2arx5_vla.trajectory.so3 import quat_to_matrix
    out = np.eye(4); out[:3, :3] = quat_to_matrix(state[3:7]); out[:3, 3] = state[:3]
    return out


def scene_smoke(root, config, budget):
    import isaacgym
    import torch
    import numpy as np
    from go2arx5_vla.target_sim.environment import build_native, SCENE
    results = []
    for n in (1, 2):
        budget.charge(env_episode_starts=n)
        wrapped, proxy, cfg = build_native(config, budget, n, pinned_diagnostic=True)
        env = wrapped.env
        try:
            ids = torch.arange(n, device=env.device)
            env.reset_idx(ids)
            proxy.refresh_actor_root_state_tensor(env.sim)
            proxy.refresh_rigid_body_state_tensor(env.sim)
            proxy.refresh_dof_state_tensor(env.sim)
            # Partial reset is checked at a paused simulator boundary, without unrelated evolution.
            before = {k: v.clone() for k, v in proxy.global_tensors.items()}
            history_before = {k: v.clone() for k, v in vars(wrapped).items() if torch.is_tensor(v)}
            budget.charge(env_episode_starts=1)
            env.reset_idx(torch.tensor([0], device=env.device))
            for name in ('root', 'dof', 'body', 'contact'):
                proxy._refresh(name, {'root': 'refresh_actor_root_state_tensor', 'dof': 'refresh_dof_state_tensor',
                                     'body': 'refresh_rigid_body_state_tensor', 'contact': 'refresh_net_contact_force_tensor'}[name], env.sim)
            checks = {}
            if n == 2:
                checks['other_robot_root_unchanged'] = torch.equal(before['root'][proxy.actor_map['robot'][1]], proxy.global_tensors['root'][proxy.actor_map['robot'][1]])
                checks['other_object_root_unchanged'] = torch.equal(before['root'][proxy.actor_map['object'][1]], proxy.global_tensors['root'][proxy.actor_map['object'][1]])
                checks['other_robot_dofs_unchanged'] = torch.equal(before['dof'][proxy.dof_map[1]], proxy.global_tensors['dof'][proxy.dof_map[1]])
                checks['nonadjacent_robot_actors'] = proxy.actor_map['robot'][1] != proxy.actor_map['robot'][0] + 1
                checks['history_unchanged'] = all(torch.equal(v[1], getattr(wrapped, k)[1]) for k, v in history_before.items() if v.ndim and len(v) == n)
            # Empty-load named gripper calibration. No leg/arm policy, no task qualification.
            names = proxy.rows[0]['dof_names']
            grip_ids = [names.index(k) for k in ('x5_gripper_joint', 'x5_joint8')]
            calibration, frames = [], []
            for opening in (.044, 0.0, .044):
                targets = env.dof_pos.clone(); targets[:, grip_ids] = opening
                props = proxy.gym.get_actor_dof_properties(proxy.rows[0]['env'], proxy.rows[0]['robot'])
                for r in proxy.rows:
                    properties = proxy.gym.get_actor_dof_properties(r['env'], r['robot'])
                    for j in grip_ids:
                        properties['driveMode'][j] = proxy.api.DOF_MODE_POS
                        properties['stiffness'][j] = 1000.; properties['damping'][j] = 100.
                        properties['effort'][j] = 100.; properties['velocity'][j] = 0.2
                    proxy.gym.set_actor_dof_properties(r['env'], r['robot'], properties)
                for tick in range(40):
                    proxy.gym.set_dof_position_target_tensor(env.sim, proxy.gymtorch.unwrap_tensor(targets))
                    proxy.simulate(env.sim); proxy.gym.fetch_results(env.sim, True)
                    proxy.refresh_dof_state_tensor(env.sim)
                    proxy.refresh_actor_root_state_tensor(env.sim)
                    proxy.refresh_rigid_body_state_tensor(env.sim)
                actual = env.dof_pos[:, grip_ids].detach().cpu().numpy()
                calibration.append(dict(command_per_finger_m=opening, actual_per_finger_m=actual.tolist()))
                for i in range(n):
                    s = proxy.robot_tensors['body'].reshape(n, -1, 13)[i, env.ee_idx].detach().cpu().numpy().copy()
                    tcp = pose(s); tcp[:3, 3] += tcp[:3, :3] @ np.array(cfg.arm.ik.ee_local_pos)
                    base = pose(proxy.robot_tensors['root'][i].detach().cpu().numpy())
                    images, cameras = proxy.rgb(i, tcp, base, float(proxy.gym.get_sim_time(env.sim)))
                    frames.append(np.concatenate(images, axis=1))
            directory = root / 'evidence' / ('scene-smoke-n' + str(n))
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / 'ACTUAL_RGB.npz', frames=np.stack(frames))
            objects = proxy.global_tensors['root'][proxy.actor_map['object']].detach().cpu().numpy()
            checks['object_table_height'] = bool(np.all(np.abs(objects[:, 2] - .44) < .01))
            checks['gripper_direction'] = calibration[0]['actual_per_finger_m'][0][0] > calibration[1]['actual_per_finger_m'][0][0] + .02
            checks['both_fingers_driven'] = bool(np.max(np.abs(np.asarray(calibration[0]['actual_per_finger_m'])[:, 0] - np.asarray(calibration[0]['actual_per_finger_m'])[:, 1])) < .002)
            result = dict(status='PASS_NO_WBC_PINNED_ASSET_DIAGNOSTIC' if all(checks.values()) else 'FAIL_ASSET_DIAGNOSTIC',
                          evidence_kind='REAL_TARGET_PHYSICS_NO_WBC_PINNED_BASE', num_envs=n,
                          checks=checks, calibration=calibration, actor_indices=proxy.actor_map,
                          robot_body_indices=proxy.body_map, robot_dof_indices=proxy.dof_map,
                          dof_names=names, object_states=objects.tolist(), scene=SCENE,
                          final_camera=cameras, WBC_actor_forwards=0, task_qualification=False)
            (directory / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
            results.append(result)
        finally:
            env.close()
    return dict(status='SCENE_DIAGNOSTICS_COMPLETED_G1_NOT_YET_QUALIFIED', results=results)


def nominal_reset(wrapped, proxy, budget, cfg):
    import torch
    env = wrapped.env
    budget.charge(env_episode_starts=env.num_envs)
    env.explicit_reset = True
    ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(ids)
    # Nominal configuration is applied once at the episode boundary, DOF first/root last.
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel[:] = 0
    env.root_states[:] = env.base_init_state
    env.root_states[:, :3] += env.env_origins
    proxy.set_dof_state_tensor_indexed(env.sim, proxy.gymtorch.unwrap_tensor(env.dof_state),
                                      proxy.gymtorch.unwrap_tensor(ids.int()), len(ids))
    proxy.set_actor_root_state_tensor_indexed(env.sim, proxy.gymtorch.unwrap_tensor(env.root_states),
                                             proxy.gymtorch.unwrap_tensor(ids.int()), len(ids))
    for key, value in vars(wrapped).items():
        if 'history' in key and torch.is_tensor(value): value.zero_()
    # One source-dt integration populates actual FK; it is charged and saved as initialization.
    proxy.simulate(env.sim); proxy.gym.fetch_results(env.sim, True)
    proxy.refresh_actor_root_state_tensor(env.sim); proxy.refresh_dof_state_tensor(env.sim)
    proxy.refresh_rigid_body_state_tensor(env.sim); proxy.refresh_net_contact_force_tensor(env.sim)
    s = proxy.robot_tensors['body'].reshape(env.num_envs, -1, 13)[:, env.ee_idx].clone()
    from isaacgym.torch_utils import quat_apply
    env.end_effector_state[:] = s
    env.end_effector_state[:, :3] += quat_apply(s[:, 3:7], env.ee_local_offset.expand(env.num_envs, -1))
    # Native first policy call follows termination/pose bookkeeping in the normal step.
    # Nominal reset has not stepped an actor yet; populate that same actual-pose state.
    env.check_termination()
    env.compute_observations()
    env.explicit_reset = False


def reference_fixtures(root, config, budget):
    import isaacgym
    import torch
    import numpy as np
    from go2arx5_vla.target_sim.environment import build_native
    from go2arx5_vla.target_sim.bridge import ExternalEEFReferenceBridge
    from go2arx5_vla.contracts.trajectory import TrajectoryChunkV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, geodesic_angle
    from go2arx5_vla.target_sim.limits import hard_limit_report
    # A completed valid fixture is never rerun. Original safety is checked per fixture.
    directory = root / 'evidence' / 'reference-fixtures'; directory.mkdir(exist_ok=True)
    names = ('hold', 'horizontal_5cm', 'vertical_5cm', 'pure_rotation_0p10rad',
             'vertical_return_5cm', 'hotswap_0p4s')
    results = []
    wrapped, proxy, cfg = build_native(config, budget, 1)
    import modules.trajectory as native
    from scripts.load_policy import load_arm_policy, load_dog_policy
    env = wrapped.env
    hard_properties = proxy.gym.get_actor_dof_properties(proxy.rows[0]['env'], proxy.rows[0]['robot'])
    arm = load_arm_policy(config['controller_run'], config['checkpoint_id'], cfg)
    dog = load_dog_policy(config['controller_run'], config['checkpoint_id'], cfg)
    try:
        nominal_reset(wrapped, proxy, budget, cfg)
        from go2arx5_vla.target_sim.environment import independent_fk
        state = env.root_states[0].detach().cpu().numpy()
        q = dict(zip(env.dof_names, env.dof_pos[0].detach().cpu().numpy()))
        fk = independent_fk(cfg.asset.file.replace('{MINI_GYM_ROOT_DIR}', config['wbc_root']),
                            pose(state), q, cfg.arm.ik.ee_local_pos)
        measured = pose(env.end_effector_state[0, :7].detach().cpu().numpy())
        fk_error = dict(XYZ_m=float(np.linalg.norm(fk[:3, 3] - measured[:3, 3])),
                        SO3_rad=float(geodesic_angle(fk[:3, :3], measured[:3, :3])))
        images, camera = proxy.rgb(0, measured, pose(state), float(proxy.gym.get_sim_time(env.sim)))
        checks = dict(independent_FK_XYZ=fk_error['XYZ_m'] <= 1e-5,
                      independent_FK_SO3=fk_error['SO3_rad'] <= 1e-4,
                      both_RGB_real=all(v.shape == (96, 96, 3) and v.std() > 1 for v in images),
                      actual_free_base=not cfg.asset.fix_base_link,
                      actual_complete_bound_controller=True)
        prior = [json.loads((root/'evidence'/('scene-smoke-n'+str(n))/'RESULT.json').read_text()) for n in (1,2)]
        checks['index_reset_object_gripper_prior_physics'] = all(all(v['checks'].values()) for v in prior)
        G1 = dict(status='PASS' if all(checks.values()) else 'PHYSICAL_ENV_NOT_READY',
                  evidence_kind='REAL_TARGET_PHYSICS', checks=checks, FK_error=fk_error,
                  FK_expected=fk.tolist(), FK_measured=measured.tolist(), camera=camera,
                  supporting_pinned_diagnostics_not_task_success=True,
                  controller_class=config['controller_class'])
        (root/'evidence'/'G1.json').write_text(json.dumps(G1,indent=2)+'\n')
        np.savez_compressed(root/'evidence'/'G1_ACTUAL_RGB.npz', frames=np.stack(images))
        if G1['status'] != 'PASS': return dict(status='PHYSICAL_ENV_NOT_READY', G1=G1)
        for name in names:
            if (directory / (name + '.json')).exists():
                results.append(json.loads((directory / (name + '.json')).read_text())); continue
            nominal_reset(wrapped, proxy, budget, cfg)
            origin = env.end_effector_state[0, :7].detach().cpu().numpy().copy()
            observed_origin = pose(origin)
            bridge = ExternalEEFReferenceBridge(env, native, tcp_semantic_id='go2_x5:gripper_center')
            times = np.arange(201) * .05
            positions = np.repeat(origin[None, :3], len(times), 0)
            rotations = np.repeat(observed_origin[None, :3, :3], len(times), 0)
            smooth = np.clip(times / 4., 0, 1); smooth = smooth * smooth * (3 - 2 * smooth)
            if name == 'horizontal_5cm': positions[:, 0] += .05 * smooth
            elif name == 'vertical_5cm': positions[:, 2] += .05 * smooth
            elif name == 'vertical_return_5cm': positions[:, 2] += .05 * np.sin(np.pi * np.clip(times / 8., 0, 1))**2
            elif name == 'pure_rotation_0p10rad':
                for j, angle in enumerate(.10 * smooth):
                    c, s = np.cos(angle), np.sin(angle)
                    rotations[j] = observed_origin[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            start = float(proxy.gym.get_sim_time(env.sim))
            def install(index=0):
                ids = np.arange(index, len(times))
                now = float(proxy.gym.get_sim_time(env.sim))
                ts = now + times[ids] - times[index]
                if len(ts) < 2: return
                chunk = TrajectoryChunkV1(chunk_id=name + ':' + str(index), source='PI0287_FIXED_FIXTURE',
                    observation_timestamp=float(ts[0]), point_timestamps=ts, eef_position=positions[ids],
                    eef_orientation=matrix_to_quat(rotations[ids]), source_frame='world',
                    tcp_semantic_id='go2_x5:gripper_center', validity_start=float(ts[0]),
                    validity_end=float(ts[-1]), gripper=np.ones(len(ts)))
                v = bridge.submit(0, chunk, issue_timestamp=float(ts[0]), install_timestamp=float(ts[0]))
                if v['status'] != 'INSTALLED': raise RuntimeError('Reference rejected ' + str(v))
            install()
            trace, frames, camera_trace = [], [], []
            failure = None
            try:
                for tick in range(int(round(10 / env.dt))):
                    now = float(proxy.gym.get_sim_time(env.sim)); relative = now - start
                    if name == 'hotswap_0p4s' and tick and tick % 20 == 0: install(int(round(relative / .05)))
                    bridge.boundary(now)
                    env.independent_gripper_fraction = bridge.gripper_at(0, now)
                    with torch.no_grad():
                        budget.charge(WBC_actor_forwards=1)
                        action_arm = arm(wrapped.get_arm_observations()).to(env.device)
                        if env.num_plan_actions: wrapped.plan(action_arm)
                        budget.charge(WBC_actor_forwards=1)
                        action_dog = dog(wrapped.get_dog_observations()).to(env.device)
                    wrapped.step(action_dog, action_arm[..., :env.num_actions_arm])
                    actual = env.end_effector_state[0, :7].detach().cpu().numpy().copy()
                    native_goal = np.r_[env.arm_goal_pos_world[0].detach().cpu().numpy(), env.arm_goal_quat_world[0].detach().cpu().numpy()]
                    hard_state = hard_limit_report(env.dof_names, env.dof_pos[0].detach().cpu().numpy(), hard_properties)
                    hard = hard_state['hard_limit']
                    trace.append(dict(time_s=float(proxy.gym.get_sim_time(env.sim)) - start,
                                      actual_tcp=actual.tolist(), installed_tcp=native_goal.tolist(), hard_limit=hard,
                                      actual_DOF_hard_limit_state=hard_state,
                                      base=env.root_states[0].detach().cpu().numpy().tolist()))
                    if tick % 5 == 0:
                        images, cams = proxy.rgb(0, pose(actual), pose(trace[-1]['base']),
                                                 float(proxy.gym.get_sim_time(env.sim)))
                        frames.append(np.concatenate(images, axis=1)); camera_trace.append(cams)
                    if hard: raise RuntimeError('ACTUAL_HARD_LIMIT_PENETRATION')
            except RuntimeError as e:
                failure = str(e)
            late = [row for row in trace if row['time_s'] >= 8. - 1e-6]
            if late:
                xyz = [np.linalg.norm(np.asarray(v['actual_tcp'][:3]) - v['installed_tcp'][:3]) for v in late]
                so3 = [float(geodesic_angle(pose(v['actual_tcp'])[:3, :3], pose(v['installed_tcp'])[:3, :3])) for v in late]
                metrics = dict(position_RMS_m=float(np.sqrt(np.mean(np.square(xyz)))), position_max_m=float(max(xyz)),
                               SO3_RMS_rad=float(np.sqrt(np.mean(np.square(so3)))), SO3_max_rad=float(max(so3)))
            else: metrics = None
            passed = (failure is None and len(trace) == 500 and metrics is not None
                      and metrics['position_RMS_m'] <= .015 and metrics['position_max_m'] <= .030
                      and metrics['SO3_RMS_rad'] <= .15 and metrics['SO3_max_rad'] <= .30)
            result = dict(name=name, status='PASS' if passed else 'VALID_REFERENCE_QUALITY_FAIL',
                          evidence_kind='REAL_TARGET_PHYSICS', controller_class=config['controller_class'],
                          failure=failure, metrics=metrics, ticks=len(trace), native_safety_preserved=True,
                          raw_position=positions.tolist(), raw_rotation=rotations.tolist(), trace=trace,
                          bridge_receipts=bridge.receipts, camera_trace=camera_trace)
            (directory / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
            if frames: np.savez_compressed(directory / (name + '-RGB.npz'), frames=np.stack(frames))
            results.append(result)
            # The six frozen diagnostic conditions are distinct, never quality retries.
            # Their aggregate verdict gates every subsequent teacher/data operation.
    finally: env.close()
    return dict(status='PASS' if len(results) == 6 and all(v['status'] == 'PASS' for v in results)
                else 'REFERENCE_QUALITY_FAIL', evidence_kind='REAL_TARGET_PHYSICS',
                fixtures=results, not_run=[n for n in names if n not in [v['name'] for v in results]],
                GR00T_forwards=0, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('scene-smoke', 'reference-fixtures', 'teacher-pilot', 'collect'), required=True)
    parser.add_argument('--config', required=True); parser.add_argument('--output-root', required=True)
    parser.add_argument('--seed', type=int, default=287001)
    args = parser.parse_args(); root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'SIM_LEASE.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from go2arx5_vla.target_sim.budget import Budget
        config = json.loads(Path(args.config).read_text())
        budget = Budget(root / 'evidence')
        starts = root / 'runtime' / 'LAUNCHES.jsonl'
        starts.parent.mkdir(parents=True, exist_ok=True)
        count = len(starts.read_text().splitlines()) if starts.exists() else 0
        if count >= 3: raise RuntimeError('BUDGET_STOP sequential_launch_attempts')
        start = time.time()
        with starts.open('a') as f: f.write(json.dumps(dict(mode=args.mode, pid=__import__('os').getpid(), started_epoch=start, seed=args.seed)) + '\n')
        try:
            if args.mode == 'scene-smoke': result = scene_smoke(root, config, budget)
            elif args.mode == 'reference-fixtures': result = reference_fixtures(root, config, budget)
            else: raise RuntimeError('ACTUAL_GATE_INCOMPLETE: G2/G3 must PASS before task execution')
        except Exception as e:
            result = dict(status='TECHNICAL_BLOCKED', error=repr(e))
            raise
        finally:
            cpu = resource.getrusage(resource.RUSAGE_SELF)
            evidence = dict(locals().get('result', {}), totals=budget.totals,
                            elapsed_service_s=time.time() - start, actual_self_CPU_s=cpu.ru_utime + cpu.ru_stime,
                            selected='NONE', deployment='NONE', GR00T_forwards=0, optimizer_updates=0)
            (root / 'evidence' / ('RUN_' + str(count + 1) + '.json')).write_text(json.dumps(evidence, indent=2) + '\n')
            print(json.dumps({k: evidence[k] for k in ('status', 'totals', 'elapsed_service_s')}), flush=True)


if __name__ == '__main__': main()
