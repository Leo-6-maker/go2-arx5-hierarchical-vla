"""Original multi-actor adapter for the delivered, unchanged private Isaac Gym WBC."""
import numpy as np


def independent_fk(urdf, root_world, joint_positions, tcp_offset):
    """URDF serial transforms, independent of native body gather and TCP hook."""
    import xml.etree.ElementTree as ET
    document = ET.parse(urdf).getroot()
    joints = {j.find('child').get('link'): j for j in document.findall('joint')}
    chain, link = [], 'x5_link6'
    while link in joints:
        joint = joints[link]; chain.insert(0, joint); link = joint.find('parent').get('link')
    transform = np.asarray(root_world, dtype=float).copy()
    for joint in chain:
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        roll, pitch, yaw = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        c, s = np.cos, np.sin
        rx = np.array([[1, 0, 0], [0, c(roll), -s(roll)], [0, s(roll), c(roll)]])
        ry = np.array([[c(pitch), 0, s(pitch)], [0, 1, 0], [-s(pitch), 0, c(pitch)]])
        rz = np.array([[c(yaw), -s(yaw), 0], [s(yaw), c(yaw), 0], [0, 0, 1]])
        step = np.eye(4); step[:3, :3] = rz @ ry @ rx; step[:3, 3] = xyz
        if joint.get('type') not in ('fixed', 'floating'):
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            value = float(joint_positions[joint.get('name')])
            motion = np.eye(4)
            if joint.get('type') == 'prismatic': motion[:3, 3] = axis * value
            else:
                cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                motion[:3, :3] = np.eye(3) + s(value) * cross + (1 - c(value)) * (cross @ cross)
            step = step @ motion
        transform = transform @ step
    transform[:3, 3] += transform[:3, :3] @ np.asarray(tcp_offset)
    return transform


SCENE = dict(task_id='GO2_PLACE_V1', cube_edge_m=0.04, cube_mass_kg=0.064,
             table_top_m=0.42, table_center_xy=[0.65, 0.0], table_size_m=[0.50, 0.60, 0.05],
             source_xyz=[0.50, -0.09, 0.44], goal_xy=[0.50, 0.09], blue_zone_side_m=0.16,
             object_friction=0.8, table_friction=0.8, restitution=0.0,
             geometry_basis='Go2-X5 nominal FK TCP [0.2901,0,0.5966057]; table front x=.40 beyond base',
             geometry_selected_before_teacher=True, camera_resolution=[96, 96])


class MultiActorGym:
    """Stable robot-only caches for native views, explicit refresh and sim-index writes.

    Private native code keeps its observation/action layout. All simulator tensors
    remain global and are gathered/scattered at this single API boundary.
    """
    def __init__(self, gym, budget, pinned_diagnostic=False):
        from isaacgym import gymapi, gymtorch
        import torch
        self.gym, self.api, self.gymtorch, self.torch = gym, gymapi, gymtorch, torch
        self.budget = budget
        self.pinned_diagnostic = pinned_diagnostic
        self.rows, self.global_tensors, self.robot_tensors = [], {}, {}
        self.maps = {}
        self.assets = None
        self.sim = None

    def __getattr__(self, name):
        return getattr(self.gym, name)

    def create_sim(self, *args):
        self.sim = self.gym.create_sim(*args)
        return self.sim

    def load_asset(self, sim, root, name, options):
        if self.pinned_diagnostic and 'go2_x5' in name:
            options.fix_base_link = True
        return self.gym.load_asset(sim, root, name, options)

    def create_actor(self, env, asset, pose, name, group, collision_filter, segmentation=0):
        robot = self.gym.create_actor(env, asset, pose, name, group, collision_filter, segmentation)
        if name != 'anymal':
            return robot
        a = self.api
        if self.assets is None:
            fixed = a.AssetOptions(); fixed.fix_base_link = True
            cube = a.AssetOptions(); cube.density = 1000.0
            cube.thickness = 0.001
            self.assets = (self.gym.create_box(self.sim, *SCENE['table_size_m'], fixed),
                           self.gym.create_box(self.sim, 0.04, 0.04, 0.04, cube),
                           self.gym.create_box(self.sim, 0.16, 0.16, 0.001, fixed))
        # Native plane layout gives each environment its declared world XY offset.
        offset = np.array([pose.p.x, pose.p.y, 0.0])
        actors = []
        positions = ([0.65, 0, 0.395], SCENE['source_xyz'], [0.50, 0.09, 0.4206])
        colors = ([0.55, 0.55, 0.55], [1, 0.05, 0.05], [0.05, 0.10, 1])
        for kind, asset_, xyz, color in zip(('table', 'object', 'zone'), self.assets, positions, colors):
            tf = a.Transform(); tf.p = a.Vec3(*(np.asarray(xyz) + offset))
            # The zone is a visual marking: its shape does not obstruct placement.
            actor = self.gym.create_actor(env, asset_, tf, kind, group + 10000 if kind == 'zone' else group,
                                          0xFFFF if kind == 'zone' else 0, {'table': 20, 'object': 21, 'zone': 22}[kind])
            self.gym.set_rigid_body_color(env, actor, 0, a.MESH_VISUAL_AND_COLLISION, a.Vec3(*color))
            props = self.gym.get_actor_rigid_shape_properties(env, actor)
            for p in props:
                p.friction = SCENE['object_friction'] if kind == 'object' else SCENE['table_friction']
                p.restitution = 0.0
            self.gym.set_actor_rigid_shape_properties(env, actor, props)
            actors.append(actor)
        cameras = []
        for _ in range(2):
            props = a.CameraProperties(); props.width = props.height = 96
            props.horizontal_fov = 80.0
            cameras.append(self.gym.create_camera_sensor(env, props))
        body_names = self.gym.get_actor_rigid_body_names(env, robot)
        dof_names = self.gym.get_actor_dof_names(env, robot)
        self.rows.append(dict(env=env, robot=robot, table=actors[0], object=actors[1], zone=actors[2],
                              offset=offset, cameras=cameras, body_names=body_names, dof_names=dof_names))
        return robot

    def prepare_sim(self, sim):
        result = self.gym.prepare_sim(sim)
        a, t = self.api, self.torch
        self.actor_map = {k: [self.gym.get_actor_index(r['env'], r[k], a.DOMAIN_SIM) for r in self.rows]
                          for k in ('robot', 'table', 'object', 'zone')}
        self.body_map = [[self.gym.get_actor_rigid_body_index(r['env'], r['robot'], i, a.DOMAIN_SIM)
                          for i in range(len(r['body_names']))] for r in self.rows]
        self.dof_map = [[self.gym.get_actor_dof_index(r['env'], r['robot'], i, a.DOMAIN_SIM)
                         for i in range(len(r['dof_names']))] for r in self.rows]
        self.maps = dict(root=self.actor_map['robot'], body=np.asarray(self.body_map).reshape(-1).tolist(),
                         contact=np.asarray(self.body_map).reshape(-1).tolist(),
                         dof=np.asarray(self.dof_map).reshape(-1).tolist())
        from .indexing import TargetActorIndexMap
        bodies, dofs = self.rows[0]['body_names'], self.rows[0]['dof_names']
        self.index_map = TargetActorIndexMap(
            {k: v for k, v in self.actor_map.items() if k in ('robot', 'table', 'object')},
            self.body_map, self.dof_map, bodies, dofs, [k for k in bodies if 'foot' in k], 'x5_link6',
            dofs[:12], dofs[12:18], dofs[18:])
        return result

    def _acquire(self, kind, method, sim):
        if kind not in self.global_tensors:
            descriptor = getattr(self.gym, method)(sim)
            full = self.gymtorch.wrap_tensor(descriptor)
            self.global_tensors[kind] = full
            idx = self.torch.tensor(self.maps[kind], device=full.device, dtype=self.torch.long)
            self.robot_tensors[kind] = full.index_select(0, idx).contiguous()
        return self.gymtorch.unwrap_tensor(self.robot_tensors[kind])

    def _refresh(self, kind, method, sim):
        result = getattr(self.gym, method)(sim)
        if kind in self.global_tensors:
            full = self.global_tensors[kind]
            idx = self.torch.tensor(self.maps[kind], device=full.device, dtype=self.torch.long)
            self.robot_tensors[kind].copy_(full.index_select(0, idx))
        return result

    def acquire_actor_root_state_tensor(self, sim): return self._acquire('root', 'acquire_actor_root_state_tensor', sim)
    def acquire_dof_state_tensor(self, sim): return self._acquire('dof', 'acquire_dof_state_tensor', sim)
    def acquire_rigid_body_state_tensor(self, sim): return self._acquire('body', 'acquire_rigid_body_state_tensor', sim)
    def acquire_net_contact_force_tensor(self, sim): return self._acquire('contact', 'acquire_net_contact_force_tensor', sim)
    def refresh_actor_root_state_tensor(self, sim): return self._refresh('root', 'refresh_actor_root_state_tensor', sim)
    def refresh_dof_state_tensor(self, sim): return self._refresh('dof', 'refresh_dof_state_tensor', sim)
    def refresh_rigid_body_state_tensor(self, sim): return self._refresh('body', 'refresh_rigid_body_state_tensor', sim)
    def refresh_net_contact_force_tensor(self, sim): return self._refresh('contact', 'refresh_net_contact_force_tensor', sim)

    def _set(self, kind, method, sim, descriptor, env_descriptor=None, count=None):
        t = self.torch
        cache = self.gymtorch.wrap_tensor(descriptor).reshape(self.robot_tensors[kind].shape)
        full = self.global_tensors[kind]
        if env_descriptor is None:
            envs = t.arange(len(self.rows), device=full.device)
        else:
            envs = self.gymtorch.wrap_tensor(env_descriptor).long()
        mapping = t.tensor(self.maps[kind], device=full.device, dtype=t.long)
        if kind == 'dof':
            mapping = mapping.reshape(len(self.rows), -1)[envs].reshape(-1)
            values = cache.reshape(len(self.rows), -1, 2)[envs].reshape(-1, 2)
        else:
            mapping, values = mapping[envs], cache[envs]
        full.index_copy_(0, mapping, values)
        actors = t.tensor(self.actor_map['robot'], device=full.device, dtype=t.int32)[envs].contiguous()
        # All root writes use indexed writes, preserving table/object/other environments.
        return getattr(self.gym, method)(sim, self.gymtorch.unwrap_tensor(full),
                                         self.gymtorch.unwrap_tensor(actors), len(envs))

    def set_actor_root_state_tensor_indexed(self, sim, v, ids, n):
        return self._set('root', 'set_actor_root_state_tensor_indexed', sim, v, ids, n)
    def set_actor_root_state_tensor(self, sim, v):
        return self._set('root', 'set_actor_root_state_tensor_indexed', sim, v)
    def set_dof_state_tensor_indexed(self, sim, v, ids, n):
        return self._set('dof', 'set_dof_state_tensor_indexed', sim, v, ids, n)
    def set_dof_state_tensor(self, sim, v):
        return self._set('dof', 'set_dof_state_tensor_indexed', sim, v)

    def apply_rigid_body_force_at_pos_tensors(self, sim, force, positions, frame):
        t = self.torch
        full = self.global_tensors['body']
        idx = t.tensor(self.maps['body'], device=full.device)
        f = t.zeros_like(full[:, :3]); p = full[:, :3].clone()
        f.index_copy_(0, idx, self.gymtorch.wrap_tensor(force).reshape(-1, 3))
        p.index_copy_(0, idx, self.gymtorch.wrap_tensor(positions).reshape(-1, 3))
        return self.gym.apply_rigid_body_force_at_pos_tensors(sim, self.gymtorch.unwrap_tensor(f),
                                                              self.gymtorch.unwrap_tensor(p), frame)

    def simulate(self, sim):
        self.budget.charge(physics_integrations=len(self.rows))
        return self.gym.simulate(sim)

    def rgb(self, env_id, tcp_world, base_world, timestamp):
        r, a = self.rows[env_id], self.api
        mounts = [(np.array([.18, -.10, .20]), np.array([.65, 0, .08])),
                  (np.array([0, .05, .08]), np.array([.15, 0, -.10]))]
        for camera, T, (eye, target) in zip(r['cameras'], (base_world, tcp_world), mounts):
            eye = T[:3, :3] @ eye + T[:3, 3]; target = T[:3, :3] @ target + T[:3, 3]
            self.gym.set_camera_location(camera, r['env'], a.Vec3(*eye), a.Vec3(*target))
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        images, matrices = [], []
        for camera in r['cameras']:
            images.append(np.asarray(self.gym.get_camera_image(self.sim, r['env'], camera, a.IMAGE_COLOR))
                          .reshape(96, 96, 4)[..., :3].copy())
            matrices.append(dict(view=np.asarray(self.gym.get_camera_view_matrix(self.sim, r['env'], camera)).tolist(),
                                 projection=np.asarray(self.gym.get_camera_proj_matrix(self.sim, r['env'], camera)).tolist()))
        self.budget.charge(RGB_pairs=1)
        return images, dict(timestamp_s=timestamp, valid=[True, True], matrices=matrices)


def build_native(config, budget, n_envs, pinned_diagnostic=False):
    import pickle
    from pathlib import Path
    import sys
    from isaacgym import gymapi
    sys.path.insert(0, config['wbc_root'])
    sys.path.insert(0, str(Path(__file__).resolve().parents[3].parent / 'scripts/pi0287_target_sim'))
    from doctor import inference_package
    inference_package(config['wbc_root'])
    from go1_gym.envs.config import build_roboduet_config, RoboDuetRuntimeOptions, apply_config_snapshot, recompute_observation_dims
    from go1_gym.envs.roboduet.wbc_env import WBCEnv
    from go1_gym.envs.roboduet.wbc_env_wrapper import HistoryWrapper
    from go1_gym.utils.global_switch import global_switch
    snapshot = pickle.loads((Path(config['controller_run']) / 'parameters.pkl').read_bytes())['Cfg']
    cfg = build_roboduet_config(options=RoboDuetRuntimeOptions(num_envs=n_envs, robot='go2_x5'))
    apply_config_snapshot(cfg, snapshot, drop_unknown=True)
    recompute_observation_dims(cfg)
    recorded = snapshot['arm']['arm_num_observations']
    if recorded == cfg.arm.arm_num_observations + 16:
        cfg.arm.checkpoint_observation_layout = 'goal_reaching_extended_v1'
        cfg.arm.arm_num_observations = recorded
        cfg.arm.arm_num_obs_history = recorded * cfg.arm.arm_num_observation_history
    cfg.env.num_envs = n_envs
    cfg.env.record_video = False
    cfg.env.num_recording_envs = 0
    cfg.terrain.mesh_type = 'plane'
    cfg.terrain.teleport_robots = False
    cfg.asset.render_sphere = False
    # Evaluation disables input randomization; gains, dynamics and native safety remain unchanged.
    for name in vars(cfg.domain_rand):
        if name.startswith('randomize_') or name == 'push_robots':
            if isinstance(getattr(cfg.domain_rand, name), bool): setattr(cfg.domain_rand, name, False)
    global_switch.open_switch()
    class ExternalWBC(WBCEnv):
        external_task_reference = True
        explicit_reset = True
        independent_gripper_fraction = 1.0

        def _compute_torques(self, actions):
            target = super()._compute_torques(actions)
            for name in ('x5_gripper_joint', 'x5_joint8'):
                target[:, self.dof_names.index(name)] = float(self.independent_gripper_fraction) * .044
            return target

        def _traj_tracking_init_hook(self):
            # Allocate the delivered native buffers without generating unused bank trajectories.
            import modules.curriculum as curriculum
            bank = curriculum.TrajectoryBank
            curriculum.TrajectoryBank = lambda *a, **k: None
            try:
                super()._traj_tracking_init_hook()
            finally:
                curriculum.TrajectoryBank = bank

        def _reset_trajectories(self, env_ids):
            import torch
            from modules.trajectory import to_canonical
            if not hasattr(self, 'traj_batch'): return
            for i in env_ids.tolist():
                state = self.end_effector_state[i].detach().cpu().numpy()
                from go2arx5_vla.trajectory.so3 import quat_to_matrix
                gamma, law = to_canonical(np.repeat(state[None, :3], 2, 0),
                                          np.repeat(quat_to_matrix(state[3:7])[None], 2, 0), .05)
                self.traj_batch.load(torch.tensor([i], device=self.device), [gamma], [law])
                for key in ('traj_s', 'traj_s_prev', 'traj_sim_time', 'traj_d_lat',
                            'traj_sdot_meas', 'traj_sdot_ref', 'traj_timing_err', 'traj_twist_err'):
                    getattr(self, key)[i] = 0

        def reset_idx(self, ids):
            if not self.explicit_reset and len(ids):
                # Observe the original safety result before native auto-reset erases it.
                raise RuntimeError('NATIVE_ABORT ' + str(ids.detach().cpu().tolist()))
            return super().reset_idx(ids)
    proxy = MultiActorGym(gymapi.acquire_gym(), budget, pinned_diagnostic)
    acquire = gymapi.acquire_gym
    gymapi.acquire_gym = lambda: proxy
    try:
        env = ExternalWBC(sim_device='cuda:0', headless=True, cfg=cfg)
    finally:
        gymapi.acquire_gym = acquire
    return HistoryWrapper(env), proxy, cfg
