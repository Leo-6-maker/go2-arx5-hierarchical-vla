"""Isolated PI0275 finite-effort CPU-PhysX environment. No learner or WBC."""
from isaacgym import gymapi
import numpy as np
from scipy.spatial.transform import Rotation
from contract import DT, REPO, efforts, measured_opening, obb_depth


def vec(v):
    return gymapi.Vec3(*map(float, v))


def xyz(v):
    return np.array([v['x'], v['y'], v['z']], float)


class Environment:
    def __init__(self, family, freefall=False):
        self.gym = gymapi.acquire_gym()
        p = gymapi.SimParams(); p.dt = DT; p.substeps = 2
        p.up_axis = gymapi.UP_AXIS_Z; p.gravity = vec([0, 0, -9.81])
        p.use_gpu_pipeline = False; p.physx.use_gpu = False
        p.physx.solver_type = 1; p.physx.num_position_iterations = 8; p.physx.num_velocity_iterations = 2
        # Pair records represent the last substep, not an invented sum of substep impulses.
        p.physx.contact_collection = gymapi.ContactCollection.CC_LAST_SUBSTEP
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, p)
        if self.sim is None:
            raise RuntimeError('NATIVE_SIM_CREATION_FAILED')
        self.env = self.gym.create_env(self.sim, vec([-2, -2, 0]), vec([2, 2, 2]), 1)
        self.family = family; self.tick = 0; self.bodies = {}; self.boxes = {}; self.actors = {}
        self.setter_attempts = []
        self._box('table', [.6, .6, .025], [0, 0, .625], static=True, color=[.55, .55, .55])
        for side, tray in family['trays'].items():
            x, y = tray['interior_center_xy_m']; w = .008; h = .04
            for edge, half, pos in [
                ('left', [w / 2, .08 + w, h / 2], [x - .08 - w / 2, y, .65 + h / 2]),
                ('right', [w / 2, .08 + w, h / 2], [x + .08 + w / 2, y, .65 + h / 2]),
                ('front', [.08, w / 2, h / 2], [x, y - .08 - w / 2, .65 + h / 2]),
                ('back', [.08, w / 2, h / 2], [x, y + .08 + w / 2, .65 + h / 2])]:
                self._box('tray_' + side + '_' + edge, half, pos, static=True, color=[.65, .65, .65])
        for color in ('red', 'blue'):
            o = family['objects'][color]; pos = np.array(o['position_m'])
            if freefall and color == 'red':
                pos = pos + [0, 0, .10]
            self._box(color, np.array(o['dimensions_m']) / 2, pos, mass=o['mass_kg'],
                      color=[.9, .1, .1] if color == 'red' else [.1, .1, .9])
        options = gymapi.AssetOptions(); options.fix_base_link = True; options.collapse_fixed_joints = False
        options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT; options.disable_gravity = False
        asset = self.gym.load_asset(self.sim, str(REPO / 'assets/object_transfer'), 'draft_tool.urdf', options)
        if asset is None:
            raise RuntimeError('NATIVE_TOOL_ASSET_LOAD_FAILED')
        self.tool = self.gym.create_actor(self.env, asset, gymapi.Transform(), 'tool', 0, 0, 20)
        self.actors['tool'] = self.tool; self._shapes(self.tool)
        names = self.gym.get_actor_rigid_body_names(self.env, self.tool)
        dofs = self.gym.get_actor_dof_names(self.env, self.tool)
        expected = ['tool_x', 'tool_y', 'tool_z', 'tool_yaw', 'tool_pitch', 'tool_roll', 'finger_left', 'finger_right']
        if len(names) != 9 or dofs != expected:
            raise RuntimeError('NATIVE_TOOL_INVENTORY_MISMATCH: ' + repr((names, dofs)))
        self.tool_names = names; self.dof_names = dofs
        for i, name in enumerate(names):
            self.bodies[self.gym.get_actor_rigid_body_index(self.env, self.tool, i, gymapi.DOMAIN_ENV)] = ('tool', name)
        for name, half, center in [('palm_tcp', [.03, .048, .01], [0, 0, -.01]),
                                   ('finger_left', [.01, .004, .02], [0, 0, 0]),
                                   ('finger_right', [.01, .004, .02], [0, 0, 0])]:
            bid = self.gym.find_actor_rigid_body_index(self.env, self.tool, name, gymapi.DOMAIN_ENV)
            self.boxes[bid] = (np.array(half), np.array(center))
        self.props = self.gym.get_actor_dof_properties(self.env, self.tool)
        self.props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
        self.props['stiffness'].fill(0); self.props['damping'].fill(0)
        self.gym.set_actor_dof_properties(self.env, self.tool, self.props)
        s = self.gym.get_actor_dof_states(self.env, self.tool, gymapi.STATE_ALL)
        s['pos'] = [*family['initial_tcp_position_m'], 0, 0, -np.pi, .04, .04]
        s['vel'].fill(0)
        self.gym.set_actor_dof_states(self.env, self.tool, s, gymapi.STATE_ALL)
        self.masses = self.gym.get_actor_rigid_body_properties(self.env, self.tool)
        self.palm_index = names.index('palm_tcp')
        self.cameras = {}
        for key, fov in [('body', 55), ('wrist', 65)]:
            cp = gymapi.CameraProperties(); cp.width = cp.height = 96; cp.horizontal_fov = fov
            self.cameras[key] = self.gym.create_camera_sensor(self.env, cp)
        self.gym.prepare_sim(self.sim)
        self.contact_dtype = None
        self.raw_gym = self.gym
        self.gym = SetterGuard(self.raw_gym, self)

    def _shapes(self, actor):
        props = self.gym.get_actor_rigid_shape_properties(self.env, actor)
        for p in props:
            p.friction = .8; p.restitution = 0.; p.contact_offset = .001; p.rest_offset = 0.
        self.gym.set_actor_rigid_shape_properties(self.env, actor, props)
        actual = self.gym.get_actor_rigid_shape_properties(self.env, actor)
        if any(abs(p.friction - .8) > 1e-6 or abs(p.contact_offset - .001) > 1e-7 or abs(p.rest_offset) > 1e-7 for p in actual):
            raise RuntimeError('NATIVE_SHAPE_PROPERTIES_MISMATCH')

    def _box(self, name, half, position, static=False, mass=None, color=None):
        options = gymapi.AssetOptions(); options.fix_base_link = static; options.disable_gravity = static
        asset = self.gym.create_box(self.sim, *map(float, 2 * np.array(half)), options)
        pose = gymapi.Transform(); pose.p = vec(position)
        actor = self.gym.create_actor(self.env, asset, pose, name, 0, 0, len(self.actors) + 1)
        self.actors[name] = actor; self._shapes(actor)
        if mass is not None:
            props = self.gym.get_actor_rigid_body_properties(self.env, actor)
            props[0].mass = mass
            self.gym.set_actor_rigid_body_properties(self.env, actor, props, True)
        if color:
            self.gym.set_rigid_body_color(self.env, actor, 0, gymapi.MESH_VISUAL, vec(color))
        bid = self.gym.get_actor_rigid_body_index(self.env, actor, 0, gymapi.DOMAIN_ENV)
        self.bodies[bid] = (name, name); self.boxes[bid] = (np.asarray(half), np.zeros(3))

    def state(self):
        dof = self.gym.get_actor_dof_states(self.env, self.tool, gymapi.STATE_ALL)
        states = self.gym.get_env_rigid_body_states(self.env, gymapi.STATE_ALL).copy()
        ids = {v: k for k, v in self.bodies.items()}; p = states[ids[('tool', 'palm_tcp')]]
        def pose(s):
            q = s['pose']['r']; return xyz(s['pose']['p']), Rotation.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
        tcp, rotation = pose(p)
        lp, _ = pose(states[ids[('tool', 'finger_left')]]); rp, _ = pose(states[ids[('tool', 'finger_right')]])
        width = float(np.dot(lp - rp, rotation[:, 1]) - .008)
        measured_opening(width)
        if np.max(abs(tcp - dof['pos'][:3])) > 1e-5:
            raise RuntimeError('NATIVE_TCP_FK_MISMATCH')
        contacts = self.gym.get_env_rigid_contacts(self.env).copy()
        if self.tick and not {'body0', 'body1', 'lambda', 'normal'}.issubset(contacts.dtype.names or ()):
            raise RuntimeError('NATIVE_EVIDENCE_INCOMPLETE: ' + str(contacts.dtype))
        self.contact_dtype = repr(contacts.dtype.descr)
        positions, rotations, velocities = [], [], []
        for color in ('red', 'blue'):
            s = states[ids[(color, color)]]; op, rr = pose(s)
            positions.append(op); rotations.append(rr); velocities.append(xyz(s['vel']['linear']))
        return dict(time=self.tick * DT, q=dof['pos'].astype(float), qd=dof['vel'].astype(float), p=tcp, r=rotation,
                    v=xyz(p['vel']['linear']), omega=xyz(p['vel']['angular']), width=width,
                    object_position=np.array(positions), object_rotation=np.array(rotations), object_velocity=np.array(velocities),
                    states=states, contacts=contacts)

    def contact_flags(self, state, target):
        touching = set(); tool_contact = False; dangerous = False; depth = 0.; supported = False
        for c in state['contacts']:
            a, b = int(c['body0']), int(c['body1'])
            if a not in self.bodies or b not in self.bodies:
                raise RuntimeError('NATIVE_UNMAPPED_BODY_PAIR')
            aa, bb = self.bodies[a], self.bodies[b]
            if float(c['lambda']) >= 1e-3:
                pair = {aa, bb}
                if (aa[0] == 'tool' and bb[0] == target) or (bb[0] == 'tool' and aa[0] == target):
                    tool_contact = True
                for finger in ('finger_left', 'finger_right'):
                    if pair == {('tool', finger), (target, target)}:
                        touching.add(finger)
                if pair == {('table', 'table'), (target, target)}:
                    supported = True
                tool = aa[0] == 'tool' or bb[0] == 'tool'
                internal = aa[0] == bb[0] == 'tool'
                grasp = (aa[0] == 'tool' and aa[1].startswith('finger') and bb[0] == target) or (bb[0] == 'tool' and bb[1].startswith('finger') and aa[0] == target)
                if tool and not internal and not grasp:
                    dangerous = True
                if {aa[0], bb[0]} == {'red', 'blue'}:
                    dangerous = True
            if a in self.boxes and b in self.boxes and not (aa[0] == bb[0] == 'tool'):
                def box(bid):
                    s = state['states'][bid]; qq = s['pose']['r']
                    rr = Rotation.from_quat([qq['x'], qq['y'], qq['z'], qq['w']]).as_matrix()
                    half, offset = self.boxes[bid]; return xyz(s['pose']['p']) + rr @ offset, rr, half
                pa, ra, ha = box(a); pb, rb, hb = box(b)
                depth = max(depth, obb_depth(pa, ra, ha, pb, rb, hb))
        return dict(bilateral_contact=len(touching) == 2, target_tool_contact=tool_contact, supported=supported,
                    dangerous_contact=dangerous, penetration_m=depth)

    def step(self, label, state):
        # Only real tool link COMs contribute; no object attachment flags or task truth.
        gravity = np.zeros(3); moment = np.zeros(3)
        ids = {v: k for k, v in self.bodies.items()}
        for index, (name, props) in enumerate(zip(self.tool_names, self.masses)):
            if name == 'anchor':
                continue
            mass = props.mass
            if index >= 3: gravity[2] += mass * 9.81
            if index >= self.palm_index:
                s = state['states'][ids[('tool', name)]]; qq = s['pose']['r']
                rr = Rotation.from_quat([qq['x'], qq['y'], qq['z'], qq['w']]).as_matrix()
                com = xyz(s['pose']['p']) + rr @ np.array([props.com.x, props.com.y, props.com.z])
                moment -= np.cross(com - state['p'], [0, 0, -mass * 9.81])
        desired = np.array(label[3:9]).reshape(2, 3); r = np.vstack([desired, np.cross(*desired)])
        error = Rotation.from_matrix(r @ state['r'].T).as_rotvec()
        tau, force, torque = efforts(label, state, gravity, moment, error)
        self.gym.apply_actor_dof_efforts(self.env, self.tool, tau.astype(np.float32))
        self.gym.simulate(self.sim); self.gym.fetch_results(self.sim, True); self.tick += 1
        return dict(efforts=tau, world_force=force, world_torque=torque, gravity_compensation=gravity, gravity_moment=moment)

    def render(self, state):
        self.gym.set_camera_location(self.cameras['body'], self.env, vec([0, -.50, 1]), vec([0, 0, .70]))
        self.gym.set_camera_location(self.cameras['wrist'], self.env, vec(state['p'] + state['r'] @ [.06, 0, -.03]), vec(state['p'] + state['r'] @ [0, 0, .027]))
        self.gym.step_graphics(self.sim); self.gym.render_all_camera_sensors(self.sim)
        arrays, matrices = {}, {}
        for key, camera in self.cameras.items():
            arrays[key + '_rgb'] = np.asarray(self.gym.get_camera_image(self.sim, self.env, camera, gymapi.IMAGE_COLOR)).reshape(96, 96, 4)[..., :3].copy()
            arrays[key + '_mask'] = np.asarray(self.gym.get_camera_image(self.sim, self.env, camera, gymapi.IMAGE_SEGMENTATION)).reshape(96, 96).copy()
            arrays[key + '_depth'] = np.asarray(self.gym.get_camera_image(self.sim, self.env, camera, gymapi.IMAGE_DEPTH)).reshape(96, 96).copy()
            matrices[key] = dict(view=np.asarray(self.gym.get_camera_view_matrix(self.sim, self.env, camera)).tolist(),
                                 projection=np.asarray(self.gym.get_camera_proj_matrix(self.sim, self.env, camera)).tolist())
        return arrays, matrices

    def close(self):
        self.gym.destroy_sim(self.sim)


class SetterGuard:
    """Only initialization may set native state; effort/camera/physics calls pass through."""
    def __init__(self, gym, environment):
        self.gym, self.environment = gym, environment

    def __getattr__(self, name):
        if name.startswith('set_') and any(k in name for k in ('state', 'transform', 'pose')):
            def reject(*args, **kwargs):
                self.environment.setter_attempts.append(dict(tick=self.environment.tick, api=name))
                raise RuntimeError('RUNTIME_STATE_SETTER_FORBIDDEN: ' + name)
            return reject
        return getattr(self.gym, name)
