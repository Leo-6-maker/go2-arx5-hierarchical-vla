"""Cold source-engine extension; import alone cannot step physics.

Construction/stepping must occur inside an explicitly qualified source capture
owner. The existing S1 target lease is not authority for these source tasks.
"""
from itertools import product

import numpy as np

from .contact_scene import overlay_source_xml
from .object_relations import pose


def make_engine(card, asset_root):
    """Reuse the original finite source arm/gripper driver with passive tasks.

    No executable CLI/dispatcher is introduced. The source joint-torque driver
    is inherited unchanged; this is not the floating Go2/WBC environment.
    """
    import mujoco
    from scripts.pi0291_arx5_data.engine import Engine as BaseEngine
    from scripts.pi0291_arx5_data.config import PARAMETERS as P
    from scripts.pi0291_arx5_data.scene import ARM_NAMES, FINGER_NAMES
    from scripts.pi0294.scene import xml_scene

    class SourceEngine(BaseEngine):
        def __init__(self):
            self.card, self.mode = card, 'normal'
            self.xml, self.task_binding = overlay_source_xml(xml_scene(card['source_base_card'], asset_root), card)
            self.model = mujoco.MjModel.from_xml_string(self.xml)
            self.data, self.ik_data = mujoco.MjData(self.model), mujoco.MjData(self.model)
            self.ids = {key: self._id(mujoco.mjtObj.mjOBJ_BODY, name) for key, name in (
                ('tool', 'x5_link6'), ('cube', self.task_binding['object_body']),
                ('left_finger', 'x5_link7'), ('right_finger', 'x5_link8'))}
            self.geoms = {key: self._id(mujoco.mjtObj.mjOBJ_GEOM, name) for key, name in (
                ('cube_geom', self.task_binding['grasp_geom']), ('left_pad', 'left_pad'),
                ('right_pad', 'right_pad'), ('table', 'table'), ('floor', 'floor'))}
            self.obstacles = {self._id(mujoco.mjtObj.mjOBJ_GEOM, name)
                              for name in self.task_binding['obstacle_geoms']}
            self.object_geoms = {self.geoms['cube_geom']} | (self.obstacles if card['task'] == 'DOOR' else set())
            base_id = self._id(mujoco.mjtObj.mjOBJ_BODY, 'base_link')
            self.robot_bodies = set()
            for i in range(self.model.nbody):
                parent = i
                while parent and parent != base_id: parent = int(self.model.body_parentid[parent])
                if parent == base_id: self.robot_bodies.add(i)
            self.tcp_site = self._id(mujoco.mjtObj.mjOBJ_SITE, 'x5_ee')
            self.joints = {name: self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_NAMES+FINGER_NAMES}
            self.arm_q = np.array([self.model.jnt_qposadr[self.joints[x]] for x in ARM_NAMES], dtype=int)
            self.arm_v = np.array([self.model.jnt_dofadr[self.joints[x]] for x in ARM_NAMES], dtype=int)
            self.finger_q = np.array([self.model.jnt_qposadr[self.joints[x]] for x in FINGER_NAMES], dtype=int)
            self.finger_v = np.array([self.model.jnt_dofadr[self.joints[x]] for x in FINGER_NAMES], dtype=int)
            self.arm_ctrl = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, x) for x in ARM_NAMES])
            self.finger_ctrl = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, x) for x in FINGER_NAMES])
            self.arm_bounds = self.model.jnt_range[[self.joints[x] for x in ARM_NAMES]].copy()
            self.arm_bounds[:, 0] += .005; self.arm_bounds[:, 1] -= .005
            self.data.qpos[self.finger_q] = .042
            initial = pose(card['initial_tcp'])  # Explicit common home; never object-relative auto-reset.
            self.ik_data.qpos[:] = self.data.qpos
            solution, error = self.solve_ik(initial, np.array([0., 2.13, 1.73, -1.17, 0., 0.]), iterations=150)
            if error[0] > .0002 or error[1] > .002:
                raise ValueError('SOURCE_COMMON_HOME_UNREACHABLE:'+str(error))
            self.data.qpos[self.arm_q] = solution
            self.arm_targets, self.finger_targets = solution.copy(), np.full(2, .042)
            self.steps, self.safety, self.renderer, self.last_ik_error = 0, [], None, error
            self.reset_ik_cache()
            if self.model.neq != 0 or self.model.nmocap != 0 or self.model.nu != 8 or self.model.opt.timestep != P['dt']:
                raise ValueError('SOURCE_FINITE_EXECUTOR_OR_PASSIVE_TASK_IDENTITY_CHANGED')
            mujoco.mj_forward(self.model, self.data)

        def contacts(self):
            left, right, opposing, illegal, entries = super().contacts()
            for c in self.data.contact:
                pair = {int(c.geom1), int(c.geom2)}
                if pair & self.obstacles and any(int(self.model.geom_bodyid[g]) in self.robot_bodies for g in pair):
                    illegal = True
            return left, right, opposing, illegal, entries

        def snapshot(self):
            s = super().snapshot()
            spec = card['book' if card['task'] == 'BOOK' else 'door']
            size = (np.asarray(spec['book_size_m']) if card['task'] == 'BOOK' else
                    np.array([2*spec['handle_radius_m'], 2*spec['handle_radius_m'], spec['handle_length_m']]))
            points = np.array(list(product((-1., 1.), repeat=3)))*size/2
            s['object_points'] = points @ s['object_pose'][:3, :3].T+s['object_pose'][:3, 3]
            s['robot_object_contact'] = any(
                ({r['geom0'], r['geom1']} & self.object_geoms)
                and any(int(self.model.geom_bodyid[g]) in self.robot_bodies for g in (r['geom0'], r['geom1']))
                for r in s['contacts'])
            if card['task'] == 'DOOR':
                j = self._id(mujoco.mjtObj.mjOBJ_JOINT, 'skill_hinge')
                s['hinge_angle_rad'] = float(self.data.qpos[self.model.jnt_qposadr[j]]+spec['initial_angle_rad'])
                s['hinge_velocity_rad_s'] = float(self.data.qvel[self.model.jnt_dofadr[j]])
            else:
                support = self._id(mujoco.mjtObj.mjOBJ_GEOM, self.task_binding['support_geom'])
                s['object_supported_by_shelf'] = any({r['geom0'], r['geom1']} == {support, self.geoms['cube_geom']}
                    and r['normal_force_N'] >= P['minimum_contact_force_N'] for r in s['contacts'])
            s['task'] = card['task']; s['domain'] = 'SOURCE_CONTACT_PHYSICS'
            return s

    return SourceEngine()
