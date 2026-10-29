"""Source object compiler and synthetic feedback tests; zero integrations."""
from copy import deepcopy
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from scripts.pi0294.multitask.contact_scene import overlay_source_xml, task_only_xml
from scripts.pi0294.multitask.feedback_teacher import SourceSkillTeacher, OPEN
from scripts.pi0294.multitask.capture_core import capture
from scripts.pi0294.multitask.object_relations import axis_rotation


def translation(x, y, z):
    p = np.eye(4); p[:3, 3] = [x, y, z]
    return p


def card(task):
    common = dict(task=task)
    offset = translation(0, 0, .004)
    offset[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    if task == 'DOOR':
        common['door'] = dict(hinge_world=(translation(.35, .08, .46) @ axis_rotation([0, 0, 1], .4)).tolist(),
            width_m=.16, height_m=.18, thickness_m=.012, panel_mass_kg=.2,
            initial_angle_rad=.15, target_angle_rad=1.047, angle_limits_rad=[0, 1.4],
            handle_offset_m=[.13, -.045, .09], handle_radius_m=.009, handle_length_m=.04,
            object_to_tcp_grasp=offset.tolist(), approach_direction_object=[0, -1, 0])
    else:
        common['book'] = dict(initial_book_world=translation(.44, -.08, .52).tolist(),
            target_book_world=translation(.52, .08, .52).tolist(), shelf_world=translation(.44, .08, .44).tolist(),
            book_size_m=[.12, .018, .16], book_mass_kg=.08, wall_thickness_m=.01,
            cavity_bounds_shelf_m=[[0, .18], [-.025, .025], [0, .2]],
            object_to_tcp_grasp=offset.tolist(), approach_direction_object=[0, 0, 1])
    return common


SOURCE_FIXTURE = '''<mujoco model="PI0294_FIXED_BASE_ARX5_MULTITASK">
<compiler angle="radian"/><option timestep=".002" gravity="0 0 -9.81"/>
<worldbody><body name="base_link"><body name="x5_link1"><joint name="arm"/>
<geom type="sphere" size=".01" mass=".1"/></body></body>
<body name="cube"><freejoint name="cube_free"/><geom name="cube_geom" type="box" size=".02 .02 .02" mass=".05"/></body>
<geom name="goal_marker_0" type="box" size=".01 .01 .001" contype="0" conaffinity="0"/>
</worldbody><actuator><motor joint="arm"/></actuator><contact/><equality/>
</mujoco>'''


class SceneTests(unittest.TestCase):
    def test_overlay_preserves_source_robot_driver_and_physics_xml(self):
        before = ET.fromstring(SOURCE_FIXTURE)
        for task in ('DOOR', 'BOOK'):
            xml, binding = overlay_source_xml(SOURCE_FIXTURE, card(task))
            after = ET.fromstring(xml)
            for query in ("worldbody/body[@name='base_link']", 'actuator', 'option', 'contact', 'equality'):
                self.assertEqual(ET.tostring(before.find(query)), ET.tostring(after.find(query)))
            self.assertIsNone(after.find("worldbody/body[@name='cube']"))
            self.assertIsNone(after.find("worldbody/geom[@name='goal_marker_0']"))
            self.assertEqual(binding['new_actuators'], 0)

    def test_rejects_target_floating_robot_or_degree_conversion(self):
        for xml in (SOURCE_FIXTURE.replace('base_link">', 'base_link"><freejoint/>'),
                    SOURCE_FIXTURE.replace('angle="radian"', 'angle="degree"'),
                    SOURCE_FIXTURE.replace('PI0294_FIXED_BASE_ARX5_MULTITASK', 'TARGET_WBC')):
            with self.assertRaises(ValueError):
                overlay_source_xml(xml, card('DOOR'))

    def test_native_door_fixture_is_passive_and_actual_handle_pose_matches_geometry(self):
        try: import mujoco
        except ImportError: self.skipTest('MuJoCo unavailable; run in the existing CPU environment')
        c = card('DOOR'); spec = c['door']
        model = mujoco.MjModel.from_xml_string(task_only_xml(c)); data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual((model.nq, model.nv, model.nu, model.neq, model.nmocap), (1, 1, 0, 0, 0))
        self.assertEqual(data.time, 0)
        expected = np.asarray(spec['hinge_world']) @ axis_rotation([0, 0, 1], spec['initial_angle_rad'])
        expected = expected @ translation(*spec['handle_offset_m'])
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'skill_handle')
        np.testing.assert_allclose(data.xpos[i], expected[:3, 3], atol=1e-9)
        np.testing.assert_allclose(data.xmat[i].reshape(3, 3), expected[:3, :3], atol=1e-9)
        np.testing.assert_allclose(model.jnt_range[0], np.asarray(spec['angle_limits_rad'])-spec['initial_angle_rad'])

    def test_native_book_fixture_has_free_dynamics_and_real_support_geometry(self):
        try: import mujoco
        except ImportError: self.skipTest('MuJoCo unavailable; run in the existing CPU environment')
        c = card('BOOK')
        # Non-identity pose catches xyzw-versus-wxyz mistakes in MJCF serialization.
        c['book']['initial_book_world'] = (translation(.44, -.08, .52) @ axis_rotation([0, 0, 1], .3)).tolist()
        model = mujoco.MjModel.from_xml_string(task_only_xml(c)); data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual((model.nq, model.nv, model.nu, model.neq, model.nmocap), (7, 6, 0, 0, 0))
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'skill_book')
        np.testing.assert_allclose(data.xmat[i].reshape(3, 3), np.asarray(c['book']['initial_book_world'])[:3, :3], atol=1e-9)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'skill_shelf_floor')
        self.assertAlmostEqual(data.geom_xpos[floor, 2]+model.geom_size[floor, 2], .44)
        self.assertEqual(data.time, 0)

    def test_physical_scene_rejects_handle_in_panel_and_book_outside_slot(self):
        c = card('DOOR'); c['door']['handle_offset_m'][1] = -.01
        with self.assertRaises(ValueError): task_only_xml(c)
        c = card('BOOK'); c['book']['target_book_world'][1][3] += .1
        with self.assertRaises(ValueError): task_only_xml(c)


def state(task, t=0., **changes):
    s = dict(time_s=t, tcp=translation(.1, 0, .3), object_pose=translation(.4, 0, .55),
        safety_valid=True, native_grasp=False, gripper_fraction=OPEN, robot_object_contact=False,
        object_linear_velocity=np.zeros(3), object_angular_velocity=np.zeros(3),
        object_supported_by_shelf=False, hinge_angle_rad=.15, hinge_velocity_rad_s=0.)
    if task == 'BOOK': s['object_pose'] = np.asarray(card(task)['book']['initial_book_world'])
    s.update(changes)
    return s


class FeedbackTests(unittest.TestCase):
    def test_stationary_actual_robot_never_advances_by_reference_timer(self):
        teacher = SourceSkillTeacher(card('DOOR'))
        for i in range(641):
            teacher.observe(state('DOOR', i*.05))
            if i % 8 == 0 and i < 640: teacher.plan()
        self.assertEqual(teacher.stage, 'approach')
        self.assertFalse(teacher.candidate_complete)

    def test_persistent_world_path_does_not_reanchor_every_observation(self):
        teacher = SourceSkillTeacher(card('DOOR')); teacher.observe(state('DOOR'))
        first = teacher.plan()
        for i in range(1, 9):
            teacher.observe(state('DOOR', i*.05, tcp=translation(.103, 0, .3)))
            if i < 8:
                with self.assertRaises(ValueError): teacher.plan()
        second = teacher.plan()
        np.testing.assert_allclose(first['poses'][8:], second['poses'][:32], atol=1e-10)
        self.assertEqual(second['persistent_path_start_observation'], 0)

    def test_stale_or_missing_actual_clock_and_native_safety_stop(self):
        teacher = SourceSkillTeacher(card('BOOK')); teacher.observe(state('BOOK'))
        with self.assertRaises(ValueError): teacher.observe(state('BOOK', .1))
        with self.assertRaises(PermissionError): teacher.observe(state('BOOK', .05, safety_valid=False))
        with self.assertRaises(PermissionError): teacher.plan()

    def test_grasp_loss_stops_transport_and_preserves_actual_failure_state(self):
        teacher = SourceSkillTeacher(card('DOOR')); teacher.stage = 'turn'
        failed = state('DOOR', 17., hinge_angle_rad=.65)
        with self.assertRaisesRegex(PermissionError, 'ACTUAL_GRASP_LOST'):
            teacher.observe(failed)
        self.assertEqual(teacher.state['hinge_angle_rad'], .65)
        recovery = SourceSkillTeacher(card('DOOR'))
        recovery.observe(failed); recovery.plan()
        np.testing.assert_allclose(recovery.grasp_goal, failed['object_pose'] @ recovery.grasp_offset)

    def test_book_release_requires_real_support_and_all_actual_corners_inside(self):
        for supported, shift, expect in ((False, 0., 'insert'), (True, .02, 'insert'), (True, 0., 'release')):
            teacher = SourceSkillTeacher(card('BOOK')); teacher.stage = 'insert'
            tool = translation(.5, .08, .6); teacher.goal = tool.copy()
            book = np.asarray(card('BOOK')['book']['target_book_world']); book[1, 3] += shift
            teacher.observe(state('BOOK', tcp=tool, object_pose=book, native_grasp=True,
                                  object_supported_by_shelf=supported))
            self.assertEqual(teacher.stage, expect)

    def test_tcp_lift_without_actual_book_lift_cannot_enter_transport(self):
        teacher = SourceSkillTeacher(card('BOOK')); teacher.stage = 'lift'
        teacher.lift_initial_object_z = .52
        teacher.goal = translation(.44, -.08, .604)
        teacher.observe(state('BOOK', tcp=teacher.goal, native_grasp=True))
        self.assertEqual(teacher.stage, 'lift')

    def test_one_intervening_contact_resets_entire_hold_duration(self):
        teacher = SourceSkillTeacher(card('DOOR')); teacher.stage = 'hold'
        for i in range(43):
            teacher.observe(state('DOOR', i*.05, tcp=translation(.1, 0, .55),
                hinge_angle_rad=1.047, robot_object_contact=(i == 1)))
            if i == 40: self.assertFalse(teacher.candidate_complete)
        self.assertTrue(teacher.candidate_complete)


class FixtureEngine:
    """Explicit CPU fixture for capture bookkeeping; not a physical executor."""
    def __init__(self, fail_at=None):
        self.steps, self.fail_at = 0, fail_at
        self.current_tcp = translation(.1, 0, .3)
        self.opening = OPEN

    def snapshot(self):
        return state('DOOR', self.steps*.002, tcp=self.current_tcp.copy(), gripper_fraction=self.opening)

    def render(self, snapshot):
        return ({name: np.zeros((256, 256, 3), dtype=np.uint8) for name in ('body_rgb', 'wrist_rgb')},
                {name: dict(timestamp_s=snapshot['time_s'], T_world_optical=np.eye(4),
                            K=np.diag([128., 128., 1.])) for name in ('body_rgb', 'wrist_rgb')})

    def step(self, target, opening):
        self.steps += 1
        if self.steps == self.fail_at: raise RuntimeError('SYNTHETIC_NATIVE_STOP')
        self.current_tcp, self.opening = target.copy(), opening


class CaptureTests(unittest.TestCase):
    def test_only_executed_prefix_becomes_interval_records(self):
        engine = FixtureEngine(); teacher = SourceSkillTeacher(card('DOOR')); records = []
        result = capture(engine, teacher, instruction='Open the cabinet door.', guard=lambda: None,
                         emit=lambda kind, row: records.append((kind, row)), duration_s=.4)
        self.assertEqual((result['physics_integrations'], result['RGB_pairs'], result['H40_plans']), (200, 9, 1))
        completed = [r for k, r in records if k == 'completed_interval']
        planned = [r for k, r in records if k == 'planned_H40']
        self.assertEqual(len(completed), 8)
        self.assertEqual(planned[0]['rows10'].shape, (40, 10))
        self.assertEqual([r['prefix_index'] for r in completed], list(range(8)))
        np.testing.assert_array_equal(np.stack([r['action_EEF10'] for r in completed]), planned[0]['rows10'][:8])
        self.assertTrue(all(r['end_native_step']-r['start_native_step'] == 25 for r in completed))
        self.assertFalse(result['training_admitted'])
        for k, r in records:
            if k == 'observation':
                self.assertEqual(set(r['policy_obs']), {'instruction', 'actual_tcp', 'actual_gripper', 'body_rgb', 'wrist_rgb'})

    def test_mid_interval_native_stop_keeps_attempted_command_out_of_completed_labels(self):
        engine = FixtureEngine(fail_at=7); teacher = SourceSkillTeacher(card('DOOR')); records = []
        with self.assertRaisesRegex(RuntimeError, 'SYNTHETIC_NATIVE_STOP'):
            capture(engine, teacher, instruction='Open.', guard=lambda: None,
                    emit=lambda kind, row: records.append((kind, row)), duration_s=.4)
        self.assertEqual(engine.steps, 7)
        self.assertEqual(sum(k == 'submitted_interval' for k, _ in records), 1)
        self.assertEqual(sum(k == 'completed_interval' for k, _ in records), 0)
        self.assertEqual([r['native_step'] for k, r in records if k == 'native_tick_failure'], [7])

    def test_recorder_failure_prevents_further_physics(self):
        engine = FixtureEngine(); teacher = SourceSkillTeacher(card('DOOR'))
        def fail(kind, row):
            if kind == 'planned_H40': raise OSError('SYNTHETIC_DISK_FAILURE')
        with self.assertRaises(OSError):
            capture(engine, teacher, instruction='Open.', guard=lambda: None, emit=fail, duration_s=.4)
        self.assertEqual(engine.steps, 0)

    def test_owner_guard_is_checked_before_every_native_tick(self):
        engine = FixtureEngine(); teacher = SourceSkillTeacher(card('DOOR'))
        def guard():
            if engine.steps == 3: raise PermissionError('SYNTHETIC_OWNER_LIMIT')
        with self.assertRaises(PermissionError):
            capture(engine, teacher, instruction='Open.', guard=guard, emit=lambda kind, row: None, duration_s=.4)
        self.assertEqual(engine.steps, 3)

    def test_high_rate_actual_safety_flag_stops_even_without_driver_exception(self):
        class Unsafe(FixtureEngine):
            def snapshot(self):
                return dict(super().snapshot(), safety_valid=self.steps < 3)
        engine = Unsafe(); records = []
        with self.assertRaisesRegex(PermissionError, 'HIGH_RATE_SAFETY_STOP'):
            capture(engine, SourceSkillTeacher(card('DOOR')), instruction='Open.', guard=lambda: None,
                    emit=lambda kind, row: records.append((kind, row)), duration_s=.4)
        self.assertEqual(engine.steps, 3)
        self.assertFalse(any(k == 'completed_interval' for k, _ in records))

    def test_invalid_camera_calibration_stops_before_first_integration(self):
        class BadCamera(FixtureEngine):
            def render(self, snapshot):
                images, cal = super().render(snapshot)
                cal['wrist_rgb']['T_world_optical'][0, 0] = -1
                return images, cal
        engine = BadCamera()
        with self.assertRaises(ValueError):
            capture(engine, SourceSkillTeacher(card('DOOR')), instruction='Open.', guard=lambda: None,
                    emit=lambda kind, row: None, duration_s=.4)
        self.assertEqual(engine.steps, 0)


if __name__ == '__main__': unittest.main()
