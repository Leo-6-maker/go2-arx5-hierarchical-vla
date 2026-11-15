import copy
import unittest

import mujoco
import numpy as np

from go2arx5_vla.world_model import sla_target_coverage_v5 as coverage


def model(declared=None, *, robot_mass=1., friction=.8):
    d = declared or dict(size_m=[.04]*3, mass_kg=.05, rgba=[.82, .04, .04, 1.],
        initial_xyz_m=[1., 0., .12], initial_quaternion_wxyz=[1., 0., 0., 0.])
    text = lambda a: ' '.join(map(str, a))
    return mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep=".002"/><worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <geom name="task_table" type="box" pos=".4 0 .575" size=".17 .24 .025" friction="{friction} .02 .002"/>
      <body name="base" pos="0 0 .4"><freejoint name="root"/><geom name="base_geom" type="box" size=".1 .1 .1" mass="{robot_mass}"/></body>
      <body name="task_cube" pos="{text(d['initial_xyz_m'])}" quat="{text(d['initial_quaternion_wxyz'])}"><freejoint name="task_cube_free"/>
      <geom name="task_cube_geom" type="box" size="{text(np.asarray(d['size_m'])/2)}" mass="{d['mass_kg']}" rgba="{text(d['rgba'])}"/></body>
      </worldbody></mujoco>''')


class FiveObjectStaticContract(unittest.TestCase):
    def test_fifty_geometries_and_hundred_task_cells_remain_unadmitted(self):
        catalog = coverage.catalogue()
        self.assertEqual(len(catalog['cases']), 50)
        self.assertEqual(len({tuple(c['size_m']) for c in catalog['cases']}), 5)
        self.assertEqual(sum(len(c['tasks']) for c in catalog['cases']), 100)
        self.assertFalse(catalog['production_admitted'])
        self.assertTrue(all(not c['formal_D100_admitted'] for c in catalog['cases']))

    def test_all_declared_variants_preserve_original_robot_and_caller_model(self):
        original = model()
        for d in coverage.catalogue()['cases']:
            with self.subTest(object=d['object_id'], pose=d['pose_id']):
                candidate = model(d)
                before = [candidate.body_mass.copy(), candidate.body_inertia.copy(), candidate.geom_size.copy(), candidate.qpos0.copy()]
                result = coverage.check_robot_and_declared_object(original, candidate, d)
                self.assertTrue(result['normalization_only_in_separate_read_only_copy'])
                for actual, expected in zip((candidate.body_mass, candidate.body_inertia, candidate.geom_size, candidate.qpos0), before):
                    np.testing.assert_array_equal(actual, expected)

    def test_undeclared_object_and_robot_changes_reject(self):
        d = coverage.declaration('BLUE_BOX50_30_40', 'POSE01')
        for label, candidate in [
            ('robot_mass', model(d, robot_mass=1.01)),
            ('support_friction', model(d, friction=.81)),
        ]:
            with self.subTest(label=label), self.assertRaises(AssertionError):
                coverage.check_robot_and_declared_object(model(), candidate, d)
        for field, index in [('geom_friction', 'task_cube_geom'), ('geom_size', 'task_cube_geom')]:
            candidate = model(d)
            getattr(candidate, field)[candidate.geom(index).id, 0] += .001
            with self.assertRaises(AssertionError):
                coverage.check_robot_and_declared_object(model(), candidate, d)
        candidate = model(d); candidate.geom('task_table').pos[0] += .001
        with self.assertRaises(AssertionError):
            coverage.check_robot_and_declared_object(model(), candidate, d)
        changed = copy.deepcopy(d); changed['mass_kg'] *= 1.1
        with self.assertRaisesRegex(ValueError, 'EXACT_OBJECT_CONTRACT'):
            coverage.check_robot_and_declared_object(model(), model(d), changed)


if __name__ == '__main__':
    unittest.main()
