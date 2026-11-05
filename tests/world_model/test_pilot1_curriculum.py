import unittest
import mujoco
from go2arx5_vla.world_model.pilot1_curriculum import robot_contract


def model(*,mass=1.,friction=.8,table_x=1.,cube_z=.12):
    return mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep=".002"/><worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <geom name="task_table" type="box" pos="{table_x} 0 .05" size=".2 .2 .05" friction="{friction} .02 .002"/>
      <body name="base" pos="0 0 .4"><freejoint name="root"/><geom name="base_geom" type="box" size=".1 .1 .1" mass="{mass}"/></body>
      <body name="task_cube" pos="1 0 {cube_z}"><freejoint name="task_cube_free"/><geom name="task_cube_geom" type="box" size=".02 .02 .02" mass=".05"/></body>
      </worldbody></mujoco>''')


class SceneFrozenRobot(unittest.TestCase):
    def test_only_scene_layout_can_change(self):
        proof=robot_contract(model(),model(table_x=1.1,cube_z=.13))
        self.assertTrue(proof['robot_mass_inertia_friction_actuators_unchanged'])
        self.assertFalse(proof['controller_or_safety_changes'])

    def test_robot_mass_change_rejected(self):
        with self.assertRaises(AssertionError):robot_contract(model(),model(mass=1.01))

    def test_even_support_friction_change_rejected(self):
        with self.assertRaises(AssertionError):robot_contract(model(),model(friction=.81))


if __name__=='__main__':unittest.main()
