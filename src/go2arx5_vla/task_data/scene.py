from __future__ import annotations

import numpy as np
from .config import PARAMETERS as P
from ..trajectory.so3 import matrix_to_quat


def camera_axes(position, target):
    back = np.asarray(position, float) - np.asarray(target, float)
    back /= np.linalg.norm(back)
    right = np.cross([0., 0., 1.], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return np.stack([right, up, back], axis=1)


def camera_quat(position, target):
    q = matrix_to_quat(camera_axes(position, target))
    return ' '.join(map(str, q[[3, 0, 1, 2]]))


def xml_scene(card):
    sx, sy = card['source_xy']; gx, gy = card['goal_xy']
    yaw = card['cube_yaw']
    cube_quat = f'{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}'
    body_cam = camera_quat([.45, -.65, .92], [0, 0, .43])
    wrist_cam = camera_quat([.0, -.10, .15], [0, 0, 0])
    return f'''<mujoco model="PI0290_independent_dynamic_tool">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" solver="Newton" iterations="50" tolerance="1e-10"/>
  <size nconmax="128" njmax="512"/>
  <visual><global offwidth="256" offheight="256"/><quality shadowsize="1024"/></visual>
  <default><geom condim="4" friction=".8 .005 .0001" solref=".004 1" solimp=".95 .99 .001"/>
    <joint damping=".05" armature=".0001"/></default>
  <worldbody>
    <light pos="0 -.3 1.4" dir="0 .2 -1" diffuse=".8 .8 .8"/>
    <light pos="-.5 .4 1.2" dir=".4 -.3 -1" diffuse=".5 .5 .5"/>
    <geom name="floor" type="plane" size="2 2 .1" rgba=".78 .80 .82 1"/>
    <geom name="table" type="box" pos="0 0 .3" size=".4 .32 .1" rgba=".62 .66 .68 1"/>
    <geom name="zone" type="box" pos="{gx} {gy} .4005" size=".08 .08 .0005" contype="0" conaffinity="0" rgba=".1 .3 .85 .7"/>
    <camera name="body_rgb" pos=".45 -.65 .92" quat="{body_cam}" fovy="48"/>
    <body name="cube" pos="{sx} {sy} .42" quat="{cube_quat}">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size=".02 .02 .02" mass=".05" rgba=".88 .08 .06 1"/>
    </body>
    <body name="tool" pos="{sx} {sy} .56" quat="{cube_quat}">
      <freejoint name="tool_free"/>
      <geom name="palm" type="box" pos="0 0 .065" size=".038 .062 .012" mass=".36" rgba=".12 .15 .18 1"/>
      <site name="tcp" pos="0 0 0" size=".003" rgba=".1 .8 .2 1"/>
      <camera name="wrist_rgb" pos="0 -.10 .15" quat="{wrist_cam}" fovy="62"/>
      <body name="left_finger" pos="0 .010 0">
        <joint name="left_slide" type="slide" axis="0 1 0" range="0 .044" damping=".2"/>
        <geom name="left_pad" type="box" size=".018 .0065 .016" mass=".06" friction="1.2 .005 .0001" rgba=".26 .3 .32 1"/>
      </body>
      <body name="right_finger" pos="0 -.010 0">
        <joint name="right_slide" type="slide" axis="0 -1 0" range="0 .044" damping=".2"/>
        <geom name="right_pad" type="box" size=".018 .0065 .016" mass=".06" friction="1.2 .005 .0001" rgba=".26 .3 .32 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="left_motor" joint="left_slide" ctrllimited="true" ctrlrange="-8 8"/>
    <motor name="right_motor" joint="right_slide" ctrllimited="true" ctrlrange="-8 8"/>
  </actuator>
</mujoco>'''


def freeze_description():
    return dict(parameters=P, tool_source='NEW_PROCEDURAL_BOX_GEOMETRY_NO_PRIVATE_MESH',
                gripper='SIM_ONLY_PROVISIONAL_GRIPPER', tcp='dynamic tool body origin at pad center',
                archive_reference={'sha256':'6dd13098de5a58327fb0a0a5b3ff749e0a5534510375b92fba3f8fb50a322d52',
                    'used':'read-only 44mm opposing slide and original TCP declaration reference',
                    'original_tcp_link6_m':[.1424, 0, .0001057], 'private_source_copied':False},
                actuator='finite Cartesian world wrench on tool body; separate limited finger motors',
                object_manipulation='MuJoCo contact only; no object pose write after initial construction',
                camera_convention='K plus T_world_optical; optical x right, y down, z forward',
                camera_parent={'body_rgb':'fixed world rig', 'wrist_rgb':'rigid dynamic tool'},
                friction_and_force='explicit provisional simulation assumptions, not hardware certification')
