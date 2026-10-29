"""Raw cabinet task checks; no simulator, Teacher or stored-goal trust."""
import hashlib,json
import xml.etree.ElementTree as ET
import numpy as np

from .robocasa_cabinet_cards import cards
from .robocasa_cabinet_scene import VERSION
from .robocasa_cabinet_assets import native_door_status
from scripts.pi0291_arx5_data.config import PARAMETERS as P


class CabinetChecks:
    def __init__(self,folder,qualification):
        q=qualification;card_bytes=(folder/'CARD.json').read_bytes();card=json.loads(card_bytes)
        if card not in cards().values()or q['source_port']!=VERSION or q['card_sha256']!=hashlib.sha256(card_bytes).hexdigest():
            raise ValueError('EXACT_PREDECLARED_CABINET_CARD_NOT_A_CALLER_THRESHOLD')
        scene=(folder/'NATIVE_SCENE.private.xml').read_bytes()
        if hashlib.sha256(scene).hexdigest()!=q['native_scene_sha256']:
            raise ValueError('EXACT_ACTUAL_NATIVE_SCENE_BYTES')
        root=ET.fromstring(scene);j=root.find(".//joint[@name='rc_doorhinge']")
        self.name='rc_doorhinge';self.qadr=q['passive_joint_qposadr'];self.vadr=q['passive_joint_dofadr']
        self.ranges={self.name:card['door']['angle_limits_rad']}
        if (root.get('model')!=VERSION or j is None or q['passive_joint_name']!=self.name
                or q['passive_joint_range']!=self.ranges[self.name]
                or not np.array_equal(np.fromstring(j.get('range'),sep=' '),self.ranges[self.name])
                or not all(type(x)is int and x>=0 for x in(self.qadr,self.vadr))):
            raise ValueError('ACTUAL_MODEL_PASSIVE_JOINT_MAPPING_AND_ORIGINAL_RANGE')
        self.geoms=q['geom_bindings'];self.robots=set(q['robot_geom_ids'])
        self.objects=set(q['obstacle_geom_ids'])|{self.geoms['cube_geom']}

    def native(self,s):
        q=float(s['qpos'][self.qadr]);v=float(s['qvel'][self.vadr])
        expected=native_door_status({self.name:q},self.ranges)
        if (not np.isfinite(v)or s['native_door_joint_positions']!={self.name:q}
                or s['native_door_joint_ranges']!=self.ranges
                or s['hinge_angle_rad']!=q or s['hinge_velocity_rad_s']!=v
                or s['native_robocasa_door_status']!=expected):
            raise ValueError('RECORDED_DOOR_FIELDS_MUST_MATCH_ACTUAL_RAW_QPOS_QVEL')
        return expected

    def grasp(self,s):
        sides={k:[]for k in('left','right')};target=self.geoms['cube_geom']
        for c in s['contacts']:
            normal=np.asarray(c['normal'],dtype=float);force=c['normal_force_N']
            if normal.shape!=(3,)or not np.isfinite(normal).all()or not np.isfinite(force):
                raise ValueError('FINITE_ACTUAL_CONTACT_NORMAL_AND_FORCE')
            for side in sides:
                if {c['geom0'],c['geom1']}=={self.geoms[side+'_pad'],target}and force>=P['minimum_contact_force_N']:
                    sides[side].append(normal if c['geom1']==target else-normal)
        opposing=bool(sides['left']and sides['right']and any(np.dot(a,b)<=P['opposing_contact_dot_max']for a in sides['left']for b in sides['right']))
        finger=np.asarray(s['finger_q']);valid=np.isfinite(finger).all()and np.all(finger>=-.001)and np.all(finger<=P['finger_travel']+.001)
        relative=np.linalg.norm(np.asarray(s['object_linear_velocity'])-np.asarray(s['tool_linear_velocity']))<=P['relative_grasp_speed_max_m_s']
        actual=bool(opposing and valid and relative)
        if s['native_grasp']!=actual:raise ValueError('STORED_GRASP_FLAG_WITHOUT_ORIGINAL_CONTACT_EVIDENCE')
        return actual

    def validate(self,s):
        self.native(s);self.grasp(s)

    def goal_and_extra(self,s):
        goal=self.native(s)['native_is_open']
        contact=any(bool({c['geom0'],c['geom1']}&self.robots)and bool({c['geom0'],c['geom1']}&self.objects)for c in s['contacts'])
        return bool(goal and not contact and s['gripper_fraction']>=.85
            and np.linalg.norm(np.asarray(s['tcp'])[:3,3]-np.asarray(s['object_pose'])[:3,3])>=.08
            and abs(float(s['qvel'][self.vadr]))<=.05)
