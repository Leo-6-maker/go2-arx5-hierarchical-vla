from copy import deepcopy
import unittest

import numpy as np

from scripts.pi0291_arx5_data.teacher import Teacher
from scripts.pi0294.scene_cards import task_card
from scripts.pi0294.vap1.plan import candidate
from scripts.pi0294.vap1.teacher_audit import replay


class SavedTeacherReplayTests(unittest.TestCase):
    def packet(self):
        card=task_card(candidate(0,'AP0','+Y','NEAR',0),'T1')
        initial=np.eye(4);initial[:3,3]=[*card['source_xy'],.46]
        teacher=Teacher(card,initial);tcp=np.eye(4);tcp[:3,3]=[.45,0.,.8]
        samples=[];references=[]
        physics={k:[] for k in ('time_s','object_pose','native_grasp','left_object_contact',
            'right_object_contact','robot_object_contact','object_linear_velocity','object_angular_velocity')}
        for i in range(641):
            t=i*.05;s=dict(time_s=t,object_pose=initial.copy(),tcp=tcp.copy(),native_grasp=False,
                left_object_contact=False,right_object_contact=False,robot_object_contact=False,
                object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),gripper_fraction=1.)
            teacher.observe(s)
            for k in physics:physics[k].append(s[k])
            samples.append(dict(policy_obs=dict(timestamp_s=t,actual_gripper=1.),
                causal_history=dict(episode_id=card['id']),execution=dict(actual_tcp=tcp.tolist()),
                privileged=dict(teacher_stage=teacher.stage,ever_lifted=teacher.lifted,ever_carried=teacher.carried)))
            if i%8==0 and i<640:
                p=teacher.plan(s);references.append(dict(decision_index=i//8,privileged_teacher_stage=p['stage'],
                    planned_H40=p['poses'].tolist(),gripper_H40=p['opening'].tolist()))
        physics={k:np.array(x) for k,x in physics.items()}
        physics.update(tcp_world=np.repeat(tcp[None],641,axis=0),contact_time_s=np.arange(1,16001)*.002,
                       contact_audit=np.column_stack([np.zeros((16000,3)),np.full(16000,.46),np.zeros(16000)]))
        return card,initial,samples,physics,references,teacher.events

    def test_actual_original_teacher_pure_replay(self):
        result=replay(*self.packet())
        self.assertEqual(result['observations'],641)
        self.assertEqual(result['references'],80)
        self.assertFalse(result['recovery_qualification'])
        self.assertEqual(result['new_physics'],0)

    def test_phase_or_reference_corruption_is_not_accepted(self):
        args=list(self.packet());args[2][0]['privileged']['teacher_stage']='hold'
        with self.assertRaisesRegex(ValueError,'STAGE_REPLAY'):replay(*args)
        args=list(self.packet());args[4][0]['planned_H40'][0][0][3]+=.001
        with self.assertRaisesRegex(ValueError,'H40_REPLAY'):replay(*args)

    def test_eval_data_or_missing_push_geometry_rejected(self):
        args=list(self.packet());args[0]['split']='DEV'
        with self.assertRaises(PermissionError):replay(*args)
        args=list(self.packet());args[0]['task']='T2'
        with self.assertRaisesRegex(PermissionError,'PAD_GEOMETRY'):replay(*args)


if __name__=='__main__':unittest.main()
