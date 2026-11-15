import copy
import unittest
from unittest.mock import patch

from go2arx5_vla.world_model import sla_five_object_hold_v5 as profile
from go2arx5_vla.world_model import corelab_expert_semantics_v3 as expert
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import FROZEN_TEACHER, teacher_fingerprint


def spec(obj, version=profile.VERSION):
    s = dict(sla_five_object_hold_v5=dict(version=version, object_id=obj, pose_id='POSE01'),
        task_learning_role=profile.ROLE, task_learning_teacher_mode='NORMAL_TEACHER_V7',
        expert_eef_stream_v2=dict(version='EEF_STREAM_V2', prefix_atoms=10),
        expert_recorded_success_tail_v3=expert.RECORDED_TAIL,
        expert_dense_observation_v3=dict(version='NATIVE_EXPERT_DENSE_OBSERVATION_V3'))
    s.update(phase=task.PHASE, kind=task.KIND, task_learning_grasp=profile.protocol(s),
        task_command=profile.command(obj, version), live_skill_start_step=task.CALIBRATION_B,
        max_ticks=profile.ENDPOINT, physics_dt_s=.002, WBC_changes=False)
    return s


class FiveObjectExpertContract(unittest.TestCase):
    def test_repair_changes_only_grasp_point_for_three_failed_geometry_conditions(self):
        import numpy as np
        initial=dict(tcp=np.eye(4),object_pose=np.eye(4))
        initial['object_pose'][:3,3]=[.28,-.014,.62]
        old=profile.make_teacher(spec('RED_CUBE40'),initial)
        self.assertIs(type(old),task.Teacher)
        for obj in profile.REPAIRED_OBJECTS:
            s=spec(obj,profile.REPAIRED_VERSION); profile.declared(s)
            new=profile.make_teacher(s,initial)
            np.testing.assert_allclose(new.grasp_goal()-old.grasp_goal(),[-.005,0.,0.],atol=1e-16,rtol=0)
            self.assertEqual(new.stage,old.stage)
            self.assertEqual(new.grip,old.grip)
            self.assertEqual(s['task_learning_grasp']['policy_queries'],60)
            self.assertEqual(s['task_learning_grasp']['prefix_atoms'],10)
            self.assertTrue(s['task_learning_grasp']['original_score_unchanged'])
            self.assertEqual(profile.restart(s),12)
        for obj in ['RED_CUBE40','YELLOW_BOX35_45_40']:
            with self.assertRaises(ValueError):spec(obj,profile.REPAIRED_VERSION)

    def test_each_object_has_its_real_language_and_unchanged_clock_teacher_score(self):
        self.assertEqual(teacher_fingerprint(), FROZEN_TEACHER)
        for obj, name in profile.NAMES.items():
            s = spec(obj)
            self.assertTrue(profile.configuration(s)); profile.declared(s)
            self.assertIn(name, profile.language(s))
            self.assertEqual(expert.native_protocol(s), profile.protocol(s))
            self.assertEqual(expert.native_endpoint(s), 21703)
            self.assertEqual((21703-task.B)//25, 640)
            self.assertFalse(s['task_learning_grasp']['D100_new'])

    def test_no_flag_leaves_original_protocol_unchanged(self):
        s = dict(task_learning_role=expert.ROLE)
        self.assertFalse(profile.configuration(s))
        self.assertEqual(expert.native_protocol(s), task.protocol(expert.ROLE))

    def test_original_initializer_gets_exact_scene_fields_without_rewriting_evidence(self):
        proof = dict(schema='new_static_evidence', scene_sha256='c'*64, passes=True)
        original = copy.deepcopy(proof)
        value = profile.scene_binding(proof, 'PURPLE_BOX45_35_50')
        self.assertEqual(value['parent'], profile.scene_id('PURPLE_BOX45_35_50'))
        self.assertEqual(value['compiled']['scene_sha256'], proof['scene_sha256'])
        self.assertEqual(value['split'], 'TRAIN')
        self.assertEqual(proof, original)
        self.assertFalse(value['original_static_result_modified'])

    def test_conflicting_task_or_privileged_student_rejected(self):
        for extra in [dict(sla_pick_place_v5=dict(version='other')),
            dict(target_gr00t_policy=dict(kind='student')), dict(task_learning_control='hold')]:
            s = dict(spec('PURPLE_BOX45_35_50'), **extra)
            with self.assertRaises(ValueError): profile.configuration(s)
        s = spec('RED_CUBE40'); s['task_command'] = profile.command('GREEN_CUBE35')
        with self.assertRaises(ValueError): task.declared(s)

    def test_panel_keeps_all_five_and_fresh_request_identities(self):
        panel = dict(schema=profile.PANEL, source_sha='a'*40, tasks=['PICK_HOLD'], D100_new=False,
            base_grasp_Teacher_sha256=FROZEN_TEACHER, base_Teacher_qualification={}, assignments=[
                dict(object_id=obj, pose_id='POSE01', parent=profile.scene_id(obj), split='TRAIN',
                    restart=11, prefix_atoms=10, command=profile.command(obj),
                    model_sha256='b'*64, native_run='fresh-'+obj.lower()) for obj in profile.NAMES])
        with patch('go2arx5_vla.world_model.task_learning_collection_v1.qualified_base'):
            self.assertEqual(len(profile.validate_panel(panel)), 5)
            for field, value in [('object_id', 'RED_CUBE40'), ('restart', 10), ('pose_id', 'POSE02')]:
                changed = copy.deepcopy(panel); changed['assignments'][-1][field] = value
                with self.assertRaises(ValueError): profile.validate_panel(changed)

    def test_repair_panel_does_not_reopen_the_two_completed_objects(self):
        panel=dict(schema=profile.REPAIRED_PANEL,expert_version=profile.REPAIRED_VERSION,
            source_sha='a'*40,tasks=['PICK_HOLD'],D100_new=False,base_grasp_Teacher_sha256=FROZEN_TEACHER,
            base_Teacher_qualification={},assignments=[dict(object_id=obj,pose_id='POSE01',parent=profile.scene_id(obj),
                split='TRAIN',restart=12,prefix_atoms=10,command=profile.command(obj,profile.REPAIRED_VERSION),
                model_sha256='b'*64,native_run='repair-'+obj.lower())for obj in profile.REPAIRED_OBJECTS])
        with patch('go2arx5_vla.world_model.task_learning_collection_v1.qualified_base'):
            self.assertEqual(len(profile.validate_panel(panel)),3)
            bad=copy.deepcopy(panel);bad['assignments'][0]['restart']=11
            with self.assertRaises(ValueError):profile.validate_panel(bad)


if __name__ == '__main__': unittest.main()
