import copy
import unittest
from unittest.mock import patch

from go2arx5_vla.world_model import sla_five_object_hold_v5 as profile
from go2arx5_vla.world_model import corelab_expert_semantics_v3 as expert
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model.task_learning_teacher_panel_v1 import FROZEN_TEACHER, teacher_fingerprint


def spec(obj):
    s = dict(sla_five_object_hold_v5=dict(version=profile.VERSION, object_id=obj, pose_id='POSE01'),
        task_learning_role=profile.ROLE, task_learning_teacher_mode='NORMAL_TEACHER_V7',
        expert_eef_stream_v2=dict(version='EEF_STREAM_V2', prefix_atoms=10),
        expert_recorded_success_tail_v3=expert.RECORDED_TAIL,
        expert_dense_observation_v3=dict(version='NATIVE_EXPERT_DENSE_OBSERVATION_V3'))
    s.update(phase=task.PHASE, kind=task.KIND, task_learning_grasp=profile.protocol(s),
        task_command=profile.command(obj), live_skill_start_step=task.CALIBRATION_B,
        max_ticks=profile.ENDPOINT, physics_dt_s=.002, WBC_changes=False)
    return s


class FiveObjectExpertContract(unittest.TestCase):
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


if __name__ == '__main__': unittest.main()
