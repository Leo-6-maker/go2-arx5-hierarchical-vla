import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from go2arx5_vla.world_model.fusion_contract import PLAN
from go2arx5_vla.world_model.fusion_episode import SCHEMA
from go2arx5_vla.world_model.fusion_scope import binding_contract, SCOPE
from go2arx5_vla.world_model.pilot1_episode import choose, episode_config, score
from go2arx5_vla.world_model.storage import digest


class FusionNative(unittest.TestCase):
    def test_new_episode_requires_frozen_scene_known_intent_and_no_test_training(self):
        from go2arx5_vla.world_model.next1_episode import RULE
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            def save(name,value):
                p=root/name;p.write_text(json.dumps(value));return str(p)
            binary=save('binary',{});model=save('model',{})
            assets=save('assets',dict(native_binary_sha256=digest(binary),scenes=[
                dict(scene_id='FUSION_FIXTURE',parent='FUSION_FIXTURE_PARENT',split='TRAIN',
                     model_path=model,model_sha256=digest(model),static_check='PASSED_NO_INTEGRATION')]))
            e=dict(schema=SCHEMA,run=str(root),phase='collect-target-pilot1',purpose='SCENE_QUALIFICATION',
                split='TRAIN',reference_rule=RULE,task_atoms=64,prehold_atoms=40,screen_atom=48,
                screen=dict(window_s=.4,relative_TCP_diameter_m=.002,max_relative_TCP_speed_m_s=.025),
                method='B0_DIRECT',offset_xyz_m=[0,0,0],nominal_action=1,
                desired_outcome_provenance='CALIBRATION_INTENT',inference_mode='CALIBRATION_NO_VLA',
                desired_delta_xy_m=[.001,0],scene_id='FUSION_FIXTURE',parent='FUSION_FIXTURE_PARENT')
            manifest=save('episode',e);s=dict(episode_manifest=manifest,episode_manifest_sha256=digest(manifest),
                registered_root=str(root),fusion_plan_sha=PLAN,scene_allowlist=assets,
                scene_allowlist_sha256=digest(assets),model_path=model,native_binary=binary)
            for name in ['task','ARX','teacher_metadata','train_mean_response_model']:
                s[name]=save(name,{});s[name+'_sha256']=digest(s[name])
            with patch('go2arx5_vla.world_model.fusion_scope.validate'):
                self.assertEqual(episode_config(s),e)
                for change in [dict(desired_outcome_provenance='SEMANTICS_UNRESOLVED'),
                        dict(desired_delta_xy_m=[.1,0]),dict(nominal_action=4),
                        dict(purpose='TRAIN_FEEDBACK',split='TEST'),dict(method='TRAINED_ACTOR')]:
                    with self.subTest(change=change):
                        Path(manifest).write_text(json.dumps(dict(e,**change)))
                        s['episode_manifest_sha256']=digest(manifest)
                        with self.assertRaises(PermissionError):episode_config(s)
                Path(manifest).write_text(json.dumps(e));s['episode_manifest_sha256']=digest(manifest)
                Path(binary).write_text('changed')
                with self.assertRaisesRegex(PermissionError,'EXECUTOR'):episode_config(s)

    def test_new_calibration_path_saves_all_predictions_before_install_and_scores_actual_motion(self):
        from tests.world_model.test_pilot1_actor import fixture
        from go2arx5_vla.task_data.codec import pose10
        from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
        meta, model, _ = fixture(); sample = dict(qpos=np.r_[0, 0, .4, 1, np.zeros(30)],
            qvel=np.zeros(32), ctrl=np.zeros(20), tcp=np.eye(4), object_pose=np.eye(4), finger_q=np.zeros(2))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); arx=root/'arx.npz'; np.savez_compressed(arx, **model.arrays())
            def save(name, value):
                p=root/name;p.write_text(json.dumps(value));return str(p)
            task=save('task',dict(schema='fixture',nonhold_action_cost_m=.00002,success_tolerance_m=.0002))
            metadata=save('meta',meta)
            mean=save('mean',dict(schema='WM02_PILOT2_PARENT_BALANCED_MEAN_RESPONSE_V1',fit_split='TRAIN',
                task_sha256=digest(task),exact_fork_oracle=False,rule_fit_changes_task_or_score=False,
                mean_response_xy_m=[[0,0],[.001,0],[-.001,0],[0,.001]]))
            spec=dict(task=task,task_sha256=digest(task),ARX=str(arx),ARX_sha256=digest(arx),
                teacher_metadata=metadata,teacher_metadata_sha256=digest(metadata),
                train_mean_response_model=mean,train_mean_response_model_sha256=digest(mean),source_sha='fixture')
            events=[]
            def event(v):
                self.assertTrue((root/'FUSION_DECISION.private.json').exists());events.append(v)
            ref=np.repeat(pose10(np.eye(4),float(teacher_physical_fraction(1.)))[None],40,0)
            adapter=SimpleNamespace(spec=spec,output=root,wait_source=lambda:None,
                observe=lambda:dict(step=3703,time_s=7.406),latest=dict(ee=SimpleNamespace(time=7.406)),
                next1_samples=[sample]*9,active_reference=dict(action_EEF10=ref),
                base_pose_hold=[0,0,.4,0,0,0,1],fusion_initial_tcp=np.eye(4),
                measured_task_state=lambda:sample,command_event=event)
            episode=dict(schema=SCHEMA,method='B0_DIRECT',nominal_action=1,
                desired_delta_xy_m=[.001,0],desired_outcome_provenance='CALIBRATION_INTENT',
                purpose='EVALUATION',parent='synthetic',split='TRAIN')
            np.testing.assert_array_equal(choose(adapter,episode),[.001,0,0])
            d=json.loads((root/'FUSION_DECISION.private.json').read_text())
            self.assertEqual(np.asarray(d['predictions']).shape,(4,16,106))
            self.assertEqual(d['installed_reference'],'NOT_YET_INSTALLED')
            self.assertFalse(d['future_response_observed']);self.assertIsNone(d['vla_raw_action'])
            self.assertEqual(adapter.fusion_selected_action,1)
            sample['tcp'][0,3]=.001
            measured=score(adapter,episode)
            self.assertAlmostEqual(measured['actual_XY_error_m'],0.)
            self.assertTrue(measured['local_XY_success'])
            self.assertFalse(measured['full_task_success'])
            np.testing.assert_allclose(measured['actual_BASE_relative_displacement_m'],[.001,0,0])

    def test_schema_routes_to_fusion_validation_without_relaxing_old_episode_contract(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'episode';p.write_text(json.dumps(dict(schema=SCHEMA)))
            with patch('go2arx5_vla.world_model.fusion_episode.episode_config',return_value='dedicated')as f:
                self.assertEqual(episode_config(dict(episode_manifest=str(p))),'dedicated');f.assert_called_once()
            p.write_text(json.dumps(dict(schema='UNKNOWN')))
            with self.assertRaises((PermissionError,KeyError)):episode_config(dict(episode_manifest=str(p)))

    def test_original_safety_entry_rejects_bad_full_preview_before_ros_or_source_wait(self):
        from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
        adapter=TargetSequentialAdapter({},'.')
        adapter.fusion_initial_tcp=np.eye(4);adapter.fusion_selected_action=0
        adapter.fusion_install_action_count=0;adapter.base_pose_hold=[0,0,0,0,0,0,1]
        adapter.fusion_base_pose_hold=adapter.base_pose_hold.copy()
        with self.assertRaisesRegex(ValueError,'FULL_FINAL_H40'):
            adapter.install_reference(np.zeros((40,10)))

    def test_decoder_selector_and_native_module_import_do_not_load_torch_or_ros(self):
        code="import sys;import go2arx5_vla.world_model.fusion_contract;import go2arx5_vla.world_model.fusion_corrector;import go2arx5_vla.world_model.fusion_episode;raise SystemExit(1 if 'torch'in sys.modules or 'rclpy'in sys.modules else 0)"
        result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH='src'),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_original_deadline_reserve_and_scale_limit_are_preserved(self):
        v=dict(schema='WM02_FUSION_NIGHT1_SCOPE_V1',plan_sha=PLAN,scope=SCOPE,
            owner='EXISTING_LOCAL_PC',host='corelab-4080s',uid=2009,max_native_slots=2,
            GR00T_updates=0,WBC_changes=False,physics_dt_s=.002,
            native_reference_domain='SINGLE_DECISION_LOCAL_XY4_RAMP8_H16_OPEN_GRIPPER',
            activation_epoch=100,deadline_epoch=2000,parent_deadline_epoch=4000)
        binding_contract(v,101)
        for change in [dict(deadline_epoch=3000),dict(max_native_slots=4),dict(GR00T_updates=1),
                       dict(WBC_changes=True),dict(physics_dt_s=.004),dict(plan_sha='old')]:
            with self.subTest(change=change),self.assertRaises(PermissionError):binding_contract(dict(v,**change),101)
        with self.assertRaises(PermissionError):binding_contract(v,2000)


if __name__=='__main__':unittest.main()
