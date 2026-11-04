import unittest,tempfile,json,copy
from pathlib import Path
from go2arx5_vla.world_model.target_manifest import episode_limit,validate_night1,model_fingerprint,PLAN_SHA
from go2arx5_vla.world_model.storage import digest

class ManifestContract(unittest.TestCase):
    def test_duration_ladder_and_immutable_scene_identity(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);(p/'model').write_bytes(b'fixture');(p/'binary').write_bytes(b'fixture-binary')
            assets=dict(robot_fingerprint='fp',native_binary_sha256=digest(p/'binary'),scenes=[dict(scene_id='original',model_path=str(p/'model'),model_sha256=digest(p/'model'),parent='p',split='TRAIN',robot_fingerprint='fp',static_check='PASSED_NO_INTEGRATION')]);(p/'assets').write_text(json.dumps(assets))
            e=dict(schema='WM02_NIGHT1_FINITE_EPISODE_V1',run=str(p),phase='collect-target-night1',split='TRAIN',parent='p',scene_id='original',task_atoms=48,purpose='DURATION_SINGLE',reference_rule='PREDECLARED_RAMP8_THEN_FIXED_GOAL_V1',offset_xyz_m=[.001,0,0]);(p/'episode').write_text(json.dumps(e))
            s=dict(registered_root=str(p),episode_manifest=str(p/'episode'),episode_manifest_sha256=digest(p/'episode'),scene_allowlist=str(p/'assets'),scene_allowlist_sha256=digest(p/'assets'),model_path=str(p/'model'),native_binary=str(p/'binary'))
            prev=dict(status='NATIVE_STAGE_PASSED',shutdown_clean=True,owned_process_exits=[dict(exit_code=0)],actual_target_integrations=3103,scene_sha256=digest(p/'model'))
            self.assertEqual(episode_limit(s,prev),3703)
            with self.assertRaises(PermissionError):episode_limit(s,dict(prev,actual_target_integrations=2703))
            (p/'model').write_bytes(b'changed')
            with self.assertRaises(PermissionError):episode_limit(s,prev)
    def test_uncapped_project_budget_retains_real_window_and_registry(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'night';b=dict(plan_sha=PLAN_SHA,owner='EXISTING_LOCAL_PC',campaign_root=tmp,max_native_slots=2,engineering_budget_caps=None,activation_epoch=1793812000,deadline_epoch=1793840400,dispatch_cutoff_epoch=1793839200,original_campaign_deadline_epoch=1794052461);p.write_text(json.dumps(b));s=dict(night1_binding=str(p),night1_binding_sha256=digest(p),campaign_root=tmp,registered_root=tmp+'/run',source_sha='fixture',deadline_epoch=1793812400);r=dict(entries=[dict(path=s['registered_root'],source_sha='fixture',night1_binding=str(p))]);validate_night1(s,r,now=1793812100)
            with self.assertRaises(PermissionError):validate_night1(s,r,now=b['dispatch_cutoff_epoch'])
            with self.assertRaises(PermissionError):validate_night1(s,dict(entries=[]),now=1793812100)
    def test_only_initial_cube_xy_can_change_in_model_fingerprint(self):
        import mujoco
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="task_cube" pos="0 0 .5"><freejoint/><geom type="box" size=".02 .02 .02"/></body></worldbody></mujoco>');first,_=model_fingerprint(m);m.qpos0[0]+=.01;m.qpos0[1]-=.01;self.assertEqual(model_fingerprint(m)[0],first);m.geom_friction[0,0]+=.01;self.assertNotEqual(model_fingerprint(m)[0],first)
    def test_home_layout_survives_binary_save_and_native_keyframe_reset(self):
        import mujoco,numpy as np
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="task_cube" pos="0 0 .5"><freejoint/><geom type="box" size=".02 .02 .02"/></body></worldbody><keyframe><key name="home" qpos="0 0 .5 1 0 0 0"/></keyframe></mujoco>')
        old,_=model_fingerprint(m);new,_=model_fingerprint(m,'HOME_KEYFRAME_XY_V2');m.key_qpos[0,:2]=[.008,-.012]
        self.assertNotEqual(model_fingerprint(m)[0],old);self.assertEqual(model_fingerprint(m,'HOME_KEYFRAME_XY_V2')[0],new)
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'scene.mjb';mujoco.mj_saveModel(m,str(p),None);m=mujoco.MjModel.from_binary_path(str(p));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d);np.testing.assert_allclose(d.xpos[1],[.008,-.012,.5]);self.assertIsInstance(m._address,int);self.assertIsInstance(d._address,int)
        m.qpos0[0]+=.001;self.assertNotEqual(model_fingerprint(m,'HOME_KEYFRAME_XY_V2')[0],new)
if __name__=='__main__':unittest.main()
