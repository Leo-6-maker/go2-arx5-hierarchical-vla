import unittest,tempfile,json
from pathlib import Path
from go2arx5_vla.world_model.next1_episode import episode_config,episode_limit,RULE
from go2arx5_vla.world_model.storage import digest


class Next1Episode(unittest.TestCase):
    def test_fixed_small_action_and_stability_screen_cannot_be_relaxed(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);(p/'binary').write_bytes(b'binary');(p/'model').write_bytes(b'model')
            assets=dict(native_binary_sha256=digest(p/'binary'),scenes=[dict(scene_id='P01',parent='physical1',split='TRAIN',model_path=str(p/'model'),model_sha256=digest(p/'model'))]);(p/'assets').write_text(json.dumps(assets))
            e=dict(schema='WM02_NEXT1_EPISODE_V1',run=str(p),phase='collect-target-next1',purpose='ACTION_PILOT',reference_rule=RULE,task_atoms=64,prehold_atoms=40,screen_atom=48,split='TRAIN',parent='physical1',scene_id='P01',offset_xyz_m=[.001,0,0],screen=dict(window_s=.4,relative_TCP_diameter_m=.002,max_relative_TCP_speed_m_s=.025))
            s=dict(registered_root=str(p),episode_manifest=str(p/'episode'),scene_allowlist=str(p/'assets'),scene_allowlist_sha256=digest(p/'assets'),model_path=str(p/'model'),native_binary=str(p/'binary'))
            def check(value):
                (p/'episode').write_text(json.dumps(value));s['episode_manifest_sha256']=digest(p/'episode');return episode_config(s)
            check(e);self.assertEqual(episode_limit(s,dict(status='NATIVE_STAGE_PASSED',shutdown_clean=True,native_binary_sha256=digest(p/'binary'))),4103)
            for change in [dict(offset_xyz_m=[.003,0,0]),dict(prehold_atoms=0),dict(screen_atom=8),dict(screen=dict(e['screen'],relative_TCP_diameter_m=.003)),dict(schema='WM02_NIGHT1_LATE_DECISION_EPISODE_V3')]:
                with self.assertRaises(PermissionError):check(dict(e,**change))
            with self.assertRaises(PermissionError):check(dict(e,purpose='STEADY_SINGLE'))


if __name__=='__main__':unittest.main()
