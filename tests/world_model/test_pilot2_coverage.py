import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from go2arx5_vla.world_model.pilot2_coverage import PLAN,coverage_contract,choose_predeclared
from go2arx5_vla.world_model.pilot1_episode import episode_config,OFFSETS
from go2arx5_vla.world_model.next1_episode import RULE
from go2arx5_vla.world_model.pilot1_numpy import goal_offsets
from go2arx5_vla.world_model.storage import digest


class Pilot2Coverage(unittest.TestCase):
    def episode(self):
        return dict(schema='WM02_PILOT2_COVERAGE_EPISODE_V1',purpose='ACTION_COVERAGE',split='TRAIN',
            method='PREDECLARED_ACTION',forced_action=3,repeat_index=1,
            selection_source='PREREGISTERED_RANDOMIZED_ACTION_MANIFEST',scene_id='synthetic',task_goal_offset_xy_m=[.0001,0.])

    def test_new_scope_and_predeclared_qualified_discrete_actions_only(self):
        e=self.episode();spec=dict(pilot2_plan_sha=PLAN);self.assertEqual(coverage_contract(spec,e),e)
        for change in [dict(forced_action=4),dict(forced_action=1.0),dict(repeat_index=2),dict(split='TEST'),dict(selection_source='TEACHER'),dict(purpose='EVALUATION')]:
            with self.subTest(change=change),self.assertRaises(PermissionError):coverage_contract(spec,dict(e,**change))
        with self.assertRaises(PermissionError):coverage_contract({},e)

    def test_actual_choice_saved_before_install_without_loading_actor_or_teacher(self):
        with tempfile.TemporaryDirectory()as tmp:
            r=Path(tmp);events=[];sample=dict(tcp=np.eye(4));s=dict(pilot2_plan_sha=PLAN,source_sha='synthetic',task_sha256='synthetic')
            a=SimpleNamespace(spec=s,output=r,wait_source=lambda:None,observe=lambda:dict(step=3703,time_s=7.406),
                latest=dict(ee=SimpleNamespace(time=7.406)),active_reference=dict(action_EEF10=np.zeros((40,10))),
                base_pose_hold=[0,0,0,0,0,0,1],next1_samples=[sample]*9,measured_task_state=lambda:sample,command_event=events.append)
            with patch('go2arx5_vla.world_model.next2_episode.state_vector',return_value=np.zeros(106)),patch('numpy.load',side_effect=AssertionError('No teacher/model loading allowed')):
                np.testing.assert_array_equal(choose_predeclared(a,self.episode()),OFFSETS[3])
            d=json.loads((r/'PILOT1_DECISION.private.json').read_text());self.assertEqual(d['selected_action'],3)
            self.assertFalse(d['future_response_observed']);self.assertIsNone(d['predictions'])
            self.assertEqual(events[0]['phase'],'pilot1-choice-before-install')

    def test_DEV_coverage_contract_does_not_relax_historical_episode_schema(self):
        with tempfile.TemporaryDirectory()as tmp:
            r=Path(tmp)
            def save(n,v):
                p=r/n;p.write_text(json.dumps(v));return str(p)
            model=save('model',{});binary=save('binary',{});actor=save('actor',{});arx=save('ARX',{})
            task_value=dict(goal_seed=36001,jitter_m=.0002,goals_per_center=64,mean_response_xy_m=[[0,0],[.001,0],[-.001,0],[0,.001]])
            task=save('task',task_value);assets=save('assets',dict(native_binary_sha256=digest(binary),scenes=[dict(scene_id='synthetic',parent='P',split='DEV',model_path=model,model_sha256=digest(model),static_check='PASSED_NO_INTEGRATION')]))
            e=dict(self.episode(),run=str(r),phase='collect-target-pilot1',purpose='SCENE_QUALIFICATION',split='DEV',
                forced_action=0,parent='P',reference_rule=RULE,task_atoms=64,prehold_atoms=40,screen_atom=48,
                screen=dict(window_s=.4,relative_TCP_diameter_m=.002,max_relative_TCP_speed_m_s=.025),offset_xyz_m=[0,0,0],
                goal_index=0,task_goal_offset_xy_m=goal_offsets(task_value)[0].tolist())
            path=save('episode',e);s=dict(pilot2_plan_sha=PLAN,episode_manifest=path,episode_manifest_sha256=digest(path),registered_root=str(r),pilot1_binding='synthetic',scene_allowlist=assets,scene_allowlist_sha256=digest(assets),model_path=model,native_binary=binary)
            for k,p in [('task',task),('actor_numpy',actor),('ARX',arx)]:s[k]=p;s[k+'_sha256']=digest(p)
            self.assertEqual(episode_config(s)['split'],'DEV')
            e['schema']='WM02_PILOT1_EPISODE_V1';Path(path).write_text(json.dumps(e));s['episode_manifest_sha256']=digest(path)
            with self.assertRaises(PermissionError):episode_config(s)


if __name__=='__main__':unittest.main()
