import copy,json,tempfile,unittest,struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from go2arx5_vla.world_model.pilot1_supported_close import PHASE,RULE,STAGES,stage_for,score,validate,run,GraspReader
from go2arx5_vla.world_model.storage import digest


def actual_events():
    rows=[]
    for step in range(2504,6704):
        stage=stage_for((step-2504)//25);grasp=stage in('CLOSE','SUPPORTED_HOLD');released=stage in('RELEASE','OPEN_HOLD')
        rows.append(dict(native_step=step,stage=stage,native_grasp=grasp,table_support=True,contact_positive=grasp,finger_q=[.042,.042]if released else[.018,.018]))
    return rows


class SupportedClosePilot(unittest.TestCase):
    def test_existing_attachment_is_only_a_planning_hypothesis_after_actual_grasp(self):
        from go2arx5_vla.world_model.target_sequential import supported_object_hypothesis
        sample=dict(native_grasp=True,tcp=np.eye(4),object_pose=np.eye(4))
        motion=supported_object_hypothesis(PHASE,'close',sample)
        self.assertEqual(motion.mode,'ACTUAL_GRASP_ATTACHMENT');self.assertFalse(motion.evidence()['actual_physics_or_labels_modified'])
        for p,stage,value in [(PHASE,'close',False),(PHASE,'approach',True),('collect-target-pilot1','close',True)]:
            self.assertIsNone(supported_object_hypothesis(p,stage,dict(sample,native_grasp=value)))
        np.testing.assert_array_equal(sample['tcp'],np.eye(4));np.testing.assert_array_equal(sample['object_pose'],np.eye(4))
    def test_reader_uses_original_grasp_updater_not_two_contact_shortcut(self):
        def packet(step,nan=False):
            raw=struct.pack('<8sQdIII',b'GRTSTATE',step,step*.002,34,32,20)+np.zeros(86,dtype='<f8').tobytes()+struct.pack('<I',3)
            for g,f in [(7,.1),(8,.1),(1,.5)]:
                values=np.zeros(18,dtype='<f8');values[0]=float('nan')if nan else f;values[6]=1.
                raw+=struct.pack('<IId',g,9,-.0001)+values.tobytes()
            return struct.pack('<I',len(raw))+raw
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'journal';p.write_bytes(b''.join(packet(i)for i in range(1,26)))
            calls=[]
            def update(row):
                calls.append(row['step']);return dict(native_grasp=False,opposing_contact=True,relative_motion_valid=False,safety_valid=True,finger_q=np.array([.02,.02]))
            actual=SimpleNamespace(model=None,pad=[7,8],cube_geom=9,update=update)
            with patch('mujoco.mj_name2id',return_value=1):reader=GraspReader(p,actual)
            events=reader.interval(0,25);self.assertEqual(calls,list(range(1,26)));self.assertTrue(all(e['contact_positive']and e['table_support']for e in events));self.assertFalse(any(e['native_grasp']for e in events));self.assertAlmostEqual(events[0]['contact_time_s'],0.)
            p.write_bytes(packet(1,True))
            with patch('mujoco.mj_name2id',return_value=1):reader=GraspReader(p,actual)
            with self.assertRaisesRegex(ValueError,'FINITE_ORIGINAL_CONTACT'):reader.interval(0,1)
    def test_hold_and_release_require_complete_native_duration_and_actual_predicates(self):
        rows=actual_events();v=score(rows);self.assertTrue(v['qualified_supported_close_hold_release']);self.assertEqual(v['longest_continuous_supported_grasp_native_ticks'],600)
        for key in ['native_grasp','table_support']:
            bad=copy.deepcopy(rows);bad[5203-2504][key]=False
            self.assertFalse(score(bad)['qualified_supported_grasp_hold'])
        bad=copy.deepcopy(rows)
        for r in bad:
            if r['native_step']>4903:r['stage']='EARLY_RELEASE'
        self.assertFalse(score(bad)['qualified_supported_grasp_hold'])
        bad=copy.deepcopy(rows)
        for r in bad:
            if r['native_step']<=6604:r['contact_positive']=True
        self.assertFalse(score(bad)['qualified_release'])
        bad[-100]['contact_positive']=False
        self.assertTrue(score(bad)['qualified_release'])
        with self.assertRaises(ValueError):score(rows[:-1])

    def test_single_qualified_scene_and_exact_original_safe_protocol_required(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            def save(n,v):
                p=root/n;p.write_text(json.dumps(v));return str(p)
            model=save('model',{});binary=save('binary',{});proof=save('RESULT.json',dict(status='NATIVE_STAGE_PASSED',shutdown_clean=True,physics_stops=[],scene_sha256=digest(model),native_binary_sha256=digest(binary)));save('STEADY_PREFIX_RESULT.private.json',dict(qualified=True))
            assets=save('assets',dict(native_binary_sha256=digest(binary),scenes=[dict(scene_id='C',parent='P',split='TRAIN',model_path=model,model_sha256=digest(model))]));episode=dict(schema='WM02_PILOT1_SUPPORTED_CLOSE_V1',phase=PHASE,reference_rule=RULE,purpose='SUPPORTED_CLOSE_COMMISSION',split='TRAIN',task_atoms=168,prehold_atoms=40,screen_atom=48,close_logical_opening=.4,required_supported_grasp_native_ticks=500,required_released_native_ticks=100,minimum_open_finger_q_m=.040,contact_force_threshold_N=.05,EEF_offset_xyz_m=[0.,0.,0.],stages=[list(s)for s in STAGES],run=str(root),scene_id='C',parent='P');path=save('episode',episode)
            s=dict(episode_manifest=path,episode_manifest_sha256=digest(path),registered_root=str(root),pilot1_binding='actual',scene_allowlist=assets,scene_allowlist_sha256=digest(assets),model_path=model,native_binary=binary,scene_qualification=proof,scene_qualification_sha256=digest(proof));previous=json.loads(Path(proof).read_text())
            self.assertEqual(validate(s,previous),6703)
            with self.assertRaises(PermissionError):validate(dict(s,parallel_batch_spec='two-slots-not-qualified'),previous)
            episode['close_logical_opening']=0.;Path(path).write_text(json.dumps(episode));s['episode_manifest_sha256']=digest(path)
            with self.assertRaises(PermissionError):validate(s,previous)

    def test_full_same_process_schedule_and_safe_early_release_record_actual_actions(self):
        for grasp in (True,False):
            with self.subTest(grasp=grasp),tempfile.TemporaryDirectory()as tmp:
                root=Path(tmp);e=root/'episode';e.write_text(json.dumps(dict(reference_rule=RULE,screen_atom=48,close_logical_opening=.4,task_atoms=168)));a=SimpleNamespace(output=root,spec=dict(phase=PHASE,max_ticks=6703,episode_manifest=str(e),source_sha='fixture'),step=2503,actual_state=SimpleNamespace(model=None,cube_geom=9),prepare_dynamic=lambda:None,capture_loaded_identity=lambda:None,command_event=lambda x:None,cancel_reference=lambda:{},seal_trace=lambda n:None)
                a.observe=lambda:dict(step=a.step,contacts=[dict(geom0=1,geom1=9,wrench=[1.])])
                a.measured_task_state=lambda:dict(native_step=a.step,time_s=a.step*.002,tcp=np.eye(4),object_pose=np.eye(4),qpos=np.zeros(34),qvel=np.zeros(32),ctrl=np.zeros(20),finger_q=np.array([.042,.042]),native_grasp=grasp)
                a.install_reference=lambda action:setattr(a,'active_reference',dict(sequence=0,action_EEF10=action.tolist()))
                def advance(adapter,installed):a.step+=25;return dict(reference_sha256='fixture',solver_policies_with_exact_installed_target=[{}])
                def interval(start,end):
                    return [dict(native_step=s,contact_positive=grasp,native_grasp=grasp,table_support=True,finger_q=[.042,.042])for s in range(start+1,end+1)]
                with patch('go2arx5_vla.world_model.pilot1_supported_close.GraspReader')as reader,patch('go2arx5_vla.world_model.next2_contact.advance_atom',side_effect=advance),patch('go2arx5_vla.world_model.next2_contact.gripper_memory',side_effect=lambda _:dict(values=dict(steps=a.step,sequence=0))),patch('go2arx5_vla.world_model.next1_episode.assess',return_value=dict(qualified=True)):
                    reader.return_value.interval.side_effect=interval;reader.return_value.table=1;v=run(a)
                result=json.loads((root/'SUPPORTED_CLOSE_RESULT.private.json').read_text());self.assertEqual(a.step,6703);self.assertEqual(v['complete_task_atoms'],168)
                self.assertEqual(result['samples'][64]['stage'],'CLOSE')
                self.assertEqual(result['samples'][96]['stage'],'SUPPORTED_HOLD'if grasp else'EARLY_RELEASE')
                self.assertGreater(result['samples'][120]['executed_reference_EEF10'][-1],.95)
                self.assertEqual(result['first_blocker'],None if grasp else'NO_CURRENT_SUPPORTED_NATIVE_GRASP_FOR_HOLD')


if __name__=='__main__':unittest.main()
