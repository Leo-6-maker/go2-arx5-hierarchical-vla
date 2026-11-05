import copy,gzip,json,tempfile,unittest
from pathlib import Path
import numpy as np
from go2arx5_vla.world_model.pilot1_contact_wm import replay


def bundle(split,parent,count=22):
    rows=[]
    for i in range(count):
        start=2503+25*i;pose=np.eye(4);pose[0,3]=i*.00001
        positive=i%5<3
        events=[dict(native_step=step,contact_positive=positive,max_pad_cube_normal_force_N=.1 if positive else 0.)for step in range(start+1,start+26)]
        rows.append(dict(native_start_step=start,native_end_step=start+25,qpos=[0.]*34,qvel=[0.]*32,ctrl=[0.]*20,
                         TCP_m=[i*.00001,0,0],object_pose=pose.tolist(),finger_q=[.042,.042],contact_any=positive,
                         native_contact_events=events,executed_reference_EEF10=[.001 if i>=9 else 0.,0,0,1,0,0,0,1,0,.95],
                         actual_gripper_internal_memory=dict(values=dict(target=[.042,.042],opening=.95,expiry=(start+25)*.002+.4))))
    e=dict(run_id=split,parent=parent,split=split,episode=dict(task_atoms=len(rows)),samples=rows)
    return dict(schema='WM02_PILOT1_ACTUAL_CONTACT_BUNDLE_V1',complete_MPC_WBC_hidden_state=False,episodes=[e])


def save(root,name,value):
    path=root/name;path.write_bytes(gzip.compress(json.dumps(value).encode(),mtime=0));return path


class ContactCausalDataset(unittest.TestCase):
    def test_future_achieved_state_and_phase_cannot_change_prefix_inputs(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);a=bundle('TRAIN','P');p=save(root,'a.gz',a);before=replay([p])
            a['episodes'][0]['samples'][9]['qpos'][0]=99.
            a['episodes'][0]['samples'][9]['TCP_m'][0]=10.
            a['episodes'][0]['samples'][9]['stage']='future-label-must-not-be-an-input'
            after=replay([save(root,'b.gz',a)])
            np.testing.assert_array_equal(before['x'][0],after['x'][0])
            self.assertFalse(np.array_equal(before['delta_mm'][0],after['delta_mm'][0]))
            np.testing.assert_array_equal(before['x'][0,:-10],before['blind'][0,:-10])
            self.assertNotEqual(before['x'][0,-10],before['blind'][0,-10])

    def test_parent_leak_and_fabricated_contact_label_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);a=bundle('TRAIN','P');b=bundle('DEV','P')
            with self.assertRaisesRegex(ValueError,'WHOLE_PHYSICAL_PARENT'):
                replay([save(root,'a.gz',a),save(root,'b.gz',b)])
            a['episodes'][0]['samples'][0]['native_contact_events'][0]['contact_positive']=False
            with self.assertRaisesRegex(ValueError,'ORIGINAL_FORCE_THRESHOLD'):
                replay([save(root,'c.gz',a)])

    def test_actual_training_seam_saves_and_restores_complete_state(self):
        from scripts.wm02.train_pilot1_contact_wm import run
        from go2arx5_vla.world_model.storage import digest
        import time
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);paths=[save(root,'train.gz',bundle('TRAIN','P')),save(root,'dev.gz',bundle('DEV','Q'))]
            out=root/'fit';out.mkdir()
            spec=dict(source_sha='fixture-not-native',device='cpu',kind='CONTACT_WM',cpus=[0],
                      bundles=[dict(path=str(p),sha256=digest(p))for p in paths],seed=36003,hidden=8,
                      learning_rate=.001,batch_size=4,endpoint=4,deadline_epoch=time.time()+60,registered_root=str(root))
            result=run(spec,out)
            self.assertEqual(result['main_WM_updates'],8)
            self.assertEqual(result['restore_probe_optimizer_calls'],2)
            self.assertEqual(result['TRAIN_parents'],['P']);self.assertEqual(result['DEV_parents'],['Q'])
            self.assertFalse(result['free_rollout_qualified'])
            for arm in result['arms'].values():
                self.assertEqual(arm['full_restore_error'],0)
                self.assertEqual(arm['endpoint_reload_error'],0)


if __name__=='__main__':unittest.main()
