import copy,tempfile,unittest,time
from pathlib import Path
import numpy as np
from test_pilot1_contact_wm import bundle,save
from scripts.wm02.train_pilot1_contact_policy import dataset,run
from go2arx5_vla.world_model.pilot1_contact_policy_numpy import features
from go2arx5_vla.world_model.storage import digest


def two_actions(split,parent):
    a=bundle(split,parent,128);hold=a['episodes'][0];hold['run_id']=parent+'H';hold['episode']['withdrawal_xyz_m']=[0.,0.,0.];hold['score']=dict(actual_table_support_at_end=True,qualified_contact_exit=False)
    y=copy.deepcopy(hold);y['run_id']=parent+'Y';y['episode']['withdrawal_xyz_m']=[0.,.001,0.];y['score']['qualified_contact_exit']=True
    for i,row in enumerate(y['samples']):
        if i>=64:
            row['TCP_m'][1]=(i-63)*.00001;row['contact_any']=False
            for e in row['native_contact_events']:e.update(contact_positive=False,max_pad_cube_normal_force_N=0.)
    a['episodes'].append(y);return a


class ContactPolicyPilot(unittest.TestCase):
    def test_complete_training_restore_and_no_future_input(self):
        with tempfile.TemporaryDirectory()as tmp:
            r=Path(tmp);values=[two_actions('TRAIN','P'),two_actions('TRAIN','Q'),two_actions('DEV','R')];paths=[save(r,str(i)+'.gz',v)for i,v in enumerate(values)];bundles=[dict(path=str(p),sha256=digest(p))for p in paths]
            train,dev,teacher=dataset(bundles);self.assertEqual(teacher['expected_exit_return'],[0.,1.]);self.assertEqual(len(train),4)
            before=features(values[0]['episodes'][0]['samples'][:64]);values[0]['episodes'][0]['samples'][64]['TCP_m'][0]=999.
            np.testing.assert_array_equal(before,features(values[0]['episodes'][0]['samples'][:64]))
            out=r/'fit';out.mkdir();s=dict(kind='CONTACT_POLICY',device='cpu',cpus=[0],seed=36004,bundles=bundles,endpoint=4,batch_size=4,learning_rate=.001,source_sha='fixture',registered_root=str(r),deadline_epoch=time.time()+60)
            result=run(s,out);self.assertEqual(result['main_Actor_updates'],4);self.assertEqual(result['complete_restore_error'],0);self.assertEqual(result['endpoint_reload_error'],0);self.assertFalse(result['strong_rule_advantage_established'])
            with self.assertRaisesRegex(ValueError,'WHOLE_PHYSICAL_PARENT'):
                dataset(bundles+[dict(path=str(save(r,'leak.gz',two_actions('DEV','P'))))])


if __name__=='__main__':unittest.main()
