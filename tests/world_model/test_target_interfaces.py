import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from go2arx5_vla.world_model.control_snapshot import REQUIRED,save_snapshot,load_snapshot
from go2arx5_vla.world_model.standing_t1 import score_standing_t1
from go2arx5_vla.target_sim.scorer import BOOL_FIELDS


def fixture():
    times=np.arange(641)/20;n=len(times);initial=np.eye(4);initial[:3,3]=[-.16,0,.46]
    obj=np.tile(initial,(n,1,1));grasp=(times>=1)&(times<=4)
    obj[(times>=1.5)&(times<=4),2,3]+=.08;obj[times>=2.5,0,3]=.16;obj[times>4,2,3]=.46
    tcp=np.tile(np.eye(4),(n,1,1));tcp[:,:3,3]=[.16,0,.6]
    d=dict(time_s=times,object_pose=obj,tcp_world=tcp,object_linear_velocity=np.zeros((n,3)),
        object_angular_velocity=np.zeros((n,3)),base_pose=np.tile(np.eye(4),(n,1,1)),native_fullbody_safety_valid=np.ones(n,dtype=bool))
    for k in BOOL_FIELDS:d[k]=np.zeros(n,dtype=bool)
    for k in ('object_present','contact_valid','gripper_state_valid','safety_valid'):d[k][:]=True
    for k in ('native_grasp','left_object_contact','right_object_contact'):d[k]=grasp.copy()
    return d,initial


class TargetOfflineInterfaces(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'state.npz'
        self.identity=dict(backend='GO2_ARX5_MUJOCO_OCS2',frozen_files_sha256={'fixture':'a'*64},source='CPU_SYNTHETIC_NOT_NATIVE')
        self.clock=dict(native_tick=250,time_s=.5,physics_dt_s=.002)
        self.blocks={b:{f:np.arange(4,dtype=np.float64)for f in fields}for b,fields in REQUIRED.items()}
        self.blocks['rng']['generator_state']=np.arange(10,dtype=np.uint8)
    def tearDown(self):self.temp.cleanup()

    def test_offline_full_container_roundtrip_preserves_exact_types(self):
        save_snapshot(self.path,self.blocks,self.identity,self.clock)
        restored,meta=load_snapshot(self.path,self.identity)
        for b,fields in self.blocks.items():
            for f,a in fields.items():
                self.assertEqual(restored[b][f].dtype,a.dtype);np.testing.assert_array_equal(restored[b][f],a)
        self.assertFalse(meta['native_exporter_bound']);self.assertFalse(meta['live_restore_performed'])

    def test_missing_controller_or_object_payload_rejected(self):
        del self.blocks['mpc']['warmstart']
        with self.assertRaises(ValueError):save_snapshot(self.path,self.blocks,self.identity,self.clock)
        self.blocks['mpc']['warmstart']=np.array([object()],dtype=object)
        with self.assertRaises(ValueError):save_snapshot(self.path,self.blocks,self.identity,self.clock)

    def test_snapshot_identity_and_tampering_rejected(self):
        save_snapshot(self.path,self.blocks,self.identity,self.clock)
        with self.assertRaises(ValueError):load_snapshot(self.path,dict(self.identity,source='other'))
        with np.load(self.path,allow_pickle=False)as f:payload={k:f[k].copy()for k in f.files}
        payload['mpc.solver'][0]+=1
        with self.path.open('wb')as f:np.savez_compressed(f,**payload)
        with self.assertRaises(ValueError):load_snapshot(self.path,self.identity)

    def test_versioned_standing_score_positive_and_physical_counterexamples(self):
        d,initial=fixture()
        def score(x):return score_standing_t1(x,goal_xy=[.16,0],initial_object_pose=initial)
        result=score(d);self.assertTrue(result['persistent_success']);self.assertFalse(result['native_qualified'])
        for kind in ['lost_grasp','not_released','collision','not_retreated','wholebody_unsafe']:
            x=copy.deepcopy(d)
            if kind=='lost_grasp':x['native_grasp'][:]=False
            if kind=='not_released':x['native_grasp'][560]=True
            if kind=='collision':x['illegal_contact'][620]=True
            if kind=='not_retreated':x['tcp_world'][560,:3,3]=[.16,0,.46]
            if kind=='wholebody_unsafe':x['native_fullbody_safety_valid'][50]=False
            with self.subTest(kind=kind):self.assertFalse(score(x)['persistent_success'])
        self.assertTrue(score(d)['historical_fullbody_BASE_translation_score_changed']is False)


if __name__=='__main__':unittest.main()
