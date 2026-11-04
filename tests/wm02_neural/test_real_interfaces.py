"""Contract/causality tests; generated fixtures are not native evidence."""
import copy,gzip,json,pickle,tempfile,unittest
from pathlib import Path
import numpy as np
try:import torch
except ModuleNotFoundError as e:
    if e.name!='torch':raise
    raise unittest.SkipTest('Isolated E-G Torch required')
from scipy.spatial.transform import Rotation
from go2arx5_vla.world_model import replay
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.neural_contract import compile_physical,actor_primitives,tracking_residual
from go2arx5_vla.world_model.action_compiler import compile_reference,SOURCE,TARGET
from go2arx5_vla.world_model.contracts import FRAME,TCP
from go2arx5_vla.world_model.neural import WorldModel
from go2arx5_vla.world_model.real_training import loss,read_numeric,rollout,repair_legacy_metadata
from go2arx5_vla.world_model.neural_training import checkpoint,load_checkpoint,tree_error


def fixture_rows(group):
    """Handmade serialization fixture, deliberately not a simulated trajectory."""
    pose=np.eye(4);pose[:3,3]=[.4,0,.5];rows=[]
    ref=np.tile(np.r_[pose[:3,3],pose[:2,:3].flatten(),.4],(40,1))
    def state(i):
        return dict(time_s=i*.05,tcp=pose.copy(),object_pose=pose.copy(),arm_q=np.zeros(6),arm_qd=np.zeros(6),
            tool_linear_velocity=np.zeros(3),tool_angular_velocity=np.zeros(3),gripper_fraction=.4,gripper_speed=0.,
            object_linear_velocity=np.zeros(3),object_angular_velocity=np.zeros(3),qpos=np.full(15,i*.001),qvel=np.zeros(14),reset=False,
            left_object_contact=False,right_object_contact=False,native_grasp=False,illegal_contact=False)
    for i in range(24):
        point=i%8
        rows.append(dict(run_id='unit',episode_id='episode'+group,parent_id='parent'+group,scene_id='scene'+group,split='TRAIN',
            executor_id='unit_not_native',segment_id=0,frame=FRAME,tcp=TCP,actor_version='unit',t0=i*.05,t1=(i+1)*.05,physics_ticks=25,
            action_raw=np.r_[np.zeros(6),.4],reference_compiled=ref.copy(),installed_H40=ref.copy(),reference_installed=ref[point].copy(),
            reference_consumed=[dict(time_s=i*.05+k*.002,eef10=ref[point].copy())for k in range(25)],
            state_before=state(i),state_after=state(i+1),controller_memory={},native_before=dict(integration=np.array([i*.05])),
            native_after=dict(integration=np.array([(i+1)*.05])),context=dict(goal_xyz=[.5,0,.46],table_top_m=.44,cube_edge_m=.04,model_sha256=group*64),
            terminal=False,truncated=False,safety_reason=None,valid_state_target=True,control_restore_ref={'unit':True},planned_dt=.05,actual_dt=.05,
            native_progress=dict(point=point,forecast_index=(point+1)*25),domain='S_FIXED_BASE_DIAGNOSTIC'))
    return rows


class RealInterfaces(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def manifest(self,streams):
        entries=[]
        for i,rows in enumerate(streams):
            p=self.root/f'rows{i}.pkl.gz'
            with gzip.open(p,'wb')as f:pickle.dump(dict(rows=rows,events=[]),f)
            entries.append(dict(path=str(p),sha256=digest(p),stream=str(i),prefix=None,kind='unit'))
        p=self.root/'manifest.json';p.write_text(json.dumps(dict(schema='WM02_TRUSTED_SOURCE_REPLAY_MANIFEST_V1',trusted_project_pickle=True,expected_raw_transitions=sum(map(len,streams)),files=entries)));return p
    def test_compiler_nonidentity_absolute_time_full_H40_and_wire(self):
        anchor=np.eye(4);anchor[:3,:3]=Rotation.from_euler('xyz',[.4,-.3,.7]).as_matrix();anchor[:3,3]=[.3,-.1,.6]
        logits=torch.linspace(-3,3,56,dtype=torch.float64).reshape(1,8,7).requires_grad_();raw=actor_primitives(logits,target=True)
        rows,times=compile_physical(raw,torch.tensor(anchor)[None],torch.tensor([19.35],dtype=torch.float64),target=True)
        native=compile_reference(raw.detach().numpy()[0],anchor,19.35,backend=TARGET,frame='world',tcp='x5_ee')
        np.testing.assert_allclose(rows.detach().numpy()[0],native['rows10'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(times[0].numpy(),native['timestamps_s'],atol=1e-12)
        self.assertEqual(len(native['wire_document']['ee']['waypoints']),41)
        self.assertEqual(native['wire_document']['ee']['waypoints'][-1]['time'],2.)
        self.assertTrue(torch.equal(rows[:,8:],rows[:,7:8].expand(-1,32,-1)))
        residual=tracking_residual(torch.tensor(anchor)[None],rows[:,0]);self.assertGreater(float(residual.norm().detach()),0)
        (rows[:,:,:3].sum()+rows[:,:,3:9].sum()+residual.square().sum()).backward();self.assertTrue(torch.isfinite(logits.grad).all());self.assertGreater(float(logits.grad.norm()),0)
    def test_target_gripper_endpoints_and_physical_bounds_rejected(self):
        anchor=torch.eye(4,dtype=torch.float64)[None];raw=actor_primitives(torch.zeros(1,8,7,dtype=torch.float64),target=True)
        for value in (0.,1.):
            bad=raw.clone();bad[...,6]=value
            with self.assertRaises(ValueError):compile_physical(bad,anchor,2.,target=True)
        bad=raw.clone();bad[...,0]=.01001
        with self.assertRaises(ValueError):compile_physical(bad,anchor,2.,target=True)
        for v in (.002/.044,.042/.044):
            edge=raw.clone();edge[...,6]=v;compile_physical(edge,anchor,2.,target=True)
        bad_anchor=anchor.clone();bad_anchor[:,0,0]=2
        with self.assertRaises(ValueError):compile_physical(raw,bad_anchor,2.,target=True)
        for value in (-80.,80.):
            physical=actor_primitives(torch.full((1,8,7),value,dtype=torch.float32),target=True)
            native=compile_reference(physical.numpy()[0],np.eye(4),12.,backend=TARGET,frame='world',tcp='x5_ee')
            self.assertEqual(native['rows10'].shape,(40,10))
    def test_replay_deduplicates_aliases_and_holds_out_whole_physical_scene(self):
        a=fixture_rows('a');alias=copy.deepcopy(a)
        for r in alias:r.update(episode_id='renamed',parent_id='renamed',scene_id='renamed')
        manifest=self.manifest([a,alias,fixture_rows('b')]);report=replay.build(manifest,self.root/'replay')
        self.assertEqual(report['raw_record_count'],72);self.assertEqual(report['unique_observed_state_command_time_pairs'],48)
        self.assertEqual(report['continuous_windows'],18);self.assertEqual(report['fit_windows'],9);self.assertEqual(report['holdout_windows'],9)
        data,fit,hold,identity=read_numeric(self.root/'replay/REPLAY.npz',torch.device('cpu'))
        self.assertEqual(data['state'].shape[-1],50);self.assertNotIn('base_position',dict(replay.GROUPS))
    def test_tamper_raw_consumption_reset_and_terminal_boundary(self):
        rows=fixture_rows('a');manifest=self.manifest([rows]);p=Path(json.loads(manifest.read_text())['files'][0]['path']);p.write_bytes(p.read_bytes()+b'X')
        with self.assertRaises(ValueError):replay.build(manifest,self.root/'tamper')
        for field in ('raw','consumed','reset'):
            row=copy.deepcopy(rows[0])
            if field=='raw':row['action_raw'][0]=.001
            elif field=='consumed':row['reference_consumed'][0]['eef10'][0]+=.001
            else:row['state_after']['reset']=True
            with self.assertRaises(ValueError):replay.aligned(row)
        rows[7]['terminal']=True;self.assertFalse(replay.continuous(rows[7],rows[8]))
        with self.assertRaises(ValueError):replay.build(self.manifest([rows]),self.root/'postterminal')
        rows[7]['terminal']=False;rows[8]['state_before']['qpos'][0]+=.1;self.assertFalse(replay.continuous(rows[7],rows[8]))
    def test_source_checkpoint_and_no_post_terminal_or_future_state_leak(self):
        p=self.manifest([fixture_rows('a'),fixture_rows('b')]);replay.build(p,self.root/'built')
        data,fit,hold,identity=read_numeric(self.root/'built/REPLAY.npz',torch.device('cpu'))
        self.assertTrue(all(type(x)is str for x in identity['fit_groups']+identity['holdout_groups']))
        b={k:v[:2].clone()for k,v in data.items()};model=WorldModel(hidden=(32,32),history_hidden=16,groups=replay.GROUPS,context_dim=5)
        model.normalizer.fit(data['state'][torch.as_tensor(fit)])
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5);loss(model,b).backward();opt.step()
        cfg=dict(hidden=[32,32],history_hidden=16,groups=replay.GROUPS,context_dim=5);checkpoint(self.root/'cp.pt',model,opt,1,256,identity,cfg)
        clone,co,c=load_checkpoint(self.root/'cp.pt','cpu',identity);self.assertEqual(tree_error(model.state_dict(),clone.state_dict()),0)
        b['done'][:,10]=1;b['valid'][:,11:]=0;other={k:v.clone()for k,v in b.items()};other['state'][:,12:]+=100
        self.assertEqual(float(loss(model,b).detach()),float(loss(model,other).detach()))
        other={k:v.clone()for k,v in b.items()};other['state'][:,9:]+=100
        x=rollout(model,b,8)[0];y=rollout(model,other,8)[0];self.assertTrue(torch.equal(x,y))

    def test_known_numpy_string_metadata_repair_keeps_strict_loader_and_tensors(self):
        model=WorldModel(hidden=(16,),history_hidden=8,groups=replay.GROUPS,context_dim=5)
        opt=torch.optim.AdamW(model.parameters());cfg=dict(hidden=[16],history_hidden=8,groups=replay.GROUPS,context_dim=5)
        identity={'fit_groups':[np.str_('original-scene')]};expected={'fit_groups':['original-scene']}
        checkpoint(self.root/'legacy.pt',model,opt,1,256,identity,cfg)
        with self.assertRaises(Exception):torch.load(self.root/'legacy.pt',weights_only=True)
        proof=repair_legacy_metadata(self.root/'legacy.pt',self.root/'fixed.pt',digest(self.root/'legacy.pt'),expected)
        self.assertEqual(proof['numpy_metadata_scalars_normalized'],1)
        restored,ro,c=load_checkpoint(self.root/'fixed.pt','cpu',expected)
        self.assertEqual(tree_error(model.state_dict(),restored.state_dict()),0)
        self.assertTrue((self.root/'legacy.pt').exists())

if __name__=='__main__':unittest.main()
