"""Synthetic CPU fixtures only: zero real model, teacher, render or physics calls."""
import argparse,json,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import r3_recovery as r


def rejects(call):
    try:call()
    except (AssertionError,ValueError,KeyError,StopIteration):return
    raise AssertionError('Negative fixture was accepted')


class SyntheticCodec:
    identity=r.FROZEN.copy() # Synthetic interface identity fixture, not an actual native processor.
    bad=False
    def encode_actions(self,target,state):return target.copy()
    def decode_actions(self,target,state):
        target=target.copy()
        if self.bad:target[:,0]+=.001
        return target
    def cache_key(self,task,instruction,members):return r.array_hash(np.frombuffer((task+instruction+json.dumps(members,sort_keys=True)).encode(),dtype=np.uint8))


def synthetic_errors(target,decoded):
    return dict(XYZ_max_m=float(abs(target[:,:3]-decoded[:,:3]).max()),SO3_max_rad=0.,
        gripper_max_fraction=float(abs(target[:,9]-decoded[:,9]).max()),nonfinite=False,
        command_rot6d_validity=dict(degenerate_count=0),decoded_rot6d_validity=dict(degenerate_count=0))


def run():
    assert len(r.slots())==len({s['slot'] for s in r.slots()})==32
    assert {s['control_tick'] for s in r.slots()}=={60,140,220,340}
    checked=[]
    slot=r.slots()[0];tick=slot['control_tick']
    physics=dict(time_s=np.arange(tick*5+1)*.01,qpos=np.zeros((tick*5+1,9)))
    observation=dict(state_v1=np.zeros(10,dtype=np.float32),base_camera_raw=np.zeros((128,128,3),dtype=np.uint8),hand_camera_raw=np.zeros((128,128,3),dtype=np.uint8))
    bound=dict(slot=slot,initial={'q':np.zeros(9)},actions=np.zeros((tick,7)),commands=np.zeros((tick,10)),physics=physics,
        observations={k:np.broadcast_to(v,(tick+1,)+v.shape) for k,v in observation.items()},hashes={'synthetic':'NOT_REAL'},state_hash='SYNTHETIC_STATE')
    ep=SimpleNamespace(control=0,seed=slot['layout'],initial={'q':np.zeros(9)},states=[{k:v[0] for k,v in physics.items()}])
    def step(action,phase,command,teacher):
        ep.control+=1
        for n in range(len(ep.states),ep.control*5+1):ep.states.append({k:v[n] for k,v in physics.items()})
    ep.step=step;ep.observe=lambda:observation
    assert r.replay(ep,bound,time.time()+30,lambda v:v)['takeover_tick']==tick
    ep.control=0;ep.states=[{k:v[0] for k,v in physics.items()}];ep.initial['q'][0]=.01
    rejects(lambda:r.replay(ep,bound,time.time()+30,lambda v:v));ep.initial['q'][0]=0
    ep.observe=lambda:dict(observation,state_v1=np.ones(10,dtype=np.float32))
    rejects(lambda:r.replay(ep,bound,time.time()+30,lambda v:v))
    checked+=['synthetic prefix all-sample replay','wrong initial state and branch RGB/state rejected']
    with tempfile.TemporaryDirectory(prefix='PI0285_SYNTHETIC_NOT_PHYSICS_') as temporary:
        root=Path(temporary);folder=root/'raw';folder.mkdir();slot=r.slots()[-1];n=660;tick=slot['control_tick']
        state=np.tile(np.array([0,0,0,1,0,0,0,1,0,1],dtype=np.float32),(n,1))
        images=np.zeros((n,128,128,3),dtype=np.uint8)
        np.savez(folder/'observations.npz',state_v1=state,base_camera_raw=images,hand_camera_raw=images,time_s=np.arange(n)*.05)
        phases=np.minimum(np.arange(n)//132,4);np.savez(folder/'commands.npz',action=state,phase=phases)
        np.savez(folder/'physics.npz',is_cubeA_grasped=np.zeros(n*5+1,dtype=bool),time_s=np.arange(n*5+1)*.01,qpos=np.zeros((n*5+1,9)))
        np.savez(folder/'installed.npz',action=np.zeros((n,7)));np.savez(folder/'initial.npz',q=np.zeros(9))
        restore=dict(status='PASS_REPLAY_SYNTHETIC_ONLY',state_hash='SYNTHETIC_STATE')
        r.write(folder/'RECOVERY_RESTORE.json',restore)
        card=dict(**slot,point='PLACE4096-4096',restore=restore,teacher_sha256='SYNTHETIC_TEACHER',reset_controls=[])
        r.write(folder/'EPISODE_CARD.json',card)
        result=dict(status='VALID_SOURCE_COMPLETE',slot=slot['slot'],persistent_success=True,control_calls=n,
            files={p.name:r.sha(p) for p in folder.iterdir() if p.name!='RESULT.json'})
        r.write(folder/'RESULT.json',result)
        bound=r.prefix(folder,slot);assert len(bound['actions'])==340
        missing=dict(slot,control_tick=700);rejects(lambda:r.prefix(folder,missing))
        codec=SyntheticCodec();report=r.suffix(root/'data',folder,codec,synthetic_errors)
        assert report['status']=='PASS_FRESH_SUFFIX_STRICT_CODEC' and len(report['rows'])==281
        assert min(v['original_control_tick'] for v in report['rows'])==340 and max(v['original_control_tick']+39 for v in report['rows'])==659
        assert all(v['instruction']==r.INSTRUCTIONS['T1'] and v['source']=='corrected_suffix' for v in report['rows'])
        codec.bad=True;bad=r.suffix(root/'bad',folder,codec,synthetic_errors)
        assert bad['status']=='FAIL_SUPERVISION_CODEC_PRESERVE_RAW_STOP_BRANCH' and len(bad['failures'])==281
        assert (root/'bad/action_target'/slot['slot']/'action.npy').exists()
        checked+=['synthetic raw suffix materialization','original timestamp offsets and H40 boundary','synthetic codec fail preserves all raw labels']
        reports=[dict(report,slot=s['slot'],restore=dict(restore,state_hash=f'SYNTHETIC_{i}'),
            rows=[dict(v,episode=s['slot'],uid=f"{s['slot']}-{v['anchor']}") for v in report['rows']]) for i,s in enumerate(r.slots()[:24])]
        good=r.bank(reports,'SYNTHETIC_TEACHER');assert good['valid_branches']==24 and len(good['missing'])==8 and good['requested_branches']==32
        assert r.bank(reports[:23],'SYNTHETIC_TEACHER')['status'].startswith('FAIL')
        assert r.bank(reports+[reports[0]],'SYNTHETIC_TEACHER')['duplicates']
        reports[1]['restore']=reports[0]['restore'];assert r.bank(reports,'SYNTHETIC_TEACHER')['status'].startswith('FAIL')
        extra=dict(reports[-1],slot=r.slots()[24]['slot'],restore=dict(restore,state_hash='SYNTHETIC_EXTRA'),
            rows=[dict(v,episode=r.slots()[24]['slot'],uid=f"EXTRA-{v['anchor']}") for v in reports[-1]['rows']])
        assert r.bank(reports+[extra],'SYNTHETIC_TEACHER')['valid_branches']==24
        assert r.bank(reports+[extra],'SYNTHETIC_TEACHER')['status'].startswith('PASS_ACTUAL')
        assert r.bank([bad],'SYNTHETIC_TEACHER')['codec_failure_stops_branch']
        codec.identity=dict(r.FROZEN,**{'statistics.json':'WRONG'});rejects(lambda:r.suffix(root/'wrong',folder,codec,synthetic_errors))
    checked+=['24/32 admission retains eight missing slots','duplicate ID/state and any supervision codec failure rejected']
    rows=[dict(task='T1',split='TRAIN',layout=285100+i,episode=f'branch{i}',phase=p,anchor=a,uid=f'b{i}:{p}:{a}',
        instruction=r.INSTRUCTIONS['T1'],source='corrected_suffix',event_first8=a==0,event_tail_only=a==1) for i in range(24) for p in range(5) for a in range(4)]
    experts=[dict(v,source='expert',episode='expert'+str(v['layout'])) for v in rows]
    admission=dict(status='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC',rows=rows)
    mixed=r.recovery_plan(experts,admission)
    assert len(mixed)==16384 and sum(v['source']=='corrected_suffix' for v in mixed)==8192
    assert [sum(v['phase']==p for v in mixed) for p in range(5)]==r.quotas(16384,r.WEIGHTS)
    assert mixed==r.recovery_plan(experts,admission)
    partial=[v for v in rows if v['phase']!=4];rejects(lambda:r.correction_plan(partial,8192,r.quotas(8192,r.WEIGHTS)))
    existing=r.plan(experts,['T1'],4096)
    decision=dict(branch='DOSE2048',DOSE_allowed=True,RECOVERY_priority=False)
    dose=r.dose_plan(existing,decision);assert len(dose)==16384 and len({v['existing_plan_position'] for v in dose})==16384
    assert [sum(v['phase']==p for v in dose) for p in range(5)]==r.quotas(16384,r.WEIGHTS)
    rejects(lambda:r.dose_plan(existing,dict(decision,RECOVERY_priority=True)))
    rejects(lambda:r.collection_window(dict(branch='RECOVERY2048',RECOVERY_priority=True),1791496801,2700))
    with tempfile.TemporaryDirectory(prefix='PI0285_SYNTHETIC_GATE_') as temporary:
        saved_root=r.ROOT;saved_r2=sys.modules.get('r2');saved_time=time.time
        try:
            r.ROOT=Path(temporary);sys.modules['r2']=SimpleNamespace(admission=lambda:None)
            time.time=lambda:1791490000.
            decision=dict(branch='RECOVERY2048',RECOVERY_priority=True,status='FROZEN_R2_SINGLE_BRANCH_DECISION_NOT_DATA_OR_TRAINING_ADMISSION',proofs={})
            r.write(r.ROOT/'qualification/R2_A_SINGLE_BRANCH.json',decision)
            r.write(r.ROOT/'qualification/PLACE4096-4096_ROUTE.json',dict(next_action='SELECT_R2_SINGLE_BRANCH_FROM_FROZEN_EVIDENCE'))
            first=r.authorize_collection(2700);time.time=lambda:1791490100.
            assert r.authorize_collection(2700)['deadline_epoch']==first['deadline_epoch']
            time.time=lambda:first['deadline_epoch']+1;rejects(lambda:r.authorize_collection(2700))
            time.time=lambda:1791490100.;r.write(r.ROOT/'qualification/CONFIRM_A_CANDIDATE.json',{})
            rejects(lambda:r.authorize_collection(2700))
        finally:
            r.ROOT=saved_root;time.time=saved_time
            if saved_r2 is None:sys.modules.pop('r2',None)
            else:sys.modules['r2']=saved_r2
    checked+=['16384 draws expert/suffix50/50 equal branch quotas and fixed bucket totals','deterministic event stratum sampler and infeasible coverage rejected',
        'DOSE existing-plan positions only; RECOVERY mutually exclusive','latest collection start enforced','durable45-minute clock cannot reset; confirmation candidate blocks collection']
    return dict(status='PASS_SYNTHETIC_CPU_RECOVERY_COMPONENTS_NOT_PHYSICAL_OR_NATIVE_CODEC_PROOF',checks=checked,
        synthetic_fixtures=True,real_source_arrays_read=0,model_queries=0,physics_calls=0,teacher_calls=0,render_calls=0,optimizer_updates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report');a=p.parse_args();value=run()
    if a.report:r.write(a.report,value)
    print(json.dumps(value))
