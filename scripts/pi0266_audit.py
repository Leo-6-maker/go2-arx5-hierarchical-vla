"""Fixed 32 TRAIN / 16 saved observation audit; no dynamics or model construction."""
import copy
import os
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pi0266_common import ROOT,BASE,OLD,RAW,DATA,REF,AUTHORITY,read,sha,write,verify_sources

os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1',
    HF_HOME=str(BASE/'PI0250/runtime/hf_home'))
sys.path[:0]=[str(DATA/'R3'),str(DATA/'runtime'),str(DATA/'qa'),str(REF/'scripts'),str(REF/'src'),str(BASE.parent/'isaac-gr00t'),str(BASE/'PI0256/runtime')]
from pi0260_r3_inputs import batch,build,identity
import pi0260_r2_codec as codec
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics
from pi0266_expert import self_check as expert_check
from pi0266_metrics import self_check as metric_check,so3
from pi0258_kinematic import instructions


def main():
    import torch
    torch.set_num_threads(1)
    source=verify_sources();card=read(OLD/'design/SCENE_CARD.json');reg=card['registry']
    stats=read(DATA/'dataset/statistics.json');proc=codec.processor(stats)
    full=build(stats,BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    anchor=np.asarray(read(DATA/'design/registry.json')['initial_TCP_world'])[:3,3]
    kin=NativeArmKinematics.from_urdf(next(row['path'] for row in source['files'] if row['name']=='native_urdf'))
    selected=read(DATA/'dataset/codec_receipt.json')['selected_post_codec']['TRAIN']
    chosen=[];audits=[];ranges=[]
    for task in ('M1','M3'):
        end=6. if task=='M1' else 16.
        for command in (0,1):
            buckets={k:[] for k in range(5)}
            for family in sorted(selected):
                name=f'{family}-{task}-c{command}';rec=read(DATA/'dataset'/(name+'.json'))
                assert rec['sequence']['split']=='TRAIN'
                for row in rec['rows']:
                    if row['usable']:
                        k=min(4,int(row['time']/end*4))
                        buckets[k].append((family,row['time'],row['index'],name,row))
            for bucket,n in ((0,2),(1,2),(2,2),(3,1),(4,1)):
                candidates=sorted(buckets[bucket]);assert len(candidates)>=n,(task,command,bucket)
                for family,t,index,name,row in candidates[:n]:chosen.append((task,command,bucket,name,row))
    assert len(chosen)==32
    def inspect(state,label,body,wrist,language,metadata):
        assert np.isfinite(state).all() and np.isfinite(label).all()
        rotation=rot6d_rows_to_matrix(state[3:9]);rr=rot6d_rows_to_matrix(label[:,3:9])
        assert np.max(abs(rotation@rotation.T-np.eye(3)))<1e-5
        assert np.max(abs(rr@rr.transpose(0,2,1)-np.eye(3)))<1e-5
        ps,pa,mask=codec.encode(proc,state,label);decoded=codec.decode(proc,pa,state)
        pe=float(np.linalg.norm(decoded[:,:3]-label[:,:3],axis=-1).max());re=float(so3(rot6d_rows_to_matrix(decoded[:,3:9]),rr).max())
        assert pe<=1e-5 and re<=1e-4 and np.isfinite(ps).all() and np.isfinite(pa).all()
        packet=dict(body_rgb=body,wrist_rgb=wrist,state_v1=state,language=language)
        free=batch(full,packet)['inputs'];supervised=batch(full,packet,label)['inputs']
        assert identity(free)==identity({k:v for k,v in supervised.items() if k not in ('action','action_mask')})
        assert (mask[:,:10]==1).all() and (mask[:,10:]==0).all()
        for group,values in [('state/eef_9d',state[:9]),('state/gripper_position',state[9:]),('relative_action/eef_9d',codec.relative(label,state)),('action/gripper_position',label[:,9:])]:
            section,key=group.split('/');params=stats[codec.TAG][section][key];lo,width=codec.bounds(params)
            values=np.asarray(values);norm=codec.normalize(values,params)
            vv=values.reshape((-1,)+lo.shape);nn=norm.reshape((-1,)+lo.shape)
            ranges.append(dict(source=metadata['source'],field=group,statistics_shape=list(lo.shape),raw_min=vv.min(0).tolist(),raw_max=vv.max(0).tolist(),
                below_q01=np.sum(vv<lo,0).tolist(),above_q99=np.sum(vv>lo+width,0).tolist(),
                normalized_min=nn.min(0).tolist(),normalized_max=nn.max(0).tolist()))
        audits.append(dict(metadata,position_roundtrip_m=pe,so3_roundtrip_rad=re,actual_train_free_condition_equal=True,condition_sha256=identity(free)))
    for task,command,bucket,name,row in chosen:
        rec=read(DATA/'dataset'/(name+'.json'));path=DATA/'dataset'/(name+'.npz');assert sha(path)==rec['array_sha256']
        j=next(j for j,r in enumerate(rec['rows']) if r['index']==row['index'])
        z=dict(np.load(path,allow_pickle=False));frame=DATA/row['frame'];assert sha(frame)==row['frame_sha256']
        f=dict(np.load(frame,allow_pickle=False));state=z['state_v1'][j];label=z['raw_action'][j]
        assert np.array_equal(state,f['state_v1'].astype(np.float32))
        geom_path=DATA/'proposals'/rec['sequence']['family_id']/name/'geometry.npz'
        g=dict(np.load(geom_path,allow_pickle=False));i=row['index'];root=g['root_transform'];physical=root@kin.tcp_pose(g['joints'][i])
        assert np.linalg.norm(state[:3]+anchor-physical[:3,3])<=1e-5
        assert so3(rot6d_rows_to_matrix(state[3:9]),physical[:3,:3])<=1e-4
        indices=np.minimum(i+np.arange(40),len(g['joints'])-1)
        assert np.max(np.linalg.norm(label[:,:3]+anchor-g['desired_position'][indices],axis=-1))<=1e-5
        inspect(state,label,f['body_rgb'],f['wrist_rgb'],rec['sequence']['instruction'],dict(source='OLD_TRAIN',sequence=name,index=i,task=task,command=command,phase_bucket=bucket,
            observation_timestamp_s=row['time'],label_start_timestamp_s=row['time'],root_world=root.tolist(),frame_sha256=sha(frame),array_sha256=sha(path),
            original_frame_dtype=str(f['state_v1'].dtype),dataset_cast_dtype=str(state.dtype),original_codec_fp32_cast_exact=True))
    raw_old=RAW.parent/'PI0265/controls'
    for task in ('M1','M3'):
        for command in (0,1):
            folder=raw_old/f'pi0265-family-000-{task}-c{command}-ROW6D'
            attempts=[p.parent for p in folder.glob('attempt-*/result.json') if read(p)['validity']=='VALID' and not (p.parent/'INVALIDATION.json').exists()]
            assert len(attempts)==1;out=attempts[0];records=read(out/'rows.json');frames=sorted((out/'frames').glob('*.npz'))
            reference=dict(np.load(out/'reference.npz',allow_pickle=False));first=records[0]['time']
            for k in (0,len(frames)//3,2*len(frames)//3,len(frames)-1):
                f=dict(np.load(frames[k],allow_pickle=False));now=float(f['time']);row=next(r for r in records if abs(r['time']-now)<1e-6)
                assert not row.get('collision',{}).get('hits') and not row.get('hard_unsafe')
                root=np.asarray(row['root']);dof=np.asarray(row['DOF']).reshape(-1,2)
                # Raw native DOF order is source-bound in the saved result; arm follows twelve leg joints.
                arm=dof[12:18,0];pose=kin.tcp_pose(arm);world_r=Rotation.from_quat(root[3:7]).as_matrix()
                p=world_r@pose[:3,3]+root[:3];r=world_r@pose[:3,:3]
                assert np.linalg.norm(p-row['TCP'])<=1e-5 and so3(r,row['rotation'])<=1e-4
                state=np.r_[p-anchor,r[:2,:].reshape(6),.5].astype(np.float32)
                ts=now-first+np.arange(40)*.05
                pp=np.column_stack([np.interp(ts,reference['timestamp'],reference['position'][:,a]) for a in range(3)])
                from scipy.spatial.transform import Slerp
                rr=Slerp(reference['timestamp'],Rotation.from_quat(reference['quaternion']))(np.clip(ts,reference['timestamp'][0],reference['timestamp'][-1])).as_matrix()
                label=np.c_[pp-anchor,rr[:,:2,:].reshape(40,6),np.full(40,.5)].astype(np.float32)
                language=instructions(task,command,card['families'][0]['factors']['colors'])[0]
                inspect(state,label,f['body_rgb'],f['wrist_rgb'],language,dict(source='PI0265_SAVED_NONCOLLIDING_NOT_TRAIN',case=out.parent.name,frame=frames[k].name,
                    observation_timestamp_s=now,label_start_timestamp_s=now,actual_root=root.tolist(),frame_sha256=sha(frames[k])))
    assert len(audits)==48
    for key in ('time','phase','target_xyz','source_id','future_action'):
        try:batch(full,dict(body_rgb=f['body_rgb'],wrist_rgb=f['wrist_rgb'],state_v1=state,language=language,**{key:1}))
        except ValueError:pass
        else:raise AssertionError(('INPUT_LEAK_ACCEPTED',key))
    write(ROOT/'evidence/OBSERVATION_CODEC_AUDIT.json',dict(status='PASS_FINITE_SOURCE_BOUND_32TRAIN_16SAVED_CODEC',samples=audits,field_ranges=ranges,
        policy_anchor_world_m=anchor.tolist(),native_anchor_conversion='policy XYZ + original renderer anchor - actual native shoulder anchor; world attitude unchanged',
        mechanical_TCP=dict(parent='x5_link6',offset_m=kin.tcp_offset_xyz.tolist(),units='metres',rotation='world matrix first two ROWS'),
        statistics_sha256=sha(DATA/'dataset/statistics.json'),old_samples_reconstructed=0,old_samples_excluded=0,normalization_refit=0,old_DEV_added_to_TRAIN=False,
        out_of_fit_range_alone_invalidates=False,new_physics_starts=0,model_queries=0))
    write(ROOT/'evidence/EXPERT_CPU.json',expert_check(reg,card['families'][:2]))
    write(ROOT/'evidence/PATH_V2_CPU.json',metric_check())
    budget=copy.deepcopy(read(OLD/'design/BUDGET_CARD.json'));dt=budget['control_dt'];sim_dt=budget['sim_dt']
    budget.update(planned_starts=340,hard_starts=348,hard_simulated_seconds=14268.,hard_control_ticks=round(14268/dt),
        hard_simulate_calls=round(14268/sim_dt),hard_integration_substeps=round(14268/sim_dt)*budget['sim_substeps'],hard_actor_calls=2*round(14268/dt),
        GR00T_queries=22000,S4_heads=88000,training_updates=8480,training_head_cap=210000)
    write(ROOT/'design/BUDGET_CARD.json',budget)
    reset=[]
    for family in card['families'][:2]:
        for repeat in range(2):reset.append(dict(case_id=f"RESET-{family['family_id']}-{repeat}",stage='RESET',family=family,task='RESET',command=0,
            reset_seed=card['families'][0]['reset_seed'],max_task_s=0.,fresh_process=True,version='ROW6D'))
    write(ROOT/'design/SCENE_CARD.json',dict(schema='PI0266_FIXED_RESET_V1',authority=AUTHORITY,source_PI0265_scene_sha256=sha(OLD/'design/SCENE_CARD.json'),
        registry=reg,families=card['families'],reset_cases=reset,reset_seed=card['families'][0]['reset_seed'],policy_anchor_world_m=anchor.tolist(),
        EVAL_metadata_only_until_both_final_frozen=True,prior_failures_unchanged=True))
    print(__import__('json').dumps(dict(status='PASS_G0_PI0266_CPU_AND_FROZEN_CARDS',old_TRAIN=32,saved_noncolliding=16,reset_checks=4,physics=0,model_queries=0)))


if __name__=='__main__':main()
