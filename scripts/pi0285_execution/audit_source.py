"""CPU readback of original contacts, scorer, actual installation and paired reset."""
import argparse,json,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from contract import ROOT,sha,write
from scorer import score


def grasp_from_contacts(data):
    fingers=[]
    for side in ('left','right'):
        force=data[side+'_force'];direction=data[side+'_direction'];norm=np.linalg.norm(force,axis=1)
        cosine=(force*direction).sum(1)/(np.maximum(norm,1e-12)*np.linalg.norm(direction,axis=1))
        fingers.append((norm>=.5)&(np.arccos(np.clip(cosine,-1,1))*180/np.pi<=85))
    return fingers[0]&fingers[1]


def audit(split):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    panel=ROOT/'source/evidence'/f'{split}.json';source=json.loads(panel.read_text());rows=[];initial={}
    for row in source['rows']:
        folder=Path(row['folder']);card=json.loads((folder/'EPISODE_CARD.json').read_text());result=json.loads((folder/'RESULT.json').read_text())
        for name,h in result['files'].items():assert sha(folder/name)==h,(folder,name)
        with np.load(folder/'physics.npz') as f:data={k:f[k].copy() for k in f.files}
        times=data['time_s'];assert np.allclose(np.diff(times),.01,atol=1e-10)
        independent=grasp_from_contacts(data);assert np.array_equal(independent,data['is_cubeA_grasped'])
        data['is_cubeA_grasped']=independent
        static=(np.linalg.norm(data['cubeA_velocity'],axis=1)<=.01)&(np.linalg.norm(data['cubeA_angular'],axis=1)<=.5)
        assert np.array_equal(static,data['is_cubeA_static'])
        offset=data['cubeA_pose'][:,:3]-data['cubeB_pose'][:,:3]
        native_top=(np.linalg.norm(offset[:,:2],axis=1)<=np.linalg.norm(np.asarray([.02,.02],dtype=np.float32))+.005)&(abs(offset[:,2]-.04)<=.005)
        assert np.array_equal(native_top,data['is_cubeA_on_cubeB']),'Pinned original object predicate'
        recalculated=score(data,row['task'],card['scene']['zone_xy'],data['cubeA_pose'][0,:3],data['cubeB_pose'][0,:3],row['status'],row['control_calls'])
        assert recalculated==row['task_score'],'Scorer must match every recorded predicate/event'
        with np.load(folder/'commands.npz') as f:commands=f['action'].copy();phase=f['phase'].copy()
        with np.load(folder/'observations.npz') as f:
            states=f['state_v1'].copy();clock=f['time_s'].copy()
            initial[row['task'],row['layout']]={k:f[k][0].copy() for k in ('base_camera_raw','hand_camera_raw','state_v1','base_camera_extrinsic_cv','hand_camera_extrinsic_cv') if k in f.files}
        assert len(states)==len(commands)==len(clock) and np.allclose(clock,np.arange(len(clock))*.05)
        assert np.allclose(states[:,:3],data['tcp'][::5][:len(states),:3,3],atol=1e-7),'Actual pre-command input clock'
        with np.load(folder/'installed.npz') as f:installed=f['installed_target_fk'].copy()
        targets=np.tile(np.eye(4),(len(commands),1,1));targets[:,:3,3]=commands[:,:3]
        x=commands[:,3:6];x=x/np.linalg.norm(x,axis=1)[:,None];y=commands[:,6:9]-(x*commands[:,6:9]).sum(1)[:,None]*x;y=y/np.linalg.norm(y,axis=1)[:,None];z=np.cross(x,y)
        targets[:,:3,:3]=np.stack((x,y,z),axis=1)
        achieved=data['tcp'][np.arange(1,len(commands)+1)*5]
        rows.append(dict(task=row['task'],layout=row['layout'],status=row['status'],persistent_success=recalculated['persistent_success'],score=recalculated,source_result_sha256=sha(folder/'RESULT.json'),force_based_native_grasp_exact=True,physical_static_exact=True,original_native_on_top_exact=True,pre_command_actual_clock_exact=True,
                         reference_to_installed_XYZ_RMS_m=float(np.sqrt(np.mean(np.sum((targets[:,:3,3]-installed[:,:3,3])**2,axis=1)))),
                         installed_to_same_issued_achieved_XYZ_RMS_m=float(np.sqrt(np.mean(np.sum((installed[:,:3,3]-achieved[:,:3,3])**2,axis=1)))),
                         phase_counts={str(k):int((phase==k).sum()) for k in np.unique(phase)}))
    pairs=[]
    for seed in sorted({key[1] for key in initial}):
        if ('T1',seed) not in initial or ('T2',seed) not in initial:continue
        a=initial['T1',seed];b=initial['T2',seed];assert set(a)==set(b)
        exact={k:bool(np.array_equal(a[k],b[k])) for k in a};assert all(exact.values()),(seed,exact)
        pairs.append(dict(layout=seed,same_initial_actual_RGB_state_and_cameras=exact))
    value=dict(status='PASS_INDEPENDENT_CPU_CONTACT_SCORE_CLOCK_INSTALLATION_AND_PAIRED_INITIAL_INPUTS',split=split,source_sha256=sha(panel),rows=rows,pairs=pairs,new_model_queries=0,new_physics=0)
    write(ROOT/'audit'/f'{split}_CPU.json',value);print(json.dumps(dict(status=value['status'],rows=len(rows),pairs=len(pairs))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();audit(a.split)
