"""Object-task tables from saved physics, with original failures and denominators."""
import csv,json,math,sys,xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from run import ROOT,write,sha
from replay import reconstruct,main as replay
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0276')
sys.path.insert(0,str(ROOT/'project-src'))

def wilson(k,n):
    z=1.959963984540054;a=k/n;d=1+z*z/n;c=(a+z*z/(2*n))/d
    h=z*math.sqrt(a*(1-a)/n+z*z/(4*n*n))/d
    return [c-h,c+h]

def episode(folder):
    rec=json.loads((folder/'RESULT.json').read_text());assert rec['status'].startswith('VALID')
    with np.load(folder/'physics.npz') as z:
        b=reconstruct(z);success=b['success'][::5];indices=np.flatnonzero(success[:601])
        first=float(indices[0]*.05) if len(indices) else None;persist=bool(len(success)>=601 and success[560:601].all())
        on=b['is_cubeA_on_cubeB'];release=b['is_cubeA_grasped'];pose=z['cubeA_pose'][:,:3]
        drop=int(np.sum(on[:-1]&~on[1:]));grasp_drop=int(np.sum(release[:-1]&~release[1:]))
        speed=float(np.linalg.norm(z['cubeA_velocity'],axis=-1).max())
        min_z=float(pose[:,2].min());q=z['qpos'].copy();tcp=z['tcp'].copy()
    assert first==rec['first_success_s'] and persist==rec['persistent_success']
    contact_points=0;max_impulse=0.;min_separation=None
    for line in (folder/'contacts.jsonl').open():
        for c in json.loads(line)['contacts']:
            for p in c['points']:
                contact_points+=1
                if 'impulse' in p:max_impulse=max(max_impulse,float(np.linalg.norm(p['impulse'])))
                if 'separation' in p:min_separation=float(p['separation']) if min_separation is None else min(min_separation,float(p['separation']))
    saturation=json.loads((folder/'GRIPPER_TRANSPORT.json').read_text()) if (folder/'GRIPPER_TRANSPORT.json').exists() else []
    inventory=json.loads((folder/'NATIVE_INVENTORY.json').read_text());urdf=ROOT/'ManiSkill/mani_skill/assets/robots/panda/panda_v3.urdf'
    assert sha(urdf)==inventory['opening_calibration']['urdf_sha256']
    tree=ET.parse(urdf).getroot();names=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2'];limits=[]
    for name in names:
        limit=tree.find(f"joint[@name='{name}']/limit");assert limit is not None
        limits.append([float(limit.get('lower')),float(limit.get('upper'))])
    limits=np.array(limits);excursion=np.maximum(limits[:,0]-q,q-limits[:,1]).clip(0)
    tracking=dict(command_actual_XYZ_RMS_m=None,command_actual_SO3_RMS_rad=None,command_installed_FK_XYZ_RMS_m=None,installed_FK_actual_XYZ_RMS_m=None)
    if rec['control_calls']:
        with np.load(folder/'commands.npz') as z:cmd=z['action'].copy()
        with np.load(folder/'installed.npz') as z:installed=z['installed_target_fk'].copy()
        actual=tcp[5::5];assert len(cmd)==len(installed)==len(actual)==rec['control_calls']
        from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
        cr=rot6d_rows_to_matrix(cmd[:,3:9]);angles=Rotation.from_matrix(actual[:,:3,:3]@cr.transpose(0,2,1)).magnitude()
        rms=lambda x:float(np.sqrt(np.mean(np.sum(x*x,axis=-1))))
        tracking=dict(command_actual_XYZ_RMS_m=rms(actual[:,:3,3]-cmd[:,:3]),command_actual_SO3_RMS_rad=float(np.sqrt(np.mean(angles*angles))),
            command_installed_FK_XYZ_RMS_m=rms(installed[:,:3,3]-cmd[:,:3]),installed_FK_actual_XYZ_RMS_m=rms(actual[:,:3,3]-installed[:,:3,3]))
    return dict(path=str(folder),status=rec['status'],first_success_s=first,persistent_success=persist,control_calls=rec['control_calls'],
        IK_holds=rec['ik_holds'],contact_points=contact_points,max_contact_impulse_Ns=max_impulse,min_contact_separation_m=min_separation,
        red_min_z_m=min_z,red_max_linear_speed_mps=speed,on_top_to_off_transitions_100Hz=drop,grasp_to_release_transitions_100Hz=grasp_drop,
        joint_limit_excursion_samples=int(np.any(excursion>0,axis=1).sum()),max_joint_limit_excursion=float(excursion.max()),
        gripper_saturation_count=sum(r['magnitude']>0 for r in saturation),gripper_saturation_max=max((r['magnitude'] for r in saturation),default=0.),
        result_sha256=sha(folder/'RESULT.json'),physics_sha256=sha(folder/'physics.npz'),**tracking)

def valid(slot):
    paths=[p.parent for p in (RAW/'episodes'/slot).glob('attempt-*/RESULT.json') if json.loads(p.read_text())['status'].startswith('VALID')]
    assert len(paths)==1,slot
    return paths[0]

def main():
    assert json.loads((ROOT/'evidence/PHYSICAL_EVALUATION_COMPLETE.json').read_text())['status']=='ALL_FIXED_PHYSICAL_LANES_EXECUTED'
    replay();rows=[];card=json.loads((ROOT/'design/CONDITIONS.json').read_text())
    for seed in card['conditions']['TEST']:
        for point in ('joint','eef','BASE0','TASK8192'):
            rows.append(dict(seed=seed,lane='TEST',point=point,**episode(valid(f'TEST-{seed}-{point}'))))
    noise=json.loads((ROOT/'design/EVALUATION_NOISE.json').read_text())
    for seed in noise['TRAIN_diagnostic_seeds']:
        rows.append(dict(seed=seed,lane='TRAIN_PHYSICS',point='TASK8192',**episode(valid(f'TRAIN_PHYSICS-{seed}-TASK8192'))))
    assert len(rows)==144
    summary={}
    for lane,point,n in [('TEST','joint',32),('TEST','eef',32),('TEST','BASE0',32),('TEST','TASK8192',32),('TRAIN_PHYSICS','TASK8192',16)]:
        own=[r for r in rows if r['lane']==lane and r['point']==point];assert len(own)==n
        qualified=lambda r:r['persistent_success'] and (point not in ('BASE0','TASK8192') or r['status']=='VALID_MODEL_PHYSICS_COMPLETE')
        k=sum(qualified(r) for r in own)
        summary[f'{lane}/{point}']=dict(denominator=n,first_success=sum(r['first_success_s'] is not None for r in own),persistent_success=k,
            predicate_persistence_without_interface_qualification=sum(r['persistent_success'] for r in own),wilson95=wilson(k,n),
            failure_statuses=dict(Counter(r['status'] for r in own)))
    paired=[]
    for seed in card['conditions']['TEST']:
        own=[next(r for r in rows if r['seed']==seed and r['lane']=='TEST' and r['point']==p) for p in ('BASE0','TASK8192')]
        a,b=[r['persistent_success'] and r['status']=='VALID_MODEL_PHYSICS_COMPLETE' for r in own]
        paired.append(dict(seed=seed,BASE0=a,TASK8192=b,gained=not a and b,lost=a and not b))
    gain=sum(r['gained'] for r in paired);loss=sum(r['lost'] for r in paired)
    verdict=dict(status='PI0276_PHYSICAL_TASK_TABLE_COMPLETE',summary=summary,paired=paired,gained=gain,lost=loss,net_gain=gain-loss,
        physical_task_quality='PASS' if summary['TEST/TASK8192']['persistent_success']>=24 else 'VALID_FAIL',learning_effect_target_met=gain-loss>=8,
        success_definition='Pinned native predicate, all41 control samples28..30s',claim='StackCube-v1 / PI0276 extended-horizon EEF protocol, NON_REALTIME source-Panda simulation',
        contact_diagnostics_are_not_hardware_safety=True,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    dst=RAW/'delivery';dst.mkdir(exist_ok=True);write(dst/'PHYSICAL_VERDICT.json',verdict)
    for name,table in [('PHYSICAL144.csv',rows),('PAIRED_TEST32.csv',paired)]:
        with (dst/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)

def check():
    assert wilson(0,32)[0]>=-1e-15 and wilson(32,32)[1]<=1+1e-15
    assert wilson(16,32)[0]<.5<wilson(16,32)[1]

if __name__=='__main__':check();main()
