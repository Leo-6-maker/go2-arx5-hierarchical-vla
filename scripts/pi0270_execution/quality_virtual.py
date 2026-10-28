"""Actual fresh-render -> native S4 -> original gate -> .4 s virtual feedback."""
import isaacgym
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pi0267_virtual_renderer import BASE,ROOT,Renderer,collision
sys.path.insert(0,str(BASE/'PI0267/code'))
from pi0267_virtual_data import state,raw_gate
from pi0267_cpu import interpolate
from pi0266_expert import reference
from pi0267_metrics_v3 import score
from pi0255_motion3 import scene,write,sha
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix

R1=BASE/'PI0270';RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')
CONTROL_RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0269')
OLD_RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')


def read(p):return json.loads(Path(p).read_text())


def advance(p,r,lp,lr,backend):
    desired_p,desired_r=interpolate(lp,lr,np.arange(40)*.05,np.arange(1,21)*.02)
    actual_p=[];actual_r=[]
    for dp,dr in zip(desired_p,desired_r):
        if backend=='E0':p,r=dp.copy(),dr.copy()
        else:
            alpha=1-np.exp(-.02/.15);p=p+alpha*(dp-p)
            r=Rotation.from_rotvec(alpha*Rotation.from_matrix(dr@r.T).as_rotvec()).as_matrix()@r
        actual_p.append(p.copy());actual_r.append(r.copy())
    return np.array(actual_p),np.array(actual_r),desired_p,desired_r


def message(worker):
    while True:
        line=worker.stdout.readline()
        if not line:raise RuntimeError('INFERENCE_WORKER_EXITED_WITHOUT_RESULT')
        try:return json.loads(line)
        except ValueError:continue


def language(reg,family,task,c):
    if task=='M1':
        templates=read(R1/'design/PAIRING_PLAN.json')['templates']
        return templates['T0'].format(color=family['factors']['colors'][c])
    start,end,direction=('black','white','counterclockwise') if c==0 else ('white','black','clockwise')
    return f'Approach the {start} endpoint, then follow the open contour {direction} to the {end} endpoint as seen from above.'


def run_episode(point,family,task,command,backend,worker,start=None):
    card=read(ROOT/'design/SCENE_CARD.json');reg=card['registry'];anchor=np.array(card['policy_anchor_world_m'])
    key=start['case'] if start else f"{family['family_id']}-{task}-c{command}-{backend}";out=RAW/('component' if start else 'virtual')/point/key;out.mkdir(parents=True,exist_ok=True)
    completed=out/'result.json'
    if completed.exists():
        prior=read(completed);assert prior['validity'] in (('VALID','PREQUERY_EXCLUDED') if start else ('VALID',)) and prior['point']==point
        assert prior['evaluation_card_sha256']==sha(R1/'design/EVALUATION_CARD.json')
        if prior['validity']=='VALID':assert prior['arrays_sha256']==sha(out/'trajectory.npz')
        return prior
    starts=R1/'evidence/virtual_starts.jsonl'
    ordinal=sum(1 for _ in starts.open())+1 if starts.exists() else 1
    assert ordinal<=304,'VIRTUAL_START_HARD_CAP'
    with starts.open('a') as stream:
        stream.write(json.dumps(dict(ordinal=ordinal,point=point,case=key,resume_existing_queries=bool(list((out/'queries').glob('*/prediction.npz')))))+'\n')
        stream.flush();os.fsync(stream.fileno())
    renderer=Renderer(reg,family);initial=np.array(reg['initial_TCP_world']);p,r=(np.array(start['initial_position_world']),np.array(start['initial_rotation_world'])) if start else (initial[:3,3].copy(),initial[:3,:3].copy())
    positions=[p.copy()];rotations=[r.copy()];installed_p=[p.copy()];installed_r=[r.copy()];plans=[];gates=[];geometry=[]
    data=scene(reg,family);half=np.array(reg['geometry']['obstacle_half_extents_m']);safety=False;reason=None
    (out/'queries').mkdir(exist_ok=True)
    try:
        for replan in range(20 if start else 100):
            folder=out/'queries'/('%03d'%replan);folder.mkdir(exist_ok=True);input_path=folder/'observation.npz';now=replan*.4
            sv=state(p,r,anchor)
            if input_path.exists():
                z=np.load(input_path);assert np.array_equal(z['state_v1'],sv) and np.array_equal(z['position_world'],p) and np.array_equal(z['rotation_world'],r)
                assert float(z['timestamp'])==now
            else:
                frame_journal=R1/'evidence/virtual_frame_starts.jsonl'
                frame_ordinal=sum(1 for _ in frame_journal.open())+1 if frame_journal.exists() else 1
                factor_journal=R1/'evidence/factor_frame_starts.jsonl'
                factors=0
                assert factors+frame_ordinal<=28672,'TOTAL_NEW_FRAME_PAIR_CAP'
                with frame_journal.open('a') as stream:
                    stream.write(json.dumps(dict(ordinal=frame_ordinal,point=point,case=key,replan=replan))+'\n');stream.flush();os.fsync(stream.fileno())
                frames,info=renderer.place(p,r,now)
                # Privileged arrays stay in observation evidence; only explicit RGB/state/language enter worker.
                np.savez_compressed(input_path,body_rgb=frames['body'],wrist_rgb=frames['wrist'],state_v1=sv,
                    position_world=p,rotation_world=r,timestamp=now)
                np.savez_compressed(folder/'debug.npz',**{k:frames[k] for k in ('body_segmentation','wrist_segmentation','body_depth','wrist_depth')})
                write(folder/'camera.json',info)
            if start and replan==0:
                from pi0267_virtual_renderer import visibility
                with np.load(folder/'debug.npz') as z:frames={k:z[k].copy() for k in z.files}
                with np.load(input_path) as z:frames.update(body=z['body_rgb'].copy(),wrist=z['wrist_rgb'].copy())
                info=read(folder/'camera.json')
                check=visibility(frames,info['scene_ids'],task,command,common_start=False)
                hits=collision(p,r,np.array(data['obstacle']),half)
                if not check['pass_necessary'] or hits:
                    record=dict(validity='PREQUERY_EXCLUDED',point=point,component_start=start,
                        family=family['family_id'],task=task,command=command,backend=backend,
                        reasons=(['VISIBILITY'] if not check['pass_necessary'] else [])+(['INITIAL_TOOL_COLLISION'] if hits else []),
                        visibility=check,collision_hits=hits,observation_sha256=sha(input_path),queries=[],
                        evaluation_card_sha256=sha(R1/'design/EVALUATION_CARD.json'),start_ordinal=ordinal,
                        qualification_authority=False,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
                    write(completed,record);return record
            response_path=folder/'prediction.npz';request=dict(input=str(input_path),output=str(response_path),language=language(reg,family,task,command),
                noise_file=str(CONTROL_RAW/'noise'/(key+'.npz')) if start else str(OLD_RAW/'noise'/(family['family_id']+'.npz')),
                noise_name=f'q{replan:03d}' if start else f'{task}_{backend}_{replan:03d}',noise_registry='COMPONENT_CARD.json' if start else 'EVALUATION_CARD.json')
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response=message(worker)
            assert response['status'] in ('VALID','VALID_REUSED') and response['output']==str(response_path)
            decoded=np.load(response_path)['decoded'][0]
            if not np.isfinite(decoded).all():
                safety=True;reason='NONFINITE_POLICY_OUTPUT';plans.append(dict(replan=replan,query=request,prediction_sha256=sha(response_path)));break
            lp=decoded[:,:3].astype(float)+anchor
            try:lr=rot6d_rows_to_matrix(decoded[:,3:9].astype(float))
            except ValueError:
                safety=True;reason='INVALID_ROT6D_POLICY_OUTPUT';plans.append(dict(replan=replan,query=request,prediction_sha256=sha(response_path)));break
            gate=raw_gate(lp,lr,p,r,anchor,now=now,source='PI0267_NATIVE_S4_POLICY',gripper=decoded[:,9]);gates.append(dict(replan=replan,result=gate));plans.append(dict(replan=replan,query=request,
                prediction_sha256=sha(response_path),observation_sha256=sha(input_path),timestamp=now))
            if gate['decision']=='REJECT':safety=True;reason='GATE_REJECT';break
            ap,ar,dp,dr=advance(p,r,lp,lr,backend)
            for q,s,desired,desired_r in zip(ap,ar,dp,dr):
                hits=collision(q,s,np.array(data['obstacle']),half)
                positions.append(q.copy());rotations.append(s.copy());installed_p.append(desired.copy());installed_r.append(desired_r.copy())
                if hits:
                    geometry.append(dict(tick=len(positions)-1,hits=hits));safety=True;reason='VIRTUAL_TOOL_COLLISION';break
            p,r=positions[-1].copy(),rotations[-1].copy()
            if safety:break
        end=6 if task=='M1' else 16;gp,gr=reference(reg,family,task,command,[end])
        spec=dict(task=task,command=command,goal=gp[0],goal_rotation=gr[0],center=data['center'],radius=data['radius'])
        t=np.arange(len(positions))*.02;result=score(t,np.array(positions),np.array(rotations),np.array(installed_p),np.array(installed_r),spec,safety)
        np.savez_compressed(out/'trajectory.npz',time=t,actual_p=positions,actual_r=rotations,installed_p=installed_p,installed_r=installed_r,
            spec_json=np.array(json.dumps({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in spec.items()})))
        record=dict(validity='VALID',point=point,family=family['family_id'],task=task,command=command,backend=backend,
            success=result['success'],scorer=result,stop_reason=reason,gates=gates,collision_events=geometry,queries=plans,
            actual_elapsed_s=float(t[-1]),virtual_steps=len(t)-1,physics_steps=0,evaluation_card_sha256=sha(R1/'design/EVALUATION_CARD.json'),
            arrays_sha256=sha(out/'trajectory.npz'),mode='VIRTUAL_EEF_KINEMATIC / NOT_WBC / NON_REALTIME_LOCKSTEP',
            start_ordinal=ordinal,component_start=start,qualification_authority=start is None,virtual_tool_gripper='fixed nonarticulated .5; model gripper retained unchanged in prediction and gate',
            teacher_used_for_control=False,reference_retimed=False,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
        write(completed,record);print(json.dumps(dict(case=key,point=point,success=record['success'],queries=len(plans))),flush=True)
        return record
    finally:renderer.close()


def selected_families(card,family_id=None):
    families=[f for f in card['families'] if f['split']=='NEW_SIM_EVAL'];assert len(families)==8
    if family_id is not None:
        families=[f for f in families if f['family_id']==family_id];assert len(families)==1,'UNKNOWN_FIXED_EVAL_FAMILY'
    return families


def main(point,family_id=None):
    from quality_gate import admitted
    admitted();assert point=='AGG4096'
    card=read(ROOT/'design/SCENE_CARD.json');families=selected_families(card,family_id)
    env=dict(os.environ);env.pop('LD_PRELOAD',None)
    # Modern torch cannot inherit IsaacGym's Python3.8 native library search path.
    env.pop('LD_LIBRARY_PATH',None);env['PYTHONPATH']=str(R1/'code')
    with (R1/'evidence'/('inference-'+point+'.log')).open('ab') as log:
        worker=subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'),str(R1/'code/quality_worker.py'),'--point',point],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
        try:
            ready=message(worker);assert ready==dict(ready=True,point=point)
            for family in families:
                for task in ('M1','M3'):
                    for command in (0,1):
                        for backend in ('E0','E1'):run_episode(point,family,task,command,backend,worker)
            worker.stdin.close();assert worker.wait()==0
        finally:
            if worker.poll() is None:worker.terminate();worker.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('AGG4096',));p.add_argument('--family');p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:
        assert a.point is None and a.family is None
        fixture=dict(families=[dict(split='NEW_SIM_EVAL',family_id=str(i)) for i in range(8)])
        assert selected_families(fixture)==fixture['families'] and selected_families(fixture,'3')==[fixture['families'][3]]
        try:selected_families(fixture,'outside')
        except AssertionError:pass
        else:raise AssertionError('UNDECLARED_FAMILY_ACCEPTED')
        p0=np.array([.3,0,.6]);r0=Rotation.from_euler('xyz',[.31,-.47,.83]).as_matrix()
        lp=p0+np.arange(40)[:,None]*.05*np.array([.02,0,0]);lr=np.repeat(r0[None],40,axis=0)
        ap,ar,dp,dr=advance(p0,r0,lp,lr,'E0');assert np.array_equal(ap,dp) and np.array_equal(ar,dr)
        ap,ar,dp,dr=advance(p0,r0,lp,lr,'E1');assert p0[0]<ap[-1,0]<dp[-1,0] and np.max(abs(ar-r0))<1e-12
        for now in (0.,12.8):
            verdict=raw_gate(lp,lr,p0,r0,np.zeros(3),now=now,gripper=np.full(40,.5))
            assert verdict['decision']!='REJECT',verdict
        print('PASS_E0_E1_NONIDENTITY_PREFIX_AND_TIMESTAMPED_ORIGINAL_GATE_ZERO_QUERY_ZERO_PHYSICS',flush=True)
    else:assert a.point;main(a.point,a.family)
