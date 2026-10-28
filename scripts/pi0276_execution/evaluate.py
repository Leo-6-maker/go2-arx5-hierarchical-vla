"""Fixed no-teacher physical policy clients; inference waits pause simulated time."""
import json,hashlib,subprocess,os
from pathlib import Path
import numpy as np
from run import ROOT,write,sha
from source_worker import Episode,RAW,retained,native_action,pose_gate,matrix,source_episode,replay_episode,flatten
from evaluation_noise import freeze_noise

class Server:
    def __init__(self,point):
        self.point=point;self.errors=(ROOT/'runtime'/f'infer-{point}.log').open('a')
        env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott')
        env.pop('CUDA_VISIBLE_DEVICES',None)
        self.process=subprocess.Popen(['/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python','-u',str(ROOT/'code/infer.py'),'--point',point],
            env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.errors,text=True,bufsize=1)
        while True:
            line=self.process.stdout.readline();assert line, 'Model server exited before ready'
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if row.get('ready'):assert row['point']==point;break
    def query(self,packet,noise,folder):
        folder.mkdir(parents=True,exist_ok=True);inp=folder/'input.npz';out=folder/'output.npz'
        if inp.exists():
            with np.load(inp) as z:
                assert set(z.files)==set(packet)
                for k,v in packet.items():assert np.array_equal(z[k],v)
        else:np.savez(inp,**packet)
        request=dict(input=str(inp),output=str(out),noise_file=noise['file'],noise_name=noise['name'],noise_sha256=noise['sha256'])
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        while True:
            line=self.process.stdout.readline();assert line,'Model server terminated during a counted request'
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if 'output' in row:assert row['output']==str(out);break
        with np.load(out) as z:return z['decoded'][0].copy(),json.loads(str(z['receipt_json']))
    def close(self):
        self.process.stdin.close();code=self.process.wait();self.errors.close();assert code==0

def policy_episode(seed,lane,server,noises):
    slot=f'{lane}-{seed}-{server.point}';old=retained(slot)
    if old:return old
    ep=Episode(slot,seed,'pd_ee_pose');queries=[];saturations=[];status='VALID_MODEL_PHYSICS_COMPLETE'
    try:
        parent=retained(f'{"TEST" if lane=="TEST" else "TRAIN"}-{seed}-eef')[0]
        with np.load(parent/'initial.npz') as z:initial={k:z[k] for k in z.files};actual=flatten(ep.initial)
        assert initial.keys()==actual.keys()
        differences={k:float(np.max(np.abs(initial[k]-actual[k]))) for k in actual};assert max(differences.values())<=1e-6
        write(ep.folder/'INITIAL_FIXED_PHYSICS_MATCH.json',dict(parent=str(parent),differences=differences))
        while ep.control<640:
            observation=ep.observe();packet=dict(body_rgb=observation['base_camera_96'],wrist_rgb=observation['hand_camera_96'],state_v1=observation['state_v1'])
            index=ep.control//8;query_folder=RAW/'quality'/server.point/lane/str(seed)/ep.folder.name/f'q{index:03d}'
            decoded,receipt=server.query(packet,noises[lane,seed,index],query_folder)
            queries.append(dict(index=index,time_s=ep.control*.05,output=str(query_folder/'output.npz'),receipt=receipt))
            if not np.isfinite(decoded).all():status='VALID_MODEL_NONFINITE_FAIL';break
            if decoded.shape!=(40,10):raise ValueError('Model output ABI drift')
            try:gate=pose_gate(decoded,packet['state_v1'],ep.control*.05,slot)
            except ValueError as err:
                status='VALID_MODEL_INVALID_SO3_FAIL';queries[-1]['geometry_error']=str(err);break
            queries[-1]['gate']=gate
            if gate['decision']=='REJECT':status='VALID_MODEL_GATE_REJECT';break
            if np.min(decoded[:,9])<-.1 or np.max(decoded[:,9])>1.1:status='VALID_MODEL_GRIPPER_ENVELOPE_FAIL';break
            from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
            rotations=rot6d_rows_to_matrix(decoded[:,3:9])
            for j in range(8):
                target=np.eye(4);target[:3,3]=decoded[j,:3];target[:3,:3]=rotations[j]
                g=float(decoded[j,9]);sent=float(np.clip(g,0,1))
                saturations.append(dict(control_index=ep.control,raw=g,sent=sent,magnitude=abs(g-sent)))
                ep.step(native_action(target,ep.root,g),phase=-1,command=decoded[j])
                assert np.max(np.abs(matrix(ep.arm._target_pose)-np.linalg.inv(ep.root)@target))<1e-5
                if not all(np.isfinite(ep.states[-1][key]).all() for key in ('tcp','qpos','qvel','cubeA_pose','cubeB_pose','cubeA_velocity','cubeB_velocity')):
                    status='VALID_MODEL_PHYSICS_NONFINITE_FAIL';break
            if status!='VALID_MODEL_PHYSICS_COMPLETE':break
        write(ep.folder/'QUERIES.json',queries);write(ep.folder/'GRIPPER_TRANSPORT.json',saturations)
        write(ep.folder/'EXECUTION_CONTRACT.json',dict(no_teacher_calls=True,no_object_fields_to_policy=True,
            simulated_control_s=.4,maximum_queries=80,NON_REALTIME=True,success_never_stops_or_replaces_policy=True))
        return ep.folder,ep.finish(status)
    except BaseException as err:
        write(ep.folder/'QUERIES.json',queries);write(ep.folder/'GRIPPER_TRANSPORT.json',saturations)
        ep.finish('ENGINEERING_INCOMPLETE',repr(err));raise

def main():
    admission=json.loads((ROOT/'evidence/FINAL_POLICY_ADMISSION.json').read_text());assert admission['status']=='PASS_PI0276_FINAL_TASK_AND_BASE_POLICY_IDENTITIES'
    for entry in json.loads((ROOT/'design/EVALUATION_FILESET.json').read_text())['files']:
        assert sha(entry['path'])==entry['sha256']
    noise=freeze_noise();noises={(r['lane'],r['condition'],r['index']):r for r in noise['rows']}
    card=json.loads((ROOT/'design/CONDITIONS.json').read_text());rows=[]
    panel=json.loads((ROOT/'design/OFFLINE_CARD.json').read_text())['rows']
    for seed in card['conditions']['TEST']:
        joint=source_episode(seed,'TEST');eef=replay_episode(seed,'TEST')
        rows.append(dict(seed=seed,lane='TEST',joint=str(joint[0]),eef=str(eef[0])))
    for point in ('BASE0','TASK8192'):
        server=Server(point)
        try:
            for row in panel:
                j=row['anchor'];folder=Path(row['episode'])
                with np.load(folder/'observations.npz') as z:
                    packet=dict(body_rgb=z['base_camera_96'][j],wrist_rgb=z['hand_camera_96'][j],state_v1=z['state_v1'][j])
                server.query(packet,noises['OFFLINE',row['uid'],0],RAW/'quality'/point/'OFFLINE'/row['uid'])
            for seed in card['conditions']['TEST']:policy_episode(seed,'TEST',server,noises)
            if point=='TASK8192':
                for seed in noise['TRAIN_diagnostic_seeds']:policy_episode(seed,'TRAIN_PHYSICS',server,noises)
        finally:server.close()
    write(ROOT/'evidence/PHYSICAL_EVALUATION_COMPLETE.json',dict(status='ALL_FIXED_PHYSICAL_LANES_EXECUTED',expert_rows=rows,
        maximum_quality_queries=6784,offline_conditions_per_policy=192,scoring_and_archival='PENDING',selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))

if __name__=='__main__':main()
