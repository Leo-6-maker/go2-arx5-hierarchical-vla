"""Freeze real packets/pairs and source identities before any new scores."""
import hashlib,json,os,time,resource
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from budget import publication

OLD=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0283')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def rotation(x):
    a=x[3:6].astype(float);a/=np.linalg.norm(a)
    b=x[6:9].astype(float);b-=a*np.dot(a,b);b/=np.linalg.norm(b)
    return np.stack([a,b,np.cross(a,b)])

def run(root):
    publication(root);assert os.environ['CUDA_VISIBLE_DEVICES']==''
    done=root/'design/PRE_SCORE_INPUTS.json';assert not done.exists(),'Reuse frozen pre-score card'
    start=time.monotonic();plan=[json.loads(x) for x in (OLD/'data/DIAG48.jsonl').read_text().splitlines()]
    prior=json.loads((OLD/'source/diagnostics/EVENT1024/RESULT.json').read_text());rows=[]
    for row,previous in zip(plan,prior['rows']):
        assert row['uid']==previous['uid']
        source=OLD/'source/diagnostics/EVENT1024'/f"{previous['index']:02d}"
        for file,key in [('input.npz','input_sha256'),('target.npz','target_sha256'),('output.npz','output_sha256')]:assert sha(source/file)==previous[key]
        folder=root/'diagnostics'/f"{previous['index']:02d}";folder.mkdir(parents=True,exist_ok=True)
        (folder/'input.npz').write_bytes((source/'input.npz').read_bytes());(folder/'target.npz').write_bytes((source/'target.npz').read_bytes())
        with np.load(source/'output.npz') as f:packet=f['noise_uint16'].copy()
        np.save(folder/'noise-0.npy',packet,allow_pickle=False)
        rows.append(dict(row,index=previous['index'],input=str(folder/'input.npz'),target=str(folder/'target.npz'),packet0=str(folder/'noise-0.npy'),reused_K4_output=str(source/'output.npz'),original=previous))
    assert len(rows)==48 and sum(x['split']=='TRAIN' for x in rows)==24
    contract=json.loads((OLD/'data/DATA_CONTRACT.json').read_text());anchors=[]
    for e in contract['admitted_episodes']:
        if e['split']!='TRAIN':continue
        ep=OLD/'data/episodes'/f"TRAIN-{e['seed']}"
        state=np.load(ep/'state_v1.npy',mmap_mode='r')[0].copy()
        commands=np.load(ep/'action.npy',mmap_mode='r')[:40].copy()
        base=np.load(ep/'base_camera_raw.npy',mmap_mode='r')[0].copy();wrist=np.load(ep/'hand_camera_raw.npy',mmap_mode='r')[0].copy()
        uid=f"TRAIN-{e['seed']}:0";folder=root/'natural'/uid;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'input.npz',base_raw=base,wrist_raw=wrist,actual_state=state)
        np.savez_compressed(folder/'target.npz',commands=commands)
        anchors.append(dict(uid=uid,seed=e['seed'],input=str(folder/'input.npz'),target=str(folder/'target.npz'),state=state.tolist(),expert_displacement7=(commands[7,:3]-state[:3]).tolist(),base_sha256=ah(base),wrist_sha256=ah(wrist),input_sha256=sha(folder/'input.npz'),target_sha256=sha(folder/'target.npz')))
    assert len(anchors)==93
    eligible=[]
    for i,a in enumerate(anchors):
        for b in anchors[i+1:]:
            x,y=np.asarray(a['state']),np.asarray(b['state']);xyz=float(np.linalg.norm(x[:3]-y[:3]));so3=float(Rotation.from_matrix(rotation(x)@rotation(y).T).magnitude());grip=float(abs(x[9]-y[9]))
            diff=float(np.linalg.norm(np.asarray(a['expert_displacement7'])-b['expert_displacement7']))
            if xyz<=.002 and so3<=.02 and grip<=.02 and diff>=.01:
                lo,hi=sorted([a,b],key=lambda q:q['uid'])
                eligible.append(dict(a=lo,b=hi,state_xyz=xyz,state_SO3=so3,state_gripper=grip,expert_difference7=diff,distance=float(np.sqrt((xyz/.002)**2+(so3/.02)**2+(grip/.02)**2))))
    eligible.sort(key=lambda x:(x['distance'],x['a']['uid'],x['b']['uid']));used=set();pairs=[]
    for pair in eligible:
        if pair['a']['uid'] in used or pair['b']['uid'] in used:continue
        pairs.append(pair);used.update([pair['a']['uid'],pair['b']['uid']])
        if len(pairs)==8:break
    dependencies={str(OLD/'code'/n):sha(OLD/'code'/n) for n in ['inference.py','native_model.py','codec.py','cpu_inputs.py','data.py']}
    official=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0280/official')
    for p in [official/'gr00t/model/gr00t_n1d7/gr00t_n1d7.py',official/'gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py']:dependencies[str(p)]=sha(p)
    card=dict(parent_root=str(OLD),parent_code=str(OLD/'code'),parent_point='EVENT1024',allowed_K=[4,16],host_caps=dict(query=6272,head=100352,GPU_seconds=14400),reserved_other_host=dict(query=128,head=2048,GPU_seconds=28800),dependencies=dependencies)
    write(root/'design/RUNTIME_CARD.json',card)
    write(root/'design/NATURAL_PAIRS.json',dict(status='FROZEN_BEFORE_NEW_SCORES',pairs=pairs,eligible_count=len(eligible),admitted_TRAIN_anchors=len(anchors),camera_and_instruction='UNCHANGED_COMMON_SOURCE_CARD',displacement_definition='command7XYZ-minus-own-actual-anchor0XYZ',minimum_weak_pairs=6,pair_count=len(pairs),no_new_render=True))
    write(done,dict(status='FROZEN_BEFORE_NEW_SCORES',rows=rows,noise_namespace=28401,packets_per_input=4,noise_generation='Original BF16 CUDA native randn; namespace+UID+packet frozen before first score',native_data_contract_sha256=sha(OLD/'data/DATA_CONTRACT.json'),codec_identity=json.loads((OLD/'design/IMPLEMENTATION.json').read_text())['codec_identity'],new_model_queries=0,new_training=0,new_physics=0))
    print(json.dumps(dict(status='PASS_PRE_SCORE_REAL_INPUTS_AND_GREEDY_NATURAL_PAIRS',rows=len(rows),natural_pairs=len(pairs),elapsed_s=time.monotonic()-start,CPU_user_s=resource.getrusage(resource.RUSAGE_SELF).ru_utime,CPU_system_s=resource.getrusage(resource.RUSAGE_SELF).ru_stime)))

if __name__=='__main__':
    import sys
    run(Path(sys.argv[1]))
