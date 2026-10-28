"""Freeze only the authorized new pose proposals and old-control context schedule."""
import hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0274'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0274')
RELEASE='495b17078cb57fdbd47d8257d33202c57beb058e'
sys.path[:0]=[str(BASE/'PI0273/code'),str(BASE/'PI0267/code'),str(BASE/'PI0266/code'),str(BASE/'PI0250/runtime/reference-317a/src')]
from quality_gate import admitted,read,sha
from pi0255_motion3 import write as replace_write
from pi0267_expert import ArclengthCorrection

def write(p,value):
    if p.exists():assert read(p)==json.loads(json.dumps(value)),str(p)
    else:replace_write(p,value)

def main():
    admitted();closed=read(BASE/'PI0273/evidence/FINAL_CLOSEOUT_RECEIPT.json')
    assert closed['status']=='PI0273_FULL_SUPPORT_REPLAY_RELIABILITY_COMPLETE' and not closed['missing_scientific_requirements']
    assert closed['terminal_canonical']==5990237456
    authority=read(ROOT/'authority/CANONICAL.json');assert authority['id']==5990552027 and RELEASE in authority['body']
    old=BASE/'PI0273/design/FITTING_CARD.json';assert sha(old)=='0f455c3f0ebf7528b95cbb4f4a4d53fa671f27e987697176cc65ce1f610344ac'
    card=read(old);scene=read(BASE/'PI0267/design/SCENE_CARD.json');reg=scene['registry']
    families=[f for f in scene['families'] if f['split']=='FEEDBACK_TRAIN'];assert len(families)==24
    templates=read(BASE/'PI0270/design/PAIRING_PLAN.json')['templates']
    names=['center','rx+','rx-','ry+','ry-','rz+','rz-','px+','px-','py+','py-','pz+','pz-']
    rows=[];noise=[]
    for family in families:
        for task in ('M1','M3'):
            for command in (0,1):
                teacher=ArclengthCorrection(reg,family,task,command);gp,gr=teacher.at_length(teacher.ell[-1]);gp,gr=gp[0],gr[0]
                cell=f"{family['family_id']}-{task}-c{command}"
                language=templates['T0'].format(color=family['factors']['colors'][command]) if task=='M1' else ('Approach the black endpoint, then follow the open contour counterclockwise to the white endpoint as seen from above.' if command==0 else 'Approach the white endpoint, then follow the open contour clockwise to the black endpoint as seen from above.')
                for index,name in enumerate(names):
                    p,r=gp.copy(),gr.copy()
                    if 1<=index<=6:
                        axis=(index-1)//2;sign=1 if index%2 else -1;r=Rotation.from_rotvec(np.eye(3)[axis]*sign*.05).as_matrix()@r
                    elif index>=7:
                        axis=(index-7)//2;sign=1 if index%2 else -1;p[axis]+=.005*sign
                    rows.append(dict(id=cell+'-'+name,cell=cell,family=family['family_id'],task=task,command=command,variant=name,variant_index=index,
                        training_permitted=index<7,position_world=p.tolist(),rotation_world=r.tolist(),goal_position_world=gp.tolist(),goal_rotation_world=gr.tolist(),field_language=language))
                directory=RAW/'noise-field';directory.mkdir(parents=True,exist_ok=True)
                for j in (0,1):
                    seed=int.from_bytes(hashlib.sha256(f"PI0274:FIELD:27403:{family['family_id']}:{task}:{command}:{j}".encode()).digest()[:8],'big')%2**32
                    value=np.random.Generator(np.random.PCG64(seed)).standard_normal((1,40,132),dtype=np.float32)
                    path=directory/(cell+f'-j{j}.npz')
                    if path.exists():assert np.array_equal(np.load(path)['noise'],value)
                    else:np.savez(path,noise=value)
                    noise.append(dict(cell=cell,j=j,file=str(path),name='noise',seed=seed,noise_sha256=hashlib.sha256(value.tobytes()).hexdigest(),file_sha256=sha(path)))
    assert len(rows)==1248 and sum(r['training_permitted'] for r in rows)==672 and len(noise)==192
    counters=Counter();stream=[]
    for visit in card['streams']['BROAD']:
        v=dict(visit);source=card['base'][visit['source_index']]
        if visit['stream']=='SUPPLEMENT':
            cell=f"{source['family']}-{source['task']}-c{source['command']}";index=counters[cell]%7;counters[cell]+=1
            v.update(pose_id=cell+'-'+names[index],language=source['language'],family=source['family'],task=source['task'],command=source['command'])
        stream.append(v)
    used=Counter(v['pose_id'] for v in stream if v['stream']=='SUPPLEMENT')
    assert len(stream)==10240 and sum(used.values())==2048 and len(used)==672
    assert set(used)=={r['id'] for r in rows if r['training_permitted']}
    write(ROOT/'design/POSE_CARD.json',dict(release=RELEASE,scene_sha256=sha(BASE/'PI0267/design/SCENE_CARD.json'),rows=rows,noise=noise,world_left_rotation=True,zero_physics=True))
    write(ROOT/'design/TRAIN_PLAN.json',dict(release=RELEASE,source_card_sha256=sha(old),source_card=str(old),base=card['base'],stream=stream,
        occurrences_per_pose=dict(used),contexts_per_cell=dict(counters),source_seed_contract='PI0273:TRAIN:27303:{update}',formal_updates=5120,base_micros=8192,supplement_micros=2048))
    frozen=[]
    for name in ('RUNTIME_EVALUATION_CARD.json','RUNTIME_OFFLINE_CARD.json','RUNTIME_COMPONENT_CARD.json','NEW_FIT192.json','NEWFIT_QUERY_CARD.json','SUPPORT_CARD.json','LATE_QUERY_CARD.json'):
        p=BASE/'PI0273/design'/name;frozen.append(dict(path=str(p),sha256=sha(p)))
    sources=[BASE/'PI0267/code/pi0267_expert.py',BASE/'PI0267/code/pi0267_virtual_renderer.py',BASE/'PI0267/code/pi0267_virtual_data.py',BASE/'PI0266/code/pi0266_expert.py']
    write(ROOT/'evidence/A0_SOURCE_BINDING.json',dict(status='PASS_CLOSED_CONTROL_AND_FROZEN_NEW_PROPOSALS',release=RELEASE,predecessor_closeout_sha256=sha(BASE/'PI0273/evidence/FINAL_CLOSEOUT_RECEIPT.json'),
        old_BROAD_sample_sha256=sha(BASE/'PI0273/evidence/training-proofs/BROAD/main/training/sample_audit.jsonl'),old_BROAD_rng_sha256=sha(BASE/'PI0273/evidence/training-proofs/BROAD/main/training/step_rng.jsonl'),
        frozen_endpoint_cards=frozen,sources=[dict(path=str(p),sha256=sha(p)) for p in sources],proposals=1248,training_permitted=672,supplement_visits=2048,model_queries=0,teacher_calls=0,renders=0))
    print('PASS_FROZEN_1248_PROPOSALS_672_TRAIN_10240_SCHEDULE_AND_192_NOISE',flush=True)

if __name__=='__main__':main()
