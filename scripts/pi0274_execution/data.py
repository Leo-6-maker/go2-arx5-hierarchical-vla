"""First-valid actual RGB/StateV1/H40 production at the registered poses, zero physics."""
import isaacgym
import fcntl,hashlib,json,os,sys
from pathlib import Path
import numpy as np
sys.path[:0]=['/home/liuyu-nfs/Projects/go2-arx5/PI0267/code','/home/liuyu-nfs/Projects/go2-arx5/PI0266/code']
from pi0267_virtual_renderer import Renderer,collision,visibility
from pi0267_virtual_data import state,raw_gate
from pi0267_expert import ArclengthCorrection
from pi0255_motion3 import scene,sha,write

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0274');RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0274')
def read(p):return json.loads(p.read_text())
def append(name,row,cap):
    path=ROOT/'evidence'/name;n=sum(1 for _ in path.open()) if path.exists() else 0;assert n<cap
    with path.open('a') as f:f.write(json.dumps(dict(ordinal=n+1,**row),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def main():
    with (ROOT/'runtime/data.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for source in read(ROOT/'evidence/A0_SOURCE_BINDING.json')['sources']:assert sha(source['path'])==source['sha256']
        card=read(ROOT/'design/POSE_CARD.json');scene_card=read(ROOT.parent/'PI0267/design/SCENE_CARD.json');reg=scene_card['registry'];anchor=np.asarray(scene_card['policy_anchor_world_m'])
        assert sha(ROOT.parent/'PI0267/design/SCENE_CARD.json')==card['scene_sha256'];half=np.asarray(reg['geometry']['obstacle_half_extents_m'])
        results=[]
        for family in [f for f in scene_card['families'] if f['split']=='FEEDBACK_TRAIN']:
            renderer=Renderer(reg,family);geometry=scene(reg,family);teachers={}
            try:
                for row in [r for r in card['rows'] if r['family']==family['family_id']]:
                    out=RAW/'data'/row['id'];out.mkdir(parents=True,exist_ok=True);receipt=out/'receipt.json';sample=out/'sample.npz'
                    if receipt.exists():
                        prior=read(receipt);assert prior['proposal']==row and sha(sample)==prior['sample_sha256'];results.append(prior);continue
                    p,r=np.asarray(row['position_world']),np.asarray(row['rotation_world'])
                    key=row['task'],row['command']
                    if key not in teachers:teachers[key]=ArclengthCorrection(reg,family,*key)
                    label=out/'label.npz';meta_path=out/'teacher.json'
                    if not label.exists():
                        append('teacher_attempts.jsonl',dict(proposal=row['id']),1280)
                        lp,lr,meta=teachers[key].label(p,r);temp=out/'label.partial.npz';np.savez_compressed(temp,p=lp,r=lr,meta_json=np.array(json.dumps(meta)));temp.replace(label);write(meta_path,meta)
                    else:
                        with np.load(label) as z:lp,lr=z['p'].copy(),z['r'].copy();meta=json.loads(str(z['meta_json']))
                        if meta_path.exists():assert read(meta_path)==meta
                        else:write(meta_path,meta)
                    observation=out/'observation.npz';camera=out/'camera.json'
                    if not observation.exists():
                        append('data_frame_attempts.jsonl',dict(proposal=row['id']),1248)
                        frames,info=renderer.place(p,r,0.);temp=out/'observation.partial.npz';np.savez_compressed(temp,**frames,camera_json=np.array(json.dumps(info)));temp.replace(observation);write(camera,info)
                    else:
                        with np.load(observation) as z:frames={k:z[k].copy() for k in z.files if k!='camera_json'};info=json.loads(str(z['camera_json']))
                        if camera.exists():assert read(camera)==info
                        else:write(camera,info)
                    assert info['physics_steps']==0 and info['timestamp']==0. and info['placement_max_m']<1e-6
                    assert np.max(abs(np.asarray(info['TCP_position_world'])-p))<1e-12 and np.max(abs(np.asarray(info['TCP_rotation_world'])-r))<1e-12
                    eye=np.linalg.inv(np.asarray(info['camera_view']['wrist']))[3,:3];expected=p+r@(np.asarray(reg['renderer']['wrist_attachment'])-[.1424,0,.0001057]);assert np.max(abs(eye-expected))<1e-5
                    gate=raw_gate(lp,lr,p,r,anchor);check=visibility(frames,renderer.scene_ids,row['task'],row['command'],common_start=False)
                    hits=sorted(set(h for q,s in zip(lp,lr) for h in collision(q,s,np.asarray(geometry['obstacle']),half)))
                    reasons=([] if meta['valid_label'] else ['UNOBSERVABLE_LABEL'])+(['GATE_REJECT'] if gate['decision']=='REJECT' else [])+(['TOOL_COLLISION'] if hits else [])+([] if check['pass_necessary'] else ['VISIBILITY'])
                    sv=state(p,r,anchor);action=np.c_[lp-anchor,lr[:,:2,:].reshape(40,6),np.full(40,.5)].astype(np.float32)
                    temp=out/'sample.partial.npz';np.savez_compressed(temp,body_rgb=frames['body'],wrist_rgb=frames['wrist'],state_v1=sv,raw_action=action,position_world=p,rotation_world=r,label_position_world=lp,label_rotation_world=lr);temp.replace(sample)
                    record=dict(proposal=row,valid=not reasons,reasons=reasons,gate=gate,visibility=check,collision_hits=hits,teacher=meta,sample_sha256=sha(sample),
                        observation_sha256=sha(observation),camera_sha256=sha(camera),label_sha256=sha(label),physics_steps=0,teacher_source_sha256=sha(ROOT.parent/'PI0267/code/pi0267_expert.py'))
                    write(receipt,record);results.append(record)
            finally:renderer.close()
            print('ACTUAL_POSE_DATA',family['family_id'],len(results),sum(not r['valid'] for r in results),flush=True)
        assert len(results)==1248
        failures=[dict(id=r['proposal']['id'],reasons=r['reasons']) for r in results if r['proposal']['training_permitted'] and not r['valid']]
        write(ROOT/'evidence/DATA_PRECODEC.json',dict(status='DATA_INCOMPLETE' if failures else 'PASS_ALL_672_TRAINING_POSES_PRECODEC',total_proposals=1248,
            valid=sum(r['valid'] for r in results),training_invalid=failures,diagnosis_only_invalid=[dict(id=r['proposal']['id'],reasons=r['reasons']) for r in results if not r['proposal']['training_permitted'] and not r['valid']],
            samples=[dict(id=r['proposal']['id'],path=str(RAW/'data'/r['proposal']['id']/'sample.npz'),sha256=r['sample_sha256'],valid=r['valid']) for r in results],new_model_queries=0,new_physics=0))
        print('DATA_PRECODEC',len(failures),'training failures',flush=True)

if __name__=='__main__':main()
