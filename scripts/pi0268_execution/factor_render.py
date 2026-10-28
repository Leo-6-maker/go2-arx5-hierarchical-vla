"""Fixed 304 evaluation-only pose factors, true matching images and labels."""
import isaacgym
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5'); ROOT=BASE/'PI0268/R1'; OLD=BASE/'PI0267'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0268/R1/factors')
SOURCE=RAW.parents[2]/'PI0267/R1/S2'
sys.path[:0]=[str(OLD/'code'),str(BASE/'PI0266/code')]
from pi0267_virtual_renderer import Renderer,collision,visibility,freeze
from pi0267_virtual_data import raw_gate,state
from pi0267_expert import ArclengthCorrection
from pi0267_virtual_episode import language
from pi0255_motion3 import scene,sha,write

def read(p):return json.loads(Path(p).read_text())
def main():
    card=freeze();reg=card['registry'];anchor=np.asarray(card['policy_anchor_world_m'])
    manifest=BASE/'PI0268/preparation/output/FACTOR_304_MANIFEST.json'
    conditions=read(manifest)['conditions'];assert len(conditions)==304
    RAW.mkdir(parents=True,exist_ok=True)
    receipt=RAW/'COMPLETE.json';assert not receipt.exists(),'first valid factor panel retained'
    frame_count=0;reused=0;results=[]
    for fid in sorted({r['base']['family'] for r in conditions}):
        family=next(f for f in card['families'] if f['family_id']==fid);assert family['split']=='FEEDBACK_TRAIN'
        folder=SOURCE/fid;done=read(folder/'COMPLETE.json')
        for name in ('proposals.npz','policy_frames.npz','debug_frames.npz','rows.json','frame_index.json'):
            assert sha(folder/name)==done['hashes'][name]
        z=np.load(folder/'proposals.npz');frames=np.load(folder/'policy_frames.npz');debug=np.load(folder/'debug_frames.npz')
        rows=read(folder/'rows.json');frame_index=read(folder/'frame_index.json')
        renderer=Renderer(reg,family);geometry=scene(reg,family)
        try:
            for condition in (r for r in conditions if r['base']['family']==fid):
                out=RAW/condition['condition_id'];out.mkdir(exist_ok=True)
                if (out/'record.json').exists():
                    record=read(out/'record.json');assert record['condition']==condition and sha(out/'input.npz')==record['input_sha256']
                    results.append(record);continue
                base=condition['base'];spec=condition['transform_spec_only'];i=base['index']
                p=z['position_world'][i].copy();r=z['rotation_world'][i].copy()
                axis=spec['axis']
                if axis is not None:
                    p[axis]+=spec['translation_m']
                    r=Rotation.from_rotvec(np.eye(3)[axis]*spec['world_left_rotation_rad']).as_matrix()@r
                key=hashlib.sha256(p.tobytes()+r.tobytes()).hexdigest()
                matches=[x for x in rows if x['frame_key']==key]
                if matches:
                    match=matches[0];j=match['frame_index'];mi=match['index']
                    assert np.array_equal(p,z['position_world'][mi]) and np.array_equal(r,z['rotation_world'][mi])
                    image=dict(body=frames['body_rgb'][j],wrist=frames['wrist_rgb'][j],**{k:debug[k][j] for k in debug.files})
                    info=frame_index[j];assert info['frame_key']==key
                    origin='EXACT_EXISTING_STATE_CAMERA_FRAME';reused+=1
                else:
                    journal=ROOT/'evidence/factor_frame_starts.jsonl'
                    ordinal=sum(1 for _ in journal.open())+1 if journal.exists() else 1
                    assert ordinal<=304,'FACTOR_FRAME_PAIR_CAP'
                    with journal.open('a') as f:f.write(json.dumps(dict(ordinal=ordinal,condition_id=condition['condition_id'],frame_key=key))+'\n')
                    image,info=renderer.place(p,r,0.);frame_count+=1;origin='NEW_MATCHING_ZERO_PHYSICS_RENDER'
                eye=np.linalg.inv(np.asarray(info['camera_view']['wrist']))[3,:3]
                expected=p+r@(np.asarray(reg['renderer']['wrist_attachment'])-np.array([.1424,0,.0001057]))
                assert np.max(abs(eye-expected))<1e-5
                teacher=ArclengthCorrection(reg,family,base['task'],base['command'])
                lp,lr,label=teacher.label(p,r);gate=raw_gate(lp,lr,p,r,anchor)
                half=np.asarray(reg['geometry']['obstacle_half_extents_m']);obstacle=np.asarray(geometry['obstacle'])
                hits=sorted(set(collision(p,r,obstacle,half)+[h for q,s in zip(lp,lr) for h in collision(q,s,obstacle,half)]))
                visible=visibility(image,renderer.scene_ids,base['task'],base['command'],common_start=False)
                reasons=[]
                if not label['valid_label']:reasons.append('UNOBSERVABLE_LABEL')
                if gate['decision']=='REJECT':reasons.append('GATE_REJECT')
                if hits:reasons.append('TOOL_COLLISION')
                if not visible['pass_necessary']:reasons.append('VISIBILITY')
                assert sha(condition['noise_file'])==condition['noise_file_sha256']
                with np.load(condition['noise_file']) as nz:noise=nz[condition['noise_key']].copy()
                raw_action=np.c_[lp-anchor,lr[:,:2,:].reshape(40,6),np.full(40,.5)].astype(np.float32)
                np.savez_compressed(out/'input.npz',body_rgb=image['body'],wrist_rgb=image['wrist'],state_v1=state(p,r,anchor),
                    raw_action=raw_action,position_world=p,rotation_world=r,noise=noise,
                    **{k:v for k,v in image.items() if k not in ('body','wrist')})
                record=dict(condition=condition,language=language(reg,family,base['task'],base['command']),status='ADMITTED_PRECODEC' if not reasons else 'EXCLUDED_KEEP_DENOMINATOR',reasons=reasons,
                    origin=origin,input_sha256=sha(out/'input.npz'),frame_key=key,camera=info,label=label,gate=gate,
                    collision_hits=hits,visibility=visible,TRAIN_ADMISSION=False,physics_steps=0,model_queries=0)
                write(out/'record.json',record);results.append(record)
            print('FACTOR_FAMILY',fid,len(results),flush=True)
        finally:renderer.close();z.close();frames.close();debug.close()
    assert len(results)==304 and {r['condition']['condition_id'] for r in results}=={r['condition_id'] for r in conditions}
    write(receipt,dict(status='COMPLETE_MATCHING_FACTOR_RENDER_PRECODEC',proposed=304,admitted_precodec=sum(r['status']=='ADMITTED_PRECODEC' for r in results),
        fresh_pairs_this_attempt=frame_count,reused_pairs_this_attempt=reused,conditions_manifest_sha256=sha(manifest),
        code_sha256=sha(__file__),renderer_sha256=sha(OLD/'code/pi0267_virtual_renderer.py'),physics_steps=0,model_queries=0,
        records={r['condition']['condition_id']:sha(RAW/r['condition']['condition_id']/'record.json') for r in results}))
    print(json.dumps(read(receipt)),flush=True)

if __name__=='__main__':main()
