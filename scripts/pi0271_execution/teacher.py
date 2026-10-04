"""Original same-pose teacher on the fixed saved AGG proposals, diagnosis only."""
import isaacgym
import argparse
import fcntl
import inspect
import json
import os
import sys
from pathlib import Path
from worker import BASE, ROOT, OLD, RAW, admitted, read, sha
from support import write


def main():
    admitted()  # Must precede every new teacher call, including engineering recovery.
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (16*2**30, 16*2**30)); os.nice(19)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    if hasattr(os, 'sched_getaffinity'): os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    import numpy as np
    from scipy.spatial.transform import Rotation
    sys.path[:0] = [str(BASE/'PI0267/code'), str(BASE/'PI0266/code'), str(BASE/'PI0250/runtime/reference-317a/src')]
    from pi0267_expert import ArclengthCorrection
    from pi0267_virtual_data import state, raw_gate
    from pi0267_virtual_renderer import Renderer, visibility, collision
    from pi0255_motion3 import scene
    assert sha(BASE/'PI0267/code/pi0267_expert.py') == '54a27e28376862e2c23e16e73368b7af7f96b088a8e2d72a66142d869b90ec79'
    assert sha(BASE/'PI0267/code/pi0267_virtual_renderer.py') == 'd0250ea351adbe4705d3ffb0d7cb4ac808f414f7ac88cf91d67373a2054fbfea'
    def forbidden(*a, **kw): raise RuntimeError('DIAGNOSIS_NO_RENDERER_OR_EPISODE')
    Renderer.__init__ = forbidden
    card_path = ROOT/'design/VISITED_STATE_CARD_V2.json'; card = read(card_path)
    assert len(card['rows']) == 64
    scene_path = BASE/'PI0267/design/SCENE_CARD.json'; source = read(scene_path)
    reg = source['registry']; anchor = np.asarray(source['policy_anchor_world_m'])
    families = {f['family_id']:f for f in source['families']}
    helpers = {Path(inspect.getsourcefile(h)) for h in (ArclengthCorrection, state, raw_gate, visibility, collision, scene)}
    sources = [dict(path=str(p), sha256=sha(p)) for p in sorted(helpers)]+[
        dict(path=str(p), sha256=sha(p)) for p in (card_path, scene_path, Path(__file__))]
    teachers = {}; checks = []; journal = ROOT/'evidence/teacher_calls.jsonl'
    with (ROOT/'evidence/teacher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for row in card['rows']:
            out = RAW/'labels-visited'/row['input_id']; out.mkdir(parents=True, exist_ok=True)
            record = out/'record.json'
            if record.exists():
                prior = read(record); assert prior['row'] == row and prior['card_sha256'] == sha(card_path)
                if prior.get('sample'): assert sha(prior['sample']) == prior['sample_sha256']
                checks.append(dict(path=str(record), sha256=sha(record))); continue
            result = dict(row=row, card_sha256=sha(card_path), source_files=sources,
                DIAGNOSIS_ONLY=True, never_train_or_blind_TEST=True, teacher_used_for_control=False,
                model_queries=0, render_calls=0, physics_steps=0)
            if row['status'] == 'MISSING_AFTER_STOP' or row.get('camera_metadata_status') != 'PASS_SAME_QUERY_TCP_TIME_AND_DEBUG':
                result.update(status='MISSING_KEEP_DENOMINATOR', eligible=False, reasons=[row.get('camera_metadata_status', row['status'])])
                write(record, result); checks.append(dict(path=str(record), sha256=sha(record))); continue
            observation = Path(row['request']['input']); assert sha(observation) == row['observation_sha256']
            assert sha(row['camera_file']) == row['camera_sha256'] and sha(row['debug_file']) == row['debug_sha256']
            with np.load(observation, allow_pickle=False) as z:
                p=z['position_world'].copy(); r=z['rotation_world'].copy(); sv=z['state_v1'].copy(); now=float(z['timestamp'])
                body=z['body_rgb'].copy(); wrist=z['wrist_rgb'].copy()
            assert now == row['designated_timestamp_s'] and np.array_equal(sv, state(p,r,anchor))
            camera = read(row['camera_file'])
            assert camera['timestamp'] == now and camera['physics_steps'] == 0 and camera['RNG_unchanged']
            assert np.array_equal(camera['TCP_position_world'],p) and np.array_equal(camera['TCP_rotation_world'],r)
            with np.load(row['debug_file'], allow_pickle=False) as z: frames={k:z[k].copy() for k in z.files}
            frames.update(body=body, wrist=wrist)
            visible=visibility(frames,camera['scene_ids'],row['task'],row['command'],common_start=row['query_index']==0)
            cell=(row['family'],row['task'],row['command']); family=families[row['family']]
            if cell not in teachers: teachers[cell]=ArclengthCorrection(reg,family,row['task'],row['command'])
            teacher=teachers[cell]; data=scene(reg,family); obstacle=np.asarray(data['obstacle']); half=np.asarray(reg['geometry']['obstacle_half_extents_m'])
            current_hits=collision(p,r,obstacle,half)
            ordinal=sum(1 for _ in journal.open())+1 if journal.exists() else 1
            assert ordinal <= 80
            with journal.open('a') as stream:
                stream.write(json.dumps(dict(ordinal=ordinal,input_id=row['input_id'],observation_sha256=sha(observation),
                    teacher_source_sha256=sha(BASE/'PI0267/code/pi0267_expert.py')),sort_keys=True)+'\n')
                stream.flush(); os.fsync(stream.fileno())
            tp,tr,meta=teacher.label(p,r)
            cp,cr=teacher.at_length(meta['ell_m']+.02*np.arange(40)*.05); decay=np.exp(-2*np.arange(40)*.05)
            assert np.max(abs(tp-(cp+decay[:,None]*(p-cp[0])))) < 1e-12
            expected_r=Rotation.from_rotvec(decay[:,None]*Rotation.from_matrix(r@cr[0].T).as_rotvec()).as_matrix()@cr
            assert np.max(abs(tr-expected_r)) < 1e-10
            assert np.linalg.norm(tp[0]-p)<1e-12 and np.max(abs(tr[0]-r))<1e-10
            gate=raw_gate(tp,tr,p,r,anchor,now=now)
            hits=sorted(set(h for a,b in zip(tp,tr) for h in collision(a,b,obstacle,half)))
            reasons=[]
            if not visible['pass_necessary']: reasons.append('VISIBILITY')
            if current_hits: reasons.append('CURRENT_TOOL_COLLISION')
            if not meta['valid_label']: reasons.append('UNOBSERVABLE_LABEL')
            if gate['decision']=='REJECT': reasons.append('LABEL_GATE_REJECT')
            if hits: reasons.append('LABEL_TOOL_COLLISION')
            raw=np.c_[tp-anchor,tr[:,:2,:].reshape(40,6),np.full(40,.5)].astype(np.float32)
            sample=out/'sample.npz'; temp=out/'sample.partial.npz'
            np.savez_compressed(temp,body_rgb=body,wrist_rgb=wrist,state_v1=sv,raw_action=raw,
                position_world=p,rotation_world=r,label_position_world=tp,label_rotation_world=tr)
            with temp.open('rb') as stream: os.fsync(stream.fileno())
            os.replace(temp,sample)
            result.update(status='VALID_SAME_STATE_LABEL' if not reasons else 'UNUSABLE_LABEL_KEEP_DENOMINATOR',
                eligible=not reasons,reasons=reasons,teacher=meta,teacher_call_ordinal=ordinal,visibility=visible,
                current_collision=current_hits,label_collision=hits,label_gate=gate,sample=str(sample),sample_sha256=sha(sample),
                current_world_pose_and_time_from_saved_observation=True,teacher_phase_or_source_time_in_policy=False,
                terminal_position_world=teacher.p[-1].tolist(),terminal_rotation_world=teacher.r[-1].tolist())
            write(record,result); checks.append(dict(path=str(record),sha256=sha(record)))
            print('SAME_SAVED_STATE_LABEL',row['input_id'],result['status'],flush=True)
        assert len(checks)==64
        write(ROOT/'evidence/VISITED_LABELS_COMPLETE.json',dict(status='ALL_64_PROPOSALS_LABELLED_OR_MISSING_RETAINED',
            records=checks,source_files=sources,teacher_attempts=sum(1 for _ in journal.open()) if journal.exists() else 0,
            model_queries=0,renders=0,episodes=0,DIAGNOSIS_ONLY=True,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--admission-only',action='store_true');args=parser.parse_args()
    if args.admission_only:
        try:admitted()
        except FileNotFoundError as error:
            assert Path(error.filename)==ROOT/'evidence/A0_DECISION.json'
            print('PASS_SAME_STATE_TEACHER_BLOCKED_BEFORE_ANY_NEW_CALL_MISSING_A0')
        else:print('ACTUAL_A0_ADMISSION_PRESENT_NO_TEACHER_CALL')
    else:main()
