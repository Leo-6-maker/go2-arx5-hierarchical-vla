"""Offline same-state teacher labels; never render or execute a learned model."""
import isaacgym
import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from admission import BASE, ROOT, read, sha, write
from plan import QUERY_INDICES

sys.path[:0] = [str(BASE / 'PI0267/code'), str(BASE / 'PI0266/code'), str(BASE / 'PI0250/runtime/reference-317a/src')]
from pi0267_expert import ArclengthCorrection
from pi0267_virtual_data import state, raw_gate
from pi0267_virtual_renderer import Renderer, visibility, collision
from pi0255_motion3 import scene
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix

RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())


def main(lane, family_id=None):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    def forbidden(*args, **kwargs): raise RuntimeError('OFFLINE_LABEL_NO_RENDERER')
    Renderer.__init__ = forbidden
    card = read(ROOT / 'design/COLLECTION_CARD.json')
    assert read(ROOT / 'evidence/A0_PREDECESSOR_CLOSEOUT.json')['status'] == 'PASS_CONTINUE_PI0270_A1'
    source_scene = read(card['source_scene']); reg = source_scene['registry']
    assert sha(card['source_scene']) == card['source_scene_sha256']
    anchor = np.array(source_scene['policy_anchor_world_m'])
    families = {f['family_id']:f for f in source_scene['families'] if f['split'] == 'FEEDBACK_TRAIN'}
    cases = card['cases'] if lane == 'original' else card['reused_components']
    if family_id is not None: cases = [c for c in cases if c['family'] == family_id]
    assert cases
    teachers = {}; journal = ROOT / 'evidence/teacher_calls.jsonl'
    for case in cases:
        fid, task, command = case['family'], case['task'], case['command']
        family = families[fid]; cell = fid, task, command
        if cell not in teachers: teachers[cell] = ArclengthCorrection(reg, family, task, command)
        teacher = teachers[cell]; data = scene(reg, family)
        if lane == 'original':
            key = case['case']; folder = RAW / 'collection/PREFIX4096' / key
            indices = QUERY_INDICES; result_path = folder / 'result.json'
        else:
            key = case['source_episode_key']; result_path = Path(case['result_path']); folder = result_path.parent
            indices = range(20); assert sha(result_path) == case['result_sha256']
        result = read(result_path)
        assert result['validity'] in ('VALID', 'PREQUERY_EXCLUDED') and result['point'] == 'PREFIX4096'
        assert (result['family'], result['task'], result['command'], result['backend']) == (fid, task, command, case['backend'])
        if result['validity'] == 'VALID': assert sha(folder / 'trajectory.npz') == result['arrays_sha256']
        observed = {q['replan']:q for q in result['queries']}
        for k in indices:
            out = RAW / 'labels' / lane / key / f'{k:03d}'
            out.mkdir(parents=True, exist_ok=True); record_path = out / 'record.json'
            if record_path.exists():
                prior = read(record_path); assert prior['source_result_sha256'] == sha(result_path)
                if prior.get('sample'): assert sha(prior['sample']) == prior['sample_sha256']
                continue
            row = dict(source_lane=lane, source_episode_key=key, family=fid, task=task, command=command,
                       backend=case['backend'], query_index=int(k), scheduled_s=float(k * .4),
                       source_result=str(result_path), source_result_sha256=sha(result_path), eligible=False,
                       start_type=case.get('start_type'), initialization=case.get('initialization', 'ORIGINAL_INITIAL_POSE'))
            if result['validity'] == 'PREQUERY_EXCLUDED' or k not in observed:
                row.update(status='PREQUERY_EXCLUDED' if result['validity'] == 'PREQUERY_EXCLUDED' else 'MISSING_AFTER_STOP',
                           reasons=result.get('reasons', [result.get('stop_reason')]))
                write(record_path, row); continue
            query = observed[k]; ip = folder / 'queries' / f'{k:03d}' / 'observation.npz'
            prediction = folder / 'queries' / f'{k:03d}' / 'prediction.npz'
            assert sha(prediction) == query['prediction_sha256']
            if query.get('observation_sha256'): assert sha(ip) == query['observation_sha256']
            with np.load(ip) as z:
                p=z['position_world'].copy(); r=z['rotation_world'].copy(); sv=z['state_v1'].copy(); now=float(z['timestamp'])
                body=z['body_rgb'].copy(); wrist=z['wrist_rgb'].copy()
            assert now == k * .4 and np.array_equal(sv, state(p, r, anchor))
            assert body.shape == wrist.shape == (96,96,3) and body.dtype == wrist.dtype == np.uint8
            assert np.isfinite(sv).all() and np.isfinite(p).all() and np.isfinite(r).all()
            camera = read(ip.parent / 'camera.json')
            assert camera['timestamp'] == now and camera['physics_steps'] == 0 and camera['RNG_unchanged']
            assert np.array_equal(camera['TCP_position_world'], p) and np.array_equal(camera['TCP_rotation_world'], r)
            with np.load(ip.parent / 'debug.npz') as z: frames = {k:z[k].copy() for k in z.files}
            frames.update(body=body, wrist=wrist)
            visible = visibility(frames, camera['scene_ids'], task, command, common_start=lane == 'original' and k == 0)
            current_hits = collision(p, r, np.array(data['obstacle']), np.array(reg['geometry']['obstacle_half_extents_m']))
            ordinal = sum(1 for _ in journal.open()) + 1 if journal.exists() else 1
            assert ordinal <= 7168
            append(journal, dict(ordinal=ordinal, source_lane=lane, source_episode_key=key, query_index=int(k),
                                 observation_sha256=sha(ip), proposal=f'{lane}/{key}/{k:03d}'))
            tp, tr, meta = teacher.label(p, r)
            cp, cr = teacher.at_length(meta['ell_m'] + .02 * np.arange(40) * .05)
            decay = np.exp(-2 * np.arange(40) * .05)
            expected_p = cp + decay[:,None] * (p - cp[0])
            expected_r = Rotation.from_rotvec(decay[:,None] * Rotation.from_matrix(r @ cr[0].T).as_rotvec()).as_matrix() @ cr
            assert np.max(abs(tp - expected_p)) < 1e-12 and np.max(abs(tr - expected_r)) < 1e-10
            assert np.linalg.norm(tp[0] - p) < 1e-12 and np.max(abs(tr[0] - r)) < 1e-10
            gate = raw_gate(tp, tr, p, r, anchor, now=now)
            hits = sorted(set(h for a,b in zip(tp,tr) for h in collision(a,b,np.array(data['obstacle']),np.array(reg['geometry']['obstacle_half_extents_m']))))
            reasons = []
            if not visible['pass_necessary']: reasons.append('VISIBILITY')
            if current_hits: reasons.append('CURRENT_TOOL_COLLISION')
            if not meta['valid_label']: reasons.append('UNOBSERVABLE_LABEL')
            if gate['decision'] == 'REJECT': reasons.append('LABEL_GATE_REJECT')
            if hits: reasons.append('LABEL_TOOL_COLLISION')
            raw = np.c_[tp-anchor, tr[:,:2,:].reshape(40,6), np.full(40,.5)].astype(np.float32)
            sample = out / 'sample.npz'; temp = out / 'sample.partial.npz'
            np.savez_compressed(temp, body_rgb=body, wrist_rgb=wrist, state_v1=sv, raw_action=raw,
                                position_world=p, rotation_world=r, label_position_world=tp, label_rotation_world=tr)
            with temp.open('rb') as f: os.fsync(f.fileno())
            os.replace(temp, sample)
            td = tp[8]-p; norm = float(np.linalg.norm(td))
            with np.load(prediction) as z: decoded = z['decoded'][0].copy()
            diagnostic = dict(prediction_finite=bool(np.isfinite(decoded).all()), hold_case=norm <= 1e-12,
                              teacher_displacement_m=td.tolist())
            if diagnostic['prediction_finite']:
                try:
                    lp=decoded[:,:3].astype(float)+anchor; lr=rot6d_rows_to_matrix(decoded[:,3:9].astype(float)); pd=lp[8]-p
                    diagnostic.update(first_position_bias_m=float(np.linalg.norm(lp[0]-p)),
                        prefix_xyz_rmse_m=float(np.sqrt(np.mean(np.sum((lp[:9]-tp[:9])**2,axis=1)))),
                        prefix_endpoint_error_m=float(np.linalg.norm(lp[8]-tp[8])), predicted_displacement_m=pd.tolist(),
                        directional_cosine=None if norm<=1e-12 or np.linalg.norm(pd)<=1e-12 else float(np.dot(pd,td)/(norm*np.linalg.norm(pd))),
                        terminal_current_error_m=float(np.linalg.norm(p-teacher.p[-1])),
                        terminal_teacher_prefix_error_m=float(np.linalg.norm(tp[8]-teacher.p[-1])),
                        terminal_predicted_prefix_error_m=float(np.linalg.norm(lp[8]-teacher.p[-1])))
                except ValueError: diagnostic['predicted_rotation_invalid'] = True
            row.update(status='VALID_SAME_STATE_LABEL' if not reasons else 'UNUSABLE_LABEL_KEEP_DENOMINATOR',
                       precodec_eligible=not reasons, reasons=reasons, teacher=meta, teacher_call_ordinal=ordinal,
                       visibility=visible, current_collision=current_hits, label_collision=hits, label_gate=gate,
                       observation_sha256=sha(ip), prediction=str(prediction), prediction_sha256=sha(prediction), actual_timestamp_s=now,
                       sample=str(sample), sample_sha256=sha(sample), current_processed_state_sha256=None,
                       first_pose_position_error_m=float(np.linalg.norm(tp[0]-p)),
                       first_pose_rotation_matrix_abs_error=float(np.max(abs(tr[0]-r))), diagnostic=diagnostic,
                       teacher_phase_or_source_time_in_policy=False, teacher_used_for_control=False,
                       model_queries=0, render_calls=0, physics_steps=0)
            write(record_path, row)
        print('LABELLED_FIXED_CASE', lane, key, len(indices), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--lane', choices=('original','component'), required=True); parser.add_argument('--family')
    args=parser.parse_args()
    with (ROOT / 'evidence/labels.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.lane,args.family)
