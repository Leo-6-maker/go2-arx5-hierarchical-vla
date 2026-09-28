#!/usr/bin/env python3
"""Read-only CPU world-TCP replay and video from retained actual frames."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0252/pre_full_sim')
VIDEO_BIN = Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/stage13c/runtime_env/public_native/python_env/bin')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    code=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/reference-317a')
    sys.path[:0]=[str(code/'scripts'),str(code/'src')]
    from stage21_native_model_receding_horizon import _model_chunk
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,rot6d_rows_to_matrix
    controls, cells = [], []
    for task in 'ABCD':
        d = ROOT / 'controls' / (task+'_attempt'+('2' if task=='A' else '1'))
        r = json.loads((d / 'control.json').read_text())
        rows = [json.loads(x) for x in (d / 'rows.jsonl').read_text().splitlines()]
        distances = np.linalg.norm(np.asarray([x['achieved_world_m'] for x in rows])-r['target_world_m'], axis=1)
        assert np.isfinite(distances).all()
        assert abs(distances.min()-r['min_world_tcp_distance_m']) < 1e-12
        assert abs(distances[-1]-r['final_world_tcp_distance_m']) < 1e-12
        assert r['status'] == 'PASS' and r['complete_duration'] and r['physical_clean']
        controls.append(dict(task=task, status='PASS', min_m=float(distances.min()), final_m=float(distances[-1]),
            rows=len(rows), first_hit_tick=int(np.flatnonzero(distances<=.035)[0]), source_sha256=sha(d/'control.json')))
    for p in sorted((ROOT/'G').glob('*/output/PI0252_G_ONE_CHUNK_MANIFEST.json')):
        r = json.loads(p.read_text())
        rows_path = p.parent/'model_receding_rows.jsonl'
        rows = [json.loads(x) for x in rows_path.read_text().splitlines()]
        target = np.asarray(r['protocol']['target_position_world_m'])
        achieved = np.asarray([x['achieved_pos'] for x in rows]).reshape(-1,3)
        desired = np.asarray([x['desired_pos'] for x in rows]).reshape(-1,3)
        distance = np.linalg.norm(achieved-target,axis=1)
        tracking = np.linalg.norm(achieved-desired,axis=1)
        with np.load(r['model_reference']['artifact'],allow_pickle=False) as data:
            decoded = data['decoded_physical_action'].reshape(40,10)
        reference = decoded[:,:3].astype(np.float64)+r['identity']['anchor_world_m']
        ref_distance = np.linalg.norm(reference-target,axis=1)
        sampler=r['identity']['actual_sampler_receipt']
        with np.load(sampler['noise_artifact'],allow_pickle=False) as data:
            assert sha(Path(sampler['noise_artifact']))
            assert hashlib.sha256(data['raw_bytes'].tobytes()).hexdigest()==sampler['noise'][0]['raw_tensor_sha256']
            assert data['float_values'].shape==(1,40,132) and np.isfinite(data['float_values']).all()
            reconstructed=(data['raw_bytes'].reshape(-1).view('<u2').astype('<u4')<<16).view('<f4').reshape(1,40,132)
            assert np.array_equal(reconstructed,data['float_values'])
        assert sampler['head_calls']==4 and sampler['noise_count']==1 and sampler['status']=='PASS'
        assert np.isfinite(reference).all() and np.isfinite(achieved).all()
        assert abs(ref_distance.min()-r['model_reference']['decoded_world_target_stats']['min_distance_m'])<1e-12
        if len(rows):
            assert abs(distance.min()-r['physical_wbc']['achieved_world_target_stats']['min_distance_m'])<1e-12
            assert abs(tracking.max()-r['physical_wbc']['tracking_error_max_m'])<1e-12
        gate_reject = r['model_reference']['safety_gate']['decision']=='REJECT'
        summary=json.loads((p.parent/'model_receding_summary.json').read_text())
        case=summary['model_cases'][0]
        chunk,_=_model_chunk(case['response'],chunk_id=case['request_id'],observation_timestamp=0.,period_s=.05)
        state=np.asarray(case['state'])
        gate=TrajectorySafetyGateV1().evaluate(chunk,now=0.,achieved_position=state[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[3:9]))).as_dict()
        assert gate['decision']==r['model_reference']['safety_gate']['decision']
        assert [x['code'] for x in gate['issues']]==[x['code'] for x in r['model_reference']['safety_gate']['issues']]
        initial=case['initial_reference_replacement']
        assert all(initial[k] for k in ('arm_policy_history_unchanged','dog_policy_history_unchanged','physics_root_state_unchanged','physics_dof_state_unchanged','episode_length_unchanged'))
        assert not initial['physics_reset'] and not initial['physics_teleport'] and not initial['policy_history_reset']
        clean = not r['physical_wbc']['native_events'] and all(not x.get('invalid_state',False) and not x.get('fall_flag',False) and not x.get('joint_limit_termination',False) and not x.get('reset_buf',False) for x in rows)
        passed = bool(len(rows) and distance.min()<=.035 and clean and r['protocol']['complete_reference_duration'] and not gate_reject)
        assert r['qualification']==('PASS' if passed else 'FAIL')
        cells.append(dict(task=r['cell'], qualification=r['qualification'], gate=r['model_reference']['safety_gate']['decision'],
            reference_min_m=float(ref_distance.min()), reference_final_m=float(ref_distance[-1]),
            achieved_min_m=float(distance.min()) if len(rows) else None, achieved_final_m=float(distance[-1]) if len(rows) else None,
            first_hit_tick=int(np.flatnonzero(distance<=.035)[0]) if np.any(distance<=.035) else None,
            tracking_max_m=float(tracking.max()) if len(rows) else None, rows=len(rows),
            complete_duration=r['protocol']['complete_reference_duration'], physical_clean=clean,
            gate_cpu_replay='PASS', initial_bind_identity='PASS', actual_reset_buf_count=sum(bool(x['reset_buf']) for x in rows),
            legacy_summary_reset_events=summary['reset_events'],
            legacy_reset_counter_note='Original helper counts wrapper reference-completion done as reset_events; actual reset_buf and native termination remain zero.',
            manifest_sha256=sha(p), rows_sha256=sha(rows_path), sampler_receipt_sha256=r['identity']['actual_sampler_receipt_sha256']))
    videos=[]
    for frames_path in sorted(ROOT.glob('**/actual_camera_frames.npz')):
        out=frames_path.with_name('actual_body_wrist.mp4')
        with np.load(frames_path,allow_pickle=False) as data:
            body,wrist=data['body_rgb'],data['wrist_rgb']
            assert body.dtype==np.uint8 and wrist.dtype==np.uint8 and body.shape==wrist.shape
            n,h,w,c=body.shape
            if not out.exists():
                proc=subprocess.Popen([str(VIDEO_BIN/'ffmpeg'),'-v','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{2*w}x{h}','-framerate','50','-i','pipe:0','-an','-c:v','mpeg4','-q:v','2','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE)
                for b,wrist_frame in zip(body,wrist): proc.stdin.write(np.concatenate((b,wrist_frame),axis=1).tobytes())
                proc.stdin.close()
                assert proc.wait()==0
        probe=json.loads(subprocess.check_output([str(VIDEO_BIN/'ffprobe'),'-v','error','-show_streams','-of','json',str(out)]))['streams'][0]
        assert int(probe['nb_frames'])==n
        videos.append(dict(path=str(out),frames=n,sha256=sha(out),source_sha256=sha(frames_path)))
    invalid_path=ROOT/'controls/A_attempt1'
    invalid=json.loads((invalid_path/'control.json').read_text())
    deviation=dict(schema='PI0252_NATIVE_CONTROL_ENGINEERING_DEVIATION_V1', status='INVALID_EXPERT_TARGET_BINDING',
        original_evidence_preserved=True, source_sha256=sha(invalid_path/'control.json'),
        reason='Unbound archived expert path used old anchor; native anchor matches original PI0226 execution. Original PI0227 expert binding reused for canonical controls.',
        raw_expert_episodes=5, canonical_valid_expert_episodes=4, invalid_engineering_wbc_ticks=invalid['rows'],
        project_model_queries=0, optimizer_updates=0)
    (ROOT/'control_engineering_deviation.json').write_text(json.dumps(deviation,indent=2)+'\n')
    report=dict(schema='PI0252_PRE_FULL_CPU_REPLAY_V1', status='PASS_INDEPENDENT_CPU_REPLAY',
        controls=controls, one_chunk=cells, videos=videos, selected_checkpoint='NONE',
        model_queries=len(cells), optimizer_updates=0,
        decision='PI0252_R1_PRE_FULL_TRAIN_SIM_FAIL' if any(c['qualification']=='FAIL' for c in cells) else 'CONTINUE')
    report['deprecated_diagnostics']={'original_canary_oracle_world_target_stats':'INVALID_OLD_ANCHOR_UNBOUND_EXPERT_REFERENCE_NOT_A_SCORER',
        'original_canary_oracle_vs_model_position_rmse_m':'INVALID_UNMATCHED_SAMPLING_AND_ANCHOR_NOT_USED_FOR_QUALIFICATION',
        'qualification_dependency':'Actual model reference and achieved TCP versus fixed world target only; unaffected by discarded oracle side diagnostics.'}
    (ROOT/'cpu_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='videos'}),flush=True)

if __name__=='__main__': main()
