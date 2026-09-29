"""Package-only media/hash/time/numeric verification; no external experiment imports."""
import hashlib
from pathlib import Path
import numpy as np
from eef_video_workflow import read,write,sha,saved_path,decode,rows
from portable_numeric import screen_metrics,preparation_series
from evidence_so3 import geodesic_angle,rot6d_rows_to_matrix

def verify(root,manifest=True):
    out=root/'delivery';checked=0
    if manifest and (out/'source_manifest.json').exists():
        for f in read(out/'source_manifest.json')['files']:
            p=(out/f['path']).resolve();assert p.is_relative_to(out.resolve()),f['path']
            assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['path'];checked+=1
    complete=read(out/'completeness.json');deltas=[];issue_codes=set()
    for cell in complete['cells']:
        folder=saved_path(out,cell['path']);r=read(folder/'record.json')
        assert sha(folder/'tensors.npz')==r['tensor_sha256']
        inp=saved_path(out,complete['inputs'][cell['window']]);assert sha(inp/'condition.pt')==r['condition_sha256'] and sha(inp/'tensors.npz')==r['source_tensor_sha256']
        with np.load(folder/'tensors.npz',allow_pickle=False) as z:
            prediction=z['decoded'][0].astype(float);expert=z['expert'].astype(float)
            m=screen_metrics(prediction,expert);differences={k:abs(v-r['metrics'][k]) for k,v in m.items()};assert max(differences.values())<1e-9
            jump=float(geodesic_angle(rot6d_rows_to_matrix(z['state'][3:9].astype(float)),rot6d_rows_to_matrix(prediction[0,3:9])))
            assert abs(jump-r['gate']['metrics']['initial_rotation_discontinuity_rad'])<1e-9
            assert hashlib.sha256(z['noise_source_float32'].tobytes()).hexdigest()==r['noise_source_sha256']
        deltas.append(dict(cell=cell['id'],max_absolute_metric_delta=max(differences.values()),initial_rotation_jump_rad=jump,metrics=m))
        issue_codes.update(x['code'] for x in r['gate']['issues'])
    assert len(deltas)==96
    for case in ('pi0256-prep-A00','pi0256-prep-B00','pi0256-prep-Bfix00'):
        preparation_series(rows(out/'preparation/PI0256/episodes'/case/'raw_ticks.jsonl'))
    videos=[decode(p) for p in sorted(out.rglob('*.mp4'))]
    for number in ('01','02','03'):
        p=next(p for p in out.glob(number+'_*.mp4') if '_1x' not in p.name)
        video=next(v for v in videos if Path(v['path'])==p);frames=read(out/f'{number}_frame_mapping.json')['frames'];assert len(frames)==video['decoded_frames']
        assert [f['output_frame'] for f in frames]==list(range(len(frames)))
        for frame in frames:
            if number=='01':assert all(p['source_timestamp_s']<=frame['reference_elapsed_s']+1e-9 for p in frame['panels'])
            elif number=='03':assert all(p['missing_before'] or p['source_timestamp_s']<=frame['simulation_time_s']+1e-9 for p in frame['panels'])
    receipt=dict(status='PASS_PORTABLE_SAVED_EVIDENCE_QA',manifest_files_checked=checked,videos=videos,numeric_cells=deltas,max_absolute_metric_delta=max(x['max_absolute_metric_delta'] for x in deltas),gate_issue_codes=sorted(issue_codes),scope='Package bytes, media, saved inputs/noise, decoded metrics, saved gate initial-jump metric and preparation telemetry only',not_reexecuted=['processor construction','raw-action decoder','full gate implementation','models','simulator'],new_queries=0,new_physics=0)
    write(root/'portable_verify_receipt.json',receipt)
    print('PASS_PORTABLE_SAVED_EVIDENCE_QA 96 cells; '+str(checked)+' manifest files',flush=True)
    return receipt
