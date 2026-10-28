"""Read only the 13 exposed source windows; missing raw collector evidence stays unresolved."""
from pathlib import Path
import json
import hashlib
import sys

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT=BASE/'PI0254'
sys.path[:0]=[str(BASE/'PI0250/runtime/reference-317a/src'),str(BASE/'PI0250/runtime/reference-317a/scripts')]


def main():
    import pyarrow.parquet as pq
    import numpy as np
    from go2arx5_vla.trajectory.so3 import geodesic_angle,rot6d_rows_to_matrix
    summary=json.loads((BASE/'PI0253/p0_review/summary.json').read_text())
    wanted=set(summary['union_sources']);assert len(wanted)==13
    data=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0253/original_calib_inputs/train_calib_dataset')
    metadata={}
    for filename,key in [('stage19_episode_provenance.jsonl','source_episode_id'),('stage19_action_timestamps.jsonl','raw_episode_id'),('stage19_calibration_bundles.jsonl','source_episode_id')]:
        rows=[json.loads(line) for line in (data/'meta'/filename).read_text().splitlines()]
        if filename=='stage19_calibration_bundles.jsonl':
            metadata[filename]=[r for r in rows if r['raw_episode_id'] in wanted]
        else:metadata[filename]={r[key]:r for r in rows if r[key] in wanted}
    result=[]
    for source in sorted(wanted):
        p=metadata['stage19_episode_provenance.jsonl'][source];t=metadata['stage19_action_timestamps.jsonl'][source]
        index=p['episode_index'];file=data/f'data/chunk-{index//1000:03d}/episode_{index:06d}.parquet'
        table=pq.read_table(file).to_pydict()
        state=np.asarray(table['observation.state'][0],np.float32);first=np.asarray(table['action'][0],np.float32)
        assert table['timestamp'][0]==t['observation_timestamp']==p['sample_timestamps'][0]
        assert p['source_frame']=='NATIVE_WBC_RELATIVE_REFERENCE' and t['label_source']=='desired'
        angle=float(geodesic_angle(rot6d_rows_to_matrix(first[None,3:9]),rot6d_rows_to_matrix(state[None,3:9]))[0])
        result.append(dict(source_episode_id=source,episode_index=index,observation_timestamp=table['timestamp'][0],
            state_first=state.tolist(),desired_H40_first=first.tolist(),desired_current_rotation_rad=angle,
            parquet_path=str(file),parquet_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            provenance=p,timestamp_record=t,raw_collector_source_json=p['source_json'],
            achieved_audit_array='UNOBSERVED_HASH_ONLY',reset_settling_order='UNRESOLVED_SOURCE_EVIDENCE',
            raw_quaternion_convention='UNRESOLVED_SOURCE_EVIDENCE',native_to_world_transform='WORLD_REFERENCE_DISTANCE_UNOBSERVED',
            status='UNRESOLVED_SOURCE_EVIDENCE' if not p['source_json'] else 'RAW_SOURCE_PATH_REQUIRES_READBACK'))
    report=dict(schema='PI0254_NARROW_SOURCE_TRACE_V1',status='13_WINDOWS_LINKED_RAW_COLLECTOR_UNRESOLVED',rows=result,
                p0_summary_sha256=hashlib.sha256((BASE/'PI0253/p0_review/summary.json').read_bytes()).hexdigest(),
                calibration_bundles=metadata['stage19_calibration_bundles.jsonl'],
                new_model_queries=0,new_optimizer_updates=0,new_physical_episodes=0)
    path=ROOT/'source_trace.json';text=json.dumps(report,sort_keys=True)+'\n'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    print(json.dumps(dict(status=report['status'],windows=len(result),raw_source_paths=sum(bool(r['raw_collector_source_json']) for r in result))),flush=True)


if __name__=='__main__':main()
