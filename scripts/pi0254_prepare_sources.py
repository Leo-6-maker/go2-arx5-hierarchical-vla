"""CPU-only materialization from the original TRAIN dataset, without model queries."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--code',type=Path,required=True)
    args=parser.parse_args()
    sys.path[:0]=[str(args.code/'scripts'),str(args.code/'src')]
    import numpy as np
    import pi0238_audit_to_sim as mature
    import cv2
    root,data=args.root,args.dataset
    builder=json.loads((data/'meta/stage19_dataset_manifest.json').read_text())
    missing=[name for name in builder['file_manifest'] if not (data/name).is_file()]
    assert not missing, missing[:5]
    rows=json.loads((root/'design/support576.json').read_text())['rows']+json.loads((root/'design/P64.json').read_text())['rows']
    assert len(rows)==len({r['source_episode_id'] for r in rows})==640
    provenance={}
    times={}
    for name,target,key in [('stage19_episode_provenance.jsonl',provenance,'source_episode_id'),('stage19_action_timestamps.jsonl',times,'raw_episode_id')]:
        for line in (data/'meta'/name).read_text().splitlines():
            row=json.loads(line);target[row[key]]=row
    directory=root/'source640';directory.mkdir(parents=True,exist_ok=True)
    entries=[]
    for ordinal,row in enumerate(rows):
        source=row['source_episode_id'];index=row['episode_index'];p=provenance[source];t=times[source]
        assert p['episode_index']==index==t['dataset_episode_index']
        assert p['source_frame']=='NATIVE_WBC_RELATIVE_REFERENCE' and p['desired_label']
        state,expert,timestamps=mature.load_window(data,row)
        assert np.array_equal(timestamps,np.asarray(t['future_action_point_timestamps']))
        assert timestamps[0]==row['observation_timestamp']==t['observation_timestamp']
        assert np.allclose(np.diff(timestamps),.05,atol=1e-9,rtol=0)
        frames=[];input_files={}
        parquet=data/f'data/chunk-{index//1000:03d}/episode_{index:06d}.parquet'
        for camera in ('external_rgb','wrist_rgb'):
            path=data/f'videos/chunk-{index//1000:03d}/observation.images.{camera}/episode_{index:06d}.mp4'
            cap=cv2.VideoCapture(str(path));ok,frame=cap.read();cap.release()
            assert ok,path
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            assert frame.shape==(96,96,3) and frame.dtype==np.uint8
            frames.append(frame);input_files[str(path.relative_to(data))]=sha(path)
        input_files[str(parquet.relative_to(data))]=sha(parquet)
        assert all(builder['file_manifest'][name]==value for name,value in input_files.items())
        path=directory/f'source-{index:06d}.npz'
        arrays=dict(state=state,expert=expert,timestamps=timestamps,body_rgb=frames[0],wrist_rgb=frames[1])
        if path.exists():
            with np.load(path,allow_pickle=False) as previous:
                assert set(previous.files)==set(arrays) and all(np.array_equal(previous[k],v) for k,v in arrays.items())
        else:
            temporary=path.with_suffix('.partial')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        entries.append(dict(source_episode_id=source,episode_index=index,path=path.name,sha256=sha(path),bytes=path.stat().st_size,
                            raw_input_files=input_files,provenance=p,timestamp_record=t))
        if ordinal%64==0:print(json.dumps(dict(status='CPU_SOURCE_PREPARED',count=ordinal+1)),flush=True)
    receipt=dict(schema='PI0254_RAW_TRAIN_INPUTS_V1',status='PASS_ORIGINAL_FILES_LABEL_STATE_TIME_BINDING',
                 dataset=str(data),eligible_windows=9792,required_original_files=len(builder['file_manifest']),missing_files=missing,
                 builder_manifest_sha256=sha(data/'meta/stage19_dataset_manifest.json'),helper_sha256=sha(Path(mature.__file__)),
                 opencv_version=cv2.__version__,files=entries,new_model_queries=0,new_optimizer_updates=0)
    target=directory/'manifest.json';text=json.dumps(receipt,sort_keys=True)+'\n'
    if target.exists():assert target.read_text()==text
    else:target.write_text(text)
    print(json.dumps(dict(status=receipt['status'],windows=len(entries),bytes=sum(e['bytes'] for e in entries))),flush=True)


if __name__=='__main__':main()
