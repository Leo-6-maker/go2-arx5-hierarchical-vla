"""Original decoder parity and descriptive pristine drift on already saved DROID32."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
from prepare_core import sha,write

def run(root,point):
    root=Path(root);assert os.environ['CUDA_VISIBLE_DEVICES']==''
    sys.path.append(json.loads((root/'design/RUNTIME_CARD.json').read_text())['parent_code'])
    from retention import original_policy
    import torch
    torch.set_num_threads(4)
    card=json.loads((root/'design/RETENTION32.json').read_text());policy=original_policy(None,card['original_model_root'],card['source_tag'])
    source=root/'retention'/point/'RESULT.json';result=json.loads(source.read_text());assert len(result['records'])==32
    lookup={r['id']:r for r in result['records']};records=[]
    for row in card['rows']:
        record=lookup[row['id']];assert sha(record['path'])==record['sha256']
        with np.load(row['input_path']) as f:state=f['state'].copy()
        states={k:state[a:b][None,None].copy() for k,(a,b) in card['state_slices'].items()}
        with np.load(record['path']) as f:
            raw=f['raw'].copy();decoded={k:f['decoded_'+k].copy() for k in card['state_slices']}
            assert str(f['input_sha256'])==row['input_sha256'] and np.array_equal(f['noise_uint16'],np.load(row['noise_path']))
        independent=policy.processor.decode_action(raw,policy.embodiment_tag,states)
        assert all(np.array_equal(independent[k].astype(np.float32),decoded[k]) for k in decoded)
        with np.load(row['pristine_output']) as f:
            old=f['raw'].copy();old_decoded={k:f['decoded_'+k].copy() for k in decoded}
        assert raw.shape==old.shape==(1,40,132)
        finite=bool(np.isfinite(raw).all() and all(np.isfinite(v).all() for v in decoded.values()))
        records.append(dict(id=row['id'],output_sha256=record['sha256'],pristine_sha256=row['pristine_output_sha256'],original_decoder_exact=True,finite=finite,raw_RMS_vs_pristine=float(np.sqrt(np.mean((raw-old)**2))) if finite else None,decoded_RMS_vs_pristine={k:float(np.sqrt(np.mean((decoded[k]-old_decoded[k])**2))) if finite else None for k in decoded}))
    report=dict(status='PASS_SAVED_DROID32_IDENTITY_NOISE_AND_ORIGINAL_DECODER_CPU',point=point,source_sha256=sha(source),records=records,finite_count=sum(r['finite'] for r in records),no_task_quality_threshold=True,interpretation='Numeric drift is descriptive; saved pristine predictions are not task-success ground truth or a complete multitask retention test.',model_instantiations=0,new_model_queries=0,new_physics=0,selected='NONE',deployment='NONE')
    write(root/'audit'/f'{point}_DROID_CPU.json',report);print(json.dumps(dict(status=report['status'],point=point,finite_count=report['finite_count'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--point',choices=('CTRL2048','LR5X2048'),required=True);a=p.parse_args();run(a.root,a.point)
