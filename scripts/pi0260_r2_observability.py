"""Saved-data observable-input / future-label audit; no model or scene query."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pi0260_r2 import ROOT,read,write,sha
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix

def run():
    result=read(ROOT/'dataset/codec_receipt.json');assert result['status']=='COMPACT_30_10'
    groups={};conflicts=[];family_max={};window_count=0;pair_count=0
    for family in result['families']:
        if not family['qualified']:continue
        for entry in family['sequences']:
            seq=entry['sequence'];rec=read(ROOT/'dataset'/(seq+'.json'))
            with np.load(ROOT/'dataset'/(seq+'.npz')) as z:
                states=z['state_v1'];labels=z['raw_action'];decoded=z['decoded_action']
            for j,row in enumerate(rec['rows']):
                if not(row['active'] and row['usable']):continue
                with np.load(ROOT/row['frame']) as z:body=z['body_rgb'];wrist=z['wrist_rgb']
                h=hashlib.sha256()
                for value in (body.tobytes(),wrist.tobytes(),states[j].astype(np.float32).tobytes(),rec['sequence']['instruction'].encode()):
                    h.update(len(value).to_bytes(8,'big'));h.update(value)
                key=h.hexdigest();window_count+=1
                item=dict(sequence=seq,family=family['family'],split=family['split'],task=rec['sequence']['task'],
                    index=row['index'],row_index=j,time=row['time'],frame=row['frame'],frame_sha256=row['frame_sha256'],
                    input_identity=key,label_sha256=hashlib.sha256(labels[j].tobytes()).hexdigest())
                prior=groups.setdefault(key,[])
                for old,old_label in prior:
                    pair_count+=1;position=np.linalg.norm(labels[j,:,:3]-old_label[:,:3],axis=1)
                    rotation=Rotation.from_matrix(rot6d_rows_to_matrix(labels[j,:,3:9])@rot6d_rows_to_matrix(old_label[:,3:9]).transpose(0,2,1)).magnitude()
                    # Use original label roundtrip tolerances to report non-identical labels;
                    # these do NOT replace the eventual model-quality thresholds.
                    if position.max()<=1e-5 and rotation.max()<=1e-4:continue
                    delta=dict(xyz_rmse_m=float(np.sqrt(np.mean(position**2))),max_xyz_m=float(position.max()),
                        max_SO3_rad=float(rotation.max()),max_distance_index=int(position.argmax()))
                    c=dict(first=old,second=item,**delta);conflicts.append(c)
                    f=family['family']
                    if f not in family_max or delta['max_xyz_m']>family_max[f]['max_xyz_m']:family_max[f]=c
                prior.append((item,labels[j].copy()))
    summary=dict(status='EXACT_ALLOWED_INPUT_HAS_NONIDENTICAL_FUTURES',scope='all active usable post-codec windows; raw allowed input bytes; model inputs checked separately',
        windows=window_count,unique_inputs=len(groups),same_input_pairs=pair_count,nonidentical_future_pairs=len(conflicts),
        affected_families=len(family_max),maximum_xyz_rmse_m=max(x['xyz_rmse_m'] for x in conflicts),
        maximum_pointwise_xyz_m=max(x['max_xyz_m'] for x in conflicts),
        family_worst=family_max,conflicts=conflicts,
        original_label_roundtrip_tolerance=dict(position_m=1e-5,rotation_rad=1e-4),
        quality_thresholds_not_evaluated=True,interpretation='Identical present inputs at different synthetic times have different future labels. No phase field or target repair added.',
        new_model_loads=0,new_model_queries=0,new_IK=0,new_render=0,codec_receipt_sha256=sha(ROOT/'dataset/codec_receipt.json'))
    write(ROOT/'evidence/observability_aliases.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('family_worst','conflicts')}),flush=True)

def actual():
    from pi0260_r2_inputs import build,batch,identity,inventory
    aliases=read(ROOT/'evidence/observability_aliases.json')
    cosmos=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0250/inputs/frozen_v1/cosmos_snapshot')
    receipt=read(ROOT/'evidence/common_tokenizer_metadata_receipt.json')
    assert receipt['revision']=='9ce19a195e423419c349abfc86fd07178b230561'
    for row in receipt['files']:
        assert (cosmos/row['path']).stat().st_size==row['bytes'] and sha(cosmos/row['path'])==row['sha256']
    p=build(read(ROOT/'dataset/statistics.json'),cosmos);out=[]
    for family,pair in sorted(aliases['family_worst'].items()):
        rows=[]
        for side in ('first','second'):
            sample=pair[side];record=read(ROOT/'dataset'/(sample['sequence']+'.json'))
            with np.load(ROOT/sample['frame']) as z:
                packet=dict(body_rgb=z['body_rgb'],wrist_rgb=z['wrist_rgb'],state_v1=z['state_v1'],language=record['sequence']['instruction'])
            processed=batch(p,packet)
            rows.append(dict(source=sample,identity=identity(processed),inventory=inventory(processed)))
        assert rows[0]['identity']==rows[1]['identity'],'raw alias did not remain an actual input alias'
        out.append(dict(family=family,rows=rows,max_xyz_m=pair['max_xyz_m'],xyz_rmse_m=pair['xyz_rmse_m']))
    write(ROOT/'evidence/actual_processor_aliases.json',dict(status='CONFIRMED_ACTUAL_PROCESSOR_IDENTICAL_INPUT_CONFLICT',
        pairs=out,processor_source_sha256=sha(Path(__file__).with_name('pi0260_r2_inputs.py')),
        statistics_sha256=sha(ROOT/'dataset/statistics.json'),tokenizer_revision=receipt['revision'],
        new_model_loads=0,new_model_queries=0))
    print(json.dumps(dict(status='CONFIRMED_ACTUAL_PROCESSOR_IDENTICAL_INPUT_CONFLICT',families=len(out))),flush=True)

if __name__=='__main__':
    import sys
    run() if sys.argv[1]=='raw' else actual()
