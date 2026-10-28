"""Actual frozen processor on new RGB/state/labels, CPU and no neural queries."""
import argparse
import hashlib
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1')
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from admission import BASE, ROOT, read, sha, write
from infer_worker import build, batch, identity
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix

RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def main(lane, family=None):
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30))
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    original_call=torch.nn.Module.__call__
    def guard(self,*a,**kw):
        if type(self).__module__.startswith('torchvision.transforms'):return original_call(self,*a,**kw)
        raise RuntimeError('NO_LEARNED_CPU_QUERY')
    torch.nn.Module.__call__=guard
    def forbidden(*a,**kw):raise RuntimeError('NO_CPU_OPTIMIZER')
    torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    stats=ROOT/'contracts/ROT/source/dataset/statistics.json'
    assert sha(stats)=='0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9'
    proc=build(read(stats),BASE/'PI0250/inputs/frozen_v1/cosmos_snapshot')
    plan=read(ROOT/'design/PAIRING_PLAN.json')
    from gr00t.data.embodiment_tags import EmbodimentTag
    import pi0242_train_infer_bridge as bridge
    records=sorted((RAW/'labels'/lane).glob('*/[0-9][0-9][0-9]/record.json'))
    assert records
    for path in records:
        row=read(path)
        if family is not None and row['family']!=family:continue
        target=RAW/'codec'/lane/row['source_episode_key']/f"{row['query_index']:03d}.json"
        if target.exists():
            assert read(target)['label_record_sha256']==sha(path)
            continue
        if not row.get('sample'):
            write(target,dict(source=row,eligible=False,label_record_sha256=sha(path),status=row['status']))
            continue
        assert sha(row['sample'])==row['sample_sha256']
        if row['task']=='M1':text=plan['templates']['T0'].format(color=plan['source_family_colors'][row['family']][row['command']])
        else:
            start,end,direction=('black','white','counterclockwise') if row['command']==0 else ('white','black','clockwise')
            text=f'Approach the {start} endpoint, then follow the open contour {direction} to the {end} endpoint as seen from above.'
        with np.load(row['sample']) as z:
            packet=dict(body_rgb=z['body_rgb'].copy(),wrist_rgb=z['wrist_rgb'].copy(),state_v1=z['state_v1'].copy(),language=text)
            label=z['raw_action'].copy()
        inputs=batch(proc,packet,label)['inputs']; condition={k:v for k,v in inputs.items() if k not in ('action','action_mask')}
        pure=batch(proc,packet)['inputs']
        assert identity(condition)==identity(pure),'TRAIN_EVAL_CONDITION_MISMATCH'
        prediction=Path(row['prediction']); receipt=read(prediction.with_suffix('.receipt.json'))
        assert sha(prediction)==row['prediction_sha256']==receipt['output_sha256']
        assert identity(condition)==receipt['input_sha256'],'ACTUAL_COLLECTOR_CROSS_PROCESS_INPUT_MISMATCH'
        state=inputs['state'].detach().cpu().numpy(); action=inputs['action'].detach().cpu().numpy(); mask=inputs['action_mask'].detach().cpu().numpy()
        assert state.shape==(1,1,132) and action.shape==mask.shape==(1,40,132)
        assert state.dtype==action.dtype==np.float32 and np.isfinite(state).all() and np.isfinite(action).all()
        assert not state[...,10:].any() and not action[...,10:].any() and np.all(mask[...,:10]==1) and not mask[...,10:].any()
        decoded=bridge._decode_physical(proc,EmbodimentTag.NEW_EMBODIMENT,action,packet['state_v1'])[0]
        xyz=float(np.max(np.linalg.norm(decoded[:,:3]-label[:,:3],axis=-1)))
        expected=rot6d_rows_to_matrix(label[:,3:9].astype(float)); actual=rot6d_rows_to_matrix(decoded[:,3:9].astype(float))
        so3=float(Rotation.from_matrix(actual@expected.transpose(0,2,1)).magnitude().max())
        grip=float(np.max(abs(decoded[:,9]-label[:,9])))
        codec_pass=xyz<=1e-5 and so3<=1e-4 and grip<=1e-5
        reasons=list(row['reasons'])+([] if codec_pass else ['CODEC_ROUNDTRIP'])
        value=dict(source=row,source_lane=lane,source_episode_key=row['source_episode_key'],family=row['family'],task=row['task'],command=row['command'],
             query_index=row['query_index'],start_type=row.get('start_type'),eligible=not reasons,
             status='ELIGIBLE_ACTUAL_PROCESSOR' if not reasons else 'UNUSABLE_KEEP_DENOMINATOR',reasons=reasons,
             label_record_sha256=sha(path),language=text,condition_sha256=identity(condition),
             processed_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),labels_masks_sha256=identity({k:inputs[k] for k in ('action','action_mask')}),
             collector_full_condition_bit_identity=True,train_eval_bit_identity=True,statistics_sha256=sha(stats),
             normalized_ranges=dict(state_min=float(state.min()),state_max=float(state.max()),action_min=float(action.min()),action_max=float(action.max())),
             roundtrip=dict(PASS=codec_pass,max_xyz_m=xyz,max_so3_rad=so3,max_gripper=grip),
             model_queries=0,optimizer_updates=0,source_code_sha256=sha(__file__))
        write(target,value)
    print('PASS_ACTUAL_NEW_SUPPORT_CODEC_NO_NEURAL_QUERY',lane,family,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lane',choices=('original','component'),required=True);p.add_argument('--family')
    a=p.parse_args();main(a.lane,a.family)
