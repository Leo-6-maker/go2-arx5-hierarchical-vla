"""Independent saved-field Euler/decode/metric/gate replay, no learned calls."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import argparse,json,resource
from pathlib import Path
import numpy as np
from quality_gate import ROOT,RAW,read,sha
from quality_worker import write

def main(point,self_check=False):
    resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));os.nice(19)
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    import torch
    from report import helpers
    _,_,scope=helpers()
    from quality_offline import physical_score
    import pi0260_r2_codec as codec
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('FIELD_CPU_NO_LEARNED_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden;assert not torch.cuda.is_available()
    proc=codec.processor(read(ROOT/'contracts/ROT/source/dataset/statistics.json'));scope.update(processors={'ROT':proc},torch=torch,codec=codec)
    card=read(ROOT/'design/POSE_CARD.json');noise={(r['cell'],r['j']):r for r in card['noise']};checks=[]
    if not self_check:assert read(ROOT/'evidence'/f'FIELD_COMPLETE_{point}.json')['status']=='PASS_ALL2496_REGISTERED_FIELD_OUTPUTS'
    for proposal in card['rows']:
        for j in (0,1):
            record=RAW/'field'/point/proposal['id']/f'j{j}'/'record.json'
            if self_check and not record.exists():continue
            rec=read(record);assert rec['row']==proposal and rec['j']==j and rec['point']==point and rec['pose_card_sha256']==sha(ROOT/'design/POSE_CARD.json')
            tensor=Path(rec['tensor']);assert sha(tensor)==rec['tensor_sha256'];n=noise[proposal['cell'],j];assert sha(n['file'])==n['file_sha256']
            with np.load(n['file'],allow_pickle=False) as z:expected=z[n['name']].copy()
            with np.load(tensor,allow_pickle=False) as z:
                assert np.array_equal(z['noise'],expected) and expected.dtype==np.float32 and expected.shape==(1,40,132)
                decoded=z['decoded'].copy();state=z['state'].copy()
            check=scope['replay'](tensor,proc,torch,codec)
            sample=RAW/'data'/proposal['id']/'sample.npz'
            with np.load(sample,allow_pickle=False) as z:expert=z['raw_action'].copy();assert np.array_equal(state,z['state_v1'])
            # The gate metrics include chunk_id: replay the actual registered ID
            # and provider dtype, rather than a generic support-panel identifier.
            _,gate=physical_score(decoded,expert,state,f"PI0274-FIELD-{point}-{proposal['id']}-{j}")
            assert gate==rec['gate']
            if rec['metrics'] is not None:
                actual=scope['independent_metrics'](decoded[0].astype(float),expert.astype(float));delta=max(abs(actual[k]-rec['metrics'][k]) for k in actual);assert delta<=1e-6
            else:delta=None;assert rec['gate']['decision']=='REJECT'
            check.update(point=point,pose_id=proposal['id'],j=j,record_sha256=sha(record),registered_noise_exact=True,original_gate_exact=True,original_gate_shared=True,actual_provider_dtype=str(decoded.dtype),independent_metrics_max_abs=delta);checks.append(check)
            if self_check:print('PASS_ONE_SAVED_FIELD_FULL_EULER_DECODE_NOISE_METRIC_GATE_FIXTURE',flush=True);return
    assert len(checks)==2496
    write(RAW/'delivery'/f'FIELD_CPU_{point}.json',dict(status='PASS_ALL2496_SAVED_FIELD_EULER_DECODE_NOISE_METRIC_GATE',point=point,checks=checks,model_queries=0,teacher_calls=0,renders=0,physics=0))
    print('PASS_ALL2496_SAVED_FIELD_CPU',point,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',choices=('BROAD5120','EQUIL5120'),required=True);p.add_argument('--self-check',action='store_true');a=p.parse_args();main(a.point,a.self_check)
