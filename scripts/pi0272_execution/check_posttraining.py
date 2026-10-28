"""Finite CPU checks of reporting and pre-model admission; no learned forwards."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
from pathlib import Path
import numpy as np
import quality_worker as worker
import evaluate
import materialize
import report
from condition_comparison import self_check


def main():
    import torch
    import pi0260_r2_codec as codec
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('CPU_CHECK_NO_NEURAL_QUERY')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    assert not torch.cuda.is_available()
    self_check();scope=report.old.metric_helpers()
    original_root=worker.ROOT;fixture=original_root/'runtime/posttraining-fixtures';fixture.mkdir(parents=True,exist_ok=True)
    worker.ROOT=fixture
    (fixture/'evidence').mkdir(exist_ok=True)
    worker.write(fixture/'evidence/PRECISION_CONTRACT_RESOLUTION.json',worker.read(original_root/'evidence/PRECISION_CONTRACT_RESOLUTION.json'))
    try:
        try:worker.admitted()
        except FileNotFoundError as error:assert str(error.filename).endswith('/MAIN_TRAINING_AUDIT.json')
        else:raise AssertionError('Premature evaluation was admitted')
    finally:worker.ROOT=original_root
    prior=worker.read(worker.BASE/'PI0271/evidence/QUERY_OUTPUTS_COMPLETE.json')
    row=next(r for r in prior['rows'] if r['point']=='AGG4096' and r['panel']=='A1')
    path=Path(row['path']);assert worker.sha(path)==row['sha256']
    proc=codec.processor(worker.read(worker.BASE/'PI0270/contracts/ROT/source/dataset/statistics.json'))
    s4=scope['replay'](path,proc,torch,codec)
    # Saved S4 bytes exercise only the unchanged decoder, not a claimed DIRECT prediction.
    with np.load(path,allow_pickle=False) as z:
        packet={name:z[name].copy() for name in ('raw','decoded','state')}
    packet['zero_query']=np.zeros_like(packet['raw'])
    target=fixture/'constructed_decode_only.npz';np.savez(target,**packet)
    direct=report.direct_replay(target,proc,codec);assert direct['codec_max_abs']<=1e-7
    packet['raw']=np.full_like(packet['raw'],np.nan);packet['decoded']=np.full_like(packet['decoded'],np.nan)
    invalid=fixture/'constructed_invalid_decode.npz';np.savez(invalid,**packet)
    assert report.direct_replay(invalid,proc,codec)['codec_max_abs'] is None
    sources=worker.read(worker.BASE/'PI0271/design/LATE_SUPPORT_CARD.json')['rows']
    source=next(r for r in sources if r['input_id']==row['input_id'])
    with np.load(source['input'],allow_pickle=False) as z:expert=z['raw_action'].copy()
    valid,_,ok=report.old.measure(path,source['input'],expert,scope);assert ok and valid['teacher_metrics_available']
    rejected,_,ok=report.old.measure(invalid,source['input'],expert,scope);assert not ok and rejected['gate']=='REJECT'
    files=[Path(m.__file__) for m in (worker,evaluate,materialize,report)]+[Path(__file__)]
    worker.write(original_root/'evidence/POSTTRAINING_CPU_CHECK.json',dict(
        status='PASS_CPU_ADMISSION_BLOCK_REPLAY_INVALID_RETENTION_AND_ARITHMETIC',source_files=[dict(path=str(p),sha256=worker.sha(p)) for p in files],
        reused_S4_fixture=s4,constructed_DIRECT_decode_only=direct,actual_DIRECT_model_result=False,
        missing_actual_training_blocks_model_loading=True,invalid_predictions_retained=True,model_queries=0,gradient_queries=0))
    print('PASS_CPU_ADMISSION_BLOCK_REPLAY_INVALID_RETENTION_AND_ARITHMETIC',flush=True)


if __name__=='__main__':main()
