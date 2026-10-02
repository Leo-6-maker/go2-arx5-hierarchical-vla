"""Exercise metadata binding against the actual failed save, without model loading."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import tempfile
from pathlib import Path
from train import ROOT,bind_checkpoint_metadata,read,sha
import json


def main():
    output=ROOT/'OLD/failed-643670-continuous/training'
    stats=ROOT/'contracts/OLD/source/dataset/statistics.json'
    import torch
    def forbidden(*args,**kw):raise RuntimeError('CPU_METADATA_CHECK_NO_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden
    with tempfile.TemporaryDirectory(dir=ROOT/'runtime',prefix='metadata-check-') as folder:
        cp=Path(folder)
        bind_checkpoint_metadata(output,cp,stats,dict(statistics_sha256=sha(stats)))
        before={str(p.relative_to(cp)):sha(p) for p in cp.rglob('*') if p.is_file()}
        bind_checkpoint_metadata(output,cp,stats,dict(statistics_sha256=sha(stats)))
        assert before=={str(p.relative_to(cp)):sha(p) for p in cp.rglob('*') if p.is_file()} and len(before)==9
        try:bind_checkpoint_metadata(output,cp,ROOT/'contracts/ROT/source/dataset/statistics.json',{})
        except AssertionError as error:assert str(error)=='SAVED_PROCESSOR_CONTRACT_MISMATCH'
        else:raise AssertionError('WRONG_CONTRACT_ACCEPTED')
    updates=(ROOT/'raw_completed_updates.jsonl').read_text().splitlines()
    heads=(ROOT/'raw_training_heads.jsonl').read_text().splitlines()
    assert len(updates)==8 and len(heads)==152
    result=dict(status='PASS_ACTUAL_NATIVE_PROCESSOR_METADATA_BINDING',model_queries=0,gradient_queries=0,optimizer_updates=0,
        historical_completed_updates=8,historical_training_heads=152,copied_sidecars=before,wrong_contract_rejected=True,
        idempotent=True,train_sha256=sha(ROOT/'code/train.py'),first_update_gradient_receipt_for_failed_run='UNOBSERVED',
        recovery='Retain failed8 and repeat only the required continuous8 canary under technical completed-update allowance; main training has not started')
    p=ROOT/'evidence/CHECKPOINT_METADATA_CPU_CHECK.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],'ZERO_NEW_MODEL_QUERY',flush=True)


if __name__=='__main__':main()
