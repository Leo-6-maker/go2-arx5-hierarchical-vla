"""Read-only donor verification and immutable PI0263 execution fileset seal."""
import json
import argparse
import shutil
from pathlib import Path

from pi0263_plan import read, sha, write, validate

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0263')
OLD=ROOT.with_name('PI0261')


def main(reseal=False):
    receipt=read(OLD/'evidence/P/checkpoint-2000.full_readback.json')
    assert receipt['status']=='PASS_FULL_FILE_READBACK' and receipt['complete_trainer']
    assert receipt['revision']=='875abdde2cc7c9e6858fac008e53f2f4b3375255'
    assert receipt['tree_sha256']=='882931d754cb25653105a9213fe0dbeb4d2b14e463efb27e49c6101fd5fc5e4b'
    donor=OLD/'P/main/training/checkpoint-2000'
    if reseal:
        prior=read(ROOT/'DONOR_LOCAL_READBACK.json')
        assert prior['status']=='PI0263_FIXED_P2000_LOCAL_READBACK_PASS' and prior['files']==receipt['files']
        assert prior['source_receipt_sha256']==sha(OLD/'evidence/P/checkpoint-2000.full_readback.json')
    for row in receipt['files']:
        path=donor/row['path']
        assert path.stat().st_size==row['bytes']
        if not reseal:
            assert sha(path)==row['sha256'],str(path)
    assert read(donor/'trainer_state.json')['global_step']==2000
    size=sum(r['bytes'] for r in receipt['files'])
    write(ROOT/'DONOR_LOCAL_READBACK.json',dict(status='PI0263_FIXED_P2000_LOCAL_READBACK_PASS',
        source_receipt_sha256=sha(OLD/'evidence/P/checkpoint-2000.full_readback.json'),
        repo_id=receipt['repo_id'],revision=receipt['revision'],prefix=receipt['prefix'],
        tree_sha256=receipt['tree_sha256'],donor_path=str(donor),files=receipt['files'],
        complete_trainer_bytes=size,initialization='WEIGHTS_ONLY_FRESH_OPTIMIZER_AND_SCHEDULER'))
    plan=read(ROOT/'PAIRING_PLAN.json');validate(plan)
    audit=read(ROOT/'PROCESSED_PLAN_AUDIT.json')
    assert audit['plan_sha256']==sha(ROOT/'PAIRING_PLAN.json') and len(audit['rows'])==8000
    assert len(read(ROOT/'registry.json')['families']) >= 40
    assert audit['processor_statistics_sha256']==sha(ROOT.with_name('PI0260')/'R3/source/dataset/statistics.json')
    required=size+10*2**30
    free=shutil.disk_usage('/scratch/e1192548').free
    assert free>=3*size+10*2**30, 'CANARY_THREE_CHECKPOINT_OVERLAP'
    external=[OLD/'code/pi0261_train.py',ROOT.with_name('PI0260')/'R3/execution/pi0260_r3_train.py',
              ROOT.with_name('PI0260')/'R3/execution/pi0260_r3_inputs.py',
              ROOT.with_name('PI0260')/'R3/execution/pi0247_u_rollout_overlay.py']
    files=[p for p in ROOT.glob('code/*') if p.is_file()]+[ROOT/n for n in
        ('PAIRING_PLAN.json','PROCESSED_PLAN_AUDIT.json','registry.json','UNION_EVAL_CARD.json','PI0263_PROTOCOL.md','DONOR_LOCAL_READBACK.json')]
    write(ROOT/'EXECUTION_FILESET.json',dict(status='PI0263_EXECUTION_FILESET_FROZEN',
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files+external)],
        donor_readback='PASS_16_FILES',minimum_next_write_free_bytes=required,
        observed_user_root_free_bytes=free,quota_headroom='UNKNOWN',
        storage_query_path='/scratch/e1192548',
        storage_policy='sequential arms; retire compared canary copies; full Trainer fixed HF readback before any retirement',
        compute_capacity_basis='unchanged qualified PI0261 full-S4 FP32 action-head/AdamW runtime; no GR00T training on Corelab',
        GPU_request='legal auto queue H200 as existing qualified runtime',
        main_update_cap=4000,canary_updates=32,recovery_completed_update_reserve=256,
        raw_completed_update_cap=4288,raw_training_head_cap=100000))
    print(json.dumps(dict(status='PASS_PI0263_LOCAL_DONOR_AND_FILESET',donor_files=len(receipt['files']),user_root_free_bytes=free)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reseal',action='store_true');main(p.parse_args().reseal)
