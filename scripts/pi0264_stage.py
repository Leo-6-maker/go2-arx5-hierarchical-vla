"""Verify fixed X2000 donor and seal the isolated PI0264 training fileset."""
import argparse
import json
import shutil
from pathlib import Path

from pi0264_plan import read,sha,write,validate

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0264')
OLD=ROOT.with_name('PI0263')


def main(donor_only=False):
    receipt=read(OLD/'evidence/X/checkpoint-2000.full_readback.json')
    assert receipt['status']=='PASS_FULL_FILE_READBACK' and receipt['complete_trainer'] and len(receipt['files'])==16
    assert receipt['revision']=='a2f9f767a11e4af269377603f3da432f4931d623'
    assert receipt['tree_sha256']=='4ec3b3a44ad40a057cfdcd7ad992a86b68889b4efba98be70ed109e571497694'
    donor=OLD/'X/main/training/checkpoint-2000'
    target=ROOT/'DONOR_LOCAL_READBACK.json'
    if not target.exists():
        for row in receipt['files']:
            path=donor/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],str(path)
        assert read(donor/'trainer_state.json')['global_step']==2000
        write(target,dict(status='PI0264_FIXED_X2000_LOCAL_READBACK_PASS',source_receipt_sha256=sha(OLD/'evidence/X/checkpoint-2000.full_readback.json'),
            repo_id=receipt['repo_id'],revision=receipt['revision'],prefix=receipt['prefix'],tree_sha256=receipt['tree_sha256'],
            donor_path=str(donor),files=receipt['files'],complete_trainer_bytes=sum(r['bytes'] for r in receipt['files']),
            initialization='WEIGHTS_ONLY_FRESH_OPTIMIZER_AND_SCHEDULER'))
    else:
        previous=read(target);assert previous['files']==receipt['files'] and previous['source_receipt_sha256']==sha(OLD/'evidence/X/checkpoint-2000.full_readback.json')
        assert all((donor/r['path']).stat().st_size==r['bytes'] for r in receipt['files'])
    if donor_only:
        print('PASS_PI0264_DONOR_16_ACTUAL_FILES');return
    validate(read(ROOT/'PAIRING_PLAN.json'))
    audit=read(ROOT/'PROCESSED_PLAN_AUDIT.json')
    assert audit['plan_sha256']==sha(ROOT/'PAIRING_PLAN.json') and len(audit['rows'])==8192
    card=read(ROOT/'UNION_EVAL_CARD.json');assert card['logical_rows']==6336 and card['model_queries_before_freeze']==0
    assert audit['challenge_card_sha256']==sha(ROOT/'UNION_EVAL_CARD.json')
    size=read(target)['complete_trainer_bytes'];free=shutil.disk_usage('/scratch/e1192548').free
    assert free>=3*size+10*2**30,'CANARY_NEXT_WRITE_OVERLAP_AND_RESERVE'
    external=[ROOT.with_name('PI0261')/'code/pi0261_train.py',ROOT.with_name('PI0260')/'R3/execution/pi0260_r3_train.py',
        ROOT.with_name('PI0260')/'R3/execution/pi0260_r3_inputs.py',ROOT.with_name('PI0260')/'R3/execution/pi0247_u_rollout_overlay.py']
    files=[p for p in (ROOT/'code').iterdir() if p.is_file()]+[ROOT/n for n in ('PAIRING_PLAN.json','PROCESSED_PLAN_AUDIT.json',
        'registry.json','UNION_EVAL_CARD.json','FRESH_NOISE_MANIFEST.json','EVAL_TEXT_INPUT_CHECKS.json','PI0264_PROTOCOL.md','DONOR_LOCAL_READBACK.json')]
    files+=list((ROOT/'noise').rglob('*.npz'))
    write(ROOT/'EXECUTION_FILESET.json',dict(status='PI0264_EXECUTION_FILESET_FROZEN',
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files+external)],
        donor_readback='PASS_16_FILES',minimum_next_write_free_bytes=size+10*2**30,observed_user_root_free_bytes=free,
        quota_headroom='UNKNOWN',storage_query_path='/scratch/e1192548',
        storage_policy='sequential arms; compared canary copies only; full Trainer fixed HF readback before scientific model retirement',
        compute_capacity_basis='Inherited full-S4 FP32 trainable action head and AdamW runtime; no Corelab GR00T training.',
        main_update_cap=4096,canary_updates=32,recovery_completed_update_reserve=256,raw_completed_update_cap=4384,
        raw_training_head_cap=110000,new_quality_attempt_cap=5440,new_inference_head_cap=21760))
    print(json.dumps(dict(status='PASS_PI0264_DONOR_AND_PRE_CANARY_EXECUTION_SEAL',user_root_free_bytes=free)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--donor-only',action='store_true');main(p.parse_args().donor_only)
