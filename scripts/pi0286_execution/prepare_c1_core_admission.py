"""C1-only V2/V3 saved-data identity admission; no codec/model/physics execution."""
import hashlib,json,os,sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'code')]
from contract import sha,write
from correction_manifest import build

def prepare():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    bp=ROOT/'correction/BANK_NATIVE_CONTRACT_V2.json'
    folder=ROOT/'correction/SAMPLER_BUCKET_CONTRACT_V3'
    pp=folder/'C1_CORRECT1024_BUCKET_CONTRACT_V3.jsonl';ap=folder/'CPU_PLAN_DRAFT.json'
    protected=[ROOT/'correction'/n for n in ('CORRECTION_BANK.json','C1_PLAN_AUDIT.json','CORRECTION_TRANSFER_MANIFEST.json')]
    protected += [ROOT/'data'/n for n in ('statistics.json','processor_config.json','embodiment_id.json')]
    before={str(p):sha(p) for p in protected}
    manifest=build(ROOT/'data')
    bank=json.loads(bp.read_text());audit=json.loads(ap.read_text());rows=[json.loads(x) for x in pp.read_text().splitlines()]
    assert bank['valid_branches']==27 and len(rows)==8192 and [r['index'] for r in rows]==list(range(8192))
    originals=json.loads((ROOT/'data/WINDOW_INDEX.json').read_text())['rows']
    index={r['uid']:r for r in originals if r['task']=='T1' and r['split']=='TRAIN'}
    corrected={r['uid']:r for r in bank['rows']}
    assert len(corrected)==len(bank['rows'])
    for row in rows:
        expected=(index if row['source']=='expert' else corrected)[row['uid']]
        assert all(row[k]==v for k,v in expected.items()),row['uid']
        assert row['task']=='T1' and row['split']=='TRAIN'
        if row['source']=='corrected_suffix':
            assert row['episode'] in bank['valid_slots'] and row['anchor']+40<=660-(80 if row['episode'].endswith('03s') else 260)
    counts=Counter(r['source'] for r in rows);assert counts=={'expert':4096,'corrected_suffix':4096}
    for p,detail in audit['within_bucket_branch_draws'].items():
        actual=Counter(r['episode'] for r in rows if r['source']=='corrected_suffix' and r['phase']==int(p))
        assert actual==detail['branch_draws']
    assert len(manifest['files'])==162 and len(manifest['branches'])==27
    after={str(p):sha(p) for p in protected};assert before==after
    result=dict(status='PASS_ACTUAL_C1_NATIVE_V2_BUCKET_V3_SAVED_DATA_AND_PLAN_IDENTITY',
        bank_path=str(bp),bank_sha256=sha(bp),plan_path=str(pp),plan_sha256=sha(pp),plan_audit_sha256=sha(ap),
        manifest_path=str(ROOT/'correction/CORRECTION_TRANSFER_MANIFEST_NATIVE_V2_BUCKET_V3.json'),
        manifest_sha256=sha(ROOT/'correction/CORRECTION_TRANSFER_MANIFEST_NATIVE_V2_BUCKET_V3.json'),
        requested=32,faithful_restores=32,physical_success=28,admitted_branches=27,
        new_array_members=162,new_array_bytes=manifest['total_bytes'],new_array_tree_sha256=manifest['tree_sha256'],
        rows=8192,source_counts=dict(counts),correction_bucket_counts=audit['correction_bucket_counts'],
        within_bucket_branch_draws=audit['within_bucket_branch_draws'],all_rows_match_sealed_original_identities=True,
        original_failure_receipts_and_statistics_before=before,original_failure_receipts_and_statistics_after=after,
        deadline_extension_owner_authority='USER_AUTHORIZED_FINISH_EXPERIMENT_TO_1400_ROOT_OWNS_GUARDS',
        old1130_stops_preserved_not_active_authority=True,failed_prefix_labels=0,
        model_calls=0,physics_calls=0,teacher_calls=0,codec_calls=0,PBS_submissions=0,
        optimizer_launch_allowed=False,canary='NEW_REAL_CONTINUOUS8_VS_FRESH4_RESTORE4_REQUIRED')
    out=ROOT/'correction/C1_DATA_ADMISSION_NATIVE_V2_BUCKET_V3.json'
    if out.exists():assert json.loads(out.read_text())==result
    else:write(out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('within_bucket_branch_draws','original_failure_receipts_and_statistics_before','original_failure_receipts_and_statistics_after')}))
    return result

if __name__=='__main__':prepare()
