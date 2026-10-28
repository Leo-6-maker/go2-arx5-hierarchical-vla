"""Owner CPU draft against existing bank; never promotes an expired training card."""
import ast,hashlib,json,os,sys,time,resource
from collections import Counter
from pathlib import Path
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
resource.setrlimit(resource.RLIMIT_AS,(16<<30,16<<30))
sys.path.insert(0,str(ROOT/'code'))
import importlib.util
OUT=ROOT/'correction/SAMPLER_BUCKET_CONTRACT_V3'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    source=OUT/'recovery_draft.py';ast.parse(source.read_text())
    spec=importlib.util.spec_from_file_location('pi0286_sampler_draft',source);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    fixture=helper.self_check();assert fixture['global_equal_branch_count_required'] is False
    bp=ROOT/'correction/BANK_NATIVE_CONTRACT_V2.json';index=ROOT/'data/WINDOW_INDEX.json';seal=ROOT/'design/CODE_SEAL.json'
    protected=[bp,index,seal,ROOT/'code/recovery.py']+[ROOT/'data'/k for k in helper.FROZEN]
    before={str(p):sha(p) for p in protected}
    bank=json.loads(bp.read_text());assert bank['valid_branches']==27 and bank['status'].startswith('PASS_')
    assert sha(bp)=='f2c7b43af6edc42795ee0595ed0a9c9f05f440369945d5b43bfc38419e21cb8d'
    assert {k:sha(ROOT/'data'/k) for k in helper.FROZEN}==helper.FROZEN
    experts=[r for r in json.loads(index.read_text())['rows'] if r['task']=='T1' and r['split']=='TRAIN']
    rows=helper.recovery_plan(experts,bank)
    assert len(rows)==8192 and Counter(r['source'] for r in rows)=={'expert':4096,'corrected_suffix':4096}
    correction=[r for r in rows if r['source']=='corrected_suffix'];assert {r['episode'] for r in correction}==set(bank['valid_slots'])
    phase_counts=helper.quotas(4096,helper.WEIGHTS);per_bucket={}
    for phase in range(5):
        bucket=[r for r in correction if r['phase']==phase];counts=Counter(r['episode'] for r in bucket)
        assert len(bucket)==phase_counts[phase] and max(counts.values())-min(counts.values())<=1
        per_bucket[str(phase)]={'draws':len(bucket),'eligible_branches':len(counts),'min_branch_draws':min(counts.values()),'max_branch_draws':max(counts.values()),'branch_draws':dict(counts)}
    assert all(r['episode'] in bank['valid_slots'] for r in correction)
    path=OUT/'C1_CORRECT1024_BUCKET_CONTRACT_V3.jsonl';assert not path.exists()
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    assert [json.loads(x) for x in path.read_text().splitlines()]==rows
    assert before=={str(p):sha(p) for p in protected}
    result=dict(status='PASS_ACTUAL_BANK_ORIGINAL_BUCKET_CONDITIONAL_EQUALITY_CPU_PLAN_DRAFT',schema='SAMPLER_BUCKET_CONTRACT_V3',rows=len(rows),source_counts=dict(Counter(r['source'] for r in rows)),correction_bucket_counts=phase_counts,within_bucket_branch_draws=per_bucket,global_branch_totals=dict(Counter(r['episode'] for r in correction)),frozen_phase_order='ORIGINAL_DATA_PLAN_ORDERED_T1',event_priority='ORIGINAL_ALTERNATING_WITHIN_BUCKET',window_sampling='ORIGINAL_EIGHT_STRATA',source_ratio=[0.5,0.5],data_bank_sha256=before[str(bp)],plan_sha256=sha(path),plan_bytes=path.stat().st_size,draft_recovery_sha256=sha(source),fixture=fixture,preserved_live_code_and_bank_hashes=before,live_runtime_modified=False,launch_allowed=False,training_admission=False,reason='ORIGINAL1130_START_GATE_EXPIRED_NO_NEW_LEASE_AUTHORITY',new_model_queries=0,new_optimizer_updates=0,new_physics=0,new_codec_calls=0,selected='NONE',deployment='NONE')
    (OUT/'CPU_PLAN_DRAFT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
