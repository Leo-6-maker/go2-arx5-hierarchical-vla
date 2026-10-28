"""Derive V2 from sealed errors only, then attempt the original fixed sampler once."""
import copy,hashlib,json,os,sys,time
from collections import Counter
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286');sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'code')]
from native_contract_boundary import checker,boundary
from recovery import bank,recovery_plan,FROZEN
from data import quotas,WEIGHTS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    assert not p.exists(),str(p);p.write_text(json.dumps(value,indent=2)+'\n')
start=time.time();bp=ROOT/'correction/CORRECTION_BANK.json';old=json.loads(bp.read_text())
assert old['valid_branches']==15 and old['status']=='CORRECTION_NOT_ADMITTED'
code=ROOT/'code/recovery.py';index=ROOT/'data/WINDOW_INDEX.json';prior=ROOT/'correction/PRECLIP_FAILED15_DETAILS.json'
protected=[bp,index,prior,ROOT/'correction/C1_PLAN_AUDIT.json',ROOT/'correction/CORRECTION_TRANSFER_MANIFEST.json']+[ROOT/'data'/k for k in FROZEN]
before={str(p):sha(p) for p in protected};assert {k:sha(ROOT/'data'/k) for k in FROZEN}==FROZEN
valid=checker(code.read_text());boundary_result=boundary(valid,json.loads(prior.read_text())['window_details'])
save(ROOT/'correction/NATIVE_CONTRACT_V2_BOUNDARY_CHECK.json',boundary_result)
reports=[];reclassified=[]
for original in old['all_requested_slot_reports']:
    r=copy.deepcopy(original)
    if r['status'] in ('PASS_FRESH_SUFFIX_STRICT_CODEC','FAIL_SUPERVISION_CODEC_PRESERVE_RAW_STOP_BRANCH'):
        assert all(r['maxima'][k]<=t for k,t in [('XYZ_max_m',1e-5),('gripper_max_fraction',1e-5)])
        failures=r.get('failures',[]);remaining=[f for f in failures if not valid(f['errors'])]
        maxima_physical=all(r['maxima'][k]<=t for k,t in [('XYZ_max_m',1e-5),('SO3_max_rad',1e-4),('gripper_max_fraction',1e-5)])
        assert bool(remaining)==(not maxima_physical)
        r['sealed_V1_status']=r['status'];r['sealed_V1_failures_preserved']=failures
        r['failures']=remaining;r['status']='FAIL_SUPERVISION_CODEC_PRESERVE_RAW_STOP_BRANCH' if remaining else 'PASS_FRESH_SUFFIX_STRICT_CODEC'
        r['native_preclip_is_diagnostic_only_under_release_contract']=True
        if r['status']!=original['status']:reclassified.append(r['slot'])
    reports.append(r)
assert len(reclassified)==12
admission=bank(reports,old['teacher_sha256']);assert admission['valid_branches']==27 and admission['status'].startswith('PASS_')
admission.update(schema='BANK_NATIVE_CONTRACT_V2',derivation='SEALED_ALL_H40_MAXIMA_AND_FAILURE_ERRORS_NO_REENCODING',
    supersedes_V1_contract_interpretation_only=True,original_V1_bank_sha256=before[str(bp)],
    original_V1_valid_branches=15,reclassified_preclip_only_slots=reclassified,
    physical_success_branches=28,source_contract_codec_valid_branches=27,
    all_requested_slot_reports=reports,CPU_readbacks=old['CPU_readbacks'],
    collection_sha256=old['collection_sha256'],failed_prefix_labels=0,statistics_refit=0,
    new_model_queries=0,new_physics=0,new_codec_calls=0,
    parent_point=old['parent_point'],learner_point=old['learner_point'],layouts=old['layouts'],branch_times_seconds=old['branch_times_seconds'],
    recovery_code_sha256=sha(code),training_admission=False,optimizer_launch_allowed=False,
    training_time_gate_owner_stopped=True)
bankpath=ROOT/'correction/BANK_NATIVE_CONTRACT_V2.json';save(bankpath,admission)
experts=[r for r in json.loads(index.read_text())['rows'] if r['task']=='T1' and r['split']=='TRAIN']
phases={ep:{str(p):sum(r['episode']==ep and r['phase']==p for r in admission['rows']) for p in range(5)} for ep in admission['valid_slots']}
draws=dict(zip(sorted(phases),quotas(4096,[1]*len(phases))));phase_draws=quotas(4096,WEIGHTS)
forced={str(p):sum(draws[ep] for ep,coverage in phases.items() if coverage[str(p)] and sum(bool(n) for n in coverage.values())==1) for p in range(5)}
try:
    rows=recovery_plan(experts,admission)
    planpath=ROOT/'correction/C1_CORRECT1024_NATIVE_CONTRACT_V2.jsonl'
    assert not planpath.exists();planpath.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    audit=dict(status='PASS_ORIGINAL_IMPLEMENTED_FIXED_SAMPLER_NATIVE_CONTRACT_V2_NOT_TRAINING_ADMISSION',rows=len(rows),
        source_counts=dict(Counter(r['source'] for r in rows)),phase_counts=dict(Counter(r['phase'] for r in rows)),plan_sha256=sha(planpath))
except AssertionError as exc:
    audit=dict(status='IMPLEMENTED_GLOBAL_EQUAL_BRANCH_FIXED_BUCKET_FLOW_INFEASIBLE_NATIVE_CONTRACT_V2',error=repr(exc),
        requested_rows=8192,source_draw_counts=dict(expert=4096,corrected_suffix=4096))
audit.update(schema='C1_PLAN_NATIVE_CONTRACT_V2',actual_sampler_attempts=1,bank_sha256=sha(bankpath),valid_branches=27,
    required_phase_draw_counts=phase_draws,equal_global_branch_draw_counts=draws,
    available_H40_windows_per_branch_phase=phases,forced_single_bucket_minimum_draws=forced,
    phase_coverage_branch_counts={str(p):sum(bool(c[str(p)]) for c in phases.values()) for p in range(5)},
    implementation_constraint='EXACT_GLOBAL_EQUAL_BRANCH_COUNTS_THEN_FIXED_PHASE_TOTALS',
    release_text='equal episode/branch weighting within source and existing behavior buckets',
    global_exact_branch_count_not_explicit_in_release=True,
    no_bucket_or_weight_change=True,no_fallback_sampler_run=True,
    training_admission=False,optimizer_launch_allowed=False,training_time_gate_owner_stopped=True,
    model_calls=0,physics_calls=0,codec_calls=0)
auditpath=ROOT/'correction/C1_PLAN_NATIVE_CONTRACT_V2.json';save(auditpath,audit)
after={str(p):sha(p) for p in protected};assert before==after
receipt=dict(status='PASS_VERSIONED_SAVED_ERROR_DERIVATION_AND_ONE_ACTUAL_IMPLEMENTED_SAMPLER_ATTEMPT',
    bank_path=str(bankpath),bank_sha256=sha(bankpath),bank_status=admission['status'],valid_branches=27,
    plan_path=str(auditpath),plan_sha256=sha(auditpath),plan_status=audit['status'],actual_sampler_attempts=1,
    preserved_hashes_before=before,preserved_hashes_after=after,original_V1_unchanged=True,
    recovery_code_sha256=sha(code),elapsed_seconds=time.time()-start,
    model_calls=0,teacher_calls=0,physics_calls=0,codec_calls=0,optimizer_launch_allowed=False)
save(ROOT/'correction/NATIVE_CONTRACT_V2_DERIVATION_RECEIPT.json',receipt)
print(json.dumps(dict(receipt=receipt,forced_draws=forced,phase_draws=phase_draws,phase_coverage=audit['phase_coverage_branch_counts'])))
