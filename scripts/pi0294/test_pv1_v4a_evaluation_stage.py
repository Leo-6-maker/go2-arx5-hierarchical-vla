"""Original-bank two-arm stage rules, no new native sampling or execution."""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import pytest
from scripts.pi0294.pv1 import evaluation_stage as s,evaluation_work as w,evaluation_platform as e
from scripts.pi0294.test_pv1_v4_evaluation_work import POP,evidence,synthetic_population_only


def case():
    work=w.remaining(POP,evidence_by_candidate=dict(B0_CORR=evidence(),A_CONTROL=[],B_COVERAGE=[]))
    sealed={s.PLAN:s.PLAN_SHA,s.CARD:s.CARD_SHA}
    m=dict(execution=dict(sealed_inputs=sealed))
    r=dict(evaluation_execution_stage=s.VERSION,evaluation_stage_plan_sha256=s.PLAN_SHA,evaluation_stage_card_sha256=s.CARD_SHA,
        evaluation_candidate='A_CONTROL',noise_bank=294101,sealed_inputs=sealed,request_id='a')
    return work,m,r,dict(PBS_submissions=[])


def test_exact_published_user_plan_and_card_bytes():
    root=Path(__file__).resolve().parents[2]
    assert hashlib.sha256((root/'docs/protocols/GR00T_WBC_V4A_EVAL_FIRST_STAGED_EXECUTION_20261024.md').read_bytes()).hexdigest()==s.PLAN_SHA
    assert hashlib.sha256((root/'reports/PI0294/design/parallel-science-20261022/V4A_EVAL_FIRST_STAGED_EXECUTION_CARD_20261024.json').read_bytes()).hexdigest()==s.CARD_SHA


def test_only_matched_AB_first_bank_funded_all_original_history_preserved(synthetic_population_only):
    work,m,r,g=case();before=deepcopy(work)
    current=s.first_stage(work,manifest=m,current_request=r,global_ledger=g,purpose='PREPARE')
    assert work==before and [j['candidate'] for j in current['jobs']]==['A_CONTROL','B_COVERAGE']
    assert [j['noise_bank'] for j in current['jobs']]==[294101,294101]
    assert [j['conditions'] for j in current['jobs']]==[24,24]
    assert current['resources']['H200_s']==57720 and current['resources']['PBS_submissions']==2
    assert current['resources']['episode_starts']==48 and current['resources']['neural_queries']==3874
    assert current['resources']['physics_ticks']==1536000 and current['resources']['optimizer_updates']==0
    assert current['resources']['Hopper_CPU_s']==175000 and current['resources']['Core_CPU_s']==187800
    assert current['resources']['local_CPU_s']==15000
    assert current['histories']==work['histories'] and current['original_matrix_protection']==w.MATRIX
    assert current['original_final_protection']==work['original_final_protection']
    assert current['full_remaining_DEV_conditions']==130 and current['full_remaining_DEV_jobs']==6
    assert current['later_panels']=='NOT_SCHEDULED_REQUIRES_FIRST_STAGE_RESULT_DECISION'


@pytest.mark.parametrize('field,value',[('evaluation_candidate','B0_CORR'),('noise_bank',294102),('evaluation_execution_stage','ALL_SIX'),('evaluation_stage_plan_sha256','a'*64),('evaluation_stage_card_sha256','a'*64)])
def test_wrong_bank_baseline_or_six_panel_dispatch_cannot_use_current_stage(synthetic_population_only,field,value):
    work,m,r,g=case();r[field]=value
    with pytest.raises(PermissionError):s.first_stage(work,manifest=m,current_request=r,global_ledger=g,purpose='PREPARE')


def test_closed_incomplete_arm_does_not_get_automatic_extra_PBS(synthetic_population_only):
    work,m,r,g=case();g['PBS_submissions']=[dict(platform_evaluation_version=e.VERSION,evaluation_execution_stage=s.VERSION,
        noise_bank=294101,evaluation_candidate='A_CONTROL',request_id='old',status='NATIVE_TERMINAL')]
    r['evaluation_candidate']='B_COVERAGE'
    with pytest.raises(PermissionError,match='CLOSED_INCOMPLETE'):s.first_stage(work,manifest=m,current_request=r,global_ledger=g,purpose='PREPARE')


def test_complete_first_arm_requires_real_history_and_leaves_only_second_arm(synthetic_population_only):
    work,m,r,g=case();work=w.remaining(POP,evidence_by_candidate=dict(B0_CORR=evidence(),A_CONTROL=evidence('A_CONTROL',24),B_COVERAGE=[]))
    g['PBS_submissions']=[dict(platform_evaluation_version=e.VERSION,evaluation_execution_stage=s.VERSION,
        noise_bank=294101,evaluation_candidate='A_CONTROL',request_id='old',status='NATIVE_TERMINAL')]
    r['evaluation_candidate']='B_COVERAGE'
    current=s.first_stage(work,manifest=m,current_request=r,global_ledger=g,purpose='PREPARE')
    assert len(current['jobs'])==1 and current['jobs'][0]['candidate']=='B_COVERAGE'
    assert current['resources']['PBS_submissions']==1 and current['resources']['episode_starts']==24


def test_native_bound_request_is_not_a_fresh_submit_or_other_stage(synthetic_population_only):
    work,m,r,g=case();g['PBS_submissions']=[dict(platform_evaluation_version=e.VERSION,evaluation_execution_stage=s.VERSION,
        noise_bank=294101,evaluation_candidate='A_CONTROL',request_id='a',status='NATIVE_RUNNING_QUALIFICATION')]
    assert len(s.first_stage(work,manifest=m,current_request=r,global_ledger=g,purpose='NATIVE_PREMODEL')['jobs'])==2
    with pytest.raises(PermissionError):s.first_stage(work,manifest=m,current_request=r,global_ledger=g,purpose='RESERVE')
    g['PBS_submissions'][0]['noise_bank']=294102
    with pytest.raises(PermissionError):s.first_stage(work,manifest=m,current_request=r,global_ledger=g,purpose='NATIVE_PREMODEL')


def test_both_panels_and_second_submission_deadline_checked_not_only_first(synthetic_population_only):
    work,m,r,g=case();current=s.first_stage(work,manifest=m,current_request=r,global_ledger=g,purpose='PREPARE')
    good=s.clock(current,now_epoch=e.STOP_NEW-28861)
    assert good['last_stage_submission_fits_original_stop_new'] is True
    late=s.clock(current,now_epoch=e.STOP_NEW-28859)
    assert late['last_stage_submission_fits_original_stop_new'] is False
    facts=dict(allocation_start_epoch=e.STOP_NEW-28861)
    assert s.clock(current,now_epoch=e.STOP_NEW-28801,current_native_facts=facts)['last_stage_submission_fits_original_stop_new'] is True
    # Once the only remaining second arm is already R, stop-new does not
    # retroactively prevent its legal before-stop-new submission from running.
    current['jobs']=current['jobs'][1:]
    assert s.clock(current,now_epoch=e.STOP_NEW+600,current_native_facts=dict(allocation_start_epoch=e.STOP_NEW+300))['last_stage_submission_fits_original_stop_new'] is True
