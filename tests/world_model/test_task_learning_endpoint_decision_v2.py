"""Scientific rule/array fixtures; none are native model or scope qualification."""
import copy
import json

import numpy as np
import pytest

from go2arx5_vla.world_model import task_learning_endpoint_decision_v2 as d
from go2arx5_vla.world_model.task_learning_probes_v1 import contract as probes, PHASES
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_task_learning_student_v1 import panel


def outcomes(p,successes=0,legal=0):
    rows=[];counts={'B0':0,'B1':0}
    for row in p['assignments']:
        method=row['method'];index=counts[method];counts[method]+=1
        success=method=='B1'and index<successes
        held=success or(method=='B1'and index<legal)
        rows.append(dict(row,attempted=True,reset_ready=True,technical_valid=True,hard_safety_stop=False,
            task_success=success,score=dict(standing_passes=True,unsafe_samples=0,
                maximum_object_lift_m=.025 if held else 0,
                longest_contiguous_grasp_lift_hold_s=1. if success else .15 if held else 0)))
    return rows


def summary(split,step,xyz):
    phases={p:{h:dict(XYZ_m=xyz,SO3_rad=.01,gripper=.02)for h in ['H40','first']}for p in PHASES}
    return dict(global_step=step,generation_id='generation',dataset_sha256='a'*64,statistics_sha256='b'*64,
        actual_unique_probes=40 if split=='TRAIN'else 20,matched_observation_noise_sha256='c'*64,
        phases=phases,key={h:{k:float(np.mean([phases[p][h][k]for p in d.KEY_PHASES]))
            for k in d.CHANNELS}for h in ['H40','first']})


def evidence(step=2000,successes=4,legal=0):
    p=panel();p['B1_global_step']=step
    before=0 if step==2000 else 2000 if step==4000 else 4000
    e=dict(contract=d.contract(),global_step=step,generation_id='generation',dataset_sha256='a'*64,
        statistics_sha256='b'*64,package_sha256=p['policies']['B1']['package']['sha256'],checkpoint_sha256='d'*64,
        native_panel=p,native_outcomes=outcomes(p,successes,legal),
        open_before={s:summary(s,before,.01)for s in ['TRAIN','DEV']},
        open_after={s:summary(s,step,.008)for s in ['TRAIN','DEV']})
    if step>2000:
        previous=panel();previous['B1_global_step']=before
        e.update(previous_native_panel=previous,previous_native_outcomes=outcomes(previous,0))
    return e


def test_first_endpoint_uses_complete_B0_population_and_measurable_native_gain():
    r=d.evaluate(evidence())
    assert r['action']=='CONTINUE_SAME_GENERATION'and r['next_endpoint']==4000
    assert r['native_success_gain']==4 and r['baseline_kind']=='COMMON_B0'
    assert all(r['guards'].values())and r['progress_OR']['native_success_gain4']


def test_partial_never_selects_or_promotes_even_when_executed_subset_is_perfect():
    e=evidence(successes=40);e['native_outcomes']=e['native_outcomes'][:20]
    r=d.evaluate(e)
    assert r['action']=='CONTINUE_UNEXECUTED_OR_REPAIR_ONLY'and not r['allows_execution']


def test_predeclared_stage_branch_uses_original_safe_grasp_lift_hold_score():
    e=evidence(successes=0,legal=4);r=d.evaluate(e)
    assert r['allows_execution']and r['progress_OR']['DEV_XYZ_and_legal_hold_gain4']
    for row in e['native_outcomes']:
        if row['method']=='B1':row['score']['standing_passes']=False
    assert not d.evaluate(e)['allows_execution']


@pytest.mark.parametrize('fault',['hard','technical','gripper','rotation','TRAIN','gain3','noise','generation','wrong_endpoint','nan','aggregate'])
def test_common_guards_and_bound_probe_identities_cannot_be_bypassed(fault):
    e=evidence()
    if fault=='hard':e['native_outcomes'][0]['hard_safety_stop']=True
    if fault=='technical':e['native_outcomes'][0]['technical_valid']=False
    if fault in ['gripper','rotation']:
        e['open_after']['DEV']['phases']['CLOSE']['first']['gripper'if fault=='gripper'else'SO3_rad']=.5
        return assert_rejected_or_no_continue(e)
    if fault=='TRAIN':e['open_after']['TRAIN']=summary('TRAIN',2000,.011)
    if fault=='gain3':e['native_outcomes']=outcomes(e['native_panel'],3)
    if fault=='noise':e['open_after']['DEV']['matched_observation_noise_sha256']='e'*64
    if fault=='generation':e['open_after']['TRAIN']['generation_id']='other'
    if fault=='wrong_endpoint':e['native_panel']['B1_global_step']=4000
    if fault=='nan':e['open_after']['DEV']['phases']['HOLD']['H40']['SO3_rad']=float('nan')
    if fault=='aggregate':e['open_after']['DEV']['key']['H40']['XYZ_m']=0
    assert_rejected_or_no_continue(e)


def assert_rejected_or_no_continue(e):
    try:r=d.evaluate(e)
    except ValueError:return
    assert r['allows_execution']is False


def test_later_endpoint_compares_previous_evaluated_B1_not_current_B0():
    e=evidence(step=4000,successes=7)
    e['previous_native_outcomes']=outcomes(e['previous_native_panel'],5)
    # +7 against B0 would pass. +2 against the prior candidate does not.
    assert not d.evaluate(e)['allows_execution']
    e['native_outcomes']=outcomes(e['native_panel'],9)
    assert d.evaluate(e)['next_endpoint']==8000
    e['previous_native_outcomes']=e['previous_native_outcomes'][:2]
    with pytest.raises(ValueError,match='PREVIOUS_COMPLETE'):d.evaluate(e)


def test_target_freezes_for_confirmation_and_8000_never_creates_a_longer_schedule():
    r=d.evaluate(evidence(successes=32))
    assert r['action']=='FREEZE_FOR_SEED_CONFIRMATION_BEFORE_TEST'and not r['allows_execution']
    r=d.evaluate(evidence(step=8000))
    assert r['next_endpoint']is None and not r['allows_execution']


def test_continuation_is_recomputed_and_binds_exact_resume_checkpoint(tmp_path):
    e=evidence();ep=tmp_path/'evidence.json';ep.write_text(json.dumps(e))
    cp=tmp_path/'contract.json';cp.write_text(json.dumps(d.contract()))
    r=d.evaluate(e)
    decision=dict(scientific_evidence=d.descriptor(ep),decision_contract=d.descriptor(cp),scientific_result=r,
        action=r['action'],next_endpoint=4000,generation_id='generation',dataset_sha256='a'*64,checkpoint_sha256='d'*64)
    spec=dict(recipe=dict(endpoint=4000,dataset_sha256='a'*64,statistics_sha256='b'*64),
        start_step=2000,generation_id='generation',resume=dict(sha256='d'*64))
    assert d.validate_continuation(decision,spec)==r
    spec['resume']['sha256']='e'*64
    with pytest.raises(ValueError,match='EXACT_RESUME'):d.validate_continuation(decision,spec)
    with pytest.raises(ValueError,match='NOT_BOOLEAN'):d.validate_continuation(dict(allows_execution=True),spec)


def test_confirmation_seeds_require_complete_first_seed_evidence_instead_of_a_permission_boolean(tmp_path):
    e=evidence(successes=32);e['training_seed']=20261109
    path=tmp_path/'evidence.json';path.write_text(json.dumps(e));rule=tmp_path/'rule.json';rule.write_text(json.dumps(d.contract()))
    result=d.evaluate(e)
    doc=dict(scientific_evidence=d.descriptor(path),decision_contract=d.descriptor(rule),scientific_result=result,
        first_seed=20261109,first_seed_generation_id='generation',dataset_sha256='a'*64,
        allowed_confirmation_seeds=[20261110,20261111])
    spec=dict(recipe=dict(seed=20261110,dataset_sha256='a'*64,statistics_sha256='b'*64))
    assert d.validate_confirmation(doc,spec)==result
    e['native_outcomes']=e['native_outcomes'][:20];path.write_text(json.dumps(e));doc['scientific_evidence']=d.descriptor(path)
    with pytest.raises(ValueError,match='COMPLETE_FIRST_SEED_GAIN'):d.validate_confirmation(doc,spec)
    with pytest.raises(ValueError,match='NOT_BOOLEAN'):d.validate_confirmation(dict(allows_execution=True),spec)


def test_real_array_error_recomputation_uses_unique_phase_queries_and_fixed_noise(tmp_path):
    from scripts.pi0291_arx5_data.native_dataset import pose10
    rows=[]
    for episode in range(8):
        for phase_index,phase in enumerate(PHASES):
            index=len(rows);name=f'train{episode:02d}';file=tmp_path/f'{name}-a{phase_index:02d}.npz'
            expert=pose10(np.repeat(np.eye(4)[None],40,axis=0),np.full(40,.5))
            prediction=expert.copy();prediction[:,0]+=.002
            state=expert[0]
            np.savez(file,physical_H40=prediction,expert_policy_H40=expert,actual_state10=state,
                noise_seed=np.asarray(2026110900+index,dtype=np.int64))
            rows.append(dict(episode=name,anchor=phase_index,split='TRAIN',seed=2026110900+index,
                phase_audit_only=phase,genuine_recovery_audit_only=episode>=4,sample_file=file.name,
                sha256=digest(file),noise=dict(fixture=True),Processor_decode_calls=1,model_global_step=0,
                model_identity=dict(global_step=0,generation_id='generation',dataset_sha256='a'*64,statistics_sha256='b'*64)))
    (tmp_path/'RESULT.json').write_text(json.dumps(dict(selection_contract=probes(),records=rows)))
    kwargs=dict(split='TRAIN',global_step=0,generation_id='generation',dataset_sha256='a'*64,statistics_sha256='b'*64)
    r=d.probe_summary(tmp_path,**kwargs)
    assert r['actual_unique_probes']==40 and r['key']['H40']['XYZ_m']==pytest.approx(.002)
    file=tmp_path/rows[0]['sample_file'];file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='PROBE_ARRAY_HASH'):d.probe_summary(tmp_path,**kwargs)
