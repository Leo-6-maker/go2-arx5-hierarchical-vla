"""Actual captured 25-row/17-attempt CPU proposals, not native admission."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.pi0294.pv1 import evaluation_core_admission as a,evaluation_core_accounting as c
from scripts.pi0294.pv1 import evaluation_work as w,evaluation_admission as old
from scripts.pi0294.test_pv1_core_evaluation_accounting import row,NOW
from scripts.pi0294.dec1 import source_cpu_once_io as once


@pytest.fixture
def captured_inputs():
    root=Path(__file__).resolve().parents[4];p=root/'work/private';t=p/'science-next/wdr1-target-identity'
    paths=[t/'v4-approved-s2-scope-adoption-01/INDEPENDENT_GLOBAL_READBACK.private.json',
        p/'pv1-20261019/PV1_RESOURCE_LEDGER.private.json',t/'V4_COMPLETE_ORIGINAL_PRIMARY17_ASSEMBLY_1792826330286141498.private.json',
        p/'pv1-20261019/POPULATION_FINAL.private.json']
    if not all(p.is_file() for p in paths):pytest.skip('private complete actual ledgers/history absent')
    gb=base64.b64decode(json.loads(paths[0].read_bytes())['global_after_base64']);lb=paths[1].read_bytes()
    capture=json.loads(paths[2].read_bytes());evidence=dict(B0_CORR=[{k:base64.b64decode(v) for k,v in r.items()}
        for r in capture['evidence_base64']],A_CONTROL=[],B_COVERAGE=[])
    population=(json.dumps(json.loads(paths[3].read_bytes())['plan'],indent=2)+'\n').encode()
    work=w.remaining(population,evidence_by_candidate=evidence);g,l=json.loads(gb),json.loads(lb)
    record=row();sealed={c.stage.PLAN:c.stage.PLAN_SHA,c.stage.CARD:c.stage.CARD_SHA}
    manifest=dict(schema='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1',scope=c.ROLE,owner='EXISTING_LOCAL_PC',
        owner_user='liuyu-nfs',owner_uid=2009,run_uuid=record['child_run_uuid'],platform_evaluation_version=c.VERSION,
        execution=dict(profile=c.PROFILE,host='corelab-4080s',uid=2009,sealed_inputs=sealed),
        code=dict(vla_commit=record['source_sha']))
    mb=old.canonical(manifest);r=deepcopy(record)
    for key in ('request_sha256','native_handle','status','reserved_neural_queries'):r.pop(key)
    job=dict(next(j for j in work['jobs'] if j['candidate']=='A_CONTROL' and j['noise_bank']==294101),native_platform='CORELAB_RTX4080')
    job['GPU_site_output_upper_bytes']+=c.MODEL_BYTES
    r.update(schema='PI0294_PV1_CORE_NATIVE_GPU_REQUEST_V1',allocation_number=1,run_uuid=record['child_run_uuid'],
        run_manifest_sha256=hashlib.sha256(mb).hexdigest(),new_bank_manifest_sha256=work['new_bank_manifest_sha256'],
        optimizer_updates=0,Teacher_takeover=False,normalizer_refit=False,evaluation_data_for_training=False,
        LOCKBOX_opened=False,numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',model_transfer_payload_bytes=c.MODEL_BYTES,
        sealed_inputs=sealed,planned_panel_sha256=a.digest(job),prior_outcomes_sha256=a.digest(work['histories']['A_CONTROL']['prior_rows']))
    used,reserved=old.resource_floor(g,l);used.update(Core_storage_bytes=115058147328,local_storage_bytes=8589934592)
    snapshot=dict(schema=old.SNAPSHOT,campaign_run_uuid=c.scope.CAMPAIGN,original_T0_epoch=c.old.T0,
        hard_close_epoch=c.old.HARD_CLOSE,observed_epoch=NOW,global_ledger_sha256=hashlib.sha256(gb).hexdigest(),
        LOCAL_ledger_sha256=hashlib.sha256(lb).hexdigest(),used=used,reserved=reserved,
        Core_storage_upper_bytes=115058147328,LOCAL_storage_upper_bytes=8589934592,
        primary_attempt_inventory={k:list(v['all_actual_attempts']) for k,v in work['histories'].items()})
    inputs=dict(manifest_bytes=mb,request_bytes=old.canonical(r),population_bytes=population,
        evidence_by_candidate=evidence,global_bytes=gb,local_bytes=lb,snapshot_bytes=old.canonical(snapshot),now_epoch=NOW)
    proof=a.recompute(**inputs,purpose='PREPARE');r['prequeue_future_admission_sha256']=proof['prequeue_context_sha256']
    inputs['request_bytes']=old.canonical(r)
    record.update(run_manifest_sha256=r['run_manifest_sha256'],request_sha256=hashlib.sha256(inputs['request_bytes']).hexdigest(),
        planned_panel_sha256=r['planned_panel_sha256'],prior_outcomes_sha256=r['prior_outcomes_sha256'],
        prequeue_future_admission_sha256=r['prequeue_future_admission_sha256'])
    return inputs,record


def test_all_resources_and_raw_histories_without_mock_or_new_caps(captured_inputs):
    inputs,record=captured_inputs;before=deepcopy(inputs)
    p=a.reserve_pair_proposal(inputs=inputs,record=record);proof=p['recomputed_proof']
    assert inputs==before and proof['shortfalls']=={} and proof['native_or_scientific_admission'] is False
    d=proof['dimensions'];assert d['Core_CPU_s']['remaining_stage_upper']==362800
    assert d['transfer_bytes']['remaining_stage_upper']==52613349376+4*c.MODEL_BYTES
    assert d['Hopper_storage_bytes']['remaining_stage_upper']==51539607552+2*c.MODEL_BYTES
    assert d['PBS_submissions']['remaining_stage_upper']==2
    assert p['proposed_global']['PBS_submissions']==json.loads(inputs['global_bytes'])['PBS_submissions']
    assert p['proposed_LOCAL']['CPU_reserved_s']['CoreLab']==80000
    assert p['proposed_LOCAL']['Core_native_GPU_reserved']['transfer_bytes']==2*c.MODEL_BYTES
    assert p['PBS_submissions_consumed']==0 and p['IO_performed'] is False
    assert c.checked(p['proposed_global'])==p['proposed_global']
    used,held=old.resource_floor(p['proposed_global'],p['proposed_LOCAL'])
    assert held['H200_s']==28860 and held['Core_CPU_s']==held['Hopper_CPU_s']==80000
    assert held['transfer_bytes']==2*c.MODEL_BYTES


@pytest.mark.parametrize('fault',['snapshot_age','request_age','late_stage','snapshot_GLOBAL','snapshot_LOCAL',
    'missing_original_attempt','Core_paid_CPU','GPU_paid','inclusive_Core_storage','fake_PBS_profile','training','rebound_context','changed_record'])
def test_all_original_unknown_storage_clock_identity_and_history_guards(captured_inputs,fault):
    inputs,record=deepcopy(captured_inputs);r=json.loads(inputs['request_bytes']);snap=json.loads(inputs['snapshot_bytes'])
    if fault=='snapshot_age':snap['observed_epoch']=NOW-121
    if fault=='request_age':r['created_epoch']=NOW-121
    if fault=='late_stage':inputs['now_epoch']=c.old.STOP_NEW-28859;snap['observed_epoch']=inputs['now_epoch'];r['created_epoch']=inputs['now_epoch']
    if fault=='snapshot_GLOBAL':snap['global_ledger_sha256']='f'*64
    if fault=='snapshot_LOCAL':snap['LOCAL_ledger_sha256']='f'*64
    if fault=='missing_original_attempt':snap['primary_attempt_inventory']['B0_CORR'].pop()
    if fault=='Core_paid_CPU':snap['used']['Core_CPU_s']-=1
    if fault=='GPU_paid':snap['used']['H200_s']-=1
    if fault=='inclusive_Core_storage':snap['used']['Core_storage_bytes']=0
    if fault=='fake_PBS_profile':r['project']='CFP04-CF-007'
    if fault=='training':r['optimizer_updates']=1
    if fault=='rebound_context':r['prequeue_future_admission_sha256']='f'*64
    if fault=='changed_record':record['checkpoint_manifest_sha256']=c.old.ENDPOINTS['B_COVERAGE']
    inputs['request_bytes']=old.canonical(r);inputs['snapshot_bytes']=old.canonical(snap)
    with pytest.raises(PermissionError):a.reserve_pair_proposal(inputs=inputs,record=record)


def test_actual_captured_pair_proposals_original_once_IO_only_in_temp_files(captured_inputs,tmp_path):
    inputs,record=captured_inputs;p=a.reserve_pair_proposal(inputs=inputs,record=record)
    for label,key in [('GLOBAL','global_bytes'),('LOCAL','local_bytes')]:
        path=tmp_path/(label+'.json');path.write_bytes(inputs[key]);intent=tmp_path/(label+'.intent');receipt=tmp_path/(label+'.receipt')
        proposed=(json.dumps(p['proposed_global' if label=='GLOBAL' else 'proposed_LOCAL'],indent=2)+'\n').encode()
        result=once.commit(path,expected_before_sha256=hashlib.sha256(inputs[key]).hexdigest(),proposed_bytes=proposed,
            request_id=record['request_id'],intent_path=intent,receipt_path=receipt,lock_path=tmp_path/(label+'.lock'))
        assert path.read_bytes()==proposed and result
        assert once.reconcile(path,intent_path=intent,receipt_path=receipt)['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
        with pytest.raises(Exception):once.commit(path,expected_before_sha256=hashlib.sha256(inputs[key]).hexdigest(),proposed_bytes=proposed,
            request_id=record['request_id'],intent_path=intent,receipt_path=receipt,lock_path=tmp_path/(label+'.lock'))
