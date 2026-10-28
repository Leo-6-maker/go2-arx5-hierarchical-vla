"""Full captured escrow plus fixture PBS identities; no native admission."""
import base64
from copy import deepcopy
import time

import pytest
from scripts.pi0294.test_v5_vanda_cohost import captured
from scripts.pi0294.test_v5_vanda_render import request as render_request
from scripts.pi0294.pv1 import mix_vanda_render as render, mix_vanda_cohost as cohost
from scripts.pi0294.pv1 import mix_vanda_eval_accounting as v, mix_vanda_eval_identity as identity
from scripts.pi0294.pv1 import mix_accounting as a, mix_core_accounting as core, successor_cd as successor


def inputs(captured):
    g=cohost.propose(**captured)['proposed_global'];r=render_request(g)
    g=render.bind(render.register(g,r),r['request_id'],'1234567.stdct-mgmt-02')
    g=render.settle(g,r['request_id'],native=dict(job_state='F',Exit_status=1,job_id='1234567.stdct-mgmt-02',
        request_id=r['request_id'],submission_nonce=r['submission_nonce']),CPU_s=2,constructor_intent=False)
    marker=cohost.validated(g);repair=cohost.repair.validated(g);now=time.time()
    cap=deepcopy(captured['captures']['C50']);cap['global_base64']=base64.b64encode(a.raw(g)).decode()
    cap['handoff']=cohost.endpoint(a.raw(g),cap)
    request=dict(platform_evaluation_version=identity.VERSION,native_platform='VANDA_A40',
        campaign_run_uuid='20261018T182956Z-localpc-pv1',
        authority_release=successor.TASK_SOURCE,scope_grant_sha256=successor.GRANT_SHA,
        project='CFP04-CF-007',job_name=identity.JOB_NAME,owner_user='e1192548',owner_uid=22182,
        stage='V5_MATCHED_DEV24',noise_bank=294101,remaining_conditions=24,checkpoint_step=1024,
        parent_checkpoint_manifest_sha256=a.PARENT,statistics_sha256=a.STATS,
        physical_GPU_kind='NVIDIA_A40',physical_GPU_count=1,reserved_H200_s=28800,accounting_pad_s=60,
        CPU_upper_s=v.CPU_UPPER,neural_CPU_upper_s=80000,physical_CPU_upper_s=93900,active_cpu_limit=8,
        query_limit=2097,optimizer_update_limit=0,reserved_optimizer_updates=0,
        optimizer_updates=0,walltime_s=28800,joint_CPU_upper_s=173900,
        numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',source_renderer='MESA_LLVMPIPE_CPU_EGL',
        source_gl_GPU_visible=False,model_payload_copies=0,allocation_number=1,submit_count=30,
        owner_manifest_sha256='f'*64,
        additional_output_upper_bytes=cohost.storage_contract()['original_gross_upper_bytes'],
        legacy_GPU_seconds_field_is_projection=True,charges_DEC1_subcaps=False,
        T_CD=marker['T_CD'],deadline_epoch=marker['deadline_epoch'],created_epoch=now,
        evaluation_candidate='C50',request_id=repair['replacement_request_id'],
        submission_nonce=repair['replacement_submission_nonce'],run_uuid=repair['replacement_run_uuid'],
        checkpoint_manifest_sha256=cap['handoff']['checkpoint_manifest_sha256'],
        source_sha='b'*40,run_manifest_sha256='c'*64,cold_input_audit_sha256='d'*64,
        planned_panel_sha256='e'*64,training_handoff_sha256=a.sha(a.raw(cap['handoff'])))
    physical=dict(scope=identity.PHYSICAL_ROLE,run_uuid=request['run_uuid'],candidate='C50',
        code=dict(vla_commit=request['source_sha']),execution=dict(profile=identity.PHYSICAL_PROFILE),
        authority=dict(protocol_ref=successor.TASK_SOURCE,T_CD=marker['T_CD'],deadline_epoch=marker['deadline_epoch']),
        limits=dict(CPU_s=93900,physics_ticks=832000,policy_episodes=26,engine_starts=2108))
    row=dict(request,child_run_uuid=request['run_uuid'],request_sha256=a.sha(a.raw(request)),
        physical_manifest_sha256=a.sha(a.raw(physical)),job_id=None,status=v.STATES[0],
        reserved_neural_queries=2097,prepaid_panel_IO_credit_sha256=a.sha(a.raw(repair)))
    return g,dict(record=row,request=request,capture=cap,physical_manifest_bytes=a.raw(physical),now_epoch=now)


def test_actual_captured_whole_package_registers_original_C_without_double_reserving_or_refunding(captured):
    g,args=inputs(captured);before=deepcopy(g);out=v.register(g,**args)
    assert g==before and len(out['PBS_submissions'])==30
    assert out['used']['PBS_submissions']==30
    for k in ('CPU_s','reserved','CPU_reserved_s','storage_reserved_bytes','new_storage_current_bytes','new_storage_peak_bytes',a.CORE_ROWS):
        assert out[k]==before[k]
    for k in ('Core_CPU_s','Core_storage_bytes','transfer_bytes','local_CPU_s'):
        assert out[a.PACKAGE]['remaining_resources'][k]==before[a.PACKAGE]['remaining_resources'][k]
    p=out[a.PACKAGE]['remaining_resources']
    assert p['PBS_submissions']==1 and p['H200_s']==28860
    assert p['Hopper_CPU_s']==before[a.PACKAGE]['remaining_resources']['Hopper_CPU_s']-173900
    assert p['Hopper_storage_bytes']==before[a.PACKAGE]['remaining_resources']['Hopper_storage_bytes']
    with pytest.raises(PermissionError):v.register(out,**args)
    # A historical LOCAL is only a conservative budget witness, not native authority.
    budget=successor.recompute(out,__import__('json').loads(captured['historical_LOCAL_bytes']),
        Core_storage_upper_bytes=cohost.first.CORE_STORAGE,local_storage_upper_bytes=cohost.first.LOCAL_STORAGE)
    assert budget['shortfalls']=={}


@pytest.mark.parametrize('change',['new_uuid','old_global','physical_CUDA','wrong_physical_limits','different_request','extra_GPU','extra_panel','wrong_clock','unpaid_D'])
def test_no_alias_replay_or_silent_expansion(captured,change):
    g,args=inputs(captured);r=args['record']
    if change=='new_uuid':r['request_id']='c9b737c5-2960-489b-9e15-fd346940e9ef'
    elif change=='old_global':g['revision']+=1
    elif change=='physical_CUDA':
        p=__import__('json').loads(args['physical_manifest_bytes']);p['execution']['profile']='4080-simulation'
        args['physical_manifest_bytes']=a.raw(p);r['physical_manifest_sha256']=a.sha(a.raw(p))
    elif change=='wrong_physical_limits':
        p=__import__('json').loads(args['physical_manifest_bytes']);p['limits']['physics_ticks']+=1
        args['physical_manifest_bytes']=a.raw(p);r['physical_manifest_sha256']=a.sha(a.raw(p))
    elif change=='different_request':args['request']['source_sha']='f'*40
    elif change=='extra_GPU':r['physical_GPU_count']=2
    elif change=='extra_panel':r['remaining_conditions']=48
    elif change=='wrong_clock':r['T_CD']+=1
    else:r['evaluation_candidate']='D25'
    with pytest.raises(PermissionError):v.register(g,**args)


def test_native_bound_once_charge_deduplicates_and_retains_all_other_escrows(captured):
    g,args=inputs(captured);g=v.register(g,**args);rid=args['record']['request_id'];jid='7654321.stdct-mgmt-02'
    g=v.bind_job(g,request_id=rid,job_id=jid,now_epoch=time.time());old=deepcopy(g)
    with pytest.raises(PermissionError):v.bind_job(g,request_id=rid,job_id=jid,now_epoch=time.time())
    with pytest.raises(PermissionError):v.charge(g,request_id=rid,job_id=jid,kind='generation_queries',count=1,effect_id='1:1:query')
    g=v.charge(g,request_id=rid,job_id=jid,kind='model_loads',count=1,effect_id='1:1:model')
    g=v.charge(g,request_id=rid,job_id=jid,kind='generation_queries',count=1,effect_id='1:2:query')
    g=v.charge(g,request_id=rid,job_id=jid,kind='CPU_s',count=13.25,effect_id='1:3:cpu')
    assert old['generation_queries']+1==g['generation_queries']
    assert g['CPU_s']==old['CPU_s']+13.25 and g['CPU_reserved_s']==old['CPU_reserved_s']-13.25
    assert g[a.PACKAGE]==old[a.PACKAGE]
    with pytest.raises(PermissionError):v.charge(g,request_id=rid,job_id=jid,kind='generation_queries',count=1,effect_id='1:2:query')
    g['unknown_effects'].append(dict(kind='unrelated'))
    with pytest.raises(PermissionError):v.charge(g,request_id=rid,job_id=jid,kind='model_loads',count=1,effect_id='1:4:model')


def zero_terminal(captured):
    g,args=inputs(captured);g=v.register(g,**args);rid=args['record']['request_id'];jid='7654321.stdct-mgmt-02'
    g=v.bind_job(g,request_id=rid,job_id=jid,now_epoch=time.time());r=v.item(g,rid)
    native=dict(schema='V5_CD_VANDA_COLOCATED_PBS_TERMINAL_V1',job_id=jid,job_state='F',
        project=r['project'],job_name=r['job_name'],owner='e1192548',ngpus=1,request_id=rid,
        submission_nonce=r['submission_nonce'],recorded_process_terminal=True,terminal_inventory_stable=True,
        all_actual_workers_terminal=True,Exit_status=1,observed_epoch=time.time(),conservative_GPU_s=10,
        conservative_CPU_s=5,new_resident_allocated_upper_bytes=r['additional_output_upper_bytes'])
    native['CPU_classification']=v.classify_PBS_CPU(raw_PBS_CPU_s=7,neural_CPU_s=3,physical_counter_CPU_s=2,physical_RPC_CPU_s=0)
    effects=dict(schema='V5_CD_VANDA_COLOCATED_NEURAL_EFFECT_SUMMARY_V1',job_id=jid,request_id=rid,
        request_sha256=r['request_sha256'],source_sha=r['source_sha'],run_manifest_sha256=r['run_manifest_sha256'],
        checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],statistics_sha256=a.STATS,evaluation_candidate='C50',
        optimizer_updates_committed=0,effect_journal_complete=True,query_attempts=0,model_load_attempts=0,unknown_query_intents=[])
    physical=dict(schema='V5_CD_VANDA_POLICY_EFFECT_LEDGER_V1',physical_site='VANDA_A40',run_uuid=r['child_run_uuid'],
        authority_release=successor.TASK_SOURCE,main_starts=0,engine_starts=0,live_and_recovery_integrations=0,
        shadow_integrations=0,Core_CPU_s=2.,reservations={},unknown_effects=[])
    return g,dict(request_id=rid,native=native,effects=effects,physical=physical,now_epoch=time.time())


def test_known_zero_financial_terminal_keeps_full_prior_fees_and_cannot_advance_stage(captured):
    g,args=zero_terminal(captured);out=v.settle(g,**args);r=v.item(out,args['request_id'])
    assert out['used']['H200_s']==g['used']['H200_s']+10
    assert out['CPU_s']==g['CPU_s']+5
    assert r['settlement_financial_deltas']['output_bytes']==0
    assert out['new_storage_current_bytes']==g['new_storage_current_bytes']
    assert out[a.PACKAGE]['completed_stages']==g[a.PACKAGE]['completed_stages']
    assert r['actual_physical_effects']['main_starts']==0
    with pytest.raises(PermissionError):v.settle(out,**args)
    with pytest.raises(PermissionError):v.complete_stage(out,request_id=args['request_id'],population_bytes=b'{}',
        evidence=[],GPU_lease_bytes=b'{}',physical_lease_bytes=b'{}',now_epoch=time.time())


@pytest.mark.parametrize('change',['not_F','missing_final','wrong_job','unknown_physics','pending_physics',
                                  'unpaid_query','too_much_cpu','unknown_gpu','missing_counter','old_native'])
def test_unknown_or_mismatched_effects_never_settle_as_zero(captured,change):
    g,args=zero_terminal(captured)
    if change=='not_F':args['native']['job_state']='R'
    elif change=='missing_final':args['native']['recorded_process_terminal']=False
    elif change=='wrong_job':args['native']['job_id']='999.stdct-mgmt-02'
    elif change=='unknown_physics':args['physical']['unknown_effects']=[dict(error='UNKNOWN')]
    elif change=='pending_physics':args['physical']['reservations']={'prefix':dict(kind='shadow_integrations',maximum=200)}
    elif change=='unpaid_query':args['effects']['query_attempts']=1
    elif change=='too_much_cpu':args['native']['conservative_CPU_s']=173901
    elif change=='unknown_gpu':args['native']['all_actual_workers_terminal']=False
    elif change=='missing_counter':args['physical'].pop('engine_starts')
    else:args['native']['observed_epoch']-=121
    with pytest.raises((PermissionError,KeyError)):v.settle(g,**args)


def test_shared_control_is_not_billed_again_as_PBS_child_CPU():
    p=v.classify_PBS_CPU(raw_PBS_CPU_s=1200,neural_CPU_s=900.2,physical_counter_CPU_s=200.3,physical_RPC_CPU_s=100)
    assert p['conservative_payload_CPU_s']==1001
    assert p['compute_control_CPU_s_covered_by_existing_prepaid_upper']==199.5
    assert p['existing_prepaid_upper_s']==7200 and not p['old_fees_refunded']
    with pytest.raises(PermissionError):v.classify_PBS_CPU(raw_PBS_CPU_s=9000,neural_CPU_s=1,physical_counter_CPU_s=0,physical_RPC_CPU_s=0)
