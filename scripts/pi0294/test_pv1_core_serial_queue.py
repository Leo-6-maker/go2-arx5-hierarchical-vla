"""Captured real budgets, simulated process receipts. No native dispatch."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts.pi0294.pv1 import evaluation_core_queue as q,evaluation_core_accounting as c
from scripts.pi0294.pv1 import evaluation_core_admission as a,evaluation_admission as old
from scripts.pi0294.pv1 import evaluation_work as w,evaluation_stage as stage
from scripts.pi0294.test_pv1_core_evaluation_admission import captured_inputs
from scripts.pi0294.test_pv1_core_evaluation_accounting import captured_global,row,process,BOOT,NOW

QUEUE_ID='52345678-1234-4234-8234-123456789abc'


def pair_fixture(captured,now=c.old.STOP_NEW-200):
    inputs,record=captured;work=w.remaining(inputs['population_bytes'],evidence_by_candidate=inputs['evidence_by_candidate'])
    pairs=[]
    for i,candidate in enumerate(stage.CANDIDATES):
        item=dict(inputs=deepcopy(inputs),record=deepcopy(record));r=item['record']
        r.update(status=c.STATES[4],queue_id=QUEUE_ID,queue_position=i,queue_dispatch_mode=q.MODE,
            request_id=('2' if i==0 else '6')+'2345678-1234-4234-8234-123456789abc',
            submission_nonce=('3' if i==0 else '7')+'2345678-1234-4234-8234-123456789abc',
            child_run_uuid='fixture-core-evaluation-'+str(i),evaluation_candidate=candidate,
            checkpoint_manifest_sha256=c.old.ENDPOINTS[candidate],created_epoch=now-5)
        job=dict(next(j for j in work['jobs'] if j['candidate']==candidate and j['noise_bank']==294101),
            native_platform='CORELAB_RTX4080')
        job['GPU_site_output_upper_bytes']+=c.MODEL_BYTES
        r['planned_panel_sha256']=a.digest(job);r['prior_outcomes_sha256']=a.digest(work['histories'][candidate]['prior_rows'])
        pairs.append(item)
    pin=c.digest(q.plan([p['record'] for p in pairs]))
    for i,item in enumerate(pairs):
        r=item['record'];r['queue_plan_sha256']=pin
        m=json.loads(inputs['manifest_bytes']);m['run_uuid']=r['child_run_uuid']
        m['execution']['sealed_inputs']['plans/CORE_RESIDENT_STAGE_QUEUE.json']=pin
        mb=old.canonical(m);r['run_manifest_sha256']=hashlib.sha256(mb).hexdigest()
        request=json.loads(inputs['request_bytes'])
        for key in ('request_id','submission_nonce','evaluation_candidate','checkpoint_manifest_sha256',
                'created_epoch','planned_panel_sha256','prior_outcomes_sha256','run_manifest_sha256',
                'queue_id','queue_position','queue_dispatch_mode','queue_plan_sha256'):request[key]=r[key]
        request.update(run_uuid=m['run_uuid'],allocation_number=i+1,sealed_inputs=m['execution']['sealed_inputs'])
        snapshot=json.loads(inputs['snapshot_bytes']);snapshot['observed_epoch']=now
        item['inputs'].update(manifest_bytes=mb,request_bytes=old.canonical(request),
            snapshot_bytes=old.canonical(snapshot),now_epoch=now)
        proof=a.recompute(**item['inputs'],purpose='PREPARE')
        request['prequeue_future_admission_sha256']=r['prequeue_future_admission_sha256']=proof['prequeue_context_sha256']
        item['inputs']['request_bytes']=old.canonical(request)
        r['request_sha256']=hashlib.sha256(item['inputs']['request_bytes']).hexdigest()
    return pairs


def acknowledgement(g,now):
    p=g[q.FIELD]['plan'];rows=sorted(g[c.FIELD].values(),key=lambda r:r['queue_position'])
    return dict(schema='PV1_CORE_RESIDENT_QUEUE_ACCEPTED_V1',hostname='corelab-4080s',uid=2009,
        source_sha=p['source_sha'],queue_id=QUEUE_ID,queue_plan_sha256=c.digest(p),process_state='RUNNING',
        observed_epoch=now,started_epoch=now-2,accepted_epoch=now,boot_id=BOOT,pid=111,proc_start=222,
        owner_handle=c.process_handle(BOOT,111,222),native_unit='pv1-core-'+QUEUE_ID+'.service',
        native_unit_MainPID=111,native_unit_ActiveState='active',native_queue_bytes_sha256=c.digest(p),
        requests=[{k:r[k] for k in ('request_id','request_sha256','submission_nonce')} for r in rows],
        native_GPU_processes_started=0,model_loads=0,neural_queries=0,physics_integrations=0)


def test_real_pair_resources_can_be_reserved_before_native_acceptance(captured_inputs):
    now=c.old.STOP_NEW-200;pair=pair_fixture(captured_inputs,now);result=a.reserve_stage_queue_proposal(pair)
    g=result['proposed_global'];local=result['proposed_LOCAL'];before=json.loads(pair[0]['inputs']['global_bytes'])
    assert g['PBS_submissions']==before['PBS_submissions'] and g['training_runs']==before['training_runs']
    assert g['used']==before['used'] and g['optimizer_updates']==11714
    assert g['reserved']['H200_s']==57720 and g['reserved']['neural_queries']==3874
    assert g['CPU_reserved_s']==160000 and local['CPU_reserved_s']['CoreLab']==160000
    assert local['Core_native_GPU_reserved']['transfer_bytes']==4*c.MODEL_BYTES
    assert result['native_queue_accepted'] is False and result['native_dispatch_admitted'] is False
    for proof in result['recomputed_proofs']:
        assert proof['shortfalls']=={} and proof['stage_clock']['last_stage_submission_fits_original_stop_new']
    work=w.remaining(pair[0]['inputs']['population_bytes'],evidence_by_candidate=pair[0]['inputs']['evidence_by_candidate'])
    work['jobs']=[j for j in work['jobs'] if j['candidate'] in stage.CANDIDATES and j['noise_bank']==294101]
    assert not stage.clock(work,now_epoch=now)['last_stage_submission_fits_original_stop_new']
    assert q.clock(work,now_epoch=now)['last_stage_submission_fits_original_stop_new']
    assert not q.clock(work,now_epoch=c.old.STOP_NEW)['last_stage_submission_fits_original_stop_new']
    assert not q.clock(work,now_epoch=c.old.STOP_PHYSICS-57719)['both_stage_panels_fit_original_physics_window']
    # Intents cannot masquerade as accepted tasks or receive model/query debit.
    r=next(iter(g[c.FIELD].values()))
    with pytest.raises(PermissionError):c.bind(g,request_id=r['request_id'],process=process(r),now_epoch=now)
    with pytest.raises(PermissionError):c.charge(g,request_id=r['request_id'],native_handle=None,
        kind='model_loads',count=1,effect_id='fixture-load')


@pytest.mark.parametrize('fault',['old_ack','after_stop_new','wrong_source','missing_owner','wrong_plan',
    'foreign_PID','inactive_owner','wrong_request','model_effect','missing_escrow','replay'])
def test_queue_acceptance_requires_exact_live_owner_and_never_refunds_unknown(captured_inputs,fault):
    now=c.old.STOP_NEW-200;g=a.reserve_stage_queue_proposal(pair_fixture(captured_inputs,now))['proposed_global']
    ack=acknowledgement(g,now);before=deepcopy(g)
    if fault=='old_ack':ack['observed_epoch']-=121
    if fault=='after_stop_new':now=c.old.STOP_NEW;ack['observed_epoch']=ack['accepted_epoch']=now
    if fault=='wrong_source':ack['source_sha']='f'*40
    if fault=='missing_owner':ack['process_state']='MISSING'
    if fault=='wrong_plan':ack['native_queue_bytes_sha256']='f'*64
    if fault=='foreign_PID':ack['native_unit_MainPID']+=1
    if fault=='inactive_owner':ack['native_unit_ActiveState']='inactive'
    if fault=='wrong_request':ack['requests'][1]['request_sha256']='f'*64
    if fault=='model_effect':ack['model_loads']=1
    if fault=='missing_escrow':g['reserved']['H200_s']-=1
    if fault=='replay':g=q.accept_two(g,native_ack=ack,now_epoch=now)
    with pytest.raises(PermissionError):q.accept_two(g,native_ack=ack,now_epoch=now)
    if fault not in ('missing_escrow','replay'):assert g==before


def test_queue_dispatch_separate_from_acceptance_and_single_actual_process(captured_inputs):
    now=c.old.STOP_NEW-200;g=a.reserve_stage_queue_proposal(pair_fixture(captured_inputs,now))['proposed_global']
    ack=acknowledgement(g,now);g=q.accept_two(g,native_ack=ack,now_epoch=now)
    ordered=sorted(g[c.FIELD].values(),key=lambda r:r['queue_position']);first,second=ordered
    # Starting later does not create a new job; the two native queue accepts
    # must already exist. This remains identity-only, never model/physics GO.
    later=c.old.STOP_NEW+10;p=process(first);p.update(observed_epoch=later,started_epoch=later-2)
    dispatch=dict(ack,schema='PV1_CORE_RESIDENT_QUEUE_DISPATCH_V1',observed_epoch=later,
        request_id=first['request_id'],request_sha256=first['request_sha256'],
        GPU_process_native_handle=c.process_handle(p['boot_id'],p['pid'],p['proc_start']))
    # At this deliberately late instant the full conservative 16h+2m no
    # longer fits. Accepting a queue cannot defeat the physics cutoff.
    with pytest.raises(PermissionError):c.bind(g,request_id=first['request_id'],process=p,now_epoch=later,queue_dispatch_proof=dispatch)
    p.update(observed_epoch=now+10,started_epoch=now+8);dispatch['observed_epoch']=now+10
    bound=c.bind(g,request_id=first['request_id'],process=p,now_epoch=now+10,queue_dispatch_proof=dispatch)
    assert bound[c.FIELD][first['request_id']]['status']==c.STATES[1]
    bad=deepcopy(dispatch);bad['pid']+=1
    with pytest.raises(PermissionError):c.bind(g,request_id=first['request_id'],process=p,now_epoch=now+10,queue_dispatch_proof=bad)
    p2=process(second);p2.update(observed_epoch=now+11,started_epoch=now+9,pid=12346)
    d2=dict(dispatch,request_id=second['request_id'],request_sha256=second['request_sha256'],
        GPU_process_native_handle=c.process_handle(p2['boot_id'],p2['pid'],p2['proc_start']))
    with pytest.raises(PermissionError):c.bind(bound,request_id=second['request_id'],process=p2,now_epoch=now+11,queue_dispatch_proof=d2)
    with pytest.raises(PermissionError):a.reserve_stage_queue_proposal([pair_fixture(captured_inputs)[0]]*2)


def native_inputs(captured):
    from scripts.pi0294.test_pv1_core_premodel_join import admission
    now=c.old.STOP_NEW-200;pair=pair_fixture(captured,now)
    proposal=a.reserve_stage_queue_proposal(pair);g=proposal['proposed_global']
    ack=acknowledgement(g,now);g=q.accept_two(g,native_ack=ack,now_epoch=now)
    inputs=deepcopy(pair[0]['inputs']);r=pair[0]['record']
    p=admission(inputs)[2]['native_process'];p.update(started_epoch=now+8,observed_epoch=now+10)
    dispatch=dict(ack,schema='PV1_CORE_RESIDENT_QUEUE_DISPATCH_V1',observed_epoch=now+10,
        request_id=r['request_id'],request_sha256=r['request_sha256'],
        GPU_process_native_handle=c.process_handle(p['boot_id'],p['pid'],p['proc_start']))
    g=c.bind(g,request_id=r['request_id'],process=p,now_epoch=now+10,queue_dispatch_proof=dispatch)
    gb=old.canonical(g);lb=old.canonical(proposal['proposed_LOCAL'])
    snapshot=json.loads(inputs['snapshot_bytes']);used,held=old.resource_floor(g,proposal['proposed_LOCAL'])
    used.update(Core_storage_bytes=115058147328,local_storage_bytes=8589934592)
    snapshot.update(global_ledger_sha256=hashlib.sha256(gb).hexdigest(),LOCAL_ledger_sha256=hashlib.sha256(lb).hexdigest(),
        used=used,reserved=held,observed_epoch=now+10)
    inputs.update(global_bytes=gb,local_bytes=lb,snapshot_bytes=old.canonical(snapshot),now_epoch=now+10,
        native_facts_bytes=old.canonical(p))
    return inputs


def test_bound_native_resource_recompute_credits_only_real_paired_escrow(captured_inputs):
    inputs=native_inputs(captured_inputs);proof=a.recompute(**inputs,purpose='NATIVE_PREMODEL')
    assert proof['shortfalls']=={} and proof['native_or_scientific_admission'] is False
    d=proof['dimensions']
    assert d['H200_s']['reserved']==57720 and d['H200_s']['remaining_stage_upper']==0
    assert d['neural_queries']['reserved']==3874 and d['neural_queries']['remaining_stage_upper']==0
    assert d['Core_CPU_s']['reserved']==160000 and d['Core_CPU_s']['remaining_stage_upper']==202800
    assert d['Hopper_CPU_s']['reserved']==160000 and d['Hopper_CPU_s']['remaining_stage_upper']==15000
    assert d['Core_storage_bytes']['reserved']==2*(c.output_upper(24)+c.MODEL_BYTES)


@pytest.mark.parametrize('fault',['intent_not_accepted','stale_process','other_PID','other_request',
    'old_snapshot','missing_LOCAL_hold','wrong_kernel_limit','busy_GPU'])
def test_native_resource_credit_cannot_be_invented_from_prepared_flags(captured_inputs,fault):
    inputs=native_inputs(captured_inputs);p=json.loads(inputs['native_facts_bytes'])
    g=json.loads(inputs['global_bytes']);l=json.loads(inputs['local_bytes']);snap=json.loads(inputs['snapshot_bytes'])
    if fault=='intent_not_accepted':
        # Preserve both real ledger bytes but supply a cold intent as facts.
        p=dict(status='PASS_PV1_OWN_NATIVE_CORELAB_GPU_ALLOCATION')
    if fault=='stale_process':p['observed_epoch']-=121
    if fault=='other_PID':p['pid']+=1
    if fault=='other_request':p['request_sha256']='f'*64
    if fault=='old_snapshot':snap['observed_epoch']-=121
    if fault=='missing_LOCAL_hold':l['Core_native_GPU_reservations'].clear()
    if fault=='wrong_kernel_limit':p['kernel_limits']['CPU_quota']=500000
    if fault=='busy_GPU':p['native_GPU_raw']['processes']['stdout']=p['GPU_uuid']+', 12345, python, 12\n'
    inputs['global_bytes']=old.canonical(g);inputs['local_bytes']=old.canonical(l)
    snap.update(global_ledger_sha256=hashlib.sha256(inputs['global_bytes']).hexdigest(),
        LOCAL_ledger_sha256=hashlib.sha256(inputs['local_bytes']).hexdigest())
    inputs['snapshot_bytes']=old.canonical(snap);inputs['native_facts_bytes']=old.canonical(p)
    with pytest.raises(PermissionError):a.recompute(**inputs,purpose='NATIVE_PREMODEL')


def test_only_preaccepted_second_panel_can_join_a_post_cutoff_native_process(captured_inputs):
    from scripts.pi0294.test_pv1_core_premodel_join import admission,joined
    now=c.old.STOP_NEW-200;pair=pair_fixture(captured_inputs,now)
    g=a.reserve_stage_queue_proposal(pair)['proposed_global'];ack=acknowledgement(g,now)
    m,r,allocation=admission(pair[1]['inputs']);p=allocation['native_process']
    p.update(started_epoch=c.old.STOP_NEW+1000,observed_epoch=c.old.STOP_NEW+1002)
    allocation.update(allocation_start_epoch=p['started_epoch'],allocation_end_epoch=p['started_epoch']+28800,
        native_process_sha256=c.digest(p),native_queue_acceptance=ack,native_queue_plan=g[q.FIELD]['plan'])
    assert joined(m,r,allocation)==allocation
    assert p['native_model_permission'] is False and p['physics_authorized'] is False
    # This only joins identity. The current owner/unit, fresh resources and
    # complete native communication qualifications still have to be observed.
    for fault in ('missing_ack','late_accept','wrong_request','wrong_plan','new_owner'):
        v=deepcopy(allocation)
        if fault=='missing_ack':v.pop('native_queue_acceptance')
        if fault=='late_accept':v['native_queue_acceptance']['accepted_epoch']=c.old.STOP_NEW
        if fault=='wrong_request':v['native_queue_acceptance']['requests'][1]['request_sha256']='f'*64
        if fault=='wrong_plan':v['native_queue_plan']['maximum_tasks']=3
        if fault=='new_owner':v['native_queue_acceptance']['started_epoch']=c.old.STOP_NEW+1
        with pytest.raises(PermissionError):joined(m,r,v)
