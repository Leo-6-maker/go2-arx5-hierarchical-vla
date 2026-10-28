"""Actual closed bytes feed a pure predecessor join, never a retry grant."""
import base64,json
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.generalization import amd_training_retry_context as c

B=Path(__file__).resolve().parents[4]/'work/private/v5-execution-20261025'
@pytest.fixture
def actual():
    path=B/'GEN1_INDEPENDENT_TRAIN_01_FAILED1497713_TERMINAL_INDEPENDENT_READBACK.private.json'
    if not path.exists():pytest.skip('private actual1497713 closed-byte capture unavailable')
    return dict(capture=json.loads((B/'GEN1_INDEPENDENT_TRAIN_01_FAILED1497713_FULL_INPUTS_READONLY.private.json').read_bytes()),readback=json.loads(path.read_bytes()))

def test_actual_settlement_both_once_pairs_and_own_lease_form_predecessor_not_permission(actual):
    before=deepcopy(actual);v=c.closed_predecessor(**actual)
    assert v['status']==c.STATUS and v['predecessor_job_id']=='1497713.stdct-mgmt-02'
    assert v['current_PBS_cap']==36 and v['required_PBS_cap']==37
    assert v['additional_explicit_human_count_authority_required']
    assert v['original_row']['scientific_endpoint_complete']is False
    assert v['previous_paid_totals']['PBS']==33 and v['previous_paid_totals']['GPU_s']==150429
    assert not v['registration_admitted']and not v['native_training_permission']and not v['native_GPU_admission']
    assert v['optimizer_start_step']==0 and not v['inherit_failed_or_C_D_weights']
    assert not v['repeat_Teacher_or_official_export']and v['bank_pack_before_model_still_required']
    assert v['closed_export']['complete_bank_ready']is False
    g=json.loads(base64.b64decode(actual['readback']['GLOBAL_base64']))
    assert c.verify_preserved_history(g,v)['all_original_paid_history_preserved']
    assert actual==before

def update_json(record,key,mutate):
    b=base64.b64decode(record[key]);v=json.loads(b);mutate(v);b=c.a.raw(v);record[key]=base64.b64encode(b).decode();record[key.replace('_base64','_sha256')]=c.a.sha(b)

@pytest.mark.parametrize('bad',['missing_receipt','unknown_intent','receipt_target','intent_hash','before_hash','refund','changed_GLOBAL','active_lease','wrong_manifest','wrong_effects','future_close','reopened_deadline','missing_original_effects_preservation'])
def test_no_ready_file_or_counter_only_closure_or_unknown_pair_can_admit_retry(actual,bad):
    p=deepcopy(actual);r=p['readback']
    if bad=='missing_receipt':del r['proofs'][c.CLOSE_KEY+'.RECEIPT.json']
    elif bad in ('unknown_intent','receipt_target','intent_hash','before_hash'):
        name=c.CLOSE_KEY+'.RECEIPT.json';row=r['proofs'][name];v=json.loads(base64.b64decode(row['base64']))
        if bad=='unknown_intent':v['status']='UNKNOWN'
        elif bad=='receipt_target':v['target']+='-other'
        elif bad=='intent_hash':v['intent_sha256']='0'*64
        else:v['before_sha256']='0'*64
        b=c.a.raw(v);row.update(base64=base64.b64encode(b).decode(),sha256=c.a.sha(b))
    elif bad in ('refund','changed_GLOBAL'):
        update_json(r,'GLOBAL_base64',lambda g:g.update(CPU_s=g['CPU_s']-1)if bad=='refund'else g.update(revision=g['revision']+1))
    else:
        def change(lease):
            if bad=='active_lease':lease['status']='ACTIVE'
            elif bad=='wrong_manifest':lease['manifest_sha256']='0'*64
            elif bad=='wrong_effects':lease['terminal_effect_summary_sha256']='0'*64
            elif bad=='future_close':lease['closed_epoch']=r['epoch']+1
            elif bad=='reopened_deadline':lease['deadline_epoch']+=1
            else:lease['original_missing_effect_journal_and_final_preserved']=False
        update_json(r,'lease_base64',change)
    with pytest.raises((PermissionError,KeyError)):c.closed_predecessor(**p)

@pytest.mark.parametrize('bad',['old_row','old_failure_duplicate','old_paid_GPU','old_paid_CPU','old_peak','escrow','marker','clock','unknown','permission_flag'])
def test_downstream_fresh_view_preserves_all_prior_costs_failures_clock_and_escrow(actual,bad):
    v=c.closed_predecessor(**actual);g=json.loads(base64.b64decode(actual['readback']['GLOBAL_base64']))
    if bad=='old_row':g['PBS_submissions'][0]['source_sha']='0'*40
    elif bad=='old_failure_duplicate':g['PBS_submissions'].append(deepcopy(v['original_row']))
    elif bad=='old_paid_GPU':g['used']['H200_s']-=1
    elif bad=='old_paid_CPU':g['CPU_s']-=1
    elif bad=='old_peak':g['new_storage_peak_bytes']-=1
    elif bad=='escrow':g[c.a.original.PACKAGE]['remaining_resources']['H200_s']-=1
    elif bad=='marker':g[c.a.authority.FIELD][c.a.FIELD]['registered_epoch']+=1
    elif bad=='clock':v['deadline_epoch']+=1
    elif bad=='unknown':g['unknown_effects']=['fixture unknown effect']
    else:v['native_training_permission']=True
    with pytest.raises(PermissionError):c.verify_preserved_history(g,v)

def test_pure_predecessor_does_not_bypass_original_single_registration_gate(actual):
    context=c.closed_predecessor(**actual);g=json.loads(base64.b64decode(actual['readback']['GLOBAL_base64']))
    # Even exact successful predecessor verification supplies no extra cap,
    # adopted scope marker, new UUID, prepay, registry entry or PBS request.
    out=c.verify_preserved_history(g,context)
    assert not out['registration_admitted']and c.a.authority.validate(g)['caps']['PBS_submissions']==36
    assert len(c.a.rows(g))==1 and c.a.rows(g)[0]['status']==c.a.STATES[-1]
    from scripts.pi0294.test_pv1_recipe import old_recipe
    p=json.loads((B/'GEN1_INDEPENDENT_TRAIN_01_PREPARED.private.json').read_bytes())
    local=base64.b64decode(json.loads(base64.b64decode(p['small']['plans/GEN1_OWNER_RESOURCES.private.json']))['historical_LOCAL_base64'])
    row=deepcopy(context['original_row']);row.update(status=c.a.STATES[0],job_id=None)
    with pytest.raises(PermissionError,match='TRAIN_REGISTER_ONLY_ONCE'):
        c.a.register(g,row,closed=context['closed_export'],original_recipe=old_recipe(),historical_LOCAL_bytes=local,now_epoch=actual['readback']['epoch'])

@pytest.fixture
def future_packet(actual):
    context=c.closed_predecessor(**actual)
    p=json.loads((B/'GEN1_INDEPENDENT_TRAIN_01_PREPARED.private.json').read_bytes())
    manifest=deepcopy(p['manifest']);seal=deepcopy(p['seal'])
    request=json.loads(base64.b64decode(actual['capture']['files']['PBS/REQUEST.json']['base64']))
    # Explicit future CPU fixtures only; neither UUID nor this packet is
    # saved as an actual request, registered role, cap grant or native proof.
    manifest.update(run_uuid='gen1-cold-fixture',output_root='/fixture/campaign/children/gen1-cold-fixture')
    manifest['code']['vla_commit']='a'*40;seal['source_sha']='a'*40
    request.update(request_id='11111111-1111-4111-8111-111111111111',
        submission_nonce='22222222-2222-4222-8222-222222222222',
        run_uuid=manifest['run_uuid'],job_id=None,status=c.a.STATES[0],source_sha='a'*40,
        retry_predecessor_context_sha256=c.a.sha(c.a.raw(context)))
    def seal_packet():
        manifest['code']['source_seal_sha256']=c.a.sha(c.a.raw(seal));request['run_manifest_sha256']=c.a.sha(c.a.raw(manifest))
    seal_packet()
    return manifest,request,seal,context,seal_packet

def test_cold_packet_uses_same_saved_inputs_new_identity_and_original_science_only(future_packet):
    m,r,s,context,_=future_packet
    v=c.verify_cold_science(m,r,s,context)
    assert v['science_and_saved_input_parity']and v['original_bank_and_model_gates_still_required']
    assert not v['registration_admitted']and not v['native_training_permission']

@pytest.mark.parametrize('bad',['old_request','old_nonce','old_source','same_root','context','manifest','bank','recipe','deadline','pack_deadline','parent','C50','R0','updates','control','agent','frozen15','overlay'])
def test_new_packet_cannot_change_parent_recipe_gate_safety_or_costs(future_packet,bad):
    m,r,s,context,seal_packet=future_packet
    if bad=='old_request':r['request_id']=context['predecessor_request_id']
    elif bad=='old_nonce':r['submission_nonce']=context['predecessor_nonce']
    elif bad=='old_source':m['code']['vla_commit']=r['source_sha']=s['source_sha']=context['predecessor_source_sha']
    elif bad=='same_root':m['output_root']=context['predecessor_root'];m['run_uuid']=r['run_uuid']=Path(m['output_root']).name
    elif bad=='context':r['retry_predecessor_context_sha256']='0'*64
    elif bad=='manifest':r['run_manifest_sha256']='0'*64
    elif bad=='bank':r['bank_manifest_sha256']='0'*64
    elif bad=='recipe':r['recipe_sha256']='0'*64
    elif bad=='deadline':r['deadline_epoch']+=1
    elif bad=='pack_deadline':r['CPU_bank_deadline_epoch']+=1
    elif bad=='parent':m['science']['parent_checkpoint_manifest_sha256']='0'*64
    elif bad=='C50':m['science']['C_D_weights_not_inherited']=False
    elif bad=='R0':m['science']['statistics_sha256']='0'*64
    elif bad=='updates':m['science']['optimizer_updates']=2048
    elif bad=='control':m['limits']['CPU_control_upper_s']+=1
    elif bad=='agent':m['execution']['agent_liveness_required']=True
    elif bad=='frozen15':s['files'][next(iter(s['frozen15']))]='0'*64
    else:s['frozen_overlay_origins']={}
    if bad!='manifest':seal_packet()
    with pytest.raises((PermissionError,KeyError)):c.verify_cold_science(m,r,s,context)
