"""Actual settled05 history; new source/identity are explicit CPU fixtures."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_core_d25_admission_repair as repair,mix_core_accounting as c
from scripts.pi0294.pv1 import mix_core_d25_local_successor as paid,mix_core_d25_cold_source as cold,mix_accounting as a


@pytest.fixture(scope='module')
def prepared():
    directory=os.environ.get('D25_NODE_LOCAL05_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('actual settled05 private captures required')
    root=Path(directory);d=root/'MAINTENANCE30_CONTROL_IO';read=lambda n:json.loads((d/(n+'.private.json')).read_bytes())
    current=read('NL05_NATIVE_ONCE_SETTLEMENT_PREPARED');gb=gzip.decompress(base64.b64decode(current['GLOBAL_gzip_base64']));lb=gzip.decompress(base64.b64decode(current['LOCAL_gzip_base64']))
    closures=sorted(d.glob('NL06_CLOSED05_FRESH_*.RESULT.json'));assert closures
    closed=json.loads(closures[-1].read_bytes());source=read('D25_SOURCE05_COPY_PREPARED');seal=deepcopy(source['seal'])
    seal['source_sha']='f'*40;seal['files']['scripts/pi0294/pv1/mix_core_admission.py']=repair.FIXED_ADMISSION
    now=closed['observed_epoch']+1
    args=dict(global_bytes=gb,local_bytes=lb,authority_bytes=(d/'DIRECT_AUTHORITY.json').read_bytes(),closed_readback=closed,
        source_seal_bytes=a.raw(seal),frozen=source['frozen'],new_request_id='02856010-e0bc-45a4-a3f8-34657ac1255d',new_nonce='f629d48e-9552-4a92-b87e-2a5cae4ebc1c',
        source_readback=dict(status='ACTUAL_COMPLETE_COLD_SOURCE_SHA_READBACK',source_sha=seal['source_sha'],source_seal_sha256=a.sha(a.raw(seal)),file_sha256=seal['files'],
            hostname='corelab-4080s',uid=2009,model_calls=0,physics_ticks=0,observed_epoch=now,CPU_fixture_only=True),now_epoch=now)
    proposal=repair.propose(**args)
    return root,args,proposal


def test_actual_settled05_remains_terminal_and_only_original_remaining_panel_is_restored(prepared):
    _,args,p=prepared;oldg,oldl=map(json.loads,(args['global_bytes'],args['local_bytes']));g,l=p['proposed_global'],p['proposed_LOCAL'];m=repair.validate(g)
    assert g[c.FIELD]==oldg[c.FIELD]and g[c.FIELD][repair.FAILED]['status']=='NATIVE_TERMINAL'
    assert g['used']==oldg['used']and g['CPU_s']==oldg['CPU_s']and l['CPU_s']==oldl['CPU_s']
    assert g['PBS_submissions']==oldg['PBS_submissions']and g['V5_CD_successor_scope']==oldg['V5_CD_successor_scope']
    assert g[a.PACKAGE]['prepaid_control']==oldg[a.PACKAGE]['prepaid_control']
    assert l['V5_CD_Core_terminal_settlements']==oldl['V5_CD_Core_terminal_settlements']
    assert p['full_future_proof']['cap_changes_required']=={}and len(p['full_future_proof']['dimensions'])==14
    assert m['maximum_new_native_requests']==1 and m['new_control_payment']==0 and not p['native_admission']
    assert cold.effective_identity(g,g[paid.FIELD])==dict(source_sha='f'*40,run_uuid=repair.RUN)
    assert cold.effective_identity(g,g[paid.FIELD],request_id=repair.FAILED)==dict(source_sha='371c5cc4fc9bebb8e823c14f6f27d1ddd41926fa',run_uuid=cold.LOCAL05_RUN)
    limits=paid.finite_control_scope(g,m['replacement_request_id']);assert limits['source_preparation_CPU_s']==605
    assert limits['Core_CPU_components_s']['original_RPC']==6595 and sum(limits['Core_CPU_components_s'].values())==9000
    assert limits['financial_costs']==paid.CONTROL and limits['retained_pregrant_owner_CPU_s']==5


@pytest.mark.parametrize('fault',['old-row','cost','authority','grant','model','old-role','clock','old-marker'])
def test_no_other_failure_or_unknown_effect_can_use_this_one_repair(prepared,fault):
    _,_,p=prepared;g=deepcopy(p['proposed_global']);m=g[repair.FIELD];old=g[c.FIELD][repair.FAILED]
    if fault=='old-row':old['native_exit_status']=0
    elif fault=='cost':old['conservative_GPU_s']=242
    elif fault=='authority':m['authority_sha256']='0'*64
    elif fault=='grant':g.setdefault('V5_CD_Core_finite_delegations',{})[repair.FAILED]={}
    elif fault=='model':old['model_load_attempts']=1
    elif fault=='old-role':m['closed05_readback']['grant_and_activation_absent']=False
    elif fault=='clock':m['adopted_epoch']=m['deadline_epoch']
    else:m['financial_marker_sha256']='0'*64
    with pytest.raises(PermissionError):repair.validate(g)


def test_real_assembler_and_original_register_create_new_request_without_reopening05(prepared):
    from scripts.pi0294.pv1 import mix_core_assembly as assembler,mix_core_context as context,mix_core_delegation as delegation
    root,args,p=prepared;g,l=p['proposed_global'],p['proposed_LOCAL'];m=repair.validate(g)
    old=json.loads((root/'MAINTENANCE30_CONTROL_IO/NODE_LOCAL05_RUNTIME_ASSEMBLY_FINAL_PREPARED.private.json').read_bytes())
    owner=deepcopy(old['owner']);owner_root=str(Path(owner['root']).with_name('v5-cd-d25-eval-owner-06'));om=owner['manifest']
    om.update(run_uuid=Path(owner_root).name,output_root=owner_root);om['code'].update(vla_commit=m['source_sha'],source_seal_sha256=m['source_seal_sha256'])
    om['execution']['sealed_inputs'][cold.BINDING]=cold.digest(m)
    capture=base64.b64decode(old['assembly']['rows']['hopper']['files'][context.CAPTURE])
    template=json.loads((root.parent/'science-next/wdr1-target-identity/v4a-core-eval-preparation-01/CORE_PAIRED_PEER_TEMPLATES_7cc75180.private.json').read_bytes())
    remote=deepcopy(owner['config']['remote_binding']);remote['remote_seal_sha256']=m['source_seal_sha256']
    packet=assembler.assemble(template,global_bytes=a.raw(g),local_bytes=a.raw(l),training_capture_bytes=capture,arm='D25',source_sha=m['source_sha'],source_seal_bytes=args['source_seal_bytes'],
        owner_root=owner_root,owner_manifest_bytes=a.raw(om),canonical_LOCAL_ledger=owner['config']['canonical_LOCAL_ledger'],remote_binding=remote,run_name=repair.RUN,
        request_id=m['replacement_request_id'],nonce=m['replacement_submission_nonce'],GPU_uuid=g[c.FIELD][repair.FAILED]['GPU_uuid'],now_epoch=args['now_epoch'],desktop_grant_bytes=base64.b64decode(owner['desktop_grant_base64']),global_accounting_mode=delegation.MODE)
    cap=json.loads(capture);terminal={k:base64.b64decode(v)for k,v in cap['terminal_capture_base64'].items()}
    registered=c.register(g,l,packet['record'],base64.b64decode(packet['rows']['core']['manifest_bytes_base64']),terminal_capture=terminal,now_epoch=args['now_epoch'],training_capture_bytes=capture)
    v,w=registered['proposed_global'],registered['proposed_LOCAL'];c.checked(v);c.paired(v,w)
    assert v[c.FIELD][repair.FAILED]==g[c.FIELD][repair.FAILED]
    assert v[c.FIELD][m['replacement_request_id']]['status']==c.STATES[0]
    assert w['V5_CD_Core_CPU_substages'][c.key('D25')]['request_id']==m['replacement_request_id']
    assert repair.FAILED in w['V5_CD_historical_closed_Core_CPU_substages']
    assert len(v[c.FIELD])==len(g[c.FIELD])+1 and not registered['native_process_or_model_admission']
    # Exercise the actual admission/finite-grant path before spending another
    # native start. Only process facts below are explicit CPU fixtures.
    from scripts.pi0294.pv1 import mix_core_admission as admission
    from scripts.pi0294.pv1.evaluation_admission import resource_floor
    cap05=json.loads((root/'MAINTENANCE30_CONTROL_IO/NL05_PRE_GRANT_TERMINAL_CAPTURE.private.json').read_bytes())
    facts=json.loads(base64.b64decode(cap05['prebind_process_base64']));record=packet['record'];now=args['now_epoch']+5
    for key in('request_id','request_sha256','submission_nonce','run_manifest_sha256','source_sha'):facts[key]=record[key]
    facts.update(started_epoch=now-2,observed_epoch=now,CPU_fixture_only=True)
    facts['device']['exact_desktop_coexistence']['kernel_read']['observed_epoch']=now
    bound=c.bind(v,w,request_id=record['request_id'],process=facts,now_epoch=now)
    v,w=bound['proposed_global'],bound['proposed_LOCAL'];used,held=resource_floor(v,w);cfg=owner['config']
    used['Core_storage_bytes']=max(used['Core_storage_bytes'],cfg['Core_storage_upper_bytes'])
    used['local_storage_bytes']=max(used['local_storage_bytes'],cfg['LOCAL_storage_upper_bytes'])
    snapshot=dict(schema=admission.SNAPSHOT,task=cfg['marker']['task'],scope_grant_sha256=om['authority']['scope_grant_sha256'],T_CD=record['T_CD'],deadline_epoch=record['deadline_epoch'],GLOBAL_sha256=a.sha(a.raw(v)),LOCAL_sha256=a.sha(a.raw(w)),observed_epoch=now,heartbeat_only_raw_ledger=False,used=used,reserved=held,Core_storage_upper_bytes=used['Core_storage_bytes'],LOCAL_storage_upper_bytes=used['local_storage_bytes'],actual_current_Core_pending={},historical_storage_witness_sha256=cfg['storage_witness_sha256'])
    row=packet['rows']['hopper'];files={k:base64.b64decode(b)for k,b in row['files'].items()}
    result=delegation.issue(manifest_bytes=base64.b64decode(row['manifest_bytes_base64']),request_bytes=a.raw(packet['request']),global_bytes=a.raw(v),local_bytes=a.raw(w),snapshot_bytes=a.raw(snapshot),training_capture_bytes=capture,population_bytes=files['plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'],plan_bytes=files['plans/SOURCE_EVALUATION_PLAN.json'],prior_bytes=files['plans/PRIOR_EVALUATION_OUTCOMES.json'],native_facts_bytes=a.raw(facts),now_epoch=now)
    assert result['grant']['body']['admission']['shortfalls']=={}
    assert len(result['grant']['body']['admission']['dimensions'])==14
    assert not result['model_permission']and not result['physics_authorized']
