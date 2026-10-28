"""Captured real closed D25 history; future controller/identity are CPU fixtures.

No site access, native adoption, model, inference or physics occurs here.
"""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_core_d25_local_successor as d,mix_core_accounting as c,mix_accounting as a

@pytest.fixture(scope='module')
def captured():
    directory=os.environ.get('ARX5_RETRY_ACTUAL_CAPTURE_DIRECTORY')
    if not directory:pytest.skip('private original closed D25 captures required')
    root=Path(directory);read=lambda name:json.loads((root/name).read_bytes())
    pair=read('D25_RETRY02_NETWORK_TERMINAL_INDEPENDENT_DUAL_READBACK.private.json')
    gb=gzip.decompress(base64.b64decode(pair['global']['GLOBAL_gzip_base64']))
    lb=gzip.decompress(base64.b64decode(pair['local']['LOCAL_gzip_base64']))
    inventory=read('D25_FAILED_ROOTS_PAID_IO_NATIVE_METADATA_READONLY_1793167030858418267.private.json')
    historical=json.loads(read('D25_DELIVERED_RESTORED_IDENTITY_AND_GPU_READONLY_1793156446934928452.private.json')['stdout'])
    closed=read('D25_RETRY02_THREE_CLOSED_READONLY_1793164979876834045.private.json')['value']
    terminal=read('D25_RETRY02_TERMINAL_INDEPENDENT_ROUTES_READONLY_1793164662708107682.private.json')['value']
    now=max(v['value']['observed_epoch']for v in inventory.values())+1
    # Only the future observation wrapper/controller/UUIDs are fixtures. The
    # complete ledger, original invoices, raw terminal and inode facts stay real.
    closed=deepcopy(closed);closed['observed_epoch']=now
    plan=dict(schema='D25_FINITE_CONTROL_SEALED_HARD_LIMIT_PLAN_V1',source_sha='b'*40,
        costs=d.CONTROL,wire_phase_upper_bytes=d.WIRE,max_source_envelope_bytes=12*d.MiB,
        no_runtime_Core_to_Vanda_RPC=True,no_new_model_transfer=True,
        sealed_controller_sha256='c'*64,sealed_installer_sha256='d'*64,
        wire_planned_bytes={k:v for k,v in d.WIRE.items()})
    return dict(global_bytes=gb,local_bytes=lb,closed_readback=closed,terminal_proof=terminal,
        inventory_capture=inventory,historical_model=historical,control_plan=plan,
        new_request_id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
        new_nonce='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb',source_sha='b'*40,now_epoch=now)

def test_actual_saved_inventory_fixed_nine_members_no_stream(captured):
    p=d.inventory_guard(captured['inventory_capture'],captured['historical_model'],now_epoch=captured['now_epoch'])
    assert p['Core_model_allocated_bytes']==12576354304 and p['Core_control_metadata_allocated_bytes']==239661056
    assert p['Vanda_control_metadata_allocated_bytes']==5251072 and p['new_tensor_transfers']==0

def test_captured_closed_history_preserved_and_all_new_control_paid(captured):
    before=deepcopy(captured);out=d.propose(**captured);g,l=out['proposed_global'],out['proposed_LOCAL']
    old,oldl=json.loads(captured['global_bytes']),json.loads(captured['local_bytes'])
    assert g[c.FIELD]==old[c.FIELD]and g['PBS_submissions']==old['PBS_submissions']
    assert g[a.PACKAGE]['prepaid_control']==old[a.PACKAGE]['prepaid_control']
    assert l['V5_CD_Core_terminal_settlements']==oldl['V5_CD_Core_terminal_settlements']
    assert g['CPU_s']==old['CPU_s']+1800 and l['CPU_s']['CoreLab']==oldl['CPU_s']['CoreLab']+9000
    assert l['CPU_s']['local_PC']==oldl['CPU_s']['local_PC']+1800
    assert g['new_storage_current_bytes']==old['new_storage_current_bytes']+64*d.MiB
    assert l['network_transfer_accounted_bytes']==oldl['network_transfer_accounted_bytes']+64*d.MiB
    assert out['full_future_proof']['cap_changes_required']=={} and not out['native_registration']
    assert d.allocation(g,'D25')['Hopper_storage_bytes']==d.allocation(g,'D25')['transfer_bytes']==0
    assert d.finite_control_scope(g,captured['new_request_id'])['total_control_CPU_s']==9000
    assert before==captured
    with pytest.raises(PermissionError):d.propose(**dict(captured,global_bytes=a.raw(g),local_bytes=a.raw(l)))

def test_original_assembler_and_register_accept_only_third_exact_D25(captured):
    from scripts.pi0294.pv1 import mix_core_assembly as assembly,mix_stage_handoff as handoff,mix_core_context as context
    from scripts.pi0294.pv1.mix_core_delegation import MODE
    root=Path(os.environ['ARX5_RETRY_ACTUAL_CAPTURE_DIRECTORY'])
    old=json.loads((root/'D25_RETRY02_CORE_PEERS_PREPARED.private.json').read_bytes())
    template=json.loads((root.parent/'science-next/wdr1-target-identity/v4a-core-eval-preparation-01/CORE_PAIRED_PEER_TEMPLATES_7cc75180.private.json').read_bytes())
    out=d.propose(**captured);g,l=out['proposed_global'],out['proposed_LOCAL'];marker=g[d.FIELD]
    owner=deepcopy(old['owner']);owner_root=str(Path(owner['root']).with_name('v5-cd-d25-eval-owner-03'))
    m=owner['manifest'];m.update(run_uuid=Path(owner_root).name,output_root=owner_root);m['code']['vla_commit']=captured['source_sha']
    binding='plans/V5_CD_D25_FINITE_LOCAL_SUCCESSOR.private.json'
    m['execution'].update(global_accounting_mode=MODE);m['execution']['sealed_inputs'][binding]=d.digest(marker)
    oldcapture=json.loads(base64.b64decode(old['assembly']['rows']['hopper']['files'][context.CAPTURE]))
    terminal={k:base64.b64decode(v)for k,v in oldcapture['terminal_capture_base64'].items()}
    oldcapture.update(global_base64=base64.b64encode(a.raw(g)).decode(),handoff=handoff.closed_endpoint(global_bytes=a.raw(g),arm='D25',**terminal))
    oldsource=json.loads((root/'D25_RETRY02_CORE_SOURCE_PREPARED.private.json').read_bytes());seal=deepcopy(oldsource['seal']);seal['source_sha']=captured['source_sha']
    remote=deepcopy(owner['config']['remote_binding']);remote['remote_seal_sha256']=a.sha(a.raw(seal))
    grant=base64.b64decode(owner['desktop_grant_base64'])
    args=dict(global_bytes=a.raw(g),local_bytes=a.raw(l),training_capture_bytes=a.raw(oldcapture),arm='D25',
        source_sha=captured['source_sha'],source_seal_bytes=a.raw(seal),owner_root=owner_root,
        owner_manifest_bytes=a.raw(m),canonical_LOCAL_ledger=owner['config']['canonical_LOCAL_ledger'],
        remote_binding=remote,run_name=d.RUN,request_id=captured['new_request_id'],nonce=captured['new_nonce'],
        GPU_uuid=old['assembly']['request']['GPU_uuid'],now_epoch=captured['now_epoch'],desktop_grant_bytes=grant,global_accounting_mode=MODE)
    packet=assembly.assemble(template,**args)
    with pytest.raises(PermissionError):assembly.assemble(template,**dict(args,global_accounting_mode=None))
    assert packet['request']['global_accounting_mode']==MODE and packet['record']['D25_local_successor_sha256']==d.digest(marker)
    assert all(v['manifest']['execution']['global_accounting_mode']==MODE for v in packet['rows'].values())
    assert all(v['manifest']['execution']['sealed_inputs'][binding]==d.digest(marker)for v in packet['rows'].values())
    registered=c.register(g,l,packet['record'],base64.b64decode(packet['rows']['core']['manifest_bytes_base64']),terminal_capture=terminal,now_epoch=captured['now_epoch'])
    result=registered['proposed_global'];local=registered['proposed_LOCAL']
    assert len(result[c.FIELD])==14 and result[c.FIELD][d.FIRST]==g[c.FIELD][d.FIRST]and result[c.FIELD][d.FAILED]==g[c.FIELD][d.FAILED]
    assert local['V5_CD_historical_closed_Core_CPU_substages'][d.FAILED]==l['V5_CD_Core_CPU_substages'][c.key('D25')]
    assert result['storage_reserved_bytes']==g['storage_reserved_bytes']
    c.checked(result);c.paired(result,local)
    from scripts.pi0294.test_v5_core_stage_accounting import started
    from scripts.pi0294.test_mix_core_delegation import terminal_inputs
    from scripts.pi0294.pv1 import mix_core_settlement as settlement
    process=started(packet['record']);process.update(started_epoch=captured['now_epoch']+1,observed_epoch=captured['now_epoch']+2)
    bound=c.bind(result,local,request_id=captured['new_request_id'],process=process,now_epoch=captured['now_epoch']+2)
    row=bound['proposed_global'][c.FIELD][captured['new_request_id']]
    args=terminal_inputs(dict(state_bytes=a.raw(dict(row=row))))
    args['native_proof'].update(native_exit_status=1,finished_epoch=captured['now_epoch']+200,observed_epoch=captured['now_epoch']+201)
    args['effects']['evaluation_candidate']='D25'
    args['delivery_receipt_bytes']=a.raw(dict(json.loads(args['delivery_receipt_bytes']),status='ACTUAL_CLOSED_D25_1024_INFERENCE_VIEW_DELIVERED_NO_MODEL_LOADED'))
    args['now_epoch']=captured['now_epoch']+202
    final=settlement.propose(a.raw(bound['proposed_global']),a.raw(bound['proposed_LOCAL']),request_id=captured['new_request_id'],**args)
    assert final['proposed_global'][c.FIELD][d.FAILED]==g[c.FIELD][d.FAILED]
    assert final['receipt']['Core_output_charged_upper_bytes']==final['receipt']['transfer_charged_upper_bytes']==0
    assert final['receipt']['same_panel_GPU_output_upper_already_paid']==c.remaining_stage()['Hopper_storage_bytes']
    c.checked(final['proposed_global']);c.paired(final['proposed_global'],final['proposed_LOCAL'])

@pytest.mark.parametrize('change',['model_inode','model_size','extra_payload','oldreceipt','valid_condition','openlease','stale','control_zero','wire_zero','wrongsource'])
def test_changed_custody_or_understated_new_cost_fails_closed(captured,change):
    p=deepcopy(captured)
    if change in('model_inode','model_size','extra_payload'):
        value=p['inventory_capture']['core']['value'];raw=gzip.decompress(base64.b64decode(value['inventory_gzip_base64']));rows=json.loads(raw)
        model=value['roots']['delivery']['path']+'/model_view/model-00001-of-00003.safetensors'
        if change=='model_inode':rows[model]['ino']+=1
        elif change=='model_size':rows[model]['bytes']+=1
        else:rows[value['roots']['neural02']['path']+'/copied.safetensors']=deepcopy(rows[model])
        raw=(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n').encode();value['inventory_sha256']=a.sha(raw);value['inventory_gzip_base64']=base64.b64encode(gzip.compress(raw)).decode()
    elif change=='oldreceipt':
        g=json.loads(p['global_bytes']);g[c.FIELD][d.FAILED]['conservative_GPU_s']+=1;p['global_bytes']=a.raw(g)
    elif change=='valid_condition':p['closed_readback']['valid_conditions']=1
    elif change=='openlease':p['closed_readback']['leases']['GPU']['value']['status']='ACTIVE'
    elif change=='stale':p['now_epoch']+=121
    elif change=='control_zero':p['control_plan']['costs']['Core_CPU_s']=0
    elif change=='wire_zero':p['control_plan']['wire_planned_bytes']['protected_closeout']=0
    else:p['source_sha']='e'*40
    with pytest.raises((PermissionError,KeyError,ValueError)):d.propose(**p)
