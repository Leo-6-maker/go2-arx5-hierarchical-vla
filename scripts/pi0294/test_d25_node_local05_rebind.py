"""Saved real closed04 bytes; new source/peer identities are CPU fixtures."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_core_d25_cold_source as n,mix_core_accounting as c
from scripts.pi0294.pv1 import mix_core_d25_local_successor as paid,mix_accounting as a


@pytest.fixture(scope='module')
def prepared():
    directory=os.environ.get('D25_NODE_LOCAL05_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('actual closed04 private captures required')
    root=Path(directory);read=lambda name:json.loads((root/name).read_bytes())
    gb=gzip.decompress(base64.b64decode(read('E2_PRE_NATIVE30_FINAL_BIND_PREPARED.private.json')['after_GLOBAL_gzip_base64']))
    lb=gzip.decompress(base64.b64decode(json.loads((root.parent/'wholebody-sequence-20261029/FRESH_CORE_SOURCE_SAVED_REPLY_QUALIFIED.private.json').read_bytes())['result']['LOCAL']['gzip_base64']))
    d=root/'MAINTENANCE30_CONTROL_IO'
    closed=json.loads((d/'D25_POST_CLEANUP05_SAVED_BYTES_SMALL_HASH_QUALIFICATION.RESULT.json').read_bytes())
    from scripts.pi0294.generalization import minimal_terminal_inventory as inventory
    fresh=json.loads((d/'POST_CLEANUP30_VANDA_GLOBAL_HASH_AND_OLD_E2_CLOSURE.RESULT.json').read_bytes())
    # Future simultaneous hash reads and the resulting LOCAL CAS are fixtures;
    # the saved GLOBAL/LOCAL rows and closure records themselves are real.
    fixture_now=max(fresh['observed_epoch'],closed['observed_epoch'])+1
    fresh=dict(fresh,observed_epoch=fixture_now,CPU_fixture_only=True)
    closed=deepcopy(closed);closed.update(observed_epoch=fixture_now,CPU_fixture_only=True)
    catchup=inventory.propose_latest(gb,lb,fresh_GLOBAL=fresh,closed_D25_readback=closed,now_epoch=fixture_now)
    lb=a.raw(catchup['proposed_LOCAL']);closed['LOCAL_sha256']=a.sha(lb)
    source=read('D25_COLD_LOCAL_CONTEXT04_SOURCE_PREPARED.private.json')
    seal=deepcopy(source['seal']);seal['source_sha']='c'*40
    now=closed['observed_epoch']+1
    args=dict(global_bytes=gb,local_bytes=lb,authority_bytes=(d/'DIRECT_AUTHORITY.json').read_bytes(),
        closed_readback=closed,source_seal_bytes=a.raw(seal),frozen=source['frozen'],
        source_readback=dict(status='ACTUAL_COMPLETE_COLD_SOURCE_SHA_READBACK',source_sha=seal['source_sha'],
            source_seal_sha256=a.sha(a.raw(seal)),file_sha256=seal['files'],hostname='corelab-4080s',uid=2009,
            model_calls=0,physics_ticks=0,observed_epoch=now,CPU_fixture_only=True),now_epoch=now)
    plan=n.prepare_node_local05(**args)
    return root,args,plan


def test_actual_closed04_remains_full_history_and_private_assembly_view_is_not_adoptable(prepared):
    _,args,p=prepared;g=json.loads(args['global_bytes']);l=json.loads(args['local_bytes'])
    m=p['assembly_global'][n.LOCAL05_FIELD]
    assert m['preserved_prestart_row']==g[c.FIELD][n.REQUEST]==l[c.FIELD][n.REQUEST]
    assert m['preserved_prestart_row']['status']==c.STATES[0]and m['preserved_prestart_row']['native_handle']is None
    assert not p['intermediate_view_may_be_persisted']and not p['native_admission']
    assert p['assembly_global'][n.FIELD]==g[n.FIELD]and p['assembly_global'][paid.FIELD]==g[paid.FIELD]
    assert p['assembly_global']['reserved']==g['reserved']and p['assembly_global']['CPU_s']==g['CPU_s']
    assert n.effective_identity(p['assembly_global'],g[paid.FIELD])==dict(source_sha='c'*40,run_uuid=n.LOCAL05_RUN)


@pytest.mark.parametrize('change',['active','native','queried','closed-proof','cost','source','authority'])
def test_exact_known_unstarted_row_and_three_real_closures_cannot_be_bypassed(prepared,change):
    _,_,p=prepared;g=deepcopy(p['assembly_global']);m=g[n.LOCAL05_FIELD]
    if change=='active':m['preserved_prestart_row']['status']=c.STATES[1]
    elif change=='native':m['preserved_prestart_row']['native_handle']='an-actual-native-handle'
    elif change=='queried':m['preserved_prestart_row']['query_attempts']=1
    elif change=='closed-proof':m['closed04_roles']['neural']['record']['live_actors']=[123]
    elif change=='cost':m['preserved_prestart_row']['reserved_GPU_s']-=1
    elif change=='source':m['source_sha']=m['old_source_sha']
    else:m['maintenance_authority_sha256']='0'*64
    with pytest.raises((PermissionError,KeyError)):n.validated(g,g[paid.FIELD])


def test_paired_local_requires_the_same_new_identity_marker(prepared):
    _,_,p=prepared;g=deepcopy(p['assembly_global']);l=deepcopy(p['assembly_LOCAL'])
    l.pop(n.LOCAL05_FIELD)
    with pytest.raises(PermissionError):paid.paired_control(g,l)


def test_real_assembler_then_finalizer_reuses_one_panel_without_any_new_financial_hold(prepared):
    from scripts.pi0294.pv1 import mix_core_assembly as assembly,mix_core_context as context
    from scripts.pi0294.pv1 import mix_core_delegation as delegation
    root,args,p=prepared;g,l=p['assembly_global'],p['assembly_LOCAL'];m=g[n.LOCAL05_FIELD]
    previous=json.loads((root/'D25_FINITE_LOCAL04_DISPATCH_PREPARED.private.json').read_bytes())
    owner=json.loads((root/'D25_FINITE_LOCAL04_OWNER_PREPARED.private.json').read_bytes())
    om=deepcopy(owner['manifest']);owner_root=str(Path(owner['root']).with_name('v5-cd-d25-eval-owner-05'))
    om.update(run_uuid=Path(owner_root).name,output_root=owner_root)
    om['code'].update(vla_commit=m['source_sha'],source_seal_sha256=m['source_seal_sha256'])
    om['execution']['sealed_inputs'][n.BINDING]=n.digest(m)
    cap=json.loads(base64.b64decode(previous['assembly']['rows']['hopper']['files'][context.CAPTURE]))
    # Keep the exact original closed-training capture. Rebuilding its GLOBAL
    # snapshot would change provenance without changing the scientific input.
    assert a.sha(a.raw(cap['handoff']))==m['preserved_prestart_row']['training_handoff_sha256']
    template=json.loads((root.parent/'science-next/wdr1-target-identity/v4a-core-eval-preparation-01/CORE_PAIRED_PEER_TEMPLATES_7cc75180.private.json').read_bytes())
    remote=deepcopy(owner['config']['remote_binding']);remote['remote_seal_sha256']=m['source_seal_sha256']
    packet=assembly.assemble(template,global_bytes=a.raw(g),local_bytes=a.raw(l),training_capture_bytes=a.raw(cap),
        arm='D25',source_sha=m['source_sha'],source_seal_bytes=args['source_seal_bytes'],owner_root=owner_root,
        owner_manifest_bytes=a.raw(om),canonical_LOCAL_ledger=owner['config']['canonical_LOCAL_ledger'],
        remote_binding=remote,run_name=n.LOCAL05_RUN,request_id=n.REQUEST,nonce=m['submission_nonce'],
        GPU_uuid=m['preserved_prestart_row']['GPU_uuid'],now_epoch=args['now_epoch'],
        desktop_grant_bytes=base64.b64decode(owner['desktop_grant_base64']),global_accounting_mode=delegation.MODE)
    final=n.finalize_node_local05(p,packet);v,w=final['proposed_global'],final['proposed_LOCAL']
    assert v[c.FIELD][n.REQUEST]['child_run_uuid']==n.LOCAL05_RUN
    assert w['V5_CD_Core_CPU_substages'][c.key('D25')]['CPU_s']==93900
    assert w['V5_CD_Core_CPU_substages'][c.key('D25')]['run_uuid']==n.LOCAL05_RUN
    assert v['reserved']==p['before_global']['reserved']and v['CPU_s']==p['before_global']['CPU_s']
    assert v['PBS_submissions']==p['before_global']['PBS_submissions']and v['used']==p['before_global']['used']
    assert final['new_native_requests']==0 and not final['closed04_roles_reopened']and not final['native_admission']
    bad=deepcopy(packet);bad['record']['reserved_neural_queries']-=1
    with pytest.raises(PermissionError):n.finalize_node_local05(p,bad)
