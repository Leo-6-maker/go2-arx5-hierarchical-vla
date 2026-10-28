"""Actual closed ledgers; future cold source/role receipts are CPU fixtures.

No site access, registry change, native role close, ledger write or GPU occurs.
"""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_core_d25_cold_source as n,mix_core_d25_local_successor as old
from scripts.pi0294.pv1 import mix_core_accounting as c,mix_accounting as a,mix_core_assembly as assembly


@pytest.fixture(scope='module')
def captured():
    directory=os.environ.get('ARX5_PARTIAL_COMPLETION_CAPTURE_DIR')
    if not directory:pytest.skip('explicit actual control-adopted GLOBAL/canonical terminal LOCAL required')
    root=Path(directory);read=lambda name:json.loads((root/(name+'.private.json')).read_bytes())
    g=read('PARTIAL_COMPLETION01_PREPARED');l=read('CORE_TERMINAL_INVENTORY01_PREPARED_REFRESHED2')
    gb=gzip.decompress(base64.b64decode(g['GLOBAL_gzip_base64']))
    lb=gzip.decompress(base64.b64decode(l['LOCAL_gzip_base64']))
    previous=read('D25_FINITE_LOCAL03_PEERS_PREPARED')
    template=json.loads((root.parent/'science-next/wdr1-target-identity/v4a-core-eval-preparation-01/CORE_PAIRED_PEER_TEMPLATES_7cc75180.private.json').read_bytes())
    frozen=template['rows']['A_CONTROL']['core']['frozen']
    seal=deepcopy(read('D25_FINITE_LOCAL03_SOURCE_PREPARED')['seal']);seal['source_sha']='b'*40
    now=l['fresh_epoch']+1
    # Future role receipts and source host reading are deliberately fixtures.
    roles={role:dict(status='ACTUAL_UNUSED_COLD_ROLE_CLOSED_WITH_NO_ACTOR',source_sha=n.OLD_SOURCE,
        request_id=n.REQUEST,root='/CPU_FIXTURE/'+role,lease='/CPU_FIXTURE/'+role+'.lease',
        manifest_sha256='1'*64,closed_lease_sha256='2'*64,native_intent_sha256='3'*64,
        native_receipt_sha256='4'*64,inventory_sha256='5'*64,
        native_start_records=0,live_actors=0,model_calls=0,physics_ticks=0)for role in('owner','physical','neural')}
    args=dict(global_bytes=gb,local_bytes=lb,source_seal_bytes=a.raw(seal),frozen=frozen,
        source_readback=dict(status='ACTUAL_COMPLETE_COLD_SOURCE_SHA_READBACK',source_sha=seal['source_sha'],
            source_seal_sha256=a.sha(a.raw(seal)),file_sha256=seal['files'],hostname='corelab-4080s',uid=2009,
            observed_epoch=now,model_calls=0,physics_ticks=0,CPU_fixture_only=True),
        closed_cold_roles=roles,fresh_readback=dict(observed_epoch=now,GLOBAL_sha256=a.sha(gb),
            LOCAL_sha256=a.sha(lb),new_mutations=0,CPU_fixture_only=True),now_epoch=now)
    return root,args,previous,template


def test_actual_history_all14_and_old_financial_marker_unchanged(captured):
    _,args,_,_=captured;before=deepcopy(args);p=n.propose(**args)
    g,l=json.loads(args['global_bytes']),json.loads(args['local_bytes'])
    v,w=p['proposed_global'],p['proposed_LOCAL']
    assert {k:x for k,x in v.items()if k not in(n.FIELD,'revision')}=={k:x for k,x in g.items()if k!='revision'}
    assert {k:x for k,x in w.items()if k!=n.FIELD}==l
    assert v[old.FIELD]==g[old.FIELD]and old.finite_control_scope(v,n.REQUEST)==old.finite_control_scope(g,n.REQUEST)
    assert p['full_future_proof']['cap_changes_required']=={} and len(p['full_future_proof']['dimensions'])==14
    assert p['native_registration']is p['native_admission']is False and args==before
    with pytest.raises(PermissionError):n.propose(**dict(args,global_bytes=a.raw(v),local_bytes=a.raw(w)))


def assembled(captured):
    from scripts.pi0294.pv1 import mix_stage_handoff as handoff,mix_core_context as context,mix_core_delegation as delegation
    _,args,previous,template=captured;p=n.propose(**args);g,l=p['proposed_global'],p['proposed_LOCAL']
    owner=deepcopy(previous['owner']);m=owner['manifest'];root=str(Path(owner['root']).with_name('v5-cd-d25-eval-owner-04'))
    m.update(run_uuid=Path(root).name,output_root=root);m['code'].update(vla_commit='b'*40,source_seal_sha256=a.sha(args['source_seal_bytes']))
    m['authority'].pop('minimal_first512_cross_scope',None)
    from scripts.pi0294.generalization.minimal_concurrency import KEY
    m['authority'].pop(KEY,None)
    m['execution']['sealed_inputs'][n.BINDING]=n.digest(g[n.FIELD])
    m.update(n.descriptor(g,g[old.FIELD]))
    cap=json.loads(base64.b64decode(previous['assembly']['rows']['hopper']['files'][context.CAPTURE]))
    terminal={k:base64.b64decode(v)for k,v in cap['terminal_capture_base64'].items()}
    cap.update(global_base64=base64.b64encode(a.raw(g)).decode(),handoff=handoff.closed_endpoint(global_bytes=a.raw(g),arm='D25',**terminal))
    remote=deepcopy(owner['config']['remote_binding']);remote['remote_seal_sha256']=a.sha(args['source_seal_bytes'])
    values=dict(global_bytes=a.raw(g),local_bytes=a.raw(l),training_capture_bytes=a.raw(cap),arm='D25',source_sha='b'*40,
        source_seal_bytes=args['source_seal_bytes'],owner_root=root,owner_manifest_bytes=a.raw(m),
        canonical_LOCAL_ledger=owner['config']['canonical_LOCAL_ledger'],remote_binding=remote,
        run_name=n.RUN,request_id=n.REQUEST,nonce=g[old.FIELD]['replacement_submission_nonce'],
        GPU_uuid=previous['assembly']['request']['GPU_uuid'],now_epoch=args['now_epoch'],
        desktop_grant_bytes=base64.b64decode(owner['desktop_grant_base64']),global_accounting_mode=delegation.MODE)
    return p,assembly.assemble(template,**values),terminal,values,template


def test_new_source_original_request_distinct_roots_real_assembler_and_registration(captured):
    p,packet,terminal,values,template=assembled(captured);g,l=p['proposed_global'],p['proposed_LOCAL']
    m=g[n.FIELD];assert packet['request']['source_sha']==m['source_sha']and packet['request']['run_uuid']==n.RUN
    assert packet['request']['request_id']==n.REQUEST and packet['request'][n.DESCRIPTOR]==n.digest(m)
    for row in packet['rows'].values():
        assert Path(row['root']).name==n.RUN and row['manifest']['execution']['sealed_inputs'][n.BINDING]==n.digest(m)
        assert row['manifest']['code']['source_seal_sha256']==m['source_seal_sha256']
        assert base64.b64decode(row['files'][n.BINDING])==a.raw(m)
    registered=c.register(g,l,packet['record'],base64.b64decode(packet['rows']['core']['manifest_bytes_base64']),terminal_capture=terminal,now_epoch=captured[1]['now_epoch'])
    v,w=registered['proposed_global'],registered['proposed_LOCAL'];c.checked(v);c.paired(v,w)
    assert v[old.FIELD]==g[old.FIELD]and v['used']==g['used']and v['PBS_submissions']==g['PBS_submissions']
    assert v[c.FIELD][n.REQUEST]['child_run_uuid']==n.RUN and v[c.FIELD][old.FAILED]==g[c.FIELD][old.FAILED]
    with pytest.raises(PermissionError):
        assembly.assemble(template,**dict(values,run_name=n.OLD_RUN))
    from scripts.pi0294.test_v5_core_stage_accounting import started
    from scripts.pi0294.test_mix_core_delegation import terminal_inputs
    from scripts.pi0294.pv1 import mix_core_settlement as settlement
    now=captured[1]['now_epoch'];process=started(packet['record'])
    process.update(started_epoch=now+1,observed_epoch=now+2)
    bound=c.bind(v,w,request_id=n.REQUEST,process=process,now_epoch=now+2)
    row=bound['proposed_global'][c.FIELD][n.REQUEST]
    args=terminal_inputs(dict(state_bytes=a.raw(dict(row=row))))
    args['native_proof'].update(native_exit_status=1,finished_epoch=now+200,observed_epoch=now+201)
    args['effects']['evaluation_candidate']='D25'
    args['delivery_receipt_bytes']=a.raw(dict(json.loads(args['delivery_receipt_bytes']),status='ACTUAL_CLOSED_D25_1024_INFERENCE_VIEW_DELIVERED_NO_MODEL_LOADED'))
    args['now_epoch']=now+202
    final=settlement.propose(a.raw(bound['proposed_global']),a.raw(bound['proposed_LOCAL']),request_id=n.REQUEST,**args)
    assert final['proposed_global'][old.FIELD]==g[old.FIELD]and final['proposed_global'][n.FIELD]==m
    assert final['receipt']['Core_output_charged_upper_bytes']==final['receipt']['transfer_charged_upper_bytes']==0
    c.checked(final['proposed_global']);c.paired(final['proposed_global'],final['proposed_LOCAL'])


def test_new_runtime_sealed_source_binding_and_no_orphan(captured,tmp_path):
    p=n.propose(**captured[1]);m=p['proposed_global'][n.FIELD];original=p['proposed_global'][old.FIELD]
    path=tmp_path/n.BINDING;path.parent.mkdir(parents=True);path.write_bytes(a.raw(m));path.chmod(0o400)
    manifest=dict(code=dict(vla_commit=m['source_sha'],source_seal_sha256=m['source_seal_sha256']),
        run_uuid=n.RUN,execution=dict(sealed_inputs={n.BINDING:n.digest(m)}),**{n.DESCRIPTOR:n.digest(m)})
    request=dict(source_sha=m['source_sha'],run_uuid=n.RUN,**{n.DESCRIPTOR:n.digest(m)})
    read=lambda path:json.loads(path.read_bytes());sha=lambda path:a.sha(path.read_bytes())
    n.sealed_context(tmp_path,manifest,request,p['proposed_LOCAL'],original,read,sha)
    with pytest.raises(PermissionError):
        n.sealed_context(tmp_path,manifest,dict(request,source_sha=n.OLD_SOURCE),p['proposed_LOCAL'],original,read,sha)
    with pytest.raises(PermissionError):
        n.sealed_context(tmp_path,manifest,request,json.loads(captured[1]['local_bytes']),original,read,sha)


from scripts.pi0294.test_d25_output_guard import guarded


def test_real_temp_closed_cold_payload_cannot_change(guarded,tmp_path):
    from scripts.pi0294.pv1 import mix_core_d25_output_guard as guard_module
    guard,_,_=guarded;guard.cold=dict(closed_cold_roles={})
    for role in('owner','physical','neural'):
        root=tmp_path/('unused-'+role);root.mkdir();(root/'original.json').write_text('{}')
        rows,_=guard_module.inventory(root,uid=os.getuid())
        body=(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n').encode()
        guard.cold['closed_cold_roles'][role]=dict(root=str(root),inventory_sha256=a.sha(body))
    assert guard.check(force=True)['new_tensor_copies']==0
    (Path(guard.cold['closed_cold_roles']['neural']['root'])/'original.json').write_text('{"changed":true}')
    with pytest.raises(PermissionError):guard.check(force=True)


@pytest.mark.parametrize('change',['stale','wrong_LOCAL','changed_old_invoice','source_masquerade','missing_source_member','numerical_change','missing_role','live_actor','model_effect','no_closed_receipt'])
def test_changes_and_missing_real_boundary_proof_fail_closed(captured,change):
    args=deepcopy(captured[1])
    if change=='stale':args['now_epoch']+=121
    elif change=='wrong_LOCAL':args['fresh_readback']['LOCAL_sha256']='0'*64
    elif change=='changed_old_invoice':
        g=json.loads(args['global_bytes']);g['CPU_s']+=1;args['global_bytes']=a.raw(g)
    elif change=='source_masquerade':
        s=json.loads(args['source_seal_bytes']);s['source_sha']=n.OLD_SOURCE;args['source_seal_bytes']=a.raw(s)
    elif change=='missing_source_member':args['source_readback']['file_sha256'].pop(next(iter(args['source_readback']['file_sha256'])))
    elif change=='numerical_change':args['frozen'][next(iter(args['frozen']))]='0'*64
    elif change=='missing_role':args['closed_cold_roles'].pop('owner')
    elif change=='live_actor':args['closed_cold_roles']['neural']['live_actors']=1
    elif change=='model_effect':args['closed_cold_roles']['neural']['model_calls']=1
    else:args['closed_cold_roles']['physical']['native_receipt_sha256']=''
    with pytest.raises((PermissionError,KeyError,ValueError)):n.propose(**args)
