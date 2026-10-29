"""Actual original onceIO metadata, not a Core/runtime/model admission test."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from scripts.pi0294.dec1 import source_cpu_once_io as once
from scripts.pi0294.pv1 import node_local_custody as c

raw=lambda x:(json.dumps(x,indent=2)+'\n').encode()


def fixture(tmp_path):
    root=tmp_path/'run-05';(root/'control').mkdir(parents=True)
    registry=tmp_path/'PATH_REGISTRY.json'
    before=raw(dict(version='ORIGINAL',entries=[dict(role='OLD',path='/old',retained='negative')]))
    registry.write_bytes(before);role='D25_CORE_05';source='a'*40
    row=dict(role=role,path=str(root),run_uuid=root.name,executor_node='corelab-4080s',owner='EXISTING_LOCAL_PC',
        source_sha=source,local_lease=str(root/'control/LEASE.json'))
    after=json.loads(before);after['entries'].append(row);after=raw(after)
    key='D25-COLD05-ORIGINAL-REGISTRATION'
    ip=registry.with_name(key+'.INTENT.json');rp=registry.with_name(key+'.RECEIPT.json')
    once.commit(registry,expected_before_sha256=c.sha(before),proposed_bytes=after,request_id=key,
        intent_path=ip,receipt_path=rp,lock_path=registry.with_name('PATH_REGISTRY.owner.lock'))
    intent,receipt=ip.read_bytes(),rp.read_bytes()
    binding=dict(run_root=str(root),run_uuid=root.name,role=role,registry=str(registry),lease=row['local_lease'])
    manifest=dict(run_uuid=root.name,code=dict(vla_commit=source),execution=dict(private_path_role=role,sealed_inputs={}))
    d=dict(schema=c.SCHEMA,run_root=str(root),run_uuid=root.name,role=role,source_sha=source,
        original_registry=str(registry),snapshot_is_new_authority=False,native_admission=False,
        registry_before_sha256=c.sha(before),registry_after_sha256=c.sha(after),registration_intent_sha256=c.sha(intent),
        registration_receipt_sha256=c.sha(receipt),registration_request_id=key,appended_roles=[role])
    return root,registry,d,before,after,intent,receipt,binding,manifest


def test_real_once_receipt_and_sealed_local_snapshot_survive_canonical_registry_unavailability(tmp_path):
    root,registry,d,before,after,intent,receipt,binding,m=fixture(tmp_path)
    captures=dict(before_capture_relative=before,after_capture_relative=after,intent_capture_relative=intent,receipt_capture_relative=receipt)
    for key,b in captures.items():
        name='control/'+key+'.json';d[key]=name;once._exclusive(root/name,b)
        m['execution']['sealed_inputs'][name]=c.sha(b)
    path=root/c.DESCRIPTOR;path.parent.mkdir(parents=True);once._exclusive(path,raw(d))
    m['execution']['sealed_inputs'][c.DESCRIPTOR]=c.sha(raw(d))
    registry.rename(registry.with_suffix('.preserved'))
    result=c.registered_role(root,m,binding)
    assert result['live_canonical_registry_read']is False and result['native_admission']is False
    assert result['local_lease']==str(root/'control/LEASE.json')


def test_forged_receipt_rewritten_history_and_historical_lease_are_rejected(tmp_path):
    root,registry,d,before,after,intent,receipt,binding,m=fixture(tmp_path)
    bad=json.loads(receipt);bad['intent_sha256']='0'*64
    d2=dict(d,registration_receipt_sha256=c.sha(raw(bad)))
    with pytest.raises(PermissionError):c.registration_bytes(raw(d2),before,after,intent,raw(bad),binding=binding,manifest=m)
    d2=dict(d,snapshot_is_new_authority=True)
    with pytest.raises(PermissionError):c.registration_bytes(raw(d2),before,after,intent,receipt,binding=binding,manifest=m)
    b2=dict(binding,lease='/historical/nfs/lease.json')
    with pytest.raises(PermissionError):c.registration_bytes(raw(d),before,after,intent,receipt,binding=b2,manifest=m)


def test_exact_original_asset_identity_all_twenty_six_and_no_asset_byte_change():
    original=raw(dict(absolute_files={'/old/assets/'+str(i):format(i,'064x')for i in range(26)}))
    aliases={p:dict(local_path=p.replace('/old/','/node-local/'),sha256=pin)for p,pin in json.loads(original)['absolute_files'].items()}
    relocation=dict(schema='CORE_FIXED_ASSET_BYTE_RELOCATION_V1',original_identity_sha256=c.sha(original),aliases=aliases)
    assert len(c.asset_relocations(original,raw(relocation),expected_original_sha256=c.sha(original)))==26
    bad=deepcopy(relocation);bad['aliases'].pop('/old/assets/0')
    with pytest.raises(PermissionError):c.asset_relocations(original,raw(bad),expected_original_sha256=c.sha(original))
    bad=deepcopy(relocation);bad['aliases']['/old/assets/0']['sha256']='f'*64
    with pytest.raises(PermissionError):c.asset_relocations(original,raw(bad),expected_original_sha256=c.sha(original))
