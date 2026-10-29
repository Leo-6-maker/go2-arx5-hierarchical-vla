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
    def forbidden_registry_read(path):
        raise AssertionError('A sealed new role must not consult NFS at runtime')
    result=c.registration(root,m,binding,legacy_read=forbidden_registry_read)
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


def test_explicit_new_role_checks_original_assets_and_entry_copies_without_legacy_reads(tmp_path):
    root=tmp_path/'new';root.mkdir();pins={};aliases={}
    for i in range(26):
        p=tmp_path/('asset-'+str(i));b=str(i).encode();p.write_bytes(b)
        old='/unavailable/original/asset-'+str(i);pins[old]=c.sha(b)
        aliases[old]=dict(local_path=str(p),sha256=c.sha(b))
    original=raw(dict(absolute_files=pins));relocation=raw(dict(schema='CORE_FIXED_ASSET_BYTE_RELOCATION_V1',
        original_identity_sha256=c.sha(original),aliases=aliases))
    m=dict(execution=dict(sealed_inputs={c.DESCRIPTOR:'a'*64},node_local_entry_relative='entry/infrastructure'))
    def seal(name,body):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);once._exclusive(p,body)
        m['execution']['sealed_inputs'][name]=c.sha(body)
    seal('plans/ASSET_IDENTITY.private.json',original);seal(c.ASSET_RELOCATION,relocation)
    for name,body in [('enter.sh',b'original entry'),('doctor.py',b'original doctor'),('ENVIRONMENTS.json',b'original env')]:
        seal('entry/infrastructure/'+name,body)
        if name!='ENVIRONMENTS.json':m['execution']['entry_sha256'if name=='enter.sh'else'doctor_sha256']=c.sha(body)
    c.native_assets(root,m,expected_original_sha256=c.sha(original))
    assert c.entry_directory(root,m,dict(registry='/unavailable/registry'))==root/'entry/infrastructure'
    (tmp_path/'asset-0').write_bytes(b'changed')
    with pytest.raises(PermissionError,match='LOCAL_NATIVE_ASSET_BYTES'):
        c.native_assets(root,m,expected_original_sha256=c.sha(original))
    m['execution']['doctor_sha256']='0'*64
    with pytest.raises(PermissionError,match='UNCHANGED_STANDARD_ENTRY_AND_DOCTOR'):
        c.entry_directory(root,m,dict(registry='/unavailable/registry'))


def test_legacy_registration_stays_live_and_orphan_alias_or_entry_is_rejected(tmp_path):
    root,registry,d,before,after,intent,receipt,binding,m=fixture(tmp_path)
    calls=[]
    def read(path):calls.append(path);return json.loads(Path(path).read_bytes())
    assert c.registration(root,m,binding,legacy_read=read)['live_canonical_registry_read']is True
    assert calls==[str(registry)]
    assert c.entry_directory(root,m,binding)==registry.parent
    m['execution']['node_local_entry_relative']='entry'
    with pytest.raises(PermissionError,match='NO_ORPHAN_LOCAL_ENTRY'):c.entry_directory(root,m,binding)


def test_node_local_gate_rejects_nfs_before_native_consumption(tmp_path,monkeypatch):
    original=Path.read_text
    def mountinfo(path,*args,**kw):
        if str(path)=='/proc/self/mountinfo':return '1 0 0:1 / / rw - ext4 /dev/local rw\n2 1 0:2 / '+str(tmp_path)+' rw - nfs4 server:/home rw\n'
        return original(path,*args,**kw)
    monkeypatch.setattr(Path,'read_text',mountinfo)
    with pytest.raises(PermissionError,match='EXISTING_NODE_LOCAL_FILESYSTEM_REQUIRED'):
        c.local_path(tmp_path/'file')
