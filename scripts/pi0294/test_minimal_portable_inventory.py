"""Real temporary files; altered host device IDs are explicit CPU fixtures."""
from copy import deepcopy
import os
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_portable_inventory as p

@pytest.fixture
def roots(tmp_path):
    old=[]
    for i in range(3):
        root=tmp_path/('closed'+str(i));root.mkdir();(root/'metadata.json').write_text('{"closed":true}');old.append(str(root))
    return old

def test_actual_full_contents_and_same_host_cache(roots):
    cache={};first=p.capture(roots,metadata_ceiling=10**6,file_hash_cache=cache)
    later=p.capture(roots,metadata_ceiling=10**6,file_hash_cache=cache)
    assert first['content_bytes_read']>0 and later['content_bytes_read']==0
    assert p.verify(first,later)['native_permission']is False
    assert all('dev'not in row for row in first['value']['rows'])

def test_distinct_client_device_namespace_needs_identical_backing_and_full_identity(roots):
    before=p.capture(roots,metadata_ceiling=10**6);other=deepcopy(before);other['kernel_device_ids']=[999999];other['hostname']='CPU_FIXTURE_OTHER_CLIENT'
    assert p.verify(before,other)['kernel_device_namespace_changed']
    other['value']['roots']['closed0']['filesystem']['filesystem_source']='OTHER_EXPORT';other['portable_sha256']=p.digest(other['value'])
    with pytest.raises(PermissionError):p.verify(before,other)

@pytest.mark.parametrize('kind',['content','inode','extra_file','mode','link','weights','ceiling'])
def test_changes_are_rejected(roots,kind,tmp_path):
    before=p.capture(roots,metadata_ceiling=10**6);root=Path(roots[0]);f=root/'metadata.json'
    if kind=='content':f.write_text('{"closed":false}')
    elif kind=='inode':
        other=tmp_path/'replacement';other.write_bytes(f.read_bytes());os.replace(other,f)
    elif kind=='extra_file':(root/'extra').write_bytes(b'new')
    elif kind=='mode':f.chmod(0o400)
    elif kind=='link':os.link(f,root/'alias')
    elif kind=='weights':(root/'not_metadata.safetensors').write_bytes(b'x')
    else:
        with pytest.raises(PermissionError):p.capture(roots,metadata_ceiling=1)
        return
    with pytest.raises(PermissionError):p.verify(before,p.capture(roots,metadata_ceiling=10**6))

def test_original_stat_only_contract_is_unchanged(roots):
    from scripts.pi0294.generalization import minimal_retry_physical_guard as original
    assert 'dev'in next(iter(original._scan(roots[0],permitted_links=set())[0].values()))


def test_portable_runtime_guard_keeps_all_three_closed_and_prewrite_bounds(tmp_path,monkeypatch):
    import json
    from scripts.pi0294.generalization import minimal_retry_runtime_guard as runtime,minimal_followup_scope as scope
    from scripts.pi0294.native_dataset import sha
    names=['arx5-minimal-raw-parent-first512-01','arx5-minimal-raw-parent-first512-02','arx5-minimal-raw-parent-first512-retry-01'];roots=[];old=[]
    for name in names:
        root=tmp_path/name;root.mkdir();lease=tmp_path/(name+'.lease.json');lease.write_text('{"status":"CLOSED"}')
        (root/'RUN_MANIFEST.json').write_text('{"test":"CPU_fixture"}');(root/'OUTPUT_BINDING.private.json').write_text(json.dumps(dict(run_root=str(root),lease=str(lease))))
        roots.append(str(root));old.append(dict(run_uuid=name,manifest_sha256=sha(root/'RUN_MANIFEST.json'),closed_lease_sha256=sha(lease)))
    root=tmp_path/scope.RUN;(root/'plans').mkdir(parents=True);capture=p.capture(roots,metadata_ceiling=192*1024**2)
    request=dict(request_id='CPU-fixture-new',submission_nonce='CPU-fixture-nonce',run_uuid=scope.RUN,continuation_generation=2)
    custody=dict(schema=runtime.PORTABLE_SCHEMA,request_id=request['request_id'],submission_nonce=request['submission_nonce'],run_uuid=scope.RUN,scope_sha256=scope.GRANT_SHA,old_metadata_ceiling=192*1024**2,paid_output_ceiling=runtime.e.native_resources()['Hopper_storage_bytes'],engineering_metadata_allowance=64*1024**2,old_roots=old,portable_inventory=capture,old_inventory_sha256=capture['portable_sha256'])
    path=root/runtime.FILE;path.write_text(json.dumps(custody));path.chmod(0o400);manifest=dict(execution=dict(sealed_inputs={runtime.FILE:sha(path)}))
    guard=runtime.Guard(root,request,manifest,lambda **_:dict(original_current_resource_guard=True))
    assert guard.reserve(1024)['old_inventory_sha256']==capture['portable_sha256']
    assert guard.fresh_inventory()['status']=='ACTUAL_THREE_CLOSED_PORTABLE_METADATA_READONLY'
    with pytest.raises(PermissionError):guard.reserve(custody['paid_output_ceiling'])
    (Path(roots[0])/'unexpected').write_text('changed')
    with pytest.raises(PermissionError):guard.reserve(1)


def test_block_allocation_changes_keep_exact_identity_and_actual_cap(roots):
    before=p.capture(roots,metadata_ceiling=10**6);current=deepcopy(before)
    # Captured native symptom was +8192 blocks-bytes only, same full content.
    current['value']['rows'][0]['allocated']+=8192
    current['value']['allocated_bytes']+=8192
    current['portable_sha256']=p.digest(current['value'])
    proof=p.verify(before,current)
    assert proof['current_allocated_bytes']==proof['historical_allocated_bytes']+8192
    assert proof['portable_identity_sha256']==p.identity_digest(before['value'])
    assert current['portable_sha256']!=before['portable_sha256']
    for key in ('inode','logical','mtime_ns','ctime_ns','links','uid','mode'):
        changed=deepcopy(current);changed['value']['rows'][0][key]+=1
        changed['portable_sha256']=p.digest(changed['value'])
        with pytest.raises(PermissionError):p.verify(before,changed)
    broken=deepcopy(current);broken['value']['allocated_bytes']-=1
    broken['portable_sha256']=p.digest(broken['value'])
    with pytest.raises(PermissionError):p.verify(before,broken)
    excess=deepcopy(current);excess['value']['rows'][0]['allocated']+=10**6
    excess['value']['allocated_bytes']+=10**6;excess['portable_sha256']=p.digest(excess['value'])
    with pytest.raises(PermissionError):p.verify(before,excess)
