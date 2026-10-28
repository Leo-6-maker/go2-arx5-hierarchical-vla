import json
from pathlib import Path
import pytest
from scripts.pi0294.generalization import minimal_execution as e,minimal_retry_physical_guard as physical
from scripts.pi0294.generalization import minimal_retry_runtime_guard as runtime
from scripts.pi0294.native_dataset import sha


@pytest.fixture
def retry(tmp_path):
    roots=[];rows=[]
    for number in ('01','02'):
        root=tmp_path/('arx5-minimal-raw-parent-first512-'+number);root.mkdir();lease=tmp_path/('lease-'+number+'.json')
        lease.write_text(json.dumps({'status':'CLOSED'}));(root/'RUN_MANIFEST.json').write_text(json.dumps({'run_uuid':root.name}))
        (root/'OUTPUT_BINDING.private.json').write_text(json.dumps({'run_root':str(root),'lease':str(lease)}))
        roots.append(root);rows.append(dict(run_uuid=root.name,manifest_sha256=sha(root/'RUN_MANIFEST.json'),closed_lease_sha256=sha(lease)))
    root=tmp_path/'arx5-minimal-raw-parent-first512-retry-01';root.mkdir();(root/'plans').mkdir()
    before=physical.check(roots,root,old_metadata_ceiling=e.CONTROL['Hopper_storage_bytes'],paid_output_ceiling=e.native_resources()['Hopper_storage_bytes'])
    request=dict(request_id='one',submission_nonce='nonce',run_uuid=root.name)
    proof=dict(schema=runtime.SCHEMA,request_id='one',submission_nonce='nonce',run_uuid=root.name,
        old_roots=rows,old_inventory_sha256=before['old_inventory_sha256'],old_metadata_ceiling=e.CONTROL['Hopper_storage_bytes'],
        paid_output_ceiling=e.native_resources()['Hopper_storage_bytes'],engineering_metadata_allowance=64*1024**2)
    path=root/runtime.FILE;path.write_text(json.dumps(proof));path.chmod(0o400)
    manifest=dict(execution=dict(sealed_inputs={runtime.FILE:sha(path)}))
    calls=[]
    def original(*,startup=False):calls.append(startup);return {'original_forecast':True}
    return root,roots,request,manifest,calls,original


def test_retry_wraps_original_guard_and_checks_reserve(retry):
    root,old,req,m,calls,original=retry;guard=runtime.Guard(root,req,m,original)
    assert guard(startup=True)=={'original_forecast':True}
    assert guard.reserve(28000000000)['next_output_upper_bytes']==28000000000+runtime.CONTROL_GROWTH_RESERVE
    assert calls==[True,False]


def test_old_inode_changes_and_lease_changes_fail(retry):
    root,old,req,m,calls,original=retry;guard=runtime.Guard(root,req,m,original);guard()
    (old[0]/'new-file').write_text('changed')
    with pytest.raises(PermissionError):guard.reserve(1)
    (old[0]/'new-file').unlink();binding=json.loads((old[1]/'OUTPUT_BINDING.private.json').read_text());Path(binding['lease']).write_text('{"status":"ACTIVE"}')
    with pytest.raises(PermissionError):runtime.Guard(root,req,m,original)


def test_changed_identity_or_unsealed_custody_fails(retry):
    root,old,req,m,calls,original=retry
    with pytest.raises(PermissionError):runtime.Guard(root,dict(req,request_id='other'),m,original)
    (root/runtime.FILE).chmod(0o600)
    with pytest.raises(PermissionError):runtime.Guard(root,req,m,original)
