"""V7 cold continuation keeps history and one live physical owner."""
from copy import deepcopy
import pytest
from scripts.pi0294.generalization import minimal_core_accounting as c,minimal_core_sharing as sharing
from scripts.pi0294.test_minimal_core_runtime import fixture,DEVICE,GPU,hardware


def v7_fixture():
    m,r,*_=fixture()
    r.update(run_uuid='arx5-e1-fixed512-n294101-04',native_unit='arx5-e1-fixed512-n294101-04-neural.service',
        execution_policy_sha256=c.V7_POLICY,shared_GPU_grant_sha256=c.V7_POLICY,
        GPU_sharing_direct_human_answer='GPU0允许共享任务挂载')
    m['run_uuid']=r['run_uuid'];m['execution']['sealed_inputs'][c.V7_PLAN]=c.V7_POLICY
    r['run_manifest_sha256']=c.digest(m)
    return m,r


def test_v7_sealed_direct_shared_permission_preserves_real_device_gate():
    from scripts.pi0294.generalization.minimal_core_identity import request_identity
    m,r=v7_fixture();request_identity(m,r,c.digest(m))
    foreign=GPU+', 4101246, /foreign/python, 5770\n'
    d=hardware.parse_GPU_rows(DEVICE,foreign,GPU_uuid=GPU,shared_request=r)
    assert d['human_authorized_shared_GPU_grant_sha256']==c.V7_POLICY
    with pytest.raises(PermissionError):hardware.parse_GPU_rows(DEVICE.replace('32760','16376'),foreign,GPU_uuid=GPU,shared_request=r)
    m['execution']['sealed_inputs'].pop(c.V7_PLAN);r['run_manifest_sha256']=c.digest(m)
    with pytest.raises(PermissionError):request_identity(m,r,c.digest(m))


@pytest.mark.parametrize('key,value',[
    ('execution_policy_sha256','0'*64),('shared_GPU_grant_sha256',sharing.GRANT_SHA),
    ('GPU_sharing_direct_human_answer','UNAPPROVED'),('GPU_uuid','GPU-845348f1-1511-dc19-0877-cc97658e3edd'),
    ('optimizer_updates',1),('checkpoint_step',1024),('checkpoint_manifest_sha256','0'*64)])
def test_v7_successor_cannot_change_authority_device_or_science(key,value):
    _,r=v7_fixture();r[key]=value
    with pytest.raises(PermissionError):c.request(r)


def test_legacy_multiple_live_rows_rejected_before_historical_markers():
    _,_,_,_,row,_=fixture();a=deepcopy(row);b=deepcopy(row)
    with pytest.raises(PermissionError,match='V7_SUCCESSORS_PRESERVE_TERMINAL_LEGACY_RUN'):
        c.validate({c.FIELD:{'a':a,'b':b}})
