"""CPU fixtures only, including actual local kernel child affinity."""
from copy import deepcopy
import json,os,subprocess,sys
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_vanda_affinity as f


def fixture(granted):
    granted=list(granted)
    o=dict(job_id='123.fixture',request_id='request',source_sha='a'*40,manifest_sha256='b'*64,
           hostname='host-fixture',boot_id='boot-fixture',cgroup='group-fixture')
    b=dict(schema=f.SCHEMA,policy=f.POLICY,active_cpu_limit=8,granted_cpu_ids=sorted(granted),roles=f.mapping(granted),
           owner_manifest_sha256=o['manifest_sha256'],**{k:v for k,v in o.items()if k!='manifest_sha256'})
    return o,b


@pytest.mark.parametrize('granted',[set(range(8)),set(range(36,72)),{2,5,11,14,35,70,83,99},set(range(1,33))])
def test_nonzero_sparse_or_larger_grant(granted):
    o,b=fixture(granted);assert f.checked(b,o)==b
    roles=[set(f.cpus(b,n))for n in('neural','physical','core_rpc','neural_rpc')]
    assert list(map(len,roles))==[4,2,1,1]
    assert sum(map(len,roles))==len(set.union(*roles))==8
    assert set.union(*roles)<=granted
    assert f.cpus(b,'owner')==f.cpus(b,'core_rpc')+f.cpus(b,'neural_rpc')


@pytest.mark.parametrize('granted',[set(range(7)),[0,1,2,3,4,5,6,6],[True,1,2,3,4,5,6,7],[-1,0,1,2,3,4,5,6],['0',1,2,3,4,5,6,7]])
def test_invalid_grant(granted):
    with pytest.raises(PermissionError):f.mapping(granted)


@pytest.mark.parametrize('key',['job_id','request_id','source_sha','hostname','boot_id','cgroup','owner_manifest_sha256'])
def test_wrong_native_binding(key):
    o,b=fixture(range(24,32));b[key]='wrong'
    with pytest.raises(PermissionError):f.checked(b,o)


def test_substitution_or_overlap():
    o,b=fixture(range(24,40));b['roles']['physical'][0]=39
    with pytest.raises(PermissionError):f.checked(b,o)
    b['roles']=f.mapping(b['granted_cpu_ids']);b['roles']['physical'][0]=24
    with pytest.raises(PermissionError):f.checked(b,o)


def test_legacy_literal_ids_unchanged():
    assert f.physical_cpus('/unused',dict(limits=dict(cpus=[4,5])))==[4,5]


def test_physical_binding_preserves_manifest(monkeypatch):
    from scripts.pi0294.pv1 import mix_vanda_eval_context as context,mix_vanda_eval_identity as identity
    from scripts.pi0292 import durable
    from scripts.pi0294 import native_dataset
    o,b=fixture(range(36,72));o['CPU_slot_binding']=b
    m=dict(scope=identity.PHYSICAL_ROLE,execution=dict(CPU_slot_policy=f.POLICY),limits=dict(cpus=[4,5],active_cores=2));before=deepcopy(m)
    monkeypatch.setattr(context,'paths',lambda *args:dict(hopper=dict(root='/fixture/neural')))
    monkeypatch.setattr(context,'request_path',lambda *args:Path('/fixture/request'))
    monkeypatch.setattr(context,'owner_current',lambda *args,**kwargs:o)
    monkeypatch.setattr(durable,'read',lambda *args:{})
    monkeypatch.setattr(native_dataset,'sha',lambda *args:'0'*64)
    monkeypatch.setattr(os,'sched_getaffinity',lambda pid:{40,41})
    assert f.physical_cpus('/fixture',m)==[40,41]and m==before
    monkeypatch.setattr(os,'sched_getaffinity',lambda pid:{40,41,42})
    with pytest.raises(PermissionError):f.physical_cpus('/fixture',m)


def test_real_kernel_child_all_roles():
    available=sorted(os.sched_getaffinity(0))
    if len(available)<8:pytest.skip('CPU fixture requires eight real local CPUs')
    ids=available[-8:]
    code='''import json,os
from scripts.pi0294.pv1 import mix_vanda_affinity as f
os.sched_setaffinity(0,set(IDS));os.environ['PBS_JOBID']='cpu-fixture-only'
b=f.capture(job_id='cpu-fixture-only',request_id='fixture',source_sha='a'*40,owner_manifest_sha256='b'*64)
seen={}
for role in ('owner','neural','physical','core_rpc','neural_rpc'):
 os.sched_setaffinity(0,set(f.cpus(b,role)));seen[role]=sorted(os.sched_getaffinity(0))
print(json.dumps(dict(binding=b,seen=seen)))
'''
    p=subprocess.run([sys.executable,'-c','IDS='+repr(ids)+'\n'+code],check=True,capture_output=True,text=True,timeout=10)
    v=json.loads(p.stdout);assert v['binding']['granted_cpu_ids']==ids and v['seen']==v['binding']['roles']


def test_bool_cannot_masquerade_as_cpu_zero():
    o,b=fixture(list(range(8)));b['roles']['neural'][0]=False
    with pytest.raises(PermissionError):f.checked(b,o)
    with pytest.raises(PermissionError):f.cpus(b,'neural')
