"""Pure accounting and native-allocation parser fixtures, never native PASS."""
from copy import deepcopy
import re

import pytest
from scripts.pi0294.test_v5_vanda_cohost import captured
from scripts.pi0294.pv1 import mix_vanda_cohost as c, mix_vanda_render as r, mix_accounting as a
from scripts.pi0294.pv1.neural_ledger import check
from scripts.pi0294.dec1.vanda_campaign import classify


def request(g):
    m = c.validated(g)
    return dict(platform_qualification_version=r.VERSION,native_platform='VANDA_A40',project='CFP04-CF-007',
        job_name=r.JOB_NAME,physical_GPU_count=0,scope_grant_sha256=m['scope_grant_sha256'],
        authority_release='5b5de6f5d26ac90a6002c83ebcb60005acf06462',T_CD=m['T_CD'],deadline_epoch=m['deadline_epoch'],
        CPU_upper_s=960,reserved_H200_s=0,query_limit=0,limits=deepcopy(r.LIMITS),
        source_sha='a'*40,request_sha256='b'*64,run_manifest_sha256='c'*64,
        request_id='4689f590-5815-499d-9a8d-12d887a801ee',submission_nonce='de7ace01-15b8-4bf5-a31f-27db8ec82f21',
        job_id=None,status=r.STATES[0])


def test_original_global_validator_accepts_only_typed_cpu_job_and_once_terminal(captured):
    g = c.propose(**captured)['proposed_global']; old = deepcopy(g)
    row = request(g); v = r.register(g,row); check(v)
    assert g == old and v['used']['PBS_submissions'] == 29
    assert v['used']['H200_s'] == g['used']['H200_s'] and v['CPU_s'] == g['CPU_s']
    with pytest.raises(PermissionError): r.register(v,row)
    with pytest.raises(PermissionError): r.bind(v,row['request_id'],'not-a-PBS-id')
    v = r.bind(v,row['request_id'],'1234567.stdct-mgmt-02'); check(v)
    queue = dict(platform='VANDA_A40',job_id='1234567.stdct-mgmt-02',state='R',
                 project='CFP04-CF-007',name=r.JOB_NAME,ngpus=0)
    campaign,outside,unknown = classify(v,[queue])
    assert len(campaign) == 1 and outside == unknown == []
    with pytest.raises(PermissionError): classify(v,[dict(queue,ngpus=1)])
    native = dict(job_state='F',Exit_status=0,job_id=queue['job_id'],request_id=row['request_id'],submission_nonce=row['submission_nonce'])
    end = r.settle(v,row['request_id'],native=native,CPU_s=17,constructor_intent=True); check(end)
    assert end['CPU_s'] == g['CPU_s']+17
    assert end['CPU_reserved_s'] == g['CPU_reserved_s']-960
    assert end['used']['H200_s'] == g['used']['H200_s']
    assert end[a.PACKAGE]['remaining_resources']['engine_starts'] == g[a.PACKAGE]['remaining_resources']['engine_starts']-2
    assert g[a.PACKAGE]['completed_stages'] == end[a.PACKAGE]['completed_stages']
    with pytest.raises(PermissionError): r.settle(end,row['request_id'],native=native,CPU_s=17,constructor_intent=True)


@pytest.mark.parametrize('field,value',[('physical_GPU_count',1),('query_limit',1),('project','OTHER'),
    ('platform_qualification_version','OLD_CLOSED_MP1'),('job_name','old-camera'),('CPU_upper_s',961),('request_id','x')])
def test_zero_integration_finite_schema_cannot_be_extended(captured, field, value):
    g = c.propose(**captured)['proposed_global']; row = request(g); row[field] = value
    with pytest.raises(PermissionError): r.register(g,row)


def job():
    return dict(job_state='R',project='CFP04-CF-007',Job_Name=r.JOB_NAME,
                Job_Owner='e1192548@stdct-login-01',exec_host='cn-123/0+cn-123/1',
                Resource_List=dict(ncpus=2,ngpus=0,walltime='00:08:00'))


def test_exact_real_qstat_parser_fixture_accepts_only_allocated_cpu_node():
    assert r.native_allocation(job(),job_id='1234567.stdct-mgmt-02',hostname='cn-123')


@pytest.mark.parametrize('change', ['login','other_host','wrong_project','queued','gpu','cpu','longer','wrong_handle'])
def test_wrong_native_allocation_cannot_construct_engine(change):
    value = job(); host='cn-123'; jid='1234567.stdct-mgmt-02'
    if change == 'login': host='stdct-login-01'
    elif change == 'other_host': host='cn-124'
    elif change == 'wrong_project': value['project']='OTHER'
    elif change == 'queued': value['job_state']='Q'
    elif change == 'gpu': value['Resource_List']['ngpus']=1
    elif change == 'cpu': value['Resource_List']['ncpus']=4
    elif change == 'longer': value['Resource_List']['walltime']='01:00:00'
    else: jid='123.hopper-m-02'
    with pytest.raises(PermissionError): r.native_allocation(value,job_id=jid,hostname=host)


def test_no_native_library_imports_at_control_or_validation_import_time():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(r.__file__).read_text())
    top = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all(not re.search(r'\b(mujoco|torch|OpenGL|numpy)\b',ast.unparse(n)) for n in top)
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'run')
    statements = [ast.unparse(n) for n in fn.body]
    assert next(i for i,x in enumerate(statements) if 'validate(root, compute=True)' in x) < next(i for i,x in enumerate(statements) if x == 'import mujoco')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('step','mj_step') for n in ast.walk(fn))
