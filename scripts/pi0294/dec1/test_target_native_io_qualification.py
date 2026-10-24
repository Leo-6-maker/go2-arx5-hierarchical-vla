"""CPU-only qualification-provider tests; no native admission or transport."""
from copy import deepcopy
import pytest
from . import target_native_io_qualification as q

def packet():
    seal=dict(source_sha='a'*40,files={'scripts/current.py':'b'*64})
    tree=q.digest(seal['files'])
    proof=dict(source=seal['source_sha'],CI_conclusion='NOT_REQUIRED_USER_SERVER_VALIDATION',
        all_12_steps_success=False,CI_steps=[],CI_runner_assigned=False,full_readback_file_sha256=dict(seal['files']))
    proof['execution_qualification']=dict(method=q.METHOD,grant_sha256=q.GRANT_SHA,
        historical_Actions=dict(source='5671da2b48a930d9a670091a5bd9847516ccf493',run_id=40052632537,
            job_id=120748430019,conclusion='failure',runner='',steps=[]),
        local_CPU=dict(exit_code=0,failed=0,passed=1826,runtime_tree_sha256=tree,output_sha256='c'*64),
        server_CPU=dict(status='ACTUAL_CORE_STANDARD_ENTRY_SOURCE_QUALIFICATION_CPU_PASS',source=proof['source'],
            runtime_tree_sha256=tree,source_files=1,qualification_cases_passed=10,native_modules_imported=False,
            new_models_physics_registry_or_ledger_writes=0,
            standard_entry_sha256='0c7973e0a3bc0e585e7ab96cfcc9033532a98f56e9c9410882282a291bb7082f'),
        server_control_and_ledger_unchanged=True,native_task_qualified=False)
    return proof,seal

def test_explicit_server_provider_preserves_ci_history():
    proof,seal=packet();before=deepcopy(proof)
    assert q.verify(proof,seal)==q.METHOD and proof==before

def test_existing_exact_ci_route_remains():
    proof,seal=packet();proof.update(CI_conclusion='success',all_12_steps_success=True)
    assert q.verify(proof,seal)=='EXACT_ACTIONS_CI'

@pytest.mark.parametrize('fault',['grant','method','source','readback','tree','local_fail','local_incomplete',
    'server_source','server_write','server_native','server_entry','server_cases','changed_history','pretend_CI',
    'changed_control','task_qualified'])
def test_missing_or_changed_qualification_rejects(fault):
    proof,seal=packet();a=proof['execution_qualification']
    if fault=='grant':a['grant_sha256']='0'*64
    elif fault=='method':a['method']='BYPASS'
    elif fault=='source':proof['source']='0'*40
    elif fault=='readback':proof['full_readback_file_sha256']={}
    elif fault=='tree':a['local_CPU']['runtime_tree_sha256']='0'*64
    elif fault=='local_fail':a['local_CPU']['failed']=1
    elif fault=='local_incomplete':a['local_CPU']['passed']=1
    elif fault=='server_source':a['server_CPU']['source']='0'*40
    elif fault=='server_write':a['server_CPU']['new_models_physics_registry_or_ledger_writes']=1
    elif fault=='server_native':a['server_CPU']['native_modules_imported']=True
    elif fault=='server_entry':a['server_CPU']['standard_entry_sha256']='0'*64
    elif fault=='server_cases':a['server_CPU']['qualification_cases_passed']=0
    elif fault=='changed_history':a['historical_Actions']['conclusion']='success'
    elif fault=='pretend_CI':proof['CI_conclusion']='success'
    elif fault=='changed_control':a['server_control_and_ledger_unchanged']=False
    elif fault=='task_qualified':a['native_task_qualified']=True
    with pytest.raises((ValueError,KeyError)):q.verify(proof,seal)
