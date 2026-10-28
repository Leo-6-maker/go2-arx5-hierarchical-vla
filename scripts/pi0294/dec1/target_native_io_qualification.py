"""Explicit human-directed server qualification for the existing zero-IO scope.

Account startup failure stays a failure. This changes a software qualification
provider only: resource admission, immutable sources and native safety remain.
Pure stdlib validation runs before every native import.
"""
import hashlib
import json

GRANT_SHA = 'e20493de14b6e8dcab16eedde2e283a8130de2ad1ca80cb28b637001a416964c'
METHOD = 'DIRECT_HUMAN_STANDARD_ENTRY_SERVER_CPU_V1'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def require(value,reason):
    if not value: raise ValueError(reason)

def verify(proof,seal):
    require(proof['source']==seal['source_sha'],'qualification source')
    if proof.get('CI_conclusion')=='success' and proof.get('all_12_steps_success') is True:
        return 'EXACT_ACTIONS_CI'
    alt=proof.get('execution_qualification',{})
    require(alt.get('method')==METHOD and alt.get('grant_sha256')==GRANT_SHA,'direct human provider change')
    require(proof.get('CI_conclusion')=='NOT_REQUIRED_USER_SERVER_VALIDATION' and proof.get('all_12_steps_success') is False
            and proof.get('CI_steps')==[] and proof.get('CI_runner_assigned') is False,
            'do not fabricate a current CI success')
    history=alt['historical_Actions']
    require(history==dict(source='5671da2b48a930d9a670091a5bd9847516ccf493',run_id=40052632537,
        job_id=120748430019,conclusion='failure',runner='',steps=[]),'preserve actual Actions startup failure')
    files=seal['files'];require(files and all(proof['full_readback_file_sha256'].get(n)==h for n,h in files.items()),'complete exact runtime readback')
    tree=digest(files)
    local=alt['local_CPU'];require(local['exit_code']==0 and local['failed']==0
        and type(local['passed']) is int and local['passed']>=1826
        and local['runtime_tree_sha256']==tree and len(local['output_sha256'])==64,'full local CPU regression')
    server=alt['server_CPU'];require(server['status']=='ACTUAL_CORE_STANDARD_ENTRY_SOURCE_QUALIFICATION_CPU_PASS'
        and server['source']==proof['source'] and server['runtime_tree_sha256']==tree
        and server['source_files']==len(files) and server['qualification_cases_passed']>=10
        and server['native_modules_imported'] is False and server['new_models_physics_registry_or_ledger_writes']==0
        and server['standard_entry_sha256']=='0c7973e0a3bc0e585e7ab96cfcc9033532a98f56e9c9410882282a291bb7082f',
        'actual standard-entry server CPU proof')
    require(alt['server_control_and_ledger_unchanged'] is True and alt['native_task_qualified'] is False,
            'CPU is not native qualification')
    return METHOD
