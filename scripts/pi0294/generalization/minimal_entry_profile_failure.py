"""Exact1503589 dispatcher rejection before the minimal runtime was called.

Derive zero effects from pinned executed control flow, causal traceback, real
PBS/wrapper/owner terminal and closed scientific namespaces. Do not create a
missing process final/effect ledger or turn absence alone into zero effects.
"""
import ast,base64,gzip,json
from . import minimal_execution as e,minimal_retry_execution as retry,minimal_terminal as terminal

JOB='1503589.stdct-mgmt-02'
REQUEST='619032fd-30a8-41ea-88de-95510a4fade9'
SOURCE='9f7966187a2e1594f0fab25484da653a4a7e76db'
ERROR='VAP1_A40_CONTEXT_EXACT_MINIMAL_VERSION_AND_TYPED_PROFILE'
CONTEXT_SHA='42f5ec3c869e907687c76839d714052c050f99e25f95b5067998ac745a0477fd'
RUNTIME_SHA='d3ead751a91646776ed4eeb3c7a859a4cf6d68834c09a1a313b73d6a2cd03630'
STATUS='PASS_EXACT1503589_ENTRY_DISPATCH_CAUSAL_ZERO_EFFECT_AUDIT'


def audit(capture):
    def blob(name):
        row=capture['files'][name];body=base64.b64decode(row['base64'],validate=True)
        e.need(len(body)==row['bytes']and e.sha(body)==row['sha256'],'ENTRY_FAIL_CAPTURE_FULL_BYTES:'+name)
        return body
    def read(name):return json.loads(blob(name))
    gb=gzip.decompress(base64.b64decode(capture['GLOBAL_gzip_base64'],validate=True));e.need(e.sha(gb)==capture['GLOBAL_sha256'],'ENTRY_FAIL_CANONICAL_GLOBAL_BYTES')
    g=json.loads(gb);mark,row=retry._selected(g);retry.validate(g)
    request=read('PBS/REQUEST.json');manifest=read('RUN_MANIFEST.json');seal=read('SOURCE_SEAL.json')
    e.need(row['request_id']==request['request_id']==REQUEST and row['job_id']==JOB
        and row['source_sha']==manifest['code']['vla_commit']==seal['source_sha']==SOURCE
        and row['status']==e.STATES[1]and row['request_sha256']==e.sha(blob('PBS/REQUEST.json'))
        and row['run_manifest_sha256']==e.sha(blob('RUN_MANIFEST.json'))
        and manifest['code']['source_seal_sha256']==e.sha(blob('SOURCE_SEAL.json')),
        'ENTRY_FAIL_EXACT_REAL_REGISTERED_REQUEST_SOURCE_MANIFEST')
    modules={}
    for name,value in capture['modules'].items():
        b=base64.b64decode(value['base64'],validate=True)
        e.need(len(b)==value['bytes']and e.sha(b)==value['sha256']==seal['files'][name],
               'ENTRY_FAIL_REAL_EXECUTED_SEALED_SOURCE:'+name);modules[name]=b
    context=modules['scripts/pi0294/dec1/a40_training_context.py'];runtime=modules['scripts/pi0294/dec1/a40_training_runtime.py']
    e.need(e.sha(context)==CONTEXT_SHA and e.sha(runtime)==RUNTIME_SHA,'ENTRY_FAIL_EXACT_CAUSAL_IMPLEMENTATIONS')
    tree=ast.parse(runtime);run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    e.need(ast.unparse(run.body[0])=='(root, request_path) = (Path(root), Path(request_path))','ENTRY_FAIL_ONLY_PATH_CONVERSION_BEFORE_HEADER')
    top=run.body[1];e.need(isinstance(top,ast.Assign)and ast.unparse(top.value)=="read(root / 'RUN_MANIFEST.json')",'ENTRY_FAIL_FIRST_READ_ONLY_HEADER')
    branch=run.body[2];calls=[n for n in ast.walk(branch.test)if isinstance(n,ast.Call)];e.need(isinstance(branch,ast.If)and len(calls)>=1
        and ast.unparse(calls[0].func)=='context.minimal_training_adapter'
        and any(isinstance(n,ast.Return)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='minimal_run'for n in branch.body),
        'ENTRY_FAIL_DISPATCH_BEFORE_MINIMAL_RUNTIME_AND_EFFECTS')
    f=next(n for n in ast.parse(context).body if isinstance(n,ast.FunctionDef)and n.name=='minimal_training_adapter')
    literal="if profile != adapters[version].PROFILE:\n            fail('EXACT_MINIMAL_VERSION_AND_TYPED_PROFILE')"
    e.need(literal in ast.unparse(f)and request['profile']!=retry.PROFILE,
           'ENTRY_FAIL_POSITIVE_LITERAL_PROFILE_REJECTION_BEFORE_DELEGATION')
    log=blob('PBS/pbs.log').decode();e.need(log.rstrip().endswith('PermissionError: '+ERROR)
        and 'minimal_training_adapter'in log and 'if profile!=adapters[version].PROFILE'in log
        and 'minimal_native_runtime.py'not in log,'ENTRY_FAIL_REAL_TRACEBACK_EXACT_UNENTERED_RUNTIME')
    native=capture['native'];wrapper=read('PBS/PBS_WRAPPER_FINAL.json');owner=read('control/RESIDENT_OWNER_OS_EXIT.json')
    e.need(native['job_state']=='F'and native['Exit_status']==wrapper['returncode']==owner['returncode']==1
        and native['run_count']==1 and native['Variable_List']['PV1_REQUEST_ID']==REQUEST
        and native['Variable_List']['PV1_SUBMISSION_NONCE']==request['submission_nonce']
        and native['Job_Name']==request['job_name']and native['project']==request['project']
        and int(native['Resource_List']['ngpus'])==1 and native['Job_Owner'].split('@')[0]=='e1192548'
        and wrapper['job_id']==JOB and wrapper['source_sha']==owner['source_sha']==SOURCE,
        'ENTRY_FAIL_REAL_NATIVE_ONE_GPU_F_AND_BOTH_OS1')
    e.need(len(capture['actors'])==2 and all(x['actual_procstart']!=x['expected_procstart']for x in capture['actors'])
        and all(read('control/resident-owner/'+n+'/RPC_EFFECT_STATE.json')['status']=='CONFIRMED'for n in('setup-rpc','running-rpc')),
        'ENTRY_FAIL_REAL_ACTORS_TERMINAL_AND_NO_UNKNOWN_RUNTIME_WRITES')
    e.need(set(capture['absent'])=={'A40_NATIVE_EFFECT_LEDGER.json'}
        and {'PBS_PROCESS_FINAL.json','ALLOCATION_ADMISSION.json','PBS_READY.json','START_NEURAL.json'}<=set(capture['absent_PBS'])
        and set(capture['science_inventory'])=={'training','inference','evidence','reservations','worker_processes','aux_worker_processes'}
        and all(v==[]for v in capture['science_inventory'].values()),'ENTRY_FAIL_CAUSAL_BOUNDARY_AND_FULL_EMPTY_SCIENCE_NAMESPACES')
    inventory=capture['inventory'];e.need(inventory['worker_inventory_sha256']==e.sha(e.raw(inventory['files'])),'ENTRY_FAIL_STABLE_COMPLETE_WORKER_INVENTORY')
    from scripts.pi0294.e0_scope import pbs_epoch
    hms=lambda text:sum(int(v)*n for v,n in zip(text.split(':'),(3600,60,1)))
    end=pbs_epoch(native['obittime']);start=pbs_epoch(native['stime']);now=capture['observed_epoch']
    # There is no process-final file: bind its digest field explicitly to the
    # independent causal missing-final witness, never fabricate that old file.
    missing=dict(schema=STATUS,original_process_final_created=False,original_effect_ledger_created=False,
        source_context_sha256=CONTEXT_SHA,source_runtime_sha256=RUNTIME_SHA,pbs_traceback_sha256=e.sha(blob('PBS/pbs.log')),
        actual_wrapper_final_sha256=e.sha(blob('PBS/PBS_WRAPPER_FINAL.json')),missing=capture['absent_PBS'])
    effects=dict(schema='ARX5_MINIMAL_ACTUAL_FIVE_SESSION_EFFECT_SUMMARY_V1',job_id=JOB,request_id=REQUEST,
        request_sha256=row['request_sha256'],source_sha=SOURCE,optimizer_updates_committed=0,query_attempts=0,
        model_load_attempts=0,pending_updates=[],unknown_query_intents=[],active_session=None,CPU_s=0.,
        all_five_sessions_runtime_complete=False,independently_audited=True,global_budget_is_only_external_reservation=True,
        new_physics=0,new_episodes=0,finished_epoch=end,derived_from_exact_entry_failure=True,
        original_native_effect_ledger_created=False,sessions={s['id']:dict(status='NOT_STARTED',optimizer_updates=0,generation_queries=0,model_loads=0)for s in e.native_sessions()})
    audit=dict(status=terminal.PARTIAL,request_sha256=row['request_sha256'],source_sha=SOURCE,job_id=JOB,
        effect_summary_sha256=e.sha(e.raw(effects)),inventory_sha256=inventory['worker_inventory_sha256'],
        raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,
        new_model_loads=0,new_queries=0,new_updates=0,new_physics=0,causal_entry_failure=missing,original_missing_final_preserved=True)
    cpu=hms(native['resources_used']['cput'])+240
    proof=dict(schema=terminal.NATIVE,job_id=JOB,job_state='F',request_id=REQUEST,submission_nonce=request['submission_nonce'],
        source_sha=SOURCE,request_sha256=row['request_sha256'],Exit_status=1,finished_epoch=end,observed_epoch=now,
        conservative_GPU_s=max(hms(native['resources_used']['walltime']),end-start)+request['accounting_pad_s'],
        new_native_conservative_CPU_s=cpu,prior_readonly_preparation_CPU_upper_s=0,conservative_CPU_s=cpu,
        all_native_workers_terminal=True,terminal_inventory_stable=True,raw_native_sha256=e.sha(e.raw(native)),
        process_final_sha256=e.sha(e.raw(missing)),wrapper_final_sha256=e.sha(blob('PBS/PBS_WRAPPER_FINAL.json')),
        inventory_sha256=inventory['worker_inventory_sha256'],process_final_digest_is_independent_missing_final_witness=True)
    terminal.terminal_values(row,proof,effects,audit,retry_global=g)
    return dict(status=STATUS,native=proof,effects=effects,audit=audit,original_missing_final_witness=missing)
