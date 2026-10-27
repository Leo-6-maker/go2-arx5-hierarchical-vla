"""Exact independent audit of CPU1497552's pre-processor missing-git failure.

Original failed counters and their conservative UNKNOWN field are evidence,
not rewritten. No generic exception is turned into zero effects here.
"""
import ast,base64,hashlib,json,math
from . import amd_export_accounting as e

SOURCE='8cc8541f219a7988ddbfb88240355487f920bebe'
JOB='1497552.stdct-mgmt-02'
REQUEST='2f925f70-0ea1-46c6-be06-8807a88789ce'
ERROR="FileNotFoundError: [Errno 2] No such file or directory: 'git'"

def sha(b):return hashlib.sha256(b).hexdigest()
def audit(capture):
    def blob(name):
        row=capture['files'][name];data=base64.b64decode(row['base64'],validate=True)
        e.a.need(len(data)==row['bytes']and sha(data)==row['sha256'],'GIT_FAIL_CAPTURE_BYTES:'+name)
        return data
    def read(name):return json.loads(blob(name))
    gb=base64.b64decode(capture['GLOBAL_base64'],validate=True)
    e.a.need(sha(gb)==capture['GLOBAL_sha256'],'GIT_FAIL_ACTUAL_GLOBAL_CAPTURE')
    g=json.loads(gb);items=e.rows(g);e.a.need(len(items)==1,'GIT_FAIL_ONE_ORIGINAL_CPU_ROLE')
    row=items[0];request=read('REQUEST.json');manifest=read('RUN_MANIFEST.json');seal=read('SOURCE_SEAL.json')
    e.a.need(row['request_id']==request['request_id']==REQUEST and row['job_id']==JOB
        and row['source_sha']==manifest['code']['vla_commit']==seal['source_sha']==SOURCE
        and row['request_sha256']==sha(blob('REQUEST.json'))
        and row['run_manifest_sha256']==sha(blob('RUN_MANIFEST.json'))
        and manifest['code']['source_seal_sha256']==sha(blob('SOURCE_SEAL.json')),
        'GIT_FAIL_EXACT_ACTUAL_REQUEST_SOURCE_MANIFEST_AND_SEAL')
    modules={}
    for name,record in capture['modules'].items():
        data=base64.b64decode(record['base64'],validate=True)
        e.a.need(sha(data)==record['sha256']==seal['files'][name]
            and len(data)==record['bytes'],'GIT_FAIL_ORIGINAL_EXECUTED_SOURCE:'+name)
        modules[name]=ast.parse(data)
    adapter=modules['scripts/pi0292/train_adapter.py'];build=next(x for x in adapter.body if isinstance(x,ast.FunctionDef)and x.name=='build_processor')
    e.a.need(isinstance(build.body[0],ast.Expr)and isinstance(build.body[0].value,ast.Call)
        and isinstance(build.body[0].value.func,ast.Name)and build.body[0].value.func.id=='official_import',
        'GIT_FAIL_OFFICIAL_GIT_CHECK_PRECEDES_PROCESSOR_CONSTRUCTOR')
    imp=next(x for x in adapter.body if isinstance(x,ast.FunctionDef)and x.name=='official_import')
    checks=[x for x in ast.walk(imp)if isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)
        and isinstance(x.func.value,ast.Name)and x.func.value.id=='subprocess'
        and x.func.attr=='check_output'and x.args and isinstance(x.args[0],ast.List)]
    e.a.need(any(len(x.args[0].elts)>=4 and isinstance(x.args[0].elts[0],ast.Constant)
        and x.args[0].elts[0].value=='git'for x in checks),'GIT_FAIL_REAL_GIT_SUBPROCESS_NOT_A_MODEL_ERROR')
    native=capture['native'][JOB];variables=native['Variable_List']
    e.a.need(set(capture['native'])=={JOB}and native['job_state']=='F'
        and type(native['Exit_status'])is int and native['Exit_status']==1
        and int(native['run_count'])==1 and native['Job_Name']==e.JOB_NAME
        and native['project']=='CFP04-CF-007'and native['Job_Owner'].split('@')[0]=='e1192548'
        and int(native['Resource_List']['ncpus'])==1 and int(native['Resource_List']['ngpus'])==0
        and variables['GEN1_REQUEST_ID']==REQUEST and variables['GEN1_SUBMISSION_NONCE']==row['submission_nonce'],
        'GIT_FAIL_REAL_EXACT_CPU_F_NOT_TIMEOUT_OR_OTHER_JOB')
    actor=read('control/CPU_COMPUTE_PROCESS.json');owner=read('control/cpu-owner-queue-repair-03/CPU_OWNER_OS_EXIT.json')
    e.a.need(actor['job_id']==JOB and actor['request_id']==REQUEST and actor['source']==SOURCE
        and actor['host'].lower()in {x.split('/')[0].lower()for x in native['exec_host'].split('+')}
        and owner['returncode']==1 and owner['site_control_CPU_s']>0
        and len(capture['actors'])==2 and all(x['actual_procstart']!=x['expected_procstart']for x in capture['actors']),
        'GIT_FAIL_NATIVE_COMPUTE_F_AND_REAL_FINITE_OWNER_OS1')
    counter=read('RESOURCE_LEDGER.json');failure=read('evidence/GEN1_EXPORT_FAILURE.private.json')
    e.a.need(counter['status']=='FAILED_PRESERVED'and counter['error']==failure['error']==ERROR
        and counter['unknown_effects']==[dict(error=ERROR,remaining_output_bound_retained=True)]
        and failure['completed']==0 and capture['datasets_members']==[]
        and not capture['export_result_exists']and not capture['frozen_raw_bindings_exist']
        and not capture['prior_settlement_intent_exists'],
        'GIT_FAIL_EXACT_PRE_FIRST_PARENT_FAILURE_WITH_ALL_ORIGINAL_UNKNOWN_EVIDENCE_RETAINED')
    for key in('model_loads','queries','updates','constructors','physics_ticks','episode_starts'):
        e.a.need(type(counter[key])is int and counter[key]==0,'GIT_FAIL_REAL_ZERO:'+key)
    log=blob('pbs/alloc-01/pbs.log').decode()
    e.a.need(log.rstrip().endswith(ERROR)and 'processor=build_processor('in log
        and "subprocess.check_output(['git'"in log,'GIT_FAIL_SAVED_NATIVE_CAUSAL_TRACEBACK')
    cpu=counter['CPU_s'];e.a.need(type(cpu)in(int,float)and math.isfinite(cpu)and 0<=cpu<=36000,'GIT_FAIL_RECORDED_PAYLOAD_CPU')
    h,m,s=map(int,native['resources_used']['cput'].split(':'))
    cpu=max(h*3600+m*60+s+1,math.ceil(cpu)+10)
    terminal=dict(job_state='F',job_id=JOB,request_id=REQUEST,submission_nonce=row['submission_nonce'],
        Exit_status=1,conservative_CPU_s=cpu,run_manifest_sha256=row['run_manifest_sha256'],native_PBS=native)
    report=dict(status='PASS_INDEPENDENT_GEN1_CPU_EXPORT_EFFECTS',request_sha256=row['request_sha256'],
        all_processes_terminal=True,unknown_effects=[],export_finished=False,training_bank_qualified=False,
        model_loads=0,queries=0,updates=0,constructors=0,physics_ticks=0,episode_starts=0,
        original_failed_counter_and_UNKNOWN_unchanged=True,causal_boundary='EXACT_GIT_EXEC_FAILED_BEFORE_PROCESSOR_CONSTRUCTOR_AND_FIRST_EXPORT',
        actual_CPU_counter_s=counter['CPU_s'],conservative_CPU_s=cpu,
        actual_source_seal_sha256=sha(blob('SOURCE_SEAL.json')),resource_ledger_sha256=sha(blob('RESOURCE_LEDGER.json')),
        failure_file_sha256=sha(blob('evidence/GEN1_EXPORT_FAILURE.private.json')),
        native_log_sha256=sha(blob('pbs/alloc-01/pbs.log')),actual_compute_identity=actor)
    return dict(audit=report,native=terminal)
