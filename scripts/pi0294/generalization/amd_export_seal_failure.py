"""Exact post-export/pre-bank file-seal failure; no original UNKNOWN rewrite."""
import ast,base64,json,math
from . import amd_export_repair_accounting as e,amd_export,amd_bank

SOURCE='3396c3adbfb0a26685ddb4f28cf7083464913e37'
JOB='1497612.stdct-mgmt-02'
REQUEST='64c3d8a9-130c-44fd-867c-e7c561b8f0b3'
ERROR='PermissionError: DATA_COV1_IMMUTABLE_0400_OWNED_REGULAR_FILE'

def audit(capture):
    def blob(n):
        x=capture['files'][n];b=base64.b64decode(x['base64'],validate=True)
        e.a.need(len(b)==x['bytes']and e.a.sha(b)==x['sha256'],'SEAL_FAIL_CAPTURE_BYTES:'+n);return b
    def read(n):return json.loads(blob(n))
    gb=base64.b64decode(capture['GLOBAL_base64'],validate=True)
    e.a.need(e.a.sha(gb)==capture['GLOBAL_sha256'],'SEAL_FAIL_REAL_GLOBAL_BYTES')
    g=json.loads(gb);row=e.rows(g)[0];request=read('REQUEST.json');m=read('RUN_MANIFEST.json');seal=read('SOURCE_SEAL.json')
    e.a.need(row['request_id']==request['request_id']==REQUEST and row['job_id']==JOB
        and row['status']==e.STATES[1]and row['source_sha']==m['code']['vla_commit']==seal['source_sha']==SOURCE
        and e.a.sha(blob('RUN_MANIFEST.json'))==row['run_manifest_sha256']
        and e.a.sha(blob('REQUEST.json'))==row['request_sha256']
        and e.a.sha(blob('SOURCE_SEAL.json'))==m['code']['source_seal_sha256'],
        'SEAL_FAIL_EXACT_SOURCE_REQUEST_BOUND_ROLE')
    modules={}
    for n,x in capture['modules'].items():
        b=base64.b64decode(x['base64'],validate=True)
        e.a.need(e.a.sha(b)==x['sha256']==seal['files'][n],'SEAL_FAIL_EXECUTED_SEALED_SOURCE:'+n)
        modules[n]=ast.parse(b)
    runtime=modules['scripts/pi0294/generalization/amd_export_runtime.py']
    assemble=next(n for n in runtime.body if isinstance(n,ast.FunctionDef)and n.name=='assemble')
    calls=[(x.lineno,x.func.id)for x in ast.walk(assemble)if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)]
    e.a.need(any(name=='qualify_all'for _,name in calls), 'SEAL_FAIL_ORIGINAL_CROSS_PARENT_QUALIFIER')
    log=blob('pbs/alloc-01/pbs.log').decode()
    e.a.need(log.rstrip().endswith(ERROR)and 'mixed=qualify_all('in log
        and "sealed_file(root/row['file'],row['sha256'])"in log,
        'SEAL_FAIL_EXACT_POST_EXPORT_NUMERIC_SEAL_TRACEBACK')
    native=capture['native'][JOB];v=native['Variable_List']
    e.a.need(set(capture['native'])=={JOB}and native['job_state']=='F'and type(native['Exit_status'])is int
        and native['Exit_status']==1 and int(native['run_count'])==1
        and native['Job_Name']==e.JOB_NAME and native['project']=='CFP04-CF-007'
        and native['Job_Owner'].split('@')[0]=='e1192548'
        and int(native['Resource_List']['ngpus'])==0 and int(native['Resource_List']['ncpus'])==1
        and v['GEN1_REQUEST_ID']==REQUEST and v['GEN1_SUBMISSION_NONCE']==row['submission_nonce'],
        'SEAL_FAIL_REAL_NATIVE_F_AND_SAME_ZERO_GPU_REQUEST')
    counter=read('RESOURCE_LEDGER.json');failure=read('evidence/GEN1_EXPORT_FAILURE.private.json')
    e.a.need(counter['status']=='FAILED_PRESERVED'and counter['error']==failure['error']==ERROR
        and counter['unknown_effects']==[dict(error=ERROR,remaining_output_bound_retained=True)]
        and counter['completed']==failure['completed']==11 and counter['admitted']==10
        and not capture['result_exists']and not capture['bank_exists'],
        'SEAL_FAIL_ALL_SAVED_EXPORTS_NO_COMPLETE_BANK_OR_SUCCESS_REWRITE')
    for key in('model_loads','queries','updates','constructors','physics_ticks','episode_starts'):
        e.a.need(type(counter[key])is int and counter[key]==0,'SEAL_FAIL_ACTUAL_ZERO:'+key)
    actor=read('control/CPU_COMPUTE_PROCESS.json');owner=read('control/cpu-owner-queue-repair-03/CPU_OWNER_OS_EXIT.json')
    e.a.need(actor['job_id']==JOB and actor['request_id']==REQUEST and actor['source']==SOURCE
        and actor['host'].lower()in {p.split('/')[0].lower()for p in native['exec_host'].split('+')}
        and owner['returncode']==1 and len(capture['actors'])==2
        and all(x['actual_procstart']!=x['expected_procstart']for x in capture['actors']),
        'SEAL_FAIL_REAL_PAYLOAD_F_AND_FINITE_OWNER_OS1')
    plan=read('plans/GEN1_SAVED_EXPORT_PLAN.private.json');packet=blob('plans/GEN1_V5_PREVIEW_V2.private.json')
    records=[read('datasets/'+case['id']+'/EXPORT_RECEIPT.json')for case in plan['cases']]
    e.a.need(len(records)==11 and len({r['id']for r in records})==11,'SEAL_FAIL_EXACT_ALL_SUCCESS_PARENT_RECEIPTS')
    support=amd_export.support_after_export(packet,plan['capture_outcomes'],records)
    value=amd_bank.manifest(packet,plan['capture_outcomes'],records)
    inventory={x['path']:x for x in capture['inventory']};members={}
    for r in value['episodes']:
        path='datasets/'+r['id']+'/'+r['file'];x=inventory[path]
        e.a.need(x['is_file']is True and x['uid']==22182 and x['mode']==0o600
            and x['sha256']==r['sha256']and x['bytes']==r['bytes']and 0<x['bytes']<=8*1024**2,
            'SEAL_FAIL_ACTUAL_SAVED_NUMERIC_BYTES_OWNED_0600_NOT_0400')
        receipt=inventory['datasets/'+r['id']+'/EXPORT_RECEIPT.json']
        e.a.need(receipt['mode']==0o400 and receipt['uid']==22182,'SEAL_FAIL_ORIGINAL_IMMUTABLE_RECEIPTS')
        members[r['id']]=dict(x)
    e.a.need(len(members)==10 and support['missing']==[],'SEAL_FAIL_RAW_R0_SUPPORT_IS_KNOWN_NOT_TRAINING_ADMISSION')
    seconds=sum(int(x)*n for x,n in zip(native['resources_used']['cput'].split(':'),(3600,60,1)))
    cpu=max(seconds+1,math.ceil(counter['CPU_s'])+10)
    terminal=dict(job_state='F',job_id=JOB,request_id=REQUEST,submission_nonce=row['submission_nonce'],
        Exit_status=1,conservative_CPU_s=cpu,run_manifest_sha256=row['run_manifest_sha256'],native_PBS=native)
    report=dict(status='PASS_INDEPENDENT_GEN1_CPU_EXPORT_EFFECTS',request_sha256=row['request_sha256'],
        all_processes_terminal=True,unknown_effects=[],export_finished=False,training_bank_qualified=False,
        model_loads=0,queries=0,updates=0,constructors=0,physics_ticks=0,episode_starts=0,
        original_failed_counter_and_UNKNOWN_unchanged=True,
        causal_boundary='ALL_ELEVEN_EXPORTS_SAVED_THEN_NUMERIC_0600_SEAL_REJECTED_BEFORE_CROSS_PARENT_BATCH',
        actual_CPU_counter_s=counter['CPU_s'],conservative_CPU_s=cpu,
        resource_ledger_sha256=e.a.sha(blob('RESOURCE_LEDGER.json')),
        failure_file_sha256=e.a.sha(blob('evidence/GEN1_EXPORT_FAILURE.private.json')),
        native_log_sha256=e.a.sha(blob('pbs/alloc-01/pbs.log')),
        completed_export_parents=11,R0_admitted_parents=10,support=support,
        preserved_R0_rejection_ids=[x['id']for x in records if not x['admitted_for_training']],
        saved_numeric_members=members,prospective_bank_manifest_sha256=e.a.sha(e.a.raw(value)))
    if 'aggregate_output'in capture:
        aggregate=capture['aggregate_output']
        e.a.need(type(aggregate['total_bytes'])is int and 0<aggregate['total_bytes']<=e.original.LIMITS['output_bytes']+e.METADATA,
            'SEAL_FAIL_ALL_RETAINED_OUTPUT_INSIDE_PAID_UPPER')
        report['shared_source_and_repair_output_bytes']=aggregate['total_bytes']
    return dict(audit=report,native=terminal,prospective_bank=value)
