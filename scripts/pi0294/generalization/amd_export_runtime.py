"""Original official CPU exporter in an explicitly registered GEN1 PBS role.

No launcher, qsub, registry writer or model release lives here. A caller must
deliver closed-source raw bytes, seal all inputs, register/prepay the original
GLOBAL CPU row and bind the actual native job before this entry can proceed.
"""
import argparse,hashlib,json,os,re,resource,signal,socket,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
from . import amd_adoption as a,amd_export_accounting as accounting
from scripts.pi0294.e1.scope import STATS

ROLE=accounting.VERSION
PROFILE=accounting.PROFILE
READY_STATUS='SEALED_GEN1_RAW_R0_OFFICIAL_BANK_READY'


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()
def member(root,name):
    root=Path(root);relative=Path(name);p=root/relative
    a.need(not relative.is_absolute()and '..'not in relative.parts
        and p.is_file()and not p.is_symlink()and p.resolve().is_relative_to(root.resolve()),
        'EXPORT_REGISTERED_FILE_NO_LINK_OR_ESCAPE:'+str(name))
    return p


def native_allocation(job,*,job_id,hostname,request):
    a.need(job['job_state']=='R'and job['project']=='CFP04-CF-007'
        and job['Job_Name']==accounting.JOB_NAME and job['Job_Owner'].split('@')[0]=='e1192548',
        'EXPORT_REAL_OWN_RUNNING_CPU_JOB')
    hosts={v.split('/')[0].split('.')[0].lower()for v in job['exec_host'].split('+')}
    a.need(hostname.split('.')[0].lower()in hosts
        and not hostname.lower().startswith(('stdct-login','vanda')),'EXPORT_ACTUAL_COMPUTE_HOST')
    r=job['Resource_List'];h,m,s=map(int,r['walltime'].split(':'))
    a.need(int(r.get('ngpus',0))==0 and int(r['ncpus'])==1
        and h*3600+m*60+s==accounting.LIMITS['walltime_s']
        and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job_id)is not None
        and all(int(x)==0 for x in re.findall(r'ngpus=([0-9]+)',str(r.get('select',''))+str(job.get('exec_vnode','')))),
        'EXPORT_GRANTED_ONE_CPU_NO_GPU_AND_FIXED_WALL')
    variables=job['Variable_List'];a.need(variables['GEN1_REQUEST_ID']==request['request_id']
        and variables['GEN1_SUBMISSION_NONCE']==request['submission_nonce'],
        'EXPORT_NATIVE_REQUEST_NONCE_NOT_OLD_ALIAS')
    return job


def validate(root,*,compute=True):
    a.need(type(compute)is bool,'EXPORT_EXPLICIT_READONLY_OR_COMPUTE_VALIDATION')
    root=Path(root);a.need(root.is_absolute()and not root.is_symlink()and os.getuid()==22182,
        'EXPORT_VANDA_OWN_ROLE_HOST_UID')
    m=read(member(root,'RUN_MANIFEST.json'));b=read(member(root,'OUTPUT_BINDING.private.json'))
    role=ROLE+'__'+root.name
    a.need(m['schema']=='GEN1_VANDA_CPU_EXPORT_RUN_MANIFEST_V1'and m['scope']==ROLE
        and m['run_uuid']==root.name=='gen1-vanda-official-export-01'
        and m['output_root']==b['run_root']==str(root)and b['uid']==m['owner_uid']==22182
        and b['role']==role and sha(root/'RUN_MANIFEST.json')==b['run_manifest_sha256']
        and m['execution']['profile']==PROFILE and m['limits']==accounting.LIMITS,
        'EXPORT_OWN_TYPED_MANIFEST_AND_CANONICAL_PROFILE')
    g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json');mark=a.validate(g);rows=accounting.rows(g)
    request=read(member(root,'REQUEST.json'));a.need(len(rows)==1,'EXPORT_ONE_REGISTERED_GLOBAL_REQUEST')
    row=rows[0]
    a.need(row['request_id']==request['request_id']and row['request_sha256']==sha(root/'REQUEST.json')
        and row['run_manifest_sha256']==b['run_manifest_sha256']
        and row['submission_nonce']==request['submission_nonce']
        and row['status']in accounting.STATES[:2],'EXPORT_SAME_REGISTERED_UNFINISHED_REQUEST')
    if compute:a.need(row['status']==accounting.STATES[1]
        and os.environ.get('PBS_JOBID')==row['job_id'],'EXPORT_ACTUALLY_BOUND_NATIVE_REQUEST')
    lease=read(b['lease']);a.need(lease['status']=='ACTIVE'and lease['run_uuid']==root.name
        and lease['manifest_sha256']==b['run_manifest_sha256']
        and m['authority']['deadline_epoch']==lease['deadline_epoch']==mark['deadline_epoch']
        and mark['T_CD']<=time.time()<min(mark['deadline_epoch'],m['execution']['operation_deadline_epoch']),
        'EXPORT_OWN_ACTIVE_LEASE_AND_EXISTING_SCOPE_CLOCK')
    registry=read(b['registry'])['entries'];a.need(sum(x.get('role')==role and x['path']==str(root)
        and x['run_uuid']==root.name for x in registry)==1,'EXPORT_REAL_ORIGINAL_REGISTRY_ENTRY')
    profiles=read(b['environment'])['profiles'];matches=[x for x in profiles if x['name']==PROFILE]
    a.need(len(matches)==1 and matches[0]['manifest']==str(root/'RUN_MANIFEST.json')
        and matches[0]['entry_module']=='scripts.pi0294.generalization.amd_export_runtime'
        and matches[0]['PBS_required']is True and matches[0]['ngpus']==0,
        'EXPORT_STANDARD_ENTRY_EXPLICIT_NEW_CPU_PROFILE')
    code=root/m['code']['capsule_relative'];seal=read(member(root,'SOURCE_SEAL.json'))
    a.need(code.resolve().is_relative_to(root.resolve())and not code.is_symlink()
        and Path(__file__).resolve()==(code/'scripts/pi0294/generalization/amd_export_runtime.py').resolve()
        and seal['source_sha']==m['code']['vla_commit']==row['source_sha']
        and sha(root/'SOURCE_SEAL.json')==m['code']['source_seal_sha256'],
        'EXPORT_EXECUTED_IMMUTABLE_SOURCE')
    for name,h in seal['files'].items():a.need(sha(member(code,name))==h,'EXPORT_SOURCE:'+name)
    from scripts.pi0294.pv1.evaluation_core_scope import ORIGINAL_NUMERICAL_TABLE_SHA
    a.need(len(seal['frozen15'])==15 and hashlib.sha256(json.dumps(seal['frozen15'],sort_keys=True).encode()).hexdigest()==ORIGINAL_NUMERICAL_TABLE_SHA,
        'EXPORT_ALL_FIFTEEN_ORIGINAL_SCIENCE_IDENTITIES')
    for name,h in seal['frozen15'].items():a.need(sha(member(code,name))==h,'EXPORT_FROZEN:'+name)
    required={'INPUT_BINDINGS.private.json','plans/GEN1_SAVED_EXPORT_PLAN.private.json',
        'plans/ALL_CAPTURE_CLOSED.private.json','plans/GEN1_V5_PREVIEW_V2.private.json',
        'plans/statistics.json','plans/DELIVERED_RAW_MEMBERS.private.json','plans/OFFICIAL_CPU_INPUT_IDENTITIES.private.json'}
    a.need(required<=set(m['execution']['sealed_inputs']),'EXPORT_ALL_REQUIRED_INPUTS_SEALED')
    for name,h in m['execution']['sealed_inputs'].items():a.need(sha(member(root,name))==h,'EXPORT_INPUT:'+name)
    for name,key in [('enter.sh','entry_sha256'),('doctor.py','doctor_sha256')]:
        a.need(sha(Path(b['registry']).parent/name)==m['execution'][key],'EXPORT_UNCHANGED_ORIGINAL_ENTRY')
    a.need(os.environ.get('CUDA_VISIBLE_DEVICES')==''and os.environ.get('HF_HUB_OFFLINE')=='1'
        and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
        and os.environ.get('TRANSFORMERS_OFFLINE')=='1'and not any(x.exists()for x in(root/'STOP.json',root.parent.parent/'STOP.json')),
        'EXPORT_CPU_OFFLINE_AND_STOP_FENCES')
    plan=read(member(root,'plans/GEN1_SAVED_EXPORT_PLAN.private.json'))
    a.need(sha(root/'plans/GEN1_SAVED_EXPORT_PLAN.private.json')==row['raw_plan_sha256']
        and sha(root/'plans/ALL_CAPTURE_CLOSED.private.json')==row['closed_capture_proof_sha256'],
        'EXPORT_EXACT_REGISTERED_RAW_AND_CLOSED_PROOF')
    packet=member(root,'plans/GEN1_V5_PREVIEW_V2.private.json').read_bytes()
    closed=read(root/'plans/ALL_CAPTURE_CLOSED.private.json')
    a.need(closed['outcomes']==plan['capture_outcomes']and accounting.raw_support(packet,plan['capture_outcomes'])['ready'],
        'EXPORT_ALL_TWELVE_TERMINAL_OUTCOMES_AND_RAW_SUPPORT')
    cases=plan['cases'];expected={x['attempt_id']for x in plan['capture_outcomes']if x['full_task_success']}
    a.need(len(cases)==len(expected)and {x['id']for x in cases}==expected,'EXPORT_ALL_AND_ONLY_RAW_ADMITTED_PARENTS')
    for case in cases:
        for key in('source_episode','audit_file'):
            p=Path(case[key]);a.need(p.is_absolute()and not p.is_symlink()
                and p.resolve().is_relative_to((root/'inputs').resolve()),'EXPORT_DELIVERED_INPUT_CUSTODY_NO_ESCAPE')
    delivered=read(member(root,'plans/DELIVERED_RAW_MEMBERS.private.json'))
    a.need(delivered['status']=='PASS_ORIGINAL_CLOSED_SOURCE_BYTES_DELIVERED_FULL_SHA'
        and delivered['closed_capture_proof_sha256']==row['closed_capture_proof_sha256'],
        'EXPORT_ACTUAL_FULL_BYTE_DELIVERY_NOT_HF_READY_HINT')
    for item in delivered['files']:
        p=member(root,item['path']);a.need(p.stat().st_uid==22182 and p.stat().st_size==item['bytes']
            and sha(p)==item['sha256'],'EXPORT_DELIVERED_RAW_SHA:'+item['path'])
    a.need(len({x['path']for x in delivered['files']})==len(delivered['files'])
        and {x['path']for x in delivered['files']}=={str(p.relative_to(root))for p in (root/'inputs').rglob('*')if p.is_file()},
        'EXPORT_EXACT_FULL_REGISTERED_RAW_INPUT_FILE_SET')
    inputs=read(root/'INPUT_BINDINGS.private.json');official=read(root/'plans/OFFICIAL_CPU_INPUT_IDENTITIES.private.json')
    a.need(inputs['official_source']==official['official_source']and inputs['cosmos']==official['cosmos']
        and official['official_source_git_sha']=='51d4c89f72fda44cbf77285c6a8114b52676b8a1'
        and len(official['cosmos_files'])>0,'EXPORT_ORIGINAL_OFFICIAL_PROCESSOR_IDENTITY')
    for item in official['cosmos_files']:
        p=Path(item['path']);a.need(str(p.resolve())==item['resolved_path']and p.is_file()
            and p.stat().st_size==item['bytes']and sha(p)==item['sha256'],
            'EXPORT_FROZEN_OFFICIAL_PROCESSOR_INPUT:'+item['path'])
    if compute:
        result=subprocess.run(['/opt/pbs/bin/qstat','-f','-F','json',row['job_id']],capture_output=True,text=True,timeout=25,check=True)
        a.need(not result.stderr,'EXPORT_NATIVE_READ_DIAGNOSTIC_NOT_PASS');jobs=json.loads(result.stdout)['Jobs']
        a.need(set(jobs)=={row['job_id']},'EXPORT_EXACT_UNFILTERED_NATIVE_JOB')
        native_allocation(jobs[row['job_id']],job_id=row['job_id'],hostname=socket.gethostname(),request=request)
    return m,plan,packet,inputs


def assemble(root,value,processor,source,cosmos,guard,raw_roots):
    from scripts.pi0294.data_cov1.mixed_batch import qualify_all
    from scripts.pi0294.fd1 import put
    from .amd_bank import validate as validate_bank
    root=Path(root);packet=(root/'plans/GEN1_V5_PREVIEW_V2.private.json').read_bytes();validate_bank(value,packet)
    bindings=[];members={}
    for row in value['episodes']:
        guard();out=root/'datasets'/row['id'];a.need(read(out/'EXPORT_RECEIPT.json')==row,'EXPORT_ASSEMBLE_EXACT_OFFICIAL_RECEIPT')
        bindings.append(dict(id=row['id'],export_root=str(out),export_receipt_sha256=sha(out/'EXPORT_RECEIPT.json')))
        # The one delivered raw namespace stays in registered inputs. Numeric
        # output may have a bank link; no second or third RGB tree is made.
        numeric=member(out,row['file']);a.need(sha(numeric)==row['sha256']and numeric.stat().st_size<=8*1024**2,
            'EXPORT_BOUNDED_ACTUAL_NUMERIC_PARENT')
        members[row['file']]=numeric
    mixed=qualify_all(bindings,processor,source,cosmos,guard,registered_GEN1_raw_roots=raw_roots);out=root/'datasets/FROZEN_GEN1_BANK'
    a.need(not out.exists(),'EXPORT_PRESERVE_FIRST_BANK_INCLUDING_PARTIAL');out.mkdir()
    os.link(root/'plans/statistics.json',out/'statistics.json')
    for name,p in members.items():
        guard();target=out/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(p,target)
    put(out/'DATASET_MANIFEST.json',value);put(out/'CROSS_PARENT_QUALIFICATION.json',mixed)
    files=[]
    for p in sorted(out.rglob('*')):
        if p.is_file():
            guard()
            if p.stat().st_mode&0o222:p.chmod(0o400)
            files.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)))
    put(out/'PAYLOAD_MANIFEST.json',files)
    ready=dict(schema='GEN1_LOCAL_FULL_R0_OFFICIAL_BANK_READY_V1',status=READY_STATUS,
        training_identity=value['training_identity'],dataset_manifest_sha256=sha(out/'DATASET_MANIFEST.json'),
        payload_manifest_sha256=sha(out/'PAYLOAD_MANIFEST.json'),
        cross_parent_qualification_sha256=sha(out/'CROSS_PARENT_QUALIFICATION.json'),
        statistics_sha256=STATS,files=len(files),logical_payload_bytes=sum(x['bytes']for x in files),
        registered_source_RGB_bindings_sha256=sha(root/'plans/GEN1_FROZEN_RAW_BINDINGS.private.json'),
        later_rows_hot_injected=False,HF_delivery_claim=False,native_training_admission=False)
    put(out/'READY.json',ready);return ready


def run(root):
    m,plan,packet,inputs=validate(root);root=Path(root)
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.fd1 import put
    from . import amd_export as exporter,amd_bank as bank
    lock=OwnerLock(root/'GEN1_OFFICIAL_EXPORT_OWNER.lock');ledger=read(root/'RESOURCE_LEDGER.json')
    a.need(ledger['status']=='RESERVED'and ledger['unknown_effects']==[],'EXPORT_FRESH_OWN_EFFECT_COUNTER')
    began=time.process_time();child=resource.getrusage(resource.RUSAGE_CHILDREN);records=[];error=None;last_storage_check=0.
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_CPU,(36000,36000));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def guard():
        nonlocal last_storage_check
        a.need(time.time()<m['execution']['operation_deadline_epoch']and time.process_time()-began<35970
            and read(read(root/'OUTPUT_BINDING.private.json')['lease'])['status']=='ACTIVE'
            and not any(x.exists()for x in(root/'STOP.json',root.parent.parent/'STOP.json')),
            'EXPORT_ORIGINAL_CPU_WALL_LEASE_STOP')
        now=time.monotonic()
        if now-last_storage_check>=30:
            a.need(sum(p.stat().st_size for p in root.rglob('*')if p.is_file())<=accounting.LIMITS['output_bytes'],
                'EXPORT_ALL_INPUT_CONTROL_NUMERIC_AND_BANK_LOGICAL_BYTES_IN_PAID_UPPER')
            last_storage_check=now
    def expired(*args):raise TimeoutError('GEN1_ORIGINAL_EXPORT_WALL')
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(m['execution']['operation_deadline_epoch']-time.time())))
    try:
        from scripts.pi0291_arx5_data.native_dataset import cpu_guard
        from scripts.pi0292.train_adapter import build_processor
        from scripts.pi0294.e1.scope import STATS
        a.need(sha(root/'plans/statistics.json')==STATS,'EXPORT_EXACT_ORIGINAL_R0_BYTES')
        cpu_guard();processor=build_processor(inputs['official_source'],inputs['cosmos'],read(root/'plans/statistics.json'))
        ledger['status']='RUNNING';write(root/'RESOURCE_LEDGER.json',ledger)
        for case in plan['cases']:
            guard();out=root/'datasets'/case['id'];out.mkdir(parents=True,exist_ok=False)
            os.link(root/'plans/statistics.json',out/'statistics.json')
            row=exporter.export_case(case,out,processor,inputs['official_source'],inputs['cosmos'],guard,
                packet_bytes=packet,statistics_bytes=(root/'plans/statistics.json').read_bytes(),
                reuse_registered_source_rgb=True)
            put(out/'EXPORT_RECEIPT.json',row);records.append(row)
            ledger.update(completed=len(records),admitted=sum(x['admitted_for_training']for x in records),worker_CPU_s=time.process_time()-began);write(root/'RESOURCE_LEDGER.json',ledger)
        support=exporter.support_after_export(packet,plan['capture_outcomes'],records);ready=None
        if support['status']=='DATA_SUPPORT_READY_FOR_FRESH_TRAINING_ADMISSION':
            value=bank.manifest(packet,plan['capture_outcomes'],records)
            raw_roots={case['id']:case['source_episode']for case in plan['cases']if case['id']in {r['id']for r in value['episodes']}}
            put(root/'plans/GEN1_FROZEN_RAW_BINDINGS.private.json',raw_roots)
            ready=assemble(root,value,processor,inputs['official_source'],inputs['cosmos'],guard,raw_roots)
        last_storage_check=0.;guard()
        import torch,sys
        a.need(not torch.cuda.is_initialized()and 'mujoco'not in sys.modules
            and not any(n.startswith('gr00t')and ('modeling_'in n or'.backbone.'in n or'.action_head.'in n)for n in sys.modules),
            'EXPORT_NO_CUDA_NEURAL_MODEL_OR_NATIVE_PHYSICS_MODULES')
        result=dict(status='KNOWN_GEN1_OFFICIAL_EXPORT_TERMINAL',rows=records,support=support,bank_ready=ready,
            constructors=0,physics_ticks=0,episode_starts=0,model_loads=0,queries=0,updates=0,
            export_finished=True,native_training_admission=False)
        put(root/'evidence/GEN1_EXPORT_RESULT.private.json',result);return result
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc);ledger['unknown_effects'].append(dict(error=error,remaining_output_bound_retained=True))
        put(root/'evidence/GEN1_EXPORT_FAILURE.private.json',dict(error=error,completed=len(records)));raise
    finally:
        signal.alarm(0);after=resource.getrusage(resource.RUSAGE_CHILDREN)
        ledger.update(status='FAILED_PRESERVED'if error else'KNOWN_GEN1_EXPORT_TERMINAL',error=error,
            CPU_s=time.process_time()-began+after.ru_utime+after.ru_stime-child.ru_utime-child.ru_stime,
            constructors=0,physics_ticks=0,episode_starts=0,model_loads=0,queries=0,updates=0)
        write(root/'RESOURCE_LEDGER.json',ledger);lock.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);print(json.dumps(run(parser.parse_args().root)))
