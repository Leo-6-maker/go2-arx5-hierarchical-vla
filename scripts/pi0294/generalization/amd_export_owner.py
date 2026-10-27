"""Finite CPU adapter around the original watch, Journal and GLOBAL onceIO.

The scientific processor/exporter remains in the parent's immutable capsule.
This owner never loads a neural model, supplies a GPU heartbeat or schedules
any subsequent experiment. An uncertain mutation stops this finite adapter.
"""
import argparse,base64,hashlib,json,math,os,re,resource,shlex,subprocess,sys,time
from pathlib import Path

def read(p):return json.loads(Path(p).read_bytes())
def raw(v):return (json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def need(ok,why):
    if not ok:raise PermissionError('GEN1_CPU_OWNER_'+why)
def once(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb')as f:f.write(raw(value));f.flush();os.fsync(f.fileno())
    return value
def hms(v):
    h,m,s=map(int,v.split(':'));return h*3600+m*60+s

def payload_terminal(job,request,manifest_sha256):
    need(job['job_state']=='F'and type(job['Exit_status'])is int
        and job['Job_Name']=='pi0294_gen1_export'and job['project']=='CFP04-CF-007'
        and job['Job_Owner'].split('@')[0]=='e1192548'
        and int(job['Resource_List'].get('ngpus',0))==0
        and int(job['Resource_List']['ncpus'])==1
        and job['Variable_List']['GEN1_REQUEST_ID']==request['request_id']
        and job['Variable_List']['GEN1_SUBMISSION_NONCE']==request['submission_nonce'],
        'EXACT_NATIVE_CPU_TERMINAL_NOT_TIMEOUT_OR_OTHER_JOB')
    return dict(job_state='F',job_id=job['id'],request_id=request['request_id'],
        submission_nonce=request['submission_nonce'],Exit_status=job['Exit_status'],
        run_manifest_sha256=manifest_sha256,native_PBS=job)

def script(root,code,request,python,*,entry_module='scripts.pi0294.generalization.amd_export_runtime'):
    """Use the original Python/PBS entry; no container drops PBS variables."""
    root,code=Path(root),Path(code)
    need(entry_module in('scripts.pi0294.generalization.amd_export_runtime',
        'scripts.pi0294.generalization.amd_export_repair_runtime'),'EXACT_OFFICIAL_CPU_ENTRY_MODULE')
    bootstrap="""import json,os,time
from pathlib import Path
r=Path(ROOT);req=json.loads((r/'REQUEST.json').read_bytes());until=time.monotonic()+180
while True:
 g=json.loads((r.parent.parent/'HOPPER_RESOURCE_LEDGER.json').read_bytes());rows=[x for x in g['PBS_submissions']if x.get('request_id')==req['request_id']]
 if len(rows)==1 and rows[0]['job_id']==os.environ['PBS_JOBID']and(r/'control/CPU_RUNNING_RELEASE.json').exists():
  release=json.loads((r/'control/CPU_RUNNING_RELEASE.json').read_bytes());assert release['job_id']==os.environ['PBS_JOBID']and release['request_id']==req['request_id']and 0<=time.time()-release['observed_epoch']<=120;break
 if time.monotonic()>=until:raise TimeoutError('CPU_NATIVE_FRESH_RELEASE_MISSING_NO_PROCESSOR')
 time.sleep(.25)
pid=os.getppid();v=dict(pid=pid,procstart=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),host=os.uname().nodename,job_id=os.environ['PBS_JOBID'],request_id=req['request_id'],source=req['source_sha'],epoch=time.time())
fd=os.open(r/'control/CPU_COMPUTE_PROCESS.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400);os.write(fd,json.dumps(v,indent=2).encode());os.close(fd)
""".replace('ROOT',repr(str(root)))
    lines=['#!/bin/bash -l','#PBS -N pi0294_gen1_export','#PBS -P CFP04-CF-007',
        '#PBS -q auto','#PBS -l select=1:ncpus=1:mem=16gb:ngpus=0',
        '#PBS -l walltime=10:10:00','#PBS -m n','#PBS -r n','#PBS -j oe',
        '#PBS -o '+str(root/'pbs/alloc-01/pbs.log'),
        '#PBS -v GEN1_REQUEST_ID='+request['request_id']+',GEN1_SUBMISSION_NONCE='+request['submission_nonce'],
        'set -euo pipefail','umask 077','export CUDA_VISIBLE_DEVICES=',
        'export PYTHONPATH='+shlex.quote(str(code)+':'+str(code/'src')),
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        'export TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1',
        shlex.join([python,'-c',bootstrap]),
        'exec '+shlex.join([python,'-u','-m',entry_module,'--root',str(root)])]
    return ('\n'.join(lines)+'\n').encode()

class Owner:
    def __init__(self,root,*,control_relative=None):
        self.root=Path(root);self.m=read(self.root/'RUN_MANIFEST.json');self.binding=read(self.root/'OUTPUT_BINDING.private.json');self.code=self.root/self.m['code']['capsule_relative']
        sys.path[:0]=[str(self.code),str(self.code/'src')]
        from scripts.pi0294.generalization import amd_export_accounting as accounting,amd_export_runtime as runtime
        from scripts.pi0294.generalization import amd_export_repair_accounting,amd_export_repair_runtime
        if self.m['scope']==amd_export_repair_accounting.VERSION:
            accounting,runtime=amd_export_repair_accounting,amd_export_repair_runtime
            self.operation_prefix='GEN1-CPU-EXPORT-GIT-REPAIR02'
        else:
            need(self.m['scope']==accounting.VERSION,'EXACT_ORIGINAL_OR_DECLARED_GIT_REPAIR_CPU_ROLE')
            self.operation_prefix='GEN1-CPU-EXPORT'
        from scripts.pi0294.pv1 import mix_owner as mature
        from scripts.pi0294.dec1.owner_transport import Journal,ReadOnlyFailure
        self.e,self.runtime,self.mature=accounting,runtime,mature;self.ReadOnlyFailure=ReadOnlyFailure
        need(control_relative in(None,'control/cpu-owner-queue-repair-03'),'EXPLICIT_COLD_CONTROL_SLOT')
        self.control=self.root/(control_relative or 'control')
        self.CONFIG=read(self.control/'CPU_OWNER_CONFIG.private.json');pin=read(self.control/'CPU_OWNER_BINDING.private.json')
        need(os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-')and os.environ.get('CUDA_VISIBLE_DEVICES')=='','ORIGINAL_REGISTERED_LOGIN_METADATA_ONLY')
        need(pin['schema']=='GEN1_CPU_EXPORT_OWNER_BINDING_V1'
            and re.fullmatch('[0-9a-f]{40}',pin['control_source_sha'])is not None
            and pin['parent_manifest_sha256']==sha((self.root/'RUN_MANIFEST.json').read_bytes())
            and pin['owner_code_sha256']==sha(Path(__file__).read_bytes())
            and pin['configuration_sha256']==sha((self.control/'CPU_OWNER_CONFIG.private.json').read_bytes())
            and pin['science_source_sha']==self.m['code']['vla_commit']
            and Path(__file__).resolve()==((self.control/'amd_export_owner.py')if control_relative else(self.root/'control-code/amd_export_owner.py')).resolve(),
            'VERSIONED_COLD_CONTROL_BINDING_NOT_PARENT_HOTPATCH')
        need(read(self.binding['lease'])['status']=='ACTIVE','OWN_CPU_LEASE_ACTIVE')
        self.runtime.validate(self.root,compute=False)
        if control_relative:
            import importlib.util
            path=self.control/'amd_export_queue.py'
            need(sha(path.read_bytes())==pin['queue_adapter_sha256'],'EXACT_COLD_QUEUE_ADAPTER')
            spec=importlib.util.spec_from_file_location('scripts.pi0294.generalization.amd_export_queue',path)
            self.queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.queue)
        else:
            from scripts.pi0294.generalization import amd_export_queue
            self.queue=amd_export_queue
        self.BASE=self.control/'original-cpu-owner';self.BASE.mkdir(mode=0o700)
        self.global_path=self.root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';self.request=read(self.root/'REQUEST.json')
        self.setup=mature.new_journal(self.BASE,'setup-rpc',dict(run_uuid=self.root.name,job_id='ONE_CPU_REQUEST',source_sha=pin['control_source_sha'],request_sha256=sha((self.root/'REQUEST.json').read_bytes())))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(175,180));os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    def record(self,name,value):return once(self.BASE/(name+'.json'),value)
    def program(self,*args,**kwargs):return self.mature.Owner.program(self,*args,**kwargs)
    def auth(self,secret=None):return self.mature.Owner.auth(self,secret)
    def apply(self,key,value,before):
        from scripts.pi0294.dec1.source_cpu_once_io import commit
        def invoke(intent,limit):
            receipt=commit(self.global_path,expected_before_sha256=sha(before),proposed_bytes=raw(value),request_id=key,intent_path=self.global_path.with_name(key+'.INTENT.json'),receipt_path=self.global_path.with_name(key+'.RECEIPT.json'),lock_path=self.global_path.with_suffix('.owner.lock'))
            return dict(receipt=receipt,GLOBAL_sha256=sha(self.global_path.read_bytes()))
        return self.setup.call('WRITE_'+sha(key.encode()).upper()[:40],'MUTATION',invoke)
    def fresh(self,secret):
        from scripts.pi0294.dec1.vanda_campaign import collect_queue,queue_rows
        from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside
        g=read(self.global_path);self.e.rows(g);project=self.auth(secret);quota=self.program('QUOTA');queue=collect_queue();rows=queue_rows(queue,now_epoch=time.time());joined=self.queue.view(g,rows,request_id=self.request['request_id']);owned,outside,unknown=joined['own'],joined['outside'],joined['unknown'];need(not unknown,'COMPLETE_ACCOUNT_QUEUE_NO_UNKNOWN_CAMPAIGN')
        checked_outside(rows,outside,base64.b64decode(self.CONFIG['outside_confirmation_base64']))
        proof=self.e.scope.full_forecast(raw(g),base64.b64decode(self.CONFIG['historical_LOCAL_base64']));need(proof['cap_changes_required']=={},'ALL_FUTURE_MATRIX_FINAL_NO_SHORTFALL')
        future=proof['dimensions'];need(project['balance_CPU_hours']>=(future['Hopper_CPU_s']['projected']-g['CPU_s'])/3600 and project['balance_GPU_hours']>=(future['H200_s']['projected']-g['used']['H200_s'])/3600,'REAL_PROJECT_BALANCE_ALL_REMAINING_WORK')
        need(quota['report_age_s']<=3600 and quota['available_bytes_lower']>=max(g['storage_reserved_bytes']+40*1024**3+9442971836,future['Hopper_storage_bytes']['projected']-g['new_storage_current_bytes']),'REAL_PROJECT_FULL_REMAINING_OUTPUT_AND_FINAL')
        return dict(GLOBAL_sha256=sha(self.global_path.read_bytes()),project=project,quota=quota,full_queue=queue,queue_rows=rows,own=owned,exact_CPU_queue_join=joined,full_remaining_proof=proof,observed_epoch=time.time())
    def submit(self,secret):
        state=self.fresh(secret);need(not state['own'],'NO_OTHER_CAMPAIGN_JOB')
        need(time.time()<self.m['execution']['operation_deadline_epoch']-36600,'TIME_FOR_THE_COMPLETE_BOUND_CPU_STAGE')
        r=read(self.global_path);items=self.e.rows(r);need(len(items)==1 and items[0]['job_id']is None,'EXACT_UNBOUND_ORIGINAL_REQUEST')
        # Root/manifest/lease and this unbound request have already been
        # validated. Create only its missing metadata parent; an existing
        # allocation still raises, so an unknown qsub can never be replayed.
        directory=self.root/'pbs/alloc-01';directory.mkdir(mode=0o700,parents=True)
        body=script(self.root,self.code,self.request,self.m['execution']['python'],
            entry_module=self.m['execution'].get('entry_module','scripts.pi0294.generalization.amd_export_runtime'));path=directory/'job.pbs'
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
        with os.fdopen(fd,'wb')as f:f.write(body);f.flush();os.fsync(f.fileno())
        self.record('FRESH_PREQUEUE',state);self.record('PBS_SCRIPT',dict(sha256=sha(body),bytes=len(body),mail='n',ngpus=0,active_cpus=1))
        def invoke(intent,limit):
            v=subprocess.run(['/opt/pbs/bin/qsub',str(path)],capture_output=True,text=True,timeout=min(30,limit));self.record('QSUB_NATIVE_REPLY',dict(returncode=v.returncode,stdout=v.stdout,stderr=v.stderr));need(v.returncode==0 and not v.stderr and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',v.stdout.strip())is not None,'QSUB_OUTCOME_UNKNOWN_NO_REPLAY');return dict(job_id=v.stdout.strip())
        result=self.setup.call('ONE_ORIGINAL_GEN1_CPU_QSUB','MUTATION',invoke);job=result['job_id'];self.record('SOURCE_QSUB_RETURNED',result)
        before=self.global_path.read_bytes();after=self.e.bind(json.loads(before),request_id=self.request['request_id'],job_id=job);self.record('GLOBAL_JOB_BOUND',self.apply(getattr(self,'operation_prefix','GEN1-CPU-EXPORT')+'-BIND-01',after,before));self.record('BOUND_JOB',dict(job_id=job));return job
    def native(self,job):
        v=subprocess.run(['/opt/pbs/bin/qstat','-x','-f','-F','json',job],text=True,capture_output=True,timeout=25)
        if v.returncode or v.stderr:raise self.ReadOnlyFailure('CPU_NATIVE_READ_FAILED_NOT_TERMINAL')
        rows=json.loads(v.stdout)['Jobs'];need(set(rows)=={job},'EXACT_NATIVE_JOB_RESPONSE');return dict(rows[job],id=job)
    def release(self,job,secret):
        state=self.fresh(secret);rows=[r for r in state['queue_rows']if r['job_id']==job];need(len(rows)==1 and rows[0]['state']=='R','FRESH_REAL_RUNNING_CPU')
        native=self.native(job);self.runtime.native_allocation(native,job_id=job,hostname=native['exec_host'].split('/')[0],request=self.request)
        value=dict(status='ACTUAL_GEN1_CPU_ONLY_FRESH_RUNNING_RELEASE',job_id=job,request_id=self.request['request_id'],request_sha256=sha((self.root/'REQUEST.json').read_bytes()),source_sha=self.m['code']['vla_commit'],observed_epoch=time.time(),native_resource_proof=state,model_permission=False,physics_permission=False)
        def invoke(intent,limit):once(self.root/'control/CPU_RUNNING_RELEASE.json',value);return dict(sha256=sha(raw(value)))
        self.record('NATIVE_CPU_RELEASED',self.setup.call('ONE_FRESH_CPU_RUNNING_RELEASE','MUTATION',invoke))
    def terminal(self,job,native):
        from scripts.pi0294.generalization import amd_bank,amd_export
        need(native['Exit_status']==0,'FAILED_CPU_REQUIRES_EXACT_EFFECT_AUDIT_NOT_AUTOMATIC_SUCCESS')
        native=payload_terminal(native,self.request,self.binding['run_manifest_sha256']);ledger=read(self.root/'RESOURCE_LEDGER.json');result=read(self.root/'evidence/GEN1_EXPORT_RESULT.private.json')
        need(ledger['status']=='KNOWN_GEN1_EXPORT_TERMINAL'and ledger['unknown_effects']==[]and ledger['error']is None and result['status']=='KNOWN_GEN1_OFFICIAL_EXPORT_TERMINAL'and result['export_finished']is True,'REAL_COMPLETE_CPU_RESULT_AND_KNOWN_EFFECTS')
        for key in('constructors','physics_ticks','episode_starts','model_loads','queries','updates'):need(ledger[key]==result[key]==0,'NO_NON_CPU_EFFECT:'+key)
        plan=read(self.root/'plans/GEN1_SAVED_EXPORT_PLAN.private.json');packet=(self.root/'plans/GEN1_V5_PREVIEW_V2.private.json').read_bytes();records=result['rows']
        need([x['id']for x in records]==[x['id']for x in plan['cases']]
            and ledger['completed']==len(records)and ledger['admitted']==sum(x['admitted_for_training']for x in records),
            'ALL_ACTUAL_RAW_SUCCESS_PARENTS_EXPORTED_NOT_PARTIAL')
        for row in records:
            need(read(self.root/'datasets'/row['id']/'EXPORT_RECEIPT.json')==row,'UNCHANGED_PER_PARENT_OFFICIAL_CHAIN_RECEIPT')
        support=amd_export.support_after_export(packet,plan['capture_outcomes'],records);need(json.dumps(support,sort_keys=True)==json.dumps(result['support'],sort_keys=True),'INDEPENDENT_RECOMPUTED_RAW_R0_TASK_LAYOUT_SUPPORT')
        actor=read(self.root/'control/CPU_COMPUTE_PROCESS.json');need(actor['job_id']==job and actor['request_id']==self.request['request_id'],'ACTUAL_COMPUTE_IDENTITY')
        # PBS F plus native compute identity is used across kernels; a login
        # /proc lookup must never pretend to inspect a different compute host.
        cpu=max(hms(native['native_PBS']['resources_used']['cput'])+1,math.ceil(ledger['CPU_s'])+10);need(cpu<=36000,'ACTUAL_CONSERVATIVE_CPU_WITHIN_RESERVED_ENVELOPE');native['conservative_CPU_s']=cpu
        qualified=result['bank_ready']is not None
        need(qualified==(support['status']=='DATA_SUPPORT_READY_FOR_FRESH_TRAINING_ADMISSION'),'SUPPORTED_BANK_REQUIRED_IFF_FULL_ORIGINAL_GATES_PASS')
        if qualified:
            root=self.root/'datasets/FROZEN_GEN1_BANK';ready=read(root/'READY.json');need(ready==result['bank_ready']and ready['dataset_manifest_sha256']==sha((root/'DATASET_MANIFEST.json').read_bytes())and ready['payload_manifest_sha256']==sha((root/'PAYLOAD_MANIFEST.json').read_bytes())and ready['cross_parent_qualification_sha256']==sha((root/'CROSS_PARENT_QUALIFICATION.json').read_bytes()),'REAL_SEALED_BANK_READY_AND_ALL_PAYLOAD_IDENTITIES')
            amd_bank.validate(read(root/'DATASET_MANIFEST.json'),(self.root/'plans/GEN1_V5_PREVIEW_V2.private.json').read_bytes())
            for row in read(root/'PAYLOAD_MANIFEST.json'):
                path=root/row['path'];need(path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(root.resolve())and path.stat().st_size==row['bytes']and sha(path.read_bytes())==row['sha256'],'ACTUAL_FINAL_BANK_MEMBER')
        audit=dict(status='PASS_INDEPENDENT_GEN1_CPU_EXPORT_EFFECTS',request_sha256=sha((self.root/'REQUEST.json').read_bytes()),all_processes_terminal=True,terminal_scope='CPU_SCIENTIFIC_PAYLOAD_BY_REAL_NATIVE_PBS_F_NOT_LOGIN_PROC_GUESS',unknown_effects=[],export_finished=True,training_bank_qualified=qualified,model_loads=0,queries=0,updates=0,constructors=0,physics_ticks=0,episode_starts=0,resource_ledger_sha256=sha((self.root/'RESOURCE_LEDGER.json').read_bytes()),result_sha256=sha((self.root/'evidence/GEN1_EXPORT_RESULT.private.json').read_bytes()),actual_compute_record=actor)
        if self.e.VERSION=='GEN1_VANDA_OFFICIAL_CPU_EXPORT_KNOWN_GIT_REPAIR_V1':
            mark=self.e.a.validate(read(self.global_path))[self.e.FIELD]
            audit['shared_source_and_repair_output_bytes']=self.runtime.aggregate_output(self.root,mark['old_source_root'])['total_bytes']
        self.record('INDEPENDENT_CPU_TERMINAL_AUDIT',audit);self.record('NATIVE_CPU_TERMINAL',native);before=self.global_path.read_bytes();after=self.e.settle(json.loads(before),request_id=self.request['request_id'],native=native,audit=audit);self.record('GLOBAL_CPU_SETTLED',self.apply(getattr(self,'operation_prefix','GEN1-CPU-EXPORT')+'-SETTLE-01',after,before))
        path=Path(self.binding['lease']);before=path.read_bytes();lease=json.loads(before);need(lease['status']=='ACTIVE'and lease['run_uuid']==self.root.name and lease['manifest_sha256']==self.binding['run_manifest_sha256'],'OWN_CPU_CLOSE_ONCE');lease.update(status='CLOSED',closed_epoch=time.time(),terminal_native_sha256=sha(raw(native)),terminal_audit_sha256=sha(raw(audit)),payload_workers_terminal=True,finite_financial_owner_finishing=True)
        from scripts.pi0294.dec1.source_cpu_once_io import commit
        key=getattr(self,'operation_prefix','GEN1-CPU-EXPORT')+'-LEASE-CLOSE-01'
        def close(intent,limit):return commit(path,expected_before_sha256=sha(before),proposed_bytes=raw(lease),request_id=key,intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_suffix('.owner.lock'))
        self.record('OWN_CPU_LEASE_CLOSED',self.setup.call('CLOSE_OWN_CPU_EXPORT_LEASE','MUTATION',close));self.record('FINITE_CPU_STAGE_COMPLETE',dict(training_bank_qualified=qualified,native_training_permission=False));return qualified
    def run(self,secret):
        start=time.process_time();job=self.submit(secret);released=False
        while time.time()<self.m['authority']['deadline_epoch']:
            try:
                v=self.native(job)
                if v['job_state']=='R'and not released:self.release(job,secret);released=True;secret=None
                elif v['job_state']=='F':return self.terminal(job,v)
            except self.ReadOnlyFailure as exc:
                self.record('READ_ONLY_FAILURE_'+str(time.time_ns()),dict(error_type=type(exc).__name__,error=str(exc),not_terminal=True))
            time.sleep(15 if not released else 60)
        raise TimeoutError('ORIGINAL_SCOPE_DEADLINE_PRESERVE_EXISTING_CPU_JOB_AND_EVIDENCE')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--control-relative');args=p.parse_args();secret=sys.stdin.readline().rstrip('\n')or None;owner=Owner(args.root,control_relative=args.control_relative)
    once(owner.control/'CPU_OWNER_PROCESS.json',dict(pid=os.getpid(),procstart=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),host=os.uname().nodename,source_control_sha=read(owner.control/'CPU_OWNER_BINDING.private.json')['control_source_sha'],epoch=time.time()))
    result=owner.run(secret);print(json.dumps(dict(status='FINITE_ORIGINAL_CPU_OWNER_NATURAL_SUCCESS',training_bank_qualified=result,native_training_permission=False)))

if __name__=='__main__':main()
