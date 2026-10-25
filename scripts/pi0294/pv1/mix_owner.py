"""Cold Vanda adapter for the ORIGINAL watch; no agent heartbeat dependency.

One registered C/D training request. Original onceIO, Journal, finite watch,
queue parser, PBS wrapper and native Trainer are reused. No second scheduler.
"""
import base64,json,os,resource,subprocess,sys,time,uuid
from pathlib import Path
from types import SimpleNamespace
from scripts.pi0292.durable import read,write,OwnerLock
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.dec1.owner_transport import Journal,ReadOnlyFailure
from scripts.pi0294.dec1.source_cpu_once_io import commit
from scripts.pi0294.dec1 import a40_allocation as native,vanda_campaign as queue,cold_owner
from . import mix_accounting as a,successor_cd as scope


def need(ok,msg):
    if not ok:raise PermissionError('V5_CD_OWNER_'+msg)

def once(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:os.fchmod(f.fileno(),0o400);f.write(a.raw(v));f.flush();os.fsync(f.fileno())
    return v


def new_journal(base,name,identity):
    path=Path(base)/name;path.mkdir(mode=0o700)
    return Journal(path,identity)


def request_directory(root,number):
    need(type(number)is int and 1<=number<=34,'EXACT_CURRENT_SUBMISSION_ORDINAL')
    path=Path(root)/'pbs'/('alloc-'+str(number).zfill(2));path.mkdir(mode=0o700,parents=True)
    return path


def prepare_request(root,template,*,source_sha,submit_count,now_epoch):
    root=Path(root);audit_path=root/'control/COLD_INPUT_AUDIT.private.json';audit=read(audit_path)
    need(audit.get('status')=='PASS_ACTUAL_V5_CD_COLD_INPUTS_NO_MODEL' and audit.get('run_manifest_sha256')==sha(root/'RUN_MANIFEST.json') and audit.get('source_sha')==source_sha,'ACTUAL_COLD_INPUT_AUDIT_BEFORE_REQUEST')
    # The manifest seals the template. Bind actual manifest/audit AFTER both
    # exist, avoiding a self-referential config/manifest hash cycle.
    need(not {'request_sha256','run_manifest_sha256','cold_input_audit_sha256'}&set(template),'UNHASHED_IMMUTABLE_TEMPLATE_ONLY')
    req=dict(template,run_manifest_sha256=sha(root/'RUN_MANIFEST.json'),cold_input_audit_sha256=sha(audit_path),request_id=str(uuid.uuid4()),submission_nonce=str(uuid.uuid4()),created_epoch=now_epoch,submit_count=submit_count)
    a.validate_request(req);return req


def validate_heartbeat_receipt(value,intent,*,now):
    import math
    m=scope.validated({scope.FIELD:value.get('scope_binding')});beat=value.get('heartbeat',{});view=value.get('view',{})
    need(value.get('schema')=='V5_CD_ACTUAL_BOUND_NATIVE_HEARTBEAT_RECEIPT_V1' and m['T_CD']<=now<m['deadline_epoch'],'EXACT_NEW_SCOPE_HEARTBEAT_KIND_AND_CLOCK')
    need(value.get('lease_status')=='ACTIVE' and value.get('run_STOP')is False and value.get('parent_STOP')is False and value.get('unknown_effects')==value.get('pending_updates')==[],'REAL_LEASE_STOP_AND_UNKNOWN_FENCES')
    need(all(beat.get(k)==v for k,v in intent['identity'].items()) and beat.get('rpc_intent_id')==intent['operation_id'] and beat.get('scope')==scope.TASK and beat.get('grant_sha256')==scope.GRANT_SHA and beat.get('global_unknown_effects')==[],'EXACT_NEW_BOUND_IDENTITY_AND_INTENT')
    observed=beat.get('observed_epoch');vtime=view.get('observed_epoch')
    need(type(observed)in(int,float) and math.isfinite(observed) and intent['started_epoch']<=observed<=now and now-observed<=300 and type(vtime)in(int,float) and math.isfinite(vtime) and 0<=now-vtime<120,'ORIGINAL_300S_OWNER_AND_120S_ACCOUNT_VIEW')
    need(view.get('status')=='PASS_V5_CURRENT_UNFILTERED_QUEUE' and view.get('unknown_campaign_jobs')==[] and view.get('Hopper_current_queue_observed')is False and view.get('unavailable_Hopper_is_fresh_or_empty')is False,'NEW_TYPED_REAL_VANDA_VIEW_NOT_FORGED_OLD_DUAL')
    proof=view['complete_account_queue'];rows=queue.queue_rows(proof,now_epoch=now);own=[x for x in rows if x['job_id']==beat['job_id']]
    need(len(own)==1 and own[0]['state']==beat.get('native_job_state') and own[0]['state']in('Q','R'),'ACTUAL_CURRENT_OWN_JOB_IN_COMPLETE_ACCOUNT_QUEUE')
    job=json.loads(proof['qstat']['stdout'])['Jobs'][beat['job_id']]
    need(job['Job_Name']=='pi0294_dec1_a40_train' and job['project']=='CFP04-CF-007' and job['Variable_List']['PV1_REQUEST_ID']==beat.get('request_id') and job['Variable_List']['PV1_SUBMISSION_NONCE']==beat.get('submission_nonce'),'EXACT_REQUEST_NONCE_FROM_NATIVE_QSTAT')
    return value


class Owner:
    native=native
    read=staticmethod(read)
    update=staticmethod(write)
    def __init__(self,root):
        self.root=Path(root);self.CONFIG=read(self.root/'MIX_OWNER_CONFIG.private.json');self.manifest=read(self.root/'RUN_MANIFEST.json');self.binding=read(self.root/'OUTPUT_BINDING.private.json');self.global_path=self.root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';g=read(self.global_path);self.successor_scope=scope.validated(g);a.package_view(g)
        need(os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-') and os.environ.get('CUDA_VISIBLE_DEVICES')=='','REGISTERED_VANDA_LOGIN_CPU_OWNER_ONLY')
        need(self.manifest['platform_training_version']==a.VERSION and self.manifest['scope']==scope.TASK and self.manifest['execution']['agent_liveness_required']is False,'EXACT_NEW_SUCCESSOR_MANIFEST')
        need(self.binding['run_manifest_sha256']==sha(self.root/'RUN_MANIFEST.json') and read(self.binding['lease'])['status']=='ACTIVE','REGISTERED_ACTIVE_OWN_STAGE')
        reg=[r for r in read(self.binding['registry'])['entries'] if r.get('role')==self.binding['role']];need(len(reg)==1 and reg[0]['path']==str(self.root),'ONE_REGISTRY_ROLE')
        need(self.manifest['execution']['sealed_inputs']['MIX_OWNER_CONFIG.private.json']==sha(self.root/'MIX_OWNER_CONFIG.private.json'),'SEALED_COLD_OWNER_CONFIGURATION')
        self.code=self.root/self.manifest['code']['capsule_relative'];seal=read(self.root/'SOURCE_SEAL.json')
        for n,h in seal['files'].items():need(sha(self.code/n)==h,'EXACT_SOURCE_BYTE:'+n)
        need(Path(__file__).resolve()==(self.code/'scripts/pi0294/pv1/mix_owner.py').resolve(),'RUN_OWN_IMMUTABLE_SOURCE_CAPSULE')
        self.BASE=self.root/'control/resident-owner';self.BASE.mkdir(mode=0o700)
        self.request_template=self.CONFIG['request_template'];self.scope_record=read(self.root/'plans/V5_CD_PAIRED_PACKAGE_RESERVATION.private.json');self.last=None;self.current=None;self.cold_rpc_bound=True;self.previous=SimpleNamespace(rpc=self.heartbeat_rpc);self.HEARTBEAT='V5_CD_LOCAL_BOUND_OWNER_HEARTBEAT'
        self.setup=new_journal(self.BASE,'setup-rpc',dict(run_uuid=self.root.name,job_id='NO_GPU_YET',source_sha=self.manifest['code']['vla_commit'],request_sha256=scope.GRANT_SHA));self.running=None
        os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));resource.setrlimit(resource.RLIMIT_CPU,(1790,1800));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def record(self,n,v):return once(self.BASE/(n+'.private.json'),v)
    def write_global(self,phase,fn):
        path=self.global_path;before=path.read_bytes();g=json.loads(before);new=fn(g)
        from .neural_ledger import check
        new['revision']=g['revision']+1;check(new)
        request='v5cd-'+uuid.uuid4().hex
        return commit(path,expected_before_sha256=a.sha(before),proposed_bytes=a.raw(new),request_id=request,intent_path=path.with_name('V5_CD_'+request+'_INTENT.json'),receipt_path=path.with_name('V5_CD_'+request+'_RECEIPT.json'),lock_path=path.with_suffix('.owner.lock'))
    def program(self,name,*,effect='READ_ONLY'):
        code=self.CONFIG['programs'][name]
        def call(intent,limit):
            r=subprocess.run([sys.executable,'-'],input=code,capture_output=True,text=True,timeout=min(limit,40),preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CPU,(20,20)))
            if r.returncode:
                error=ReadOnlyFailure('V5_'+name+'_NATIVE_READ_FAILED_NOT_EMPTY_OR_TERMINAL');error.diagnostic=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);raise error
            return json.loads(r.stdout)
        return self.setup.call(name,effect,call)
    def auth(self,secret=None):
        # Released computation retains its valid job; no recurring amgr gate.
        if (self.BASE/'FIRST_UPDATE_RELEASED.private.json').exists():return dict(status='EXISTING_RELEASED_JOB_NO_AUTH_RESTART')
        value=self.program('AUTH')
        if value.get('classification')=='TOKEN_EXPIRED' and secret:
            from .auth_lifecycle import masked_login
            masked_login(['/opt/am/python/bin/amgr','login'],secret,timeout_s=45,process_cpu_limit_s=20,controlling_terminal=True)
            value=self.program('AUTH')
        need(value.get('classification')=='SUCCESS','FRESH_PROJECT_AUTH')
        need(value['balance_GPU_hours']>8.1 and value['balance_CPU_hours']>=36*22500/3600,'REAL_SITE_PROJECT_GPU_AND_CPU_BALANCE')
        return value
    def fresh(self,number=None,job=None,*,light=False):
        g=read(self.global_path);a.validate_global(g);proof=queue.collect_queue();rows=queue.queue_rows(proof,now_epoch=time.time());jobs=json.loads(proof['qstat']['stdout'])['Jobs'] if proof['qstat']else{};term=None
        if job and job not in jobs:
            args=['/opt/pbs/bin/qstat','-x','-f','-F','json',job]
            try:r=subprocess.run(args,capture_output=True,text=True,timeout=25)
            except subprocess.TimeoutExpired as e:raise ReadOnlyFailure('V5_EXACT_TERMINAL_READ_TIMEOUT_NOT_TERMINAL') from e
            if r.returncode or r.stderr:
                error=ReadOnlyFailure('V5_EXACT_TERMINAL_READ_FAILED_NOT_TERMINAL')
                error.diagnostic=dict(argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,observed_epoch=time.time())
                raise error
            term=json.loads(r.stdout)['Jobs'][job];need(term['job_state']=='F' and type(term['Exit_status'])is int,'ACTUAL_F_HISTORY_ONLY')
            rows.append(dict(platform='VANDA_A40',job_id=job,state='F',project=term['project'],name=term['Job_Name'],ngpus=term['Resource_List']['ngpus']))
        campaign,outside,unknown=queue.classify(g,rows);need(not unknown,'NO_UNKNOWN_OR_UNREGISTERED_CAMPAIGN_JOB')
        if job:
            j=jobs.get(job,term);r=read(self.BASE/'SOURCE_REQUEST.private.json')['request'];need(j['Job_Name']==r['job_name'] and j['project']==r['project'] and j['Variable_List']['PV1_REQUEST_ID']==r['request_id'] and j['Variable_List']['PV1_SUBMISSION_NONCE']==r['submission_nonce'],'EXACT_ACTUAL_JOB_AND_REQUEST_NONCE')
        receipts={};directory=Path(read(self.BASE/'SOURCE_REQUEST.private.json')['directory']) if (self.BASE/'SOURCE_REQUEST.private.json').exists() else None
        if directory:
            for n in ('PBS_READY.json','START_NEURAL.json','A40_PRETRAIN_READY.json','A40_PRETRAIN_RELEASE.json','PBS_PROCESS_FINAL.json','PBS_WRAPPER_FINAL.json'):
                p=directory/n
                if p.is_file():receipts[n]=read(p)
        p=self.root/'A40_NATIVE_EFFECT_LEDGER.json'
        if p.is_file():receipts[p.name]=read(p)
        view=dict(schema='V5_CD_ACTUAL_VANDA_ACCOUNT_VIEW_V1',status='PASS_V5_CURRENT_UNFILTERED_QUEUE',observed_epoch=time.time(),campaign_jobs=campaign,outside_registered_campaign_jobs=outside,unknown_campaign_jobs=[],complete_account_queue=proof,Hopper_current_queue_observed=False,unavailable_Hopper_is_fresh_or_empty=False)
        state=dict(view=view,rows=dict(vanda=dict(jobs=jobs,terminal_job=term,receipts=receipts)),global_ledger_sha256=sha(self.global_path));self.current=state;return state
    def heartbeat_rpc(self,site,code,values,label):
        need(site=='vanda' and values['JOB']==read(self.BASE/'SOURCE_JOB_BOUND.private.json')['job_id'],'ONE_LOCAL_BOUND_JOB_HEARTBEAT')
        job=values['JOB'];req=read(self.BASE/'SOURCE_REQUEST.private.json')['request'];need(values['VIEW']==self.current['view'],'LATEST_REAL_VIEW_BEFORE_HEARTBEAT')
        def receipt(beat):
            g=read(self.global_path);binding=read(self.root/'OUTPUT_BINDING.private.json')
            return dict(schema='V5_CD_ACTUAL_BOUND_NATIVE_HEARTBEAT_RECEIPT_V1',heartbeat=beat,scope_binding=self.successor_scope,view=self.current['view'],lease_status=read(binding['lease'])['status'],run_STOP=(self.root/'STOP.json').exists(),parent_STOP=(self.root.parent.parent/'STOP.json').exists(),unknown_effects=g['unknown_effects'],pending_updates=g['pending_updates'])
        def publish(intent,limit):
            now=time.time();need(0<=now-values['VIEW']['observed_epoch']<120,'FRESH_COMPLETE_NATIVE_VIEW')
            mark=self.root/'control/heartbeat-intents'/(intent['operation_id']+'.json');once(mark,intent)
            state=self.current['rows']['vanda']['jobs'][job]['job_state']
            beat=dict(run_uuid=self.root.name,job_id=job,request_id=req['request_id'],submission_nonce=req['submission_nonce'],request_sha256=read(self.BASE/'SOURCE_REQUEST.private.json')['request_sha256'],source_sha=req['source_sha'],scope=scope.TASK,grant_sha256=scope.GRANT_SHA,owner='EXISTING_SOLE_VANDA_RESIDENT_OWNER',observed_epoch=time.time(),rpc_intent_id=intent['operation_id'],native_job_state=state,global_unknown_effects=[],agent_liveness_required=False)
            write(self.root/'control/OWNER_HEARTBEAT.json',beat);return receipt(beat)
        def readback(intent,limit):
            beat=read(self.root/'control/OWNER_HEARTBEAT.json');need(beat['rpc_intent_id']==intent['operation_id'],'EXACT_HEARTBEAT_INTENT_READBACK');return receipt(beat)
        return self.running.call('HEARTBEAT','HEARTBEAT',publish,readback=readback)
    def submit(self,secret):
        self.auth(secret);quota=self.program('QUOTA');state=self.fresh();need(not state['view']['campaign_jobs'],'NO_ACTIVE_TASK_GPU_BEFORE_REQUEST')
        g=read(self.global_path);need(quota['report_age_s']<=3600 and quota['available_bytes_lower']>=g['storage_reserved_bytes']+40*1024**3+9442971836,'ACTUAL_PROJECT_FILESYSTEM_WHOLE_OUTPUT_AND_FINAL')
        req=prepare_request(self.root,self.request_template,source_sha=self.manifest['code']['vla_commit'],submit_count=len(g['PBS_submissions'])+1,now_epoch=time.time())
        directory=request_directory(self.root,req['submit_count']);once(directory/'REQUEST.json',req);request_sha=sha(directory/'REQUEST.json');self.record('SOURCE_REQUEST',dict(request=req,request_sha256=request_sha,directory=str(directory)));row=dict(req,request_sha256=request_sha)
        self.record('SOURCE_REQUEST_INTENT',dict(request_id=req['request_id'],once=True));receipt=self.write_global('REGISTER_REQUEST',lambda g:a.register_request(g,row,now_epoch=time.time()));self.record('SOURCE_GLOBAL_RESERVED',receipt)
        body='\n'.join(k+'='+repr(v) for k,v in dict(TARGET=str(self.root),DIRECTORY=str(directory),REQUEST=req,RESERVATION=dict(request_sha256=request_sha,observed_epoch=time.time(),whole_package_already_reserved=True)).items())+'\n'+self.CONFIG['programs']['SUBMIT']
        def submit(intent,limit):
            env={k:v for k,v in os.environ.items() if k not in ('CUDA_VISIBLE_DEVICES','PBS_JOBID')}
            r=subprocess.run([sys.executable,'-'],input=body,capture_output=True,text=True,timeout=min(limit,40),env=env)
            if r.returncode:raise RuntimeError('V5_QSUB_PROGRAM_FAILED_OR_UNKNOWN:'+r.stderr[-2000:])
            return json.loads(r.stdout)
        self.record('SOURCE_QSUB_INTENT',dict(request_id=req['request_id'],epoch=time.time(),once=True));reply=self.setup.call('QSUB','MUTATION',submit);self.record('SOURCE_QSUB_REPLY',reply)
        import re
        need(reply['returncode']==0 and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',reply['stdout']),'UNKNOWN_QSUB_NEVER_RESEND')
        job=reply['stdout'];self.write_global('BIND',lambda g:a.bind(g,req['request_id'],job,now_epoch=time.time()));self.record('SOURCE_JOB_BOUND',dict(job_id=job,request_id=req['request_id']));self.record('TRAIN_REGISTERED',dict(root=str(self.root),manifest_sha256=sha(self.root/'RUN_MANIFEST.json')))
        self.running=new_journal(self.BASE,'running-rpc',dict(run_uuid=self.root.name,job_id=job,source_sha=req['source_sha'],request_sha256=request_sha));return job
    def release(self,state,*,secret=None,first_update=False):
        req=read(self.BASE/'SOURCE_REQUEST.private.json')['request'];directory=Path(read(self.BASE/'SOURCE_REQUEST.private.json')['directory']);job=read(self.BASE/'SOURCE_JOB_BOUND.private.json')['job_id']
        project=self.auth(secret);quota=self.program('QUOTA');state=self.fresh(req['submit_count'],job,light=True);self.heartbeat_rpc('vanda','',dict(JOB=job,VIEW=state['view']),'fresh-before-release')
        g=read(self.global_path);local=json.loads(base64.b64decode(self.scope_record['Core_LOCAL_base64'],validate=True));proof=scope.recompute(g,local,Core_storage_upper_bytes=self.scope_record['Core_storage_upper'],local_storage_upper_bytes=self.scope_record['LOCAL_storage_upper']);need(not proof['shortfalls'],'FULL_REMAINING_RESOURCES_BEFORE_NATIVE_RELEASE')
        need(project.get('classification')=='SUCCESS' and quota['available_bytes_lower']>=g['storage_reserved_bytes']+40*1024**3+9442971836,'FRESH_PROJECT_AND_PHYSICAL_SPACE_AT_RELEASE')
        native_proof=dict(global_ledger_sha256=sha(self.global_path),request_sha256=sha(directory/'REQUEST.json'),whole_remaining_package_shortfalls=proof['shortfalls'],native_own_job_id=job,project_eligible=True,actual_project_quota_covers_remaining_output=True,complete_queue_verified=True,project=project,quota=quota,queue=state['view']['complete_account_queue'],Core_reservation_witness_sha256=self.scope_record['Core_LOCAL_sha256'],Core_reservation_is_exclusive_undispatched_successor_escrow=True,Core_snapshot_is_not_claimed_a_fresh_queue=True)
        ready='A40_PRETRAIN_READY.json' if first_update else 'PBS_READY.json';name='A40_PRETRAIN_RELEASE.json' if first_update else 'START_NEURAL.json';label='FIRST_UPDATE' if first_update else 'MODEL_START'
        def publish(intent,limit):
            value=dict(observed_epoch=time.time(),job_id=job,request_sha256=sha(directory/'REQUEST.json'),ready_sha256=sha(directory/ready),scope_grant_sha256=scope.GRANT_SHA,native_proof=native_proof)
            once(directory/name,value);return value
        self.record(label+'_RELEASE_INTENT',dict(job_id=job,epoch=time.time(),once=True));value=self.running.call(label,'MUTATION',publish);self.record(label+'_RELEASED',value);return value
    def finish(self,terminal):
        native_job=terminal['native'];req=read(self.BASE/'SOURCE_REQUEST.private.json')['request'];job=read(self.BASE/'SOURCE_JOB_BOUND.private.json')['job_id'];directory=Path(read(self.BASE/'SOURCE_REQUEST.private.json')['directory']);final=read(directory/'PBS_PROCESS_FINAL.json');wrapper=read(directory/'PBS_WRAPPER_FINAL.json')
        need(final['job_id']==wrapper['job_id']==job and final['source_sha']==req['source_sha'] and final['request_sha256']==sha(directory/'REQUEST.json') and wrapper['returncode']==native_job['Exit_status'],'NATIVE_PROCESS_AND_WRAPPER_FINAL_IDENTITIES')
        effects=final['effects'];need(effects['pending_updates']==effects['unknown_query_intents']==[],'ACTUAL_EFFECTS_KNOWN_BEFORE_SETTLE')
        audit=dict(status='NO_COMPLETE_ENDPOINT_FAILED_OR_PARTIAL')
        if native_job['Exit_status']==0 and effects['last_native_step']==1024:
            from .mix_terminal import audit as independent_audit
            self.record('FULL_STATE_AUDIT_INTENT',dict(job_id=job,read_only=True));audit=independent_audit(self.root,dict(req,job_id=job));self.record('FULL_STATE_INDEPENDENT_AUDIT',audit)
        from scripts.pi0294.e0_scope import pbs_epoch
        hms=lambda x:sum(int(v)*m for v,m in zip(x.split(':'),(3600,60,1)))
        wall=hms(native_job['resources_used']['walltime']);cpu=hms(native_job['resources_used']['cput']);end=pbs_epoch(native_job.get('obittime',native_job.get('mtime')));start=pbs_epoch(native_job['stime'])
        proof=dict(job_id=job,job_state='F',request_id=req['request_id'],submission_nonce=req['submission_nonce'],Exit_status=native_job['Exit_status'],conservative_GPU_s=max(wall,end-start)+60,conservative_CPU_s=max(cpu,final['CPU_actual_s'])+120,observed_epoch=time.time(),raw_native_sha256=a.sha(a.raw(native_job)),process_final_sha256=sha(directory/'PBS_PROCESS_FINAL.json'),wrapper_final_sha256=sha(directory/'PBS_WRAPPER_FINAL.json'))
        self.record('TERMINAL_SETTLEMENT_INTENT',dict(native=proof,effects=effects,audit_sha256=a.sha(a.raw(audit))))
        receipt=self.write_global('SETTLE',lambda g:a.settle_training(g,req['request_id'],native=proof,effects=effects,audit=audit,now_epoch=time.time()));self.record('TERMINAL_SETTLED',dict(receipt=receipt,native=proof,effects=effects,audit=audit))
        binding=read(self.root/'OUTPUT_BINDING.private.json');lease=read(binding['lease']);lease.update(status='CLOSED',closed_epoch=time.time(),terminal_receipt_sha256=sha(self.BASE/'TERMINAL_SETTLED.private.json'));write(binding['lease'],lease)
        self.record('CLOSED',dict(lease_sha256=sha(binding['lease']),job_id=job,native_exit_status=native_job['Exit_status'],complete_endpoint=audit.get('status')=='PASS_INDEPENDENT_V5_CD_1024_COMPLETE_STATE',next_stage='EVAL:'+req['arm']+':N294101',automatic_new_scientific_arm=False));return audit


def main(root):
    owner=Owner(root);secret=sys.stdin.readline(257).rstrip('\n');lock=OwnerLock(owner.root/'control/STAGE_SERVER_OWNER.lock')
    try:
        owner.record('SERVER_OWNER_PROCESS',dict(pid=os.getpid(),proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],source_sha=owner.manifest['code']['vla_commit'],agent_liveness_required=False,scope=owner.successor_scope))
        owner.submit(secret);terminal=cold_owner.watch(owner,secret,scope_binding=owner.successor_scope);owner.record('RESIDENT_TERMINAL_RETURNED',terminal);result=owner.finish(terminal);owner.record('RESIDENT_EXIT',dict(returncode=0,observed_epoch=time.time(),endpoint_status=result['status']));return result
    except BaseException as e:
        owner.record('PRESERVED_STOP',dict(error_type=type(e).__name__,message=str(e)[-3000:],diagnostic=getattr(e,'diagnostic',None),observed_epoch=time.time(),no_automatic_qsub_replay=True));raise
    finally:secret=None;lock.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();main(args.root)
