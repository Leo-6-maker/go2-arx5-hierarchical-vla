"""GEN1 cold adapter for the existing detached watch, auth, Journal and onceIO."""
import base64,json,os,resource,subprocess,sys,time,uuid
from pathlib import Path
from types import SimpleNamespace
from scripts.pi0292.durable import read,write,OwnerLock
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.dec1 import cold_owner
from scripts.pi0294.dec1.source_cpu_once_io import commit
from scripts.pi0294.pv1 import mix_owner as mature,successor_cd
from . import amd_training_accounting as a,amd_adoption as authority,amd_training as training
from .amd_training_runtime import bank_gate

need=a.need
once=mature.once

def validate_heartbeat_receipt(value,intent,*,now):
    from scripts.pi0294.dec1.vanda_campaign import queue_rows
    beat=value['heartbeat'];view=value['view']
    need(value['schema']=='GEN1_ACTUAL_BOUND_NATIVE_HEARTBEAT_RECEIPT_V1'
        and all(beat.get(k)==v for k,v in intent['identity'].items())
        and beat['rpc_intent_id']==intent['operation_id']and beat['scope']==a.VERSION
        and beat['grant_sha256']==authority.GRANT_SHA and beat['global_unknown_effects']==[]
        and value['lease_status']=='ACTIVE'and value['run_STOP']is value['parent_STOP']is False
        and value['unknown_effects']==value['pending_updates']==[]
        and intent['started_epoch']<=beat['observed_epoch']<=now
        and 0<=now-beat['observed_epoch']<=300 and 0<=now-view['observed_epoch']<120,
        'TRAIN_EXACT_HEARTBEAT_IDENTITY_SCOPE_AND_ORIGINAL_CLOCKS')
    need(view['status']=='PASS_V5_CURRENT_UNFILTERED_QUEUE'and view['unknown_campaign_jobs']==[]
        and view['Hopper_current_queue_observed']is False and view['unavailable_Hopper_is_fresh_or_empty']is False,
        'TRAIN_REAL_UNFILTERED_VANDA_VIEW_NOT_FABRICATED_DUAL')
    proof=view['complete_account_queue'];rows=queue_rows(proof,now_epoch=now)
    own=[x for x in rows if x['job_id']==beat['job_id']]
    need(len(own)==1 and own[0]['state']==beat['native_job_state']and own[0]['state']in('Q','R'),
        'TRAIN_CURRENT_OWN_JOB_IN_COMPLETE_ACCOUNT_QUEUE')
    job=json.loads(proof['qstat']['stdout'])['Jobs'][beat['job_id']]
    need(job['Job_Name']=='pi0294_dec1_a40_train'and job['project']=='CFP04-CF-007'
        and job['Variable_List']['PV1_REQUEST_ID']==beat['request_id']
        and job['Variable_List']['PV1_SUBMISSION_NONCE']==beat['submission_nonce'],
        'TRAIN_NATIVE_HEARTBEAT_REQUEST_AND_NONCE')
    return value

class Owner(mature.Owner):
    def __init__(self,root):
        self.root=Path(root);self.CONFIG=read(self.root/'MIX_OWNER_CONFIG.private.json')
        self.manifest=read(self.root/'RUN_MANIFEST.json');self.binding=read(self.root/'OUTPUT_BINDING.private.json')
        self.global_path=self.root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';g=read(self.global_path)
        self.successor_scope=successor_cd.validated(g);authority.validate(g);a.rows(g)
        need(os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-')and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
            'TRAIN_ORIGINAL_METADATA_LOGIN_OWNER_ONLY')
        need(self.manifest['scope']==self.manifest['platform_training_version']==a.VERSION
            and self.manifest['execution']['profile']==a.profile(self.CONFIG['request_template']) and self.manifest['execution']['agent_liveness_required']is False
            and self.binding['run_manifest_sha256']==sha(self.root/'RUN_MANIFEST.json')
            and read(self.binding['lease'])['status']=='ACTIVE','TRAIN_STANDARD_OWN_REGISTERED_ACTIVE_GEN_ROLE')
        rows=[x for x in read(self.binding['registry'])['entries']if x.get('role')==self.binding['role']]
        need(len(rows)==1 and rows[0]['path']==str(self.root),'TRAIN_ORIGINAL_CANONICAL_REGISTRY')
        self.code=self.root/self.manifest['code']['capsule_relative'];seal=read(self.root/'SOURCE_SEAL.json')
        need(sha(self.root/'SOURCE_SEAL.json')==self.manifest['code']['source_seal_sha256']
            and Path(__file__).resolve()==(self.code/'scripts/pi0294/generalization/amd_training_owner.py').resolve(),
            'TRAIN_OWN_IMMUTABLE_GEN_SOURCE_CAPSULE')
        for n,h in seal['files'].items():need(sha(self.code/n)==h,'TRAIN_SOURCE:'+n)
        for n,h in self.manifest['execution']['sealed_inputs'].items():need(sha(self.root/n)==h,'TRAIN_SEALED_INPUT:'+n)
        if self.CONFIG['request_template'].get('retry_generation')==1:
            from .amd_training_retry import verify_registered_root
            # Submit later joins the manifest/audit hashes into the template;
            # this early same-root check uses its already sealed manifest.
            verify_registered_root(self.root,dict(self.CONFIG['request_template'],
                run_manifest_sha256=sha(self.root/'RUN_MANIFEST.json'),
                cold_input_audit_sha256=sha(self.root/'control/COLD_INPUT_AUDIT.private.json')))
        self.BASE=self.root/'control/resident-owner';self.BASE.mkdir(mode=0o700)
        self.request_template=self.CONFIG['request_template'];self.scope_record=read(self.root/'plans/GEN1_OWNER_RESOURCES.private.json')
        self.last=None;self.current=None;self.cold_rpc_bound=True;self.preserve_readonly_until_deadline=True
        self.previous=SimpleNamespace(rpc=self.heartbeat_rpc);self.HEARTBEAT='GEN1_BOUND_LOCAL_OWNER_HEARTBEAT'
        self.setup=mature.new_journal(self.BASE,'setup-rpc',dict(run_uuid=self.root.name,job_id='NO_GPU_YET',source_sha=self.manifest['code']['vla_commit'],request_sha256=authority.GRANT_SHA));self.running=None
        os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));resource.setrlimit(resource.RLIMIT_CPU,(1790,1800));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def fresh(self,*args,**kwargs):
        a.rows(read(self.global_path));return super().fresh(*args,**kwargs)
    def heartbeat_rpc(self,site,code,values,label):
        need(site=='vanda'and values['JOB']==read(self.BASE/'SOURCE_JOB_BOUND.private.json')['job_id']
            and values['VIEW']==self.current['view'],'TRAIN_SAME_CURRENT_BOUND_NATIVE_VIEW')
        req=read(self.BASE/'SOURCE_REQUEST.private.json')['request'];job=values['JOB']
        def check(beat,intent):
            now=time.time();g=read(self.global_path);binding=read(self.root/'OUTPUT_BINDING.private.json')
            need(beat['rpc_intent_id']==intent['operation_id']and beat['request_id']==req['request_id']
                and beat['request_sha256']==sha(Path(read(self.BASE/'SOURCE_REQUEST.private.json')['directory'])/'REQUEST.json')
                and beat['scope']==a.VERSION and beat['grant_sha256']==authority.GRANT_SHA
                and intent['started_epoch']<=beat['observed_epoch']<=now and now-beat['observed_epoch']<=300
                and read(binding['lease'])['status']=='ACTIVE'and g['unknown_effects']==g['pending_updates']==[]
                and not any(p.exists()for p in [self.root/'STOP.json',self.root.parent.parent/'STOP.json'])
                and 0<=now-self.current['view']['observed_epoch']<120,'TRAIN_REAL_LIVE_OWNER_RECEIPT_FENCES')
            return dict(schema='GEN1_ACTUAL_BOUND_NATIVE_HEARTBEAT_RECEIPT_V1',heartbeat=beat,
                view=self.current['view'],lease_status='ACTIVE',unknown_effects=[],pending_updates=[],
                run_STOP=False,parent_STOP=False)
        def publish(intent,limit):
            state=self.current['rows']['vanda']['jobs'][job]['job_state'];need(state in('Q','R'),'TRAIN_NATIVE_Q_OR_R')
            beat=dict(run_uuid=self.root.name,job_id=job,request_id=req['request_id'],submission_nonce=req['submission_nonce'],
                request_sha256=read(self.BASE/'SOURCE_REQUEST.private.json')['request_sha256'],source_sha=req['source_sha'],
                scope=a.VERSION,grant_sha256=authority.GRANT_SHA,owner='EXISTING_SOLE_VANDA_RESIDENT_OWNER',
                observed_epoch=time.time(),rpc_intent_id=intent['operation_id'],native_job_state=state,
                global_unknown_effects=[],agent_liveness_required=False)
            result=check(beat,intent);once(self.root/'control/heartbeat-intents'/(intent['operation_id']+'.json'),intent)
            write(self.root/'control/OWNER_HEARTBEAT.json',beat);return result
        return self.running.call('HEARTBEAT','HEARTBEAT',publish,
            readback=lambda intent,limit:check(read(self.root/'control/OWNER_HEARTBEAT.json'),intent))
    def submit(self,secret):
        self.auth(secret);quota=self.program('QUOTA');state=self.fresh();need(not state['view']['campaign_jobs'],'TRAIN_NO_OTHER_ACTIVE_TASK_GPU')
        from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside
        checked_outside(state['view']['complete_account_queue']and __import__('scripts.pi0294.dec1.vanda_campaign',fromlist=['queue_rows']).queue_rows(state['view']['complete_account_queue'],now_epoch=time.time()),
            state['view']['outside_registered_campaign_jobs'],base64.b64decode(self.CONFIG['outside_confirmation_base64']))
        g=read(self.global_path);cold=self.root/'control/COLD_INPUT_AUDIT.private.json';audit=read(cold)
        need(audit['status']=='PASS_ACTUAL_GEN1_COLD_CLOSED_SAVED_INPUTS_NO_MODEL'
            and audit['run_manifest_sha256']==sha(self.root/'RUN_MANIFEST.json')and audit['source_sha']==self.manifest['code']['vla_commit'],
            'TRAIN_ACTUAL_COLD_INPUT_BEFORE_REQUEST')
        req=dict(self.request_template,run_manifest_sha256=sha(self.root/'RUN_MANIFEST.json'),
            cold_input_audit_sha256=sha(cold),created_epoch=time.time(),submit_count=len(g['PBS_submissions'])+1)
        a.validate_request(req);d=mature.request_directory(self.root,req['submit_count']);once(d/'REQUEST.json',req)
        self.record('SOURCE_REQUEST',dict(request=req,request_sha256=sha(d/'REQUEST.json'),directory=str(d)))
        local=base64.b64decode(self.scope_record['historical_LOCAL_base64']);closed=read(self.root/'plans/GEN1_CLOSED_EXPORT_INPUT.private.json')
        proposal=a.register(g,dict(req,request_sha256=sha(d/'REQUEST.json')),closed=closed,
            original_recipe=read(self.root/'plans/FROZEN_SOURCE_PAIR_RECIPE.json'),historical_LOCAL_bytes=local,now_epoch=time.time())
        need(quota['report_age_s']<=3600 and quota['available_bytes_lower']>=proposal['full_remaining_proof']['dimensions']['Hopper_storage_bytes']['projected']-g['new_storage_current_bytes'],
            'TRAIN_REAL_PROJECT_ALL_REMAINING_OUTPUT_QUOTA')
        self.record('SOURCE_REQUEST_INTENT',dict(request_id=req['request_id'],once=True))
        receipt=self.write_global('REGISTER_REQUEST',lambda current:a.register(current,dict(req,request_sha256=sha(d/'REQUEST.json')),closed=closed,
            original_recipe=read(self.root/'plans/FROZEN_SOURCE_PAIR_RECIPE.json'),historical_LOCAL_bytes=local,now_epoch=time.time())['proposed_global'])
        self.record('SOURCE_GLOBAL_RESERVED',receipt)
        body='\n'.join(k+'='+repr(v)for k,v in dict(TARGET=str(self.root),DIRECTORY=str(d),REQUEST=req,
            RESERVATION=dict(request_sha256=sha(d/'REQUEST.json'),observed_epoch=time.time(),whole_package_already_reserved=True)).items())+'\n'+self.CONFIG['programs']['SUBMIT']
        def invoke(intent,limit):
            env={k:v for k,v in os.environ.items()if k not in('CUDA_VISIBLE_DEVICES','PBS_JOBID')}
            p=subprocess.run([sys.executable,'-'],input=body,capture_output=True,text=True,timeout=min(limit,40),env=env)
            need(p.returncode==0,'TRAIN_QSUB_PROGRAM_UNKNOWN_NO_REPLAY');return json.loads(p.stdout)
        self.record('SOURCE_QSUB_INTENT',dict(request_id=req['request_id'],epoch=time.time(),once=True))
        reply=self.setup.call('GEN1_QSUB','MUTATION',invoke);self.record('SOURCE_QSUB_REPLY',reply)
        import re
        need(reply['returncode']==0 and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',reply['stdout']),'TRAIN_UNKNOWN_QSUB_NEVER_RESEND')
        job=reply['stdout'];self.write_global('BIND',lambda g:a.bind(g,req['request_id'],job,now_epoch=time.time()))
        self.record('SOURCE_JOB_BOUND',dict(job_id=job,request_id=req['request_id']));self.record('TRAIN_REGISTERED',dict(root=str(self.root),manifest_sha256=sha(self.root/'RUN_MANIFEST.json')))
        self.running=mature.new_journal(self.BASE,'running-rpc',dict(run_uuid=self.root.name,job_id=job,source_sha=req['source_sha'],request_sha256=sha(d/'REQUEST.json')));return job
    def release(self,state,*,secret=None,first_update=False):
        req=read(self.BASE/'SOURCE_REQUEST.private.json')['request'];d=Path(read(self.BASE/'SOURCE_REQUEST.private.json')['directory']);job=read(self.BASE/'SOURCE_JOB_BOUND.private.json')['job_id']
        bank=bank_gate(self.root);project=self.auth(secret);quota=self.program('QUOTA');state=self.fresh(req['submit_count'],job,light=True)
        self.heartbeat_rpc('vanda','',dict(JOB=job,VIEW=state['view']),'fresh-before-release')
        g=read(self.global_path);proof=a.scope.full_forecast(a.raw(g),base64.b64decode(self.scope_record['historical_LOCAL_base64']))
        need(not proof['cap_changes_required']and project['classification']=='SUCCESS'
            and quota['report_age_s']<=3600 and quota['available_bytes_lower']>=proof['dimensions']['Hopper_storage_bytes']['projected']-g['new_storage_current_bytes'],
            'TRAIN_ACTUAL_FULL_REMAINING_QUOTA_AND_PROJECT')
        p=dict(global_ledger_sha256=sha(self.global_path),request_sha256=sha(d/'REQUEST.json'),
            whole_remaining_package_shortfalls={},native_own_job_id=job,project_eligible=True,
            actual_project_quota_covers_remaining_output=True,complete_queue_verified=True,project=project,quota=quota,
            queue=state['view']['complete_account_queue'],actual_GEN1_bank_ready_sha256=sha(self.root/'datasets/FROZEN_GEN1_BANK/READY.json'),
            historical_LOCAL_is_not_fresh=True,full_remaining_proof=proof)
        ready='A40_PRETRAIN_READY.json'if first_update else'PBS_READY.json';name='A40_PRETRAIN_RELEASE.json'if first_update else'START_NEURAL.json';label='FIRST_UPDATE'if first_update else'MODEL_START'
        def publish(intent,limit):
            v=dict(observed_epoch=time.time(),job_id=job,request_sha256=sha(d/'REQUEST.json'),ready_sha256=sha(d/ready),scope_grant_sha256=authority.GRANT_SHA,native_proof=p)
            once(d/name,v);return v
        self.record(label+'_RELEASE_INTENT',dict(job_id=job,epoch=time.time(),once=True));v=self.running.call(label,'MUTATION',publish);self.record(label+'_RELEASED',v);return v
    def finish(self,terminal):
        req=read(self.BASE/'SOURCE_REQUEST.private.json')['request'];job=read(self.BASE/'SOURCE_JOB_BOUND.private.json')['job_id'];d=Path(read(self.BASE/'SOURCE_REQUEST.private.json')['directory']);native=terminal['native'];final=read(d/'PBS_PROCESS_FINAL.json');wrapper=read(d/'PBS_WRAPPER_FINAL.json')
        need(final['job_id']==wrapper['job_id']==job and final['source_sha']==req['source_sha']
            and final['request_sha256']==sha(d/'REQUEST.json')and wrapper['returncode']==native['Exit_status'],
            'TRAIN_REAL_CURRENT_PROCESS_WRAPPER_AND_F_IDENTITY')
        effects=final['effects'];need(effects['pending_updates']==effects['unknown_query_intents']==[],'TRAIN_KNOWN_EFFECTS_BEFORE_SETTLE')
        audit=dict(status='NO_COMPLETE_ENDPOINT_FAILED_OR_PARTIAL')
        if native['Exit_status']==0 and effects['last_native_step']==1024:
            self.record('FULL_STATE_AUDIT_INTENT',dict(job_id=job,read_only=True));audit=training.audit(self.root,dict(req,job_id=job));self.record('FULL_STATE_INDEPENDENT_AUDIT',audit)
        from scripts.pi0294.e0_scope import pbs_epoch
        hms=lambda x:sum(int(v)*m for v,m in zip(x.split(':'),(3600,60,1)))
        proof=dict(job_id=job,job_state='F',request_id=req['request_id'],submission_nonce=req['submission_nonce'],Exit_status=native['Exit_status'],
            conservative_GPU_s=max(hms(native['resources_used']['walltime']),pbs_epoch(native.get('obittime',native['mtime']))-pbs_epoch(native['stime']))+60,
            conservative_CPU_s=max(hms(native['resources_used']['cput']),final['CPU_actual_s'])+120,
            raw_native_sha256=a.sha(a.raw(native)),process_final_sha256=sha(d/'PBS_PROCESS_FINAL.json'),wrapper_final_sha256=sha(d/'PBS_WRAPPER_FINAL.json'))
        self.record('TERMINAL_SETTLEMENT_INTENT',dict(native=proof,effects=effects,audit_sha256=a.sha(a.raw(audit))))
        receipt=self.write_global('SETTLE',lambda g:a.settle_training(g,req['request_id'],native=proof,effects=effects,audit=audit,now_epoch=time.time()))
        self.record('TERMINAL_SETTLED',dict(receipt=receipt,native=proof,effects=effects,audit=audit))
        path=Path(self.binding['lease']);before=path.read_bytes();lease=json.loads(before)
        need(lease['status']=='ACTIVE'and lease['manifest_sha256']==sha(self.root/'RUN_MANIFEST.json'),'TRAIN_OWN_LEASE_CLOSE_ONCE')
        lease.update(status='CLOSED',closed_epoch=time.time(),terminal_receipt_sha256=sha(self.BASE/'TERMINAL_SETTLED.private.json'))
        key='GEN1-TRAIN-LEASE-CLOSE-'+req['request_id']
        value=self.running.call('CLOSE_OWN_GEN1_TRAIN_LEASE','MUTATION',lambda intent,limit:commit(path,
            expected_before_sha256=a.sha(before),proposed_bytes=a.raw(lease),request_id=key,
            intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),lock_path=path.with_suffix('.owner.lock')))
        self.record('CLOSED',dict(receipt=value,lease_sha256=sha(path),job_id=job,complete_endpoint=audit.get('status')==a.AUDIT,
            next_stage='DEFERRED_ORIGINAL_EVAL_NO_AUTOMATIC_NEW_TRAINING'));return audit

def main(root):
    owner=Owner(root);secret=sys.stdin.readline(257).rstrip('\n');need(bool(secret),'TRAIN_BACKGROUND_SECRET_ONLY_IN_MEMORY')
    lock=OwnerLock(owner.root/'control/STAGE_SERVER_OWNER.lock')
    try:
        owner.record('SERVER_OWNER_PROCESS',dict(pid=os.getpid(),proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            source_sha=owner.manifest['code']['vla_commit'],agent_liveness_required=False,scope=owner.successor_scope))
        owner.submit(secret);terminal=cold_owner.watch(owner,secret,scope_binding=owner.successor_scope)
        owner.record('RESIDENT_TERMINAL_RETURNED',terminal);result=owner.finish(terminal)
        owner.record('RESIDENT_EXIT',dict(returncode=0,observed_epoch=time.time(),endpoint_status=result['status']));return result
    except BaseException as exc:
        owner.record('PRESERVED_STOP',dict(error_type=type(exc).__name__,message=str(exc)[-3000:],diagnostic=getattr(exc,'diagnostic',None),no_automatic_qsub_replay=True));raise
    finally:secret=None;lock.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
