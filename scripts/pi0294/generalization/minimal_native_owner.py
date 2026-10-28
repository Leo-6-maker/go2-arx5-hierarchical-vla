"""First512 cold adapter of the original detached owner/watch and onceIO.

One registered main PBS, five sequential sessions and13 science tickets. No
new scheduler, agent, unbounded retry, scientific recipe or budget is added.
"""
import base64,json,math,os,resource,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
from scripts.pi0292.durable import read,write,OwnerLock
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.dec1.source_cpu_once_io import commit
from scripts.pi0294.dec1.owner_transport import ReadOnlyFailure
from scripts.pi0294.dec1 import vanda_campaign as queue
from scripts.pi0294.pv1 import mix_owner as mature,successor_cd
from . import minimal_execution as e,minimal_native_admission as admission,minimal_session_gate as gates
from . import minimal_execution_adapter as adapters
from .minimal_native_context import captured_LOCAL

need=e.need;once=mature.once
HEARTBEAT='ARX5_MINIMAL_ACTUAL_BOUND_NATIVE_HEARTBEAT_RECEIPT_V1'


def worker_inventory(root):
    """Stable native-worker outputs, separate from this active CPU auditor."""
    import stat
    root=Path(root);rows=[]
    for name in('training','inference','evidence','reservations','pbs','worker_logs','A40_NATIVE_EFFECT_LEDGER.json'):
        base=root/name
        paths=[base,*base.rglob('*')]if base.is_dir()else[base]if base.exists()else[]
        for p in paths:
            rel=p.relative_to(root)
            if rel.is_relative_to('evidence/independent-terminal-reader'):continue
            st=p.lstat();need(st.st_uid==os.getuid()and(stat.S_ISREG(st.st_mode)or stat.S_ISDIR(st.st_mode)),
                'OWNER_TERMINAL_WORKER_OUTPUT_NO_SYMLINK_OR_FOREIGN_OWNER')
            if stat.S_ISREG(st.st_mode):rows.append(dict(path=str(rel),device=st.st_dev,inode=st.st_ino,
                bytes=st.st_size,allocated_bytes=st.st_blocks*512,mode=st.st_mode,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns))
    rows.sort(key=lambda v:v['path'])
    return dict(schema='ARX5_MINIMAL_ACTUAL_TERMINAL_WORKER_INVENTORY_V1',files=rows,
        worker_inventory_sha256=e.sha(e.raw(rows)),logical_bytes=sum(v['bytes']for v in rows),
        allocated_bytes=sum(v['allocated_bytes']for v in rows),active_CPU_owner_and_auditor_are_not_native_GPU_workers=True)


def validate_heartbeat_receipt(value,intent,*,now):
    beat=value.get('heartbeat',{});view=value.get('view',{})
    version=e.VERSION
    if beat.get('scope')==adapters.retry.VERSION:
        r=value.get('retry_request',{});adapter=adapters.request(r);version=adapter.VERSION
        need(adapter is adapters.retry and e.sha(e.raw(r))==beat.get('request_sha256')
            and all(r.get(k)==beat.get(k)for k in ('request_id','submission_nonce','source_sha','run_uuid')),
            'OWNER_RETRY_HEARTBEAT_EXACT_NEW_REQUEST_NOT_OLD_SCOPE')
    need(value.get('schema')==HEARTBEAT and value.get('lease_status')=='ACTIVE'
        and value.get('run_STOP')is value.get('parent_STOP')is False
        and value.get('unknown_effects')==value.get('pending_updates')==[]
        and all(beat.get(k)==v for k,v in intent['identity'].items())
        and beat.get('rpc_intent_id')==intent['operation_id'] and beat.get('scope')==version
        and beat.get('grant_sha256')==e.authority.GRANT_SHA and beat.get('global_unknown_effects')==[]
        and intent['started_epoch']<=beat.get('observed_epoch',-1)<=now
        and 0<=now-beat['observed_epoch']<=300 and 0<=now-view.get('observed_epoch',-1)<120,
        'OWNER_SAME_REAL_CURRENT_IDENTITY_LEASE_AND_ORIGINAL_HEARTBEAT')
    need(view.get('status')=='PASS_V5_CURRENT_UNFILTERED_QUEUE' and view.get('unknown_campaign_jobs')==[]
        and view.get('Hopper_current_queue_observed')is False and view.get('unavailable_Hopper_is_fresh_or_empty')is False,
        'OWNER_CURRENT_COMPLETE_QUEUE_NOT_UNAVAILABLE_HOPPER_AS_EMPTY')
    rows=queue.queue_rows(view['complete_account_queue'],now_epoch=now)
    own=[v for v in rows if v['job_id']==beat['job_id']]
    need(len(own)==1 and own[0]['state']==beat.get('native_job_state')and own[0]['state']in('Q','R'),
        'OWNER_CURRENT_OWN_JOB_Q_OR_R')
    job=json.loads(view['complete_account_queue']['qstat']['stdout'])['Jobs'][beat['job_id']]
    need(job['Job_Name']=='pi0294_dec1_a40_train'and job['project']=='CFP04-CF-007'
        and job['Variable_List']['PV1_REQUEST_ID']==beat['request_id']
        and job['Variable_List']['PV1_SUBMISSION_NONCE']==beat['submission_nonce'],
        'OWNER_NATIVE_REQUEST_AND_NONCE_IN_UNFILTERED_QSTAT')
    return value


class Owner(mature.Owner):
    def __init__(self,root):
        self.root=Path(root);self.CONFIG=read(self.root/'MIX_OWNER_CONFIG.private.json')
        self.manifest=read(self.root/'RUN_MANIFEST.json');self.binding=read(self.root/'OUTPUT_BINDING.private.json')
        self.adapter=adapters.manifest(self.manifest)
        self.global_path=self.root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';g=read(self.global_path)
        self.successor_scope=successor_cd.validated(g);e.package.validate(g);e.authority.validate(g)
        need(os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-')and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
            'OWNER_ONLY_ORIGINAL_METADATA_LOGIN_ROLE')
        need(self.manifest['scope']==e.scope.SCOPE and self.manifest['execution']['agent_liveness_required']is False
            and self.binding['run_manifest_sha256']==sha(self.root/'RUN_MANIFEST.json')
            and read(self.binding['lease'])['status']=='ACTIVE','OWNER_NEW_EXPLICIT_REGISTERED_ACTIVE_SCOPE')
        reg=[v for v in read(self.binding['registry'])['entries']if v.get('role')==self.binding['role']]
        need(len(reg)==1 and reg[0]['path']==str(self.root),'OWNER_CANONICAL_EXISTING_REGISTRY')
        inputs=self.manifest['execution']['sealed_inputs']
        need(inputs.get('MIX_OWNER_CONFIG.private.json')==sha(self.root/'MIX_OWNER_CONFIG.private.json'),
            'OWNER_SEALED_FINITE_CONFIGURATION')
        self.code=self.root/self.manifest['code']['capsule_relative'];seal=read(self.root/'SOURCE_SEAL.json')
        need(sha(self.root/'SOURCE_SEAL.json')==self.manifest['code']['source_seal_sha256']
            and seal['source_sha']==self.manifest['code']['vla_commit']
            and Path(__file__).resolve()==(self.code/'scripts/pi0294/generalization/minimal_native_owner.py').resolve(),
            'OWNER_OWN_IMMUTABLE_SOURCE')
        for n,h in seal['files'].items():need(sha(self.code/n)==h,'OWNER_SOURCE_MEMBER:'+n)
        for n,h in inputs.items():need(sha(self.root/n)==h,'OWNER_SEALED_INPUT:'+n)
        self.BASE=self.root/'control/resident-owner';self.BASE.mkdir(mode=0o700)
        self.request_template=self.CONFIG['request_template'];self.local=captured_LOCAL(self.root)
        need(adapters.select(self.request_template['platform_training_version'],self.request_template['profile']) is self.adapter,
             'OWNER_TEMPLATE_AND_MANIFEST_EXACT_SAME_EXECUTION_VERSION')
        self.retry_physical=None
        if self.adapter is adapters.retry:
            need({adapters.PHYSICAL,adapters.PREDECESSOR}<=set(inputs),
                 'OWNER_RETRY_COMPLETE_CLOSED_CAPTURE_AND_PHYSICAL_SEAL')
            self.retry_predecessor=read(self.root/adapters.PREDECESSOR)
            need(set(self.retry_predecessor)=={'native_readback','closed_readback','old_cold_close_readback','root_inventory'},
                 'OWNER_RETRY_EXACT_PREDECESSOR_INPUT_ROLES')
            from .minimal_retry_runtime_guard import Guard as Physical
            # Metadata owner calls inspect only. Native compute wraps its own
            # real R/lease/resource guard separately before any model effect.
            self.retry_physical=Physical(self.root,self.request_template,self.manifest,lambda **_:None)
        need(self.CONFIG['source_sha']==self.request_template['source_sha']==self.manifest['code']['vla_commit'],
            'OWNER_SAME_COLD_CONFIG_TEMPLATE_AND_SOURCE')
        self.last=None;self.current=None;self.cold_rpc_bound=True;self.preserve_readonly_until_deadline=True
        self.previous=SimpleNamespace(rpc=self.heartbeat_rpc);self.HEARTBEAT=HEARTBEAT
        self.setup=mature.new_journal(self.BASE,'setup-rpc',dict(run_uuid=self.root.name,job_id='NO_GPU_YET',
            source_sha=self.manifest['code']['vla_commit'],request_sha256=e.authority.GRANT_SHA));self.running=None
        os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
        resource.setrlimit(resource.RLIMIT_CPU,(3590,3600));resource.setrlimit(resource.RLIMIT_CORE,(0,0))

    def auth(self,secret=None):
        value=self.program('AUTH')
        if value.get('classification')=='TOKEN_EXPIRED'and secret:
            from scripts.pi0294.pv1.auth_lifecycle import masked_login
            self.setup.call('AUTH_RENEW','MUTATION',lambda intent,limit:masked_login(
                ['/opt/am/python/bin/amgr','login'],secret,timeout_s=min(limit,40),
                process_cpu_limit_s=20,controlling_terminal=True))
            value=self.program('AUTH')
        if value.get('classification')!='SUCCESS':
            error=ReadOnlyFailure('MINIMAL_PROJECT_UNAVAILABLE_NO_PERMISSION_OR_LOGIN_REPLAY')
            error.diagnostic=dict(classification=value.get('classification'),authentication_reply_redacted=True);raise error
        candidate=(read(self.BASE/'SOURCE_REQUEST.private.json')['request']if(self.BASE/'SOURCE_REQUEST.private.json').exists()
            else dict(self.request_template,run_manifest_sha256=sha(self.root/'RUN_MANIFEST.json'),
                cold_input_audit_sha256=sha(self.root/'control/COLD_INPUT_AUDIT.private.json')))
        hold=self.adapter.project_envelope(candidate)
        need(value['balance_GPU_hours']>=hold['GPU_hours']and value['balance_CPU_hours']>=hold['CPU_hours'],
            'OWNER_REAL_FULL_MAIN_NATIVE_CAPACITY_BALANCE')
        return value

    def fresh(self,*args,**kwargs):
        from scripts.pi0294.pv1.neural_ledger import check
        check(read(self.global_path));return super().fresh(*args,**kwargs)

    def heartbeat_rpc(self,site,code,values,label):
        job=read(self.BASE/'SOURCE_JOB_BOUND.private.json')['job_id'];saved=read(self.BASE/'SOURCE_REQUEST.private.json');r=saved['request']
        adapter=adapters.request(r)
        need(site=='vanda'and values['JOB']==job and values['VIEW']==self.current['view'],'OWNER_SAME_BOUND_LATEST_REAL_VIEW')
        def receipt(beat):
            g=read(self.global_path)
            result=dict(schema=HEARTBEAT,heartbeat=beat,view=self.current['view'],
                lease_status=read(self.binding['lease'])['status'],run_STOP=(self.root/'STOP.json').exists(),
                parent_STOP=(self.root.parent.parent/'STOP.json').exists(),unknown_effects=g['unknown_effects'],pending_updates=g['pending_updates'])
            if adapter is adapters.retry:result['retry_request']=r
            return result
        def publish(intent,limit):
            need(0<=time.time()-self.current['view']['observed_epoch']<120,'OWNER_CURRENT_QUEUE_BEFORE_HEARTBEAT')
            beat=dict(run_uuid=self.root.name,job_id=job,request_id=r['request_id'],submission_nonce=r['submission_nonce'],
                request_sha256=saved['request_sha256'],source_sha=r['source_sha'],scope=adapter.VERSION,
                grant_sha256=e.authority.GRANT_SHA,owner='EXISTING_SOLE_VANDA_RESIDENT_OWNER',observed_epoch=time.time(),
                rpc_intent_id=intent['operation_id'],native_job_state=self.current['rows']['vanda']['jobs'][job]['job_state'],
                global_unknown_effects=[],agent_liveness_required=False)
            result=receipt(beat);validate_heartbeat_receipt(result,intent,now=time.time())
            once(self.root/'control/heartbeat-intents'/(intent['operation_id']+'.json'),intent)
            write(self.root/'control/OWNER_HEARTBEAT.json',beat);return result
        return self.running.call('HEARTBEAT','HEARTBEAT',publish,
            readback=lambda intent,limit:validate_heartbeat_receipt(receipt(read(self.root/'control/OWNER_HEARTBEAT.json')),intent,now=time.time()))

    def submit(self,secret):
        project=self.auth(secret);quota=self.program('QUOTA');state=self.fresh()
        need(not state['view']['campaign_jobs'],'OWNER_SERIAL_NATIVE_MAIN_AFTER_CORE_C13_CLOSED')
        from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside
        checked_outside(queue.queue_rows(state['view']['complete_account_queue'],now_epoch=time.time()),
            state['view']['outside_registered_campaign_jobs'],base64.b64decode(self.CONFIG['outside_confirmation_base64'],validate=True))
        g=read(self.global_path);cold=self.root/'control/COLD_INPUT_AUDIT.private.json';audit=read(cold)
        need(audit['status']=='PASS_ACTUAL_MINIMAL_FIRST512_COLD_INPUTS_NO_MODEL'
            and audit['source_sha']==self.manifest['code']['vla_commit']and audit['run_manifest_sha256']==sha(self.root/'RUN_MANIFEST.json'),
            'OWNER_ACTUAL_COLD_INPUT_SOURCE_AND_MANIFEST')
        from .minimal_native_launch import effective_queue,collect_effective_queue
        queue_proof=self.setup.call('FRESH_EFFECTIVE_QUEUE','READ_ONLY',
            lambda intent,limit:collect_effective_queue(walltime_s=self.request_template['walltime_s']))
        queue_bytes=e.raw(queue_proof);created=time.time()
        effective_queue(queue_bytes,created_epoch=created,walltime_s=self.request_template['walltime_s'])
        r=dict(self.request_template,run_manifest_sha256=sha(self.root/'RUN_MANIFEST.json'),
            cold_input_audit_sha256=sha(cold),effective_queue_proof_sha256=e.sha(queue_bytes),
            created_epoch=created,submit_count=len(g['PBS_submissions'])+1)
        self.adapter.request(r);need(r['submit_count']<=e.authority.CAPS['PBS_submissions'],'OWNER_ORIGINAL_EXACT_ADOPTED_PBS_CAP')
        d=self.root/'pbs'/('alloc-'+str(r['submit_count']).zfill(2));d.mkdir(mode=0o700,parents=True)
        once(d/'EFFECTIVE_QUEUE_LIMITS.json',queue_proof)
        once(d/'REQUEST.json',r);request_bytes=(d/'REQUEST.json').read_bytes()
        self.record('SOURCE_REQUEST',dict(request=r,request_sha256=e.sha(request_bytes),directory=str(d)))
        capture=(self.root/'plans/MINIMAL_ACTUAL_CORE_MEASUREMENT.private.json').read_bytes()
        proposed=self.registration(self.global_path.read_bytes(),request_bytes,capture)
        proof=e.scope.first512_forecast(e.raw(proposed['proposed_global']),self.local)
        need(proof['cap_changes_required']=={}and quota['report_age_s']<=3600
            and quota['available_bytes_lower']>=proof['dimensions']['Hopper_storage_bytes']['projected']-g['new_storage_current_bytes'],
            'OWNER_COMPLETE_FUTURE_QUOTA_AND_ALL_EXISTING_WORK')
        self.record('SOURCE_REQUEST_INTENT',dict(request_id=r['request_id'],once=True))
        receipt=self.setup.call('REGISTER_REQUEST','MUTATION',lambda intent,limit:self.write_global('REGISTER_REQUEST',
            lambda current:self.registration(e.raw(current),request_bytes,capture)['proposed_global']))
        self.record('SOURCE_GLOBAL_RESERVED',receipt)
        # Registration may be slow. Do not dispatch with a queue observation
        # that expired after the exact request was bound; retain the registered
        # row and its costs for normal reconciliation if this guard trips.
        need(sha(d/'EFFECTIVE_QUEUE_LIMITS.json')==r['effective_queue_proof_sha256'],
            'OWNER_ORIGINAL_QUEUE_BYTES_UNCHANGED_BEFORE_QSUB')
        effective_queue((d/'EFFECTIVE_QUEUE_LIMITS.json').read_bytes(),
            created_epoch=time.time(),walltime_s=r['walltime_s'])
        body='\n'.join(k+'='+repr(v)for k,v in dict(TARGET=str(self.root),DIRECTORY=str(d),REQUEST=r,
            RESERVATION=dict(request_sha256=e.sha(request_bytes),observed_epoch=time.time(),whole_package_already_reserved=True)).items())+'\n'+self.CONFIG['programs']['SUBMIT']
        def qsub(intent,limit):
            env={k:v for k,v in os.environ.items()if k not in('CUDA_VISIBLE_DEVICES','PBS_JOBID')}
            p=subprocess.run([sys.executable,'-'],input=body,capture_output=True,text=True,timeout=min(limit,40),env=env)
            need(p.returncode==0,'OWNER_QSUB_REPLY_UNKNOWN_NO_REPLAY');return json.loads(p.stdout)
        self.record('SOURCE_QSUB_INTENT',dict(request_id=r['request_id'],once=True));reply=self.setup.call('QSUB','MUTATION',qsub)
        self.record('SOURCE_QSUB_REPLY',reply)
        import re
        need(reply['returncode']==0 and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',reply['stdout']),'OWNER_UNKNOWN_QSUB_NEVER_RESEND')
        job=reply['stdout'];self.setup.call('BIND','MUTATION',lambda intent,limit:self.write_global('BIND',
            lambda current:self.adapter.bind(current,r['request_id'],job,now_epoch=time.time())))
        self.record('SOURCE_JOB_BOUND',dict(job_id=job,request_id=r['request_id']))
        self.record('TRAIN_REGISTERED',dict(root=str(self.root),manifest_sha256=sha(self.root/'RUN_MANIFEST.json')))
        self.running=mature.new_journal(self.BASE,'running-rpc',dict(run_uuid=self.root.name,job_id=job,
            source_sha=r['source_sha'],request_sha256=e.sha(request_bytes)));return job

    def registration(self,global_bytes,request_bytes,capture):
        """Fresh actual physical inspection plus original onceIO proposal."""
        if self.adapter is e:
            return e.register(global_bytes,request_bytes=request_bytes,Core_measurement_capture_bytes=capture,now_epoch=time.time())
        from .minimal_retry_runtime_guard import CONTROL_GROWTH_RESERVE
        physical=self.retry_physical.inspect(next_output_upper_bytes=CONTROL_GROWTH_RESERVE);observed=time.time()
        return self.adapter.register(global_bytes,self.local,request_bytes=request_bytes,
            physical_custody=self.retry_physical.custody,physical_proof=physical,
            physical_observed_epoch=observed,now_epoch=time.time(),**self.retry_predecessor)

    def native_material(self,*,secret=None,ready_path=None):
        saved=read(self.BASE/'SOURCE_REQUEST.private.json');r=saved['request'];d=Path(saved['directory'])
        job=read(self.BASE/'SOURCE_JOB_BOUND.private.json')['job_id'];project=self.auth(secret);quota=self.program('QUOTA')
        state=self.fresh(r['submit_count'],job,light=True)
        self.heartbeat_rpc('vanda','',dict(JOB=job,VIEW=state['view']),'current-before-finite-release')
        native=state['rows']['vanda']['jobs'][job];need(native['job_state']=='R','OWNER_ACTUAL_R_BEFORE_RELEASE')
        need(quota['status']=='OFFICIAL_OWN_PROJECT_REPORT_AND_TARGET_FILESYSTEM_STABLE_READ'
            and quota['report_age_s']<=3600 and quota['target_NFS_export_matches_official_project_report']is True,
            'OWNER_ACTUAL_OWN_PROJECT_AND_FILESYSTEM_QUOTA')
        site=dict(schema=admission.SITE,source='VANDA_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE',
            run_uuid=r['run_uuid'],request_id=r['request_id'],submission_nonce=r['submission_nonce'],
            request_sha256=sha(d/'REQUEST.json'),source_sha=r['source_sha'],job_id=job,
            observed_epoch=project['observed_epoch'],project=project['project'],
            balance_GPU_hours=project['balance_GPU_hours'],balance_CPU_hours=project['balance_CPU_hours'],
            quota=dict(observed_epoch=quota['observed_epoch'],user='e1192548',scope='OWN_PROJECT_AND_FILESYSTEM_MINIMUM',
                available_bytes=quota['available_bytes_lower'],actual_original_report=quota))
        outside=base64.b64decode(self.CONFIG['outside_confirmation_base64'],validate=True)
        ready=Path(ready_path).read_bytes()if ready_path else None;effects=(self.root/'A40_NATIVE_EFFECT_LEDGER.json').read_bytes()if ready else None
        proof=admission.validate(global_bytes=self.global_path.read_bytes(),local_bytes=self.local,
            request_bytes=(d/'REQUEST.json').read_bytes(),manifest_bytes=(self.root/'RUN_MANIFEST.json').read_bytes(),job_id=job,
            hostname=native['exec_host'].split('+')[0].split('/')[0],user='e1192548',native_job_bytes=e.raw(native),
            queue_proof=state['view']['complete_account_queue'],site_bytes=e.raw(site),
            effective_queue_bytes=(d/'EFFECTIVE_QUEUE_LIMITS.json').read_bytes(),
            independent_job_bytes=outside,now_epoch=time.time(),ready_bytes=ready,effect_bytes=effects)
        return proof,dict(LOCAL_base64=base64.b64encode(self.local).decode(),complete_queue=state['view']['complete_account_queue'],
            site_base64=base64.b64encode(e.raw(site)).decode(),independent_confirmation_base64=base64.b64encode(outside).decode())

    def release(self,state,*,secret=None,first_update=False):
        need(first_update is False,'OWNER_MINIMAL_OPTIMIZATION_ONLY_THROUGH_FINITE_SCIENCE_TICKET')
        saved=read(self.BASE/'SOURCE_REQUEST.private.json');d=Path(saved['directory']);r=saved['request']
        proof,material=self.native_material(secret=secret)
        def publish(intent,limit):
            p=d/'INITIAL_NATIVE_MATERIAL.private.json';once(p,material)
            v=dict(schema='ARX5_MINIMAL_INITIAL_NATIVE_ALLOCATION_RELEASE_V1',job_id=proof['job_id'],
                request_sha256=proof['request_sha256'],ready_sha256=sha(d/'PBS_READY.json'),
                scope_grant_sha256=e.authority.GRANT_SHA,observed_epoch=time.time(),native_proof=proof,
                material_relative=str(p.relative_to(self.root)),material_sha256=sha(p))
            once(d/'START_NEURAL.json',v);return v
        value=self.running.call('INITIAL_NATIVE_RELEASE','MUTATION',publish);self.record('INITIAL_NATIVE_RELEASED',value);return value

    def release_science_session(self,state,*,secret=None):
        saved=read(self.BASE/'SOURCE_REQUEST.private.json');d=Path(saved['directory']);r=saved['request']
        job=read(self.BASE/'SOURCE_JOB_BOUND.private.json')['job_id']
        for spec in gates.plan():
            slot=gates.directory(d,spec['session_id'],spec['phase']);ready=slot/'READY.json'
            if not ready.exists():return False
            if(slot/'RELEASE.json').exists():continue
            proof,material=self.native_material(secret=secret,ready_path=ready)
            def publish(intent,limit):
                p=slot/'NATIVE_MATERIAL.private.json';once(p,material)
                v=dict(schema=gates.RELEASE,ticket_version=gates.VERSION,
                    status='PASS_MINIMAL_SESSION_'+spec['phase']+'_NATIVE',session_id=spec['session_id'],phase=spec['phase'],
                    job_id=job,request_sha256=saved['request_sha256'],ready_sha256=sha(ready),
                    scope_grant_sha256=e.authority.GRANT_SHA,observed_epoch=time.time(),native_proof=proof,
                    material_relative=str(p.relative_to(self.root)),material_sha256=sha(p))
                gates.check_release(v,ready.read_bytes(),r,job_id=job,now_epoch=time.time());once(slot/'RELEASE.json',v);return v
            value=self.running.call('SCIENCE_TICKET_'+str(spec['index']).zfill(2),'MUTATION',publish)
            self.record('SCIENCE_TICKET_'+str(spec['index']).zfill(2)+'_RELEASED',value);return True
        return False

    def finish(self,observed):
        from . import minimal_native_audit as independent,minimal_terminal as terminal
        from scripts.pi0294.e0_scope import pbs_epoch
        saved=read(self.BASE/'SOURCE_REQUEST.private.json');r=saved['request'];d=Path(saved['directory'])
        job=read(self.BASE/'SOURCE_JOB_BOUND.private.json')['job_id'];native=observed['native']
        final=read(d/'PBS_PROCESS_FINAL.json');wrapper=read(d/'PBS_WRAPPER_FINAL.json')
        need(native['job_state']=='F'and type(native['Exit_status'])is int and final['job_id']==wrapper['job_id']==job
            and final['source_sha']==r['source_sha']and final['request_id']==r['request_id']
            and final['request_sha256']==saved['request_sha256']and wrapper['returncode']==native['Exit_status'],
            'OWNER_KNOWN_F_AND_EXACT_WORKER_WRAPPER_FINAL_NOT_TIMEOUT')
        self.record('FULL_STATE_AUDIT_INTENT',dict(job_id=job,read_only=True,new_model_or_query_or_update=False))
        first=worker_inventory(self.root);time.sleep(.25);second=worker_inventory(self.root)
        need(first==second,'OWNER_ACTUAL_F_STABLE_COMPLETE_WORKER_INVENTORY')
        inventory_sha=first['worker_inventory_sha256']
        audit=independent.audit(self.root,d/'REQUEST.json',native=dict(job_id=job,job_state='F',
            Exit_status=native['Exit_status'],raw_native=native),inventory_sha256=inventory_sha)
        after=worker_inventory(self.root);need(after==second,'OWNER_NATIVE_WORKER_OUTPUTS_UNCHANGED_DURING_AUDIT')
        self.record('FULL_STATE_INDEPENDENT_AUDIT',dict(audit=audit,inventory=first))
        hms=lambda value:sum(int(v)*k for v,k in zip(value.split(':'),(3600,60,1)))
        end=pbs_epoch(native.get('obittime',native.get('mtime')));start=pbs_epoch(native['stime'])
        new_CPU=max(hms(native['resources_used']['cput']),final['CPU_actual_s'])+240
        current=read(self.global_path);_,row,_=adapters.registered(current,r['request_id'],version=self.adapter.VERSION)
        prior=(self.adapter.terminal_CPU_prior(current,row)if self.adapter is adapters.retry else e.PREPARATION_CPU_UPPER)
        proof=dict(schema=terminal.NATIVE,job_id=job,job_state='F',request_id=r['request_id'],
            submission_nonce=r['submission_nonce'],source_sha=r['source_sha'],request_sha256=saved['request_sha256'],
            Exit_status=native['Exit_status'],finished_epoch=end,observed_epoch=time.time(),
            conservative_GPU_s=max(hms(native['resources_used']['walltime']),end-start)+r['accounting_pad_s'],
            new_native_conservative_CPU_s=new_CPU,prior_readonly_preparation_CPU_upper_s=prior,
            conservative_CPU_s=new_CPU+prior,all_native_workers_terminal=True,terminal_inventory_stable=True,
            raw_native_sha256=e.sha(e.raw(native)),process_final_sha256=sha(d/'PBS_PROCESS_FINAL.json'),
            wrapper_final_sha256=sha(d/'PBS_WRAPPER_FINAL.json'),inventory_sha256=inventory_sha)
        effects=final['effects'];terminal.terminal_values(row,proof,effects,audit,
            retry_global=current if self.adapter is adapters.retry else None)
        self.record('TERMINAL_SETTLEMENT_INTENT',dict(native=proof,effects=effects,independent_audit_sha256=e.sha(e.raw(audit))))
        def settle(current):
            if self.adapter is e:
                return terminal.settle(current,r['request_id'],native=proof,effects=effects,audit=audit,now_epoch=time.time())
            from .minimal_retry_runtime_guard import CONTROL_GROWTH_RESERVE
            physical=self.retry_physical.inspect(next_output_upper_bytes=CONTROL_GROWTH_RESERVE);observed=time.time()
            return self.adapter.settle(current,r['request_id'],native=proof,effects=effects,audit=audit,
                physical_proof=physical,physical_observed_epoch=observed,now_epoch=time.time())
        receipt=self.running.call('SETTLE','MUTATION',lambda intent,limit:self.write_global('SETTLE',settle))
        self.record('TERMINAL_SETTLED',dict(receipt=receipt,native=proof,effects=effects,audit=audit))
        # Independent actual bytes before closing only this original own lease.
        current=read(self.global_path);_,closed_row,_=adapters.registered(current,r['request_id'],version=self.adapter.VERSION)
        need(closed_row['status']==e.STATES[2],'OWNER_INDEPENDENT_EXACT_SETTLED_ROW_BEFORE_CLOSE')
        path=Path(self.binding['lease']);before=path.read_bytes();lease=json.loads(before)
        need(lease['status']=='ACTIVE'and lease['run_uuid']==self.root.name
            and lease['manifest_sha256']==r['run_manifest_sha256'],'OWNER_ONLY_OWN_ACTIVE_STAGE_CLOSE_ONCE')
        lease.update(status='CLOSED',closed_epoch=time.time(),terminal_receipt_sha256=sha(self.BASE/'TERMINAL_SETTLED.private.json'))
        key='ARX5-MINIMAL-FIRST512-LEASE-CLOSE-'+r['request_id']
        closed=self.running.call('CLOSE_OWN_LEASE','MUTATION',lambda intent,limit:commit(path,
            expected_before_sha256=e.sha(before),proposed_bytes=e.raw(lease),request_id=key,
            intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),
            lock_path=path.with_suffix('.owner.lock')))
        need(path.read_bytes()==e.raw(lease),'OWNER_INDEPENDENT_OWN_CLOSED_LEASE_BYTES')
        self.record('CLOSED',dict(lease_sha256=sha(path),job_id=job,native_exit_status=native['Exit_status'],
            complete_endpoint=audit['status']==terminal.AUDIT,actual_once_close=closed,
            next_stage='CORE_E1_AND_E2_ORIGINAL24_AFTER_C50_D25_GEN1',automatic_new_scientific_arm=False))
        return audit


def main(root):
    from scripts.pi0294.dec1 import cold_owner
    owner=Owner(root);secret=sys.stdin.readline(257).rstrip('\n')
    need(bool(secret),'OWNER_CREDENTIAL_ONLY_HIDDEN_TTY_TO_ANONYMOUS_PIPE_AND_MEMORY')
    lock=OwnerLock(owner.root/'control/STAGE_SERVER_OWNER.lock')
    try:
        owner.record('SERVER_OWNER_PROCESS',dict(pid=os.getpid(),
            proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            source_sha=owner.manifest['code']['vla_commit'],agent_liveness_required=False,scope=owner.adapter.VERSION))
        owner.submit(secret);terminal=cold_owner.watch(owner,secret,scope_binding=owner.successor_scope)
        owner.record('RESIDENT_TERMINAL_RETURNED',terminal);result=owner.finish(terminal)
        owner.record('RESIDENT_EXIT',dict(returncode=0,observed_epoch=time.time(),endpoint_status=result['status']));return result
    except BaseException as error:
        owner.record('PRESERVED_STOP',dict(error_type=type(error).__name__,message=str(error)[-3000:],
            diagnostic=getattr(error,'diagnostic',None),observed_epoch=time.time(),no_automatic_qsub_replay=True));raise
    finally:secret=None;lock.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
