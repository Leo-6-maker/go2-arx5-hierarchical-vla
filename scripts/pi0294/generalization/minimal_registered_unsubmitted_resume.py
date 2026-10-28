"""One explicit control-role handoff for an exactly unsubmitted existing row.

The old owner is terminal. This controller never registers a request, changes
its source/bytes, retries qsub, or edits its immutable capsule. It reuses the
original owner methods and submit program after independent zero-qsub proof.
"""
import base64,json,math,os,time,hashlib,sys,subprocess,resource
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from . import minimal_execution as e,minimal_remaining_execution as remaining

FIELD='ARX5_registered_unsubmitted_owner_control_continuation'
LOCAL_FIELD=FIELD+'_explicit_historical_projection'
GRANT_SHA='d2b56371f48229df1a5c153441c21bae62464c0e0eba69e8335477e4ecba7add'
SOURCE='b0509a0bf296772e8e83856a2caf259b6ae0d908'
REQUEST='80aeab00-91c9-455d-96bc-587f716f227f'
NONCE='9fd57848-8b48-4518-af1c-dc01427e9eb2'
REQUEST_SHA='35a1f1ffb21768218fa16556ca3cd9de6a0f267b1c8e4fe7e6f36d478c5094b9'
MANIFEST='053556c595b010e5064138c07266ec67bf4671298758dd762899dff1f7ca5198'
CONTROL=e._vector(dict(Hopper_CPU_s=120,local_CPU_s=60,transfer_bytes=32768,Hopper_storage_bytes=1024**2))
read=lambda p:json.loads(Path(p).read_bytes())
need=e.need


def grant(body):
    need(type(body)is bytes and e.sha(body)==GRANT_SHA,'RESUME_EXACT_SAME_REQUEST_CONTROL_GRANT')
    h=json.loads(body)
    need(h['request_id']==REQUEST and h['submission_nonce']==NONCE and h['science_source_sha']==SOURCE
        and h['request_sha256']==REQUEST_SHA and h['run_manifest_sha256']==MANIFEST
        and h['control']==dict(site_CPU_s=120,local_CPU_s=60,transfer_bytes=32768,retained_metadata_bytes=1024**2)
        and sum(h['wire_plan'].values())==32768 and h['campaign_caps_changed']=={}
        and all(n==0 for n in h['limits'].values())and all(x is True for x in h['preserved'].values()),
        'RESUME_NO_NEW_GPU_REQUEST_SCIENCE_CAP_OR_OLD_ACTOR_RESTART')
    return h


def pending_self(g):
    rows=[x for x in g['PBS_submissions']if x.get('request_id')==REQUEST]
    need(len(rows)==1,'RESUME_EXACT_ONE_EXISTING_REQUEST');row=rows[0]
    need(row['submission_nonce']==NONCE and row['source_sha']==SOURCE and row['request_sha256']==REQUEST_SHA
        and row['run_manifest_sha256']==MANIFEST and row['submit_count']==39
        and row['status']==e.STATES[0]and row['job_id']is None
        and len(g['PBS_submissions'])==g['used']['PBS_submissions']==39
        and g['pending_updates']==g['unknown_effects']==[],
        'RESUME_ONLY_EXACT_REGISTERED_NOT_SUBMITTED_NO_GLOBAL_EFFECT_UNKNOWNS')
    return row


def predispatch(g,full,*,now_epoch):
    """Classify the genuine full queue, preserving its one known self-row gap."""
    row=pending_self(g)
    from scripts.pi0294.dec1 import vanda_campaign as q
    native=q.queue_rows(full,now_epoch=now_epoch);own,outside,unknown=q.classify(g,native)
    expected=[dict(job_id=None,reason='ACTIVE_LEDGER_WITHOUT_CURRENT_NATIVE_JOB_PROOF')]
    need(own==[]and unknown==expected,'RESUME_NO_FOREIGN_UNKNOWN_OR_ACTIVE_CAMPAIGN')
    # Classification-only view; no canonical ledger or unknown field is edited.
    view=dict(g,PBS_submissions=[x for x in g['PBS_submissions']if x is not row])
    own2,outside2,unknown2=q.classify(view,native)
    need(own2==unknown2==[]and outside2==outside,'RESUME_ONLY_THIS_EXACT_UNBOUND_SELF_ROW')
    return dict(complete_queue=full,complete_queue_rows=native,outside=outside,
        original_classification_unknown=unknown,exact_registered_self_gap=True)


def validate(g):
    m=g.get(FIELD)
    if m is None:return None
    h=grant(base64.b64decode(m['grant_base64'],validate=True));rows=[x for x in g['PBS_submissions']if x.get('request_id')==REQUEST]
    need(m.get('schema')=='ARX5_EXACT_REGISTERED_UNSUBMITTED_CONTROL_ADOPTION_V1'
        and len(rows)==1 and rows[0]['request_sha256']==REQUEST_SHA and rows[0]['source_sha']==SOURCE
        and m['control_paid_once']==CONTROL and m['request_id']==REQUEST
        and m['science_manifest_sha256']==MANIFEST and m['new_request_or_GPU_authority']is False
        and m['before']['PBS_submissions']==39 and g['used']['PBS_submissions']>=39
        and g['CPU_s']>=m['before']['CPU_s']+120
        and g['new_storage_current_bytes']>=m['before']['new_storage_current_bytes']+1024**2
        and g['new_storage_peak_bytes']>=max(m['before']['new_storage_peak_bytes'],g['new_storage_current_bytes'])
        and all(m[k]==h[k]==rows[0][k]for k in('T_CD','stop_new_training_epoch','deadline_epoch'))
        and m['T_CD']<=m['adopted_epoch']<m['stop_new_training_epoch'],
        'RESUME_ONCE_CONTROL_FEE_AND_ORIGINAL_CLOCK_SOURCE_AND_OLD_COSTS')
    return m


def budget_LOCAL(g,l):
    if validate(g)is None:return l
    proof=dict(grant_sha256=GRANT_SHA,costs=CONTROL,canonical_LOCAL_write_proven=False,
               original_historical_LOCAL_sha256=g[FIELD]['historical_LOCAL_sha256'])
    if LOCAL_FIELD in l:
        need(l[LOCAL_FIELD]==proof,'RESUME_ONCE_EXPLICIT_NONCANONICAL_PROJECTION');return l
    v=deepcopy(l);v['CPU_s']['local_PC']+=60;v['network_transfer_accounted_bytes']+=32768;v[LOCAL_FIELD]=proof
    return v


def propose(g,local,scope_bytes,*,now_epoch,zero_qsub_proof):
    h=grant(scope_bytes);pending_self(g);need(FIELD not in g and zero_qsub_proof['status']=='EXACT_TERMINAL_OWNER_NEVER_ATTEMPTED_QSUB',
        'RESUME_ONE_NEW_HANDOFF_AFTER_EXACT_ZERO_QSUB_AUDIT')
    v=deepcopy(g);before={k:g[k]for k in('CPU_s','new_storage_current_bytes','new_storage_peak_bytes')};before['PBS_submissions']=39
    v['CPU_s']+=120;v['new_storage_current_bytes']+=1024**2;v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes']);v['revision']+=1
    v[FIELD]=dict(schema='ARX5_EXACT_REGISTERED_UNSUBMITTED_CONTROL_ADOPTION_V1',grant_sha256=GRANT_SHA,grant_base64=base64.b64encode(scope_bytes).decode(),control_paid_once=deepcopy(CONTROL),
        request_id=REQUEST,science_manifest_sha256=MANIFEST,before=before,before_GLOBAL_sha256=e.sha(e.raw(g)),
        historical_LOCAL_sha256=e.sha(local),zero_qsub_proof_sha256=e.sha(e.raw(zero_qsub_proof)),
        adopted_epoch=now_epoch,new_request_or_GPU_authority=False,**{k:h[k]for k in('T_CD','stop_new_training_epoch','deadline_epoch')})
    validate(v);projected=budget_LOCAL(v,json.loads(local));proof=e.scope.first512_forecast(e.raw(v),e.raw(projected))
    need(proof['cap_changes_required']=={},'RESUME_ALL14_AND_ORIGINAL_MATRIX_FINAL_PROTECTED')
    return v,proof


def exact_zero_qsub(root,g):
    row=pending_self(g);root=Path(root);base=root/'control/resident-owner';binding=read(root/'OUTPUT_BINDING.private.json')
    need(e.sha((root/'RUN_MANIFEST.json').read_bytes())==MANIFEST and read(binding['lease'])['status']=='ACTIVE',
         'RESUME_SAME_EXISTING_ACTIVE_REGISTERED_ROLE')
    saved=read(base/'SOURCE_REQUEST.private.json');directory=Path(saved['directory']);rb=(directory/'REQUEST.json').read_bytes()
    need(e.sha(rb)==saved['request_sha256']==REQUEST_SHA and all(row[k]==v for k,v in json.loads(rb).items()),
         'RESUME_EXACT_ORIGINAL_REGISTERED_REQUEST_FILE_NOT_RESTAMPED')
    final=read(root/'control/RESIDENT_OWNER_OS_EXIT.json');wrap=read(root/'control/RESIDENT_WRAPPER_PROCESS.private.json');actor=read(base/'SERVER_OWNER_PROCESS.private.json')
    need(final['returncode']==1 and final['foreground_wait_completed']is True and final['source_sha']==SOURCE
        and read(base/'PRESERVED_STOP.private.json')['message']=='V5_CD_OWNER_NO_UNKNOWN_OR_UNREGISTERED_CAMPAIGN_JOB',
        'RESUME_ORIGINAL_EXACT_PREDISPATCH_OWNER_NATURAL_OS1')
    for pid,start in((wrap['wrapper_pid'],wrap['wrapper_proc_start_ticks']),(actor['pid'],actor['proc_start_ticks'])):
        p=Path('/proc',str(pid),'stat');f=p.read_text().rsplit(')',1)[1].split()if p.exists()else None
        need(not(f and f[0]not in('Z','X')and int(f[19])==int(start)),'RESUME_ORIGINAL_EXACT_KERNEL_ACTOR_ABSENT')
    for name in('SOURCE_QSUB_INTENT','SOURCE_QSUB_REPLY','SOURCE_JOB_BOUND'):
        need(not(base/(name+'.private.json')).exists(),'RESUME_NO_PRIOR_'+name)
    from .minimal_remaining_dispatch import FILE
    need(not(directory/FILE).exists(),'RESUME_NO_PRIOR_DISPATCH_PROOF_OR_QSUB_ATTEMPT')
    intents=[]
    for p in(base/'setup-rpc').glob('*-INTENT.json'):
        v=read(p);i=v.get('intent',v);intents.append(i.get('phase'))
    need('QSUB'not in intents and not(base/'setup-rpc/RPC_EFFECT_STATE.json').exists(),
         'RESUME_NO_QSUB_JOURNAL_OR_UNRESOLVED_ORIGINAL_MUTATION')
    prior_CPU=math.ceil(8*(final['finished_epoch']-wrap['epoch']))+60
    need(0<prior_CPU<=1350 and prior_CPU+1650<=3000,'RESUME_ORIGINAL_PREPAID3000_CPU_BANK_FULL_CONSERVATIVE_JOIN')
    return dict(status='EXACT_TERMINAL_OWNER_NEVER_ATTEMPTED_QSUB',request_sha256=REQUEST_SHA,
        old_owner_OS_exit_sha256=e.sha((root/'control/RESIDENT_OWNER_OS_EXIT.json').read_bytes()),
        old_owner_process_sha256=e.sha((base/'SERVER_OWNER_PROCESS.private.json').read_bytes()),
        prior_owner_CPU_upper_s=prior_CPU,new_owner_CPU_upper_s=1650,old_owner_and_wrapper_absent=True,
        original_request_not_recreated=True,original_setup_phases=intents)


def make_owner(root,g,*,projection):
    """Explicit resume constructor; the original fresh-only constructor is not called."""
    from . import minimal_native_owner as original,minimal_execution_adapter as adapters
    from .minimal_native_context import captured_LOCAL
    from scripts.pi0294.pv1 import mix_owner as mature,successor_cd
    root=Path(root);o=original.Owner.__new__(original.Owner);o.root=root;o.CONFIG=read(root/'MIX_OWNER_CONFIG.private.json');o.manifest=read(root/'RUN_MANIFEST.json');o.binding=read(root/'OUTPUT_BINDING.private.json')
    o.adapter=adapters.manifest(o.manifest);o.code=root/o.manifest['code']['capsule_relative'];seal=read(root/'SOURCE_SEAL.json')
    need(o.adapter is remaining and o.manifest['code']['vla_commit']==SOURCE
        and Path(original.__file__).resolve()==(o.code/'scripts/pi0294/generalization/minimal_native_owner.py').resolve()
        and e.sha((root/'SOURCE_SEAL.json').read_bytes())==o.manifest['code']['source_seal_sha256'],
        'RESUME_ORIGINAL_CLASS_AND_OWN_UNCHANGED_SCIENCE_CAPSULE')
    for name,h in seal['files'].items():need(e.sha((o.code/name).read_bytes())==h,'RESUME_SOURCE:'+name)
    for name,h in o.manifest['execution']['sealed_inputs'].items():need(e.sha((root/name).read_bytes())==h,'RESUME_INPUT:'+name)
    reg=[x for x in read(o.binding['registry'])['entries']if x.get('role')==o.binding['role']]
    need(len(reg)==1 and reg[0]['path']==str(root)and read(o.binding['lease'])['status']=='ACTIVE','RESUME_ORIGINAL_EXCLUSIVE_ACTIVE_ROLE')
    o.global_path=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';o.successor_scope=successor_cd.validated(g)
    o.BASE=root/'control/resident-owner';o.request_template=o.CONFIG['request_template'];raw_local=captured_LOCAL(root)
    need(projection==budget_LOCAL(g,json.loads(raw_local)),'RESUME_EXPLICIT_PROJECTION_OF_REAL_HISTORICAL_CAPTURE')
    o.local=e.raw(projection);o.retry_physical=None
    from .minimal_remaining_physical import Guard
    o.remaining_physical=Guard(root,o.request_template,o.manifest,lambda **_:None)
    o.last=None;o.current=None;o.cold_rpc_bound=True;o.preserve_readonly_until_deadline=True
    o.previous=SimpleNamespace(rpc=o.heartbeat_rpc);o.HEARTBEAT=original.HEARTBEAT
    o.setup=mature.new_journal(o.BASE,'setup-rpc',dict(run_uuid=root.name,job_id='NO_GPU_YET',source_sha=SOURCE,request_sha256=e.authority.GRANT_SHA));o.running=None
    return o


def run(control_root,secret):
    """One local detached control actor, then the original resident watch/finish."""
    from scripts.pi0292.durable import OwnerLock
    from scripts.pi0294.dec1.owner_transport import Journal
    from scripts.pi0294.dec1.source_cpu_once_io import commit
    from scripts.pi0294.dec1 import vanda_campaign as q,cold_owner
    from scripts.pi0294.pv1.evaluation_account_jobs import checked_outside
    from .minimal_native_context import captured_LOCAL
    from .minimal_native_launch import collect_effective_queue
    from . import minimal_remaining_dispatch as dispatch
    from scripts.pi0294.pv1 import mix_owner as mature
    control_root=Path(control_root);plan=read(control_root/'CONTROL_MANIFEST.json');root=Path(plan['science_root']);gp=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json'
    need(os.getuid()==22182 and os.environ.get('CUDA_VISIBLE_DEVICES')==''and not os.environ.get('PBS_JOBID')
        and e.sha(Path(__file__).read_bytes())==plan['controller_sha256']and plan['science_source_sha']==SOURCE,
        'RESUME_REGISTERED_METADATA_HOST_AND_EXACT_PUBLISHED_CONTROLLER')
    parent=Path(plan['registered_parent']);b=read(parent/'OUTPUT_BINDING.private.json')
    need(control_root.parent==parent/'control'and e.sha((parent/'RUN_MANIFEST.json').read_bytes())==plan['parent_manifest_sha256']
        and read(b['lease'])['status']=='ACTIVE_CONTROL_PLANE_ONLY','RESUME_ORIGINAL_REGISTERED_PARENT_CONTROL_OUTPUT')
    # New metadata preparation is in its distinct120s bank. The subsequent
    # original watch receives at most1650 further CPU seconds from the old
    # prepaid3000s bank after bounding the retired owner's use independently.
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));resource.setrlimit(resource.RLIMIT_CPU,(119,1770));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    lock=OwnerLock(root/'control/STAGE_SERVER_OWNER.lock');j=Journal(control_root,dict(run_uuid=root.name,job_id='EXACT_UNSUBMITTED_EXISTING_REQUEST',source_sha=plan['controller_source_sha'],request_sha256=REQUEST_SHA))
    def put(name,v):
        p=control_root/name;fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
        with os.fdopen(fd,'wb')as f:f.write(e.raw(v));f.flush();os.fsync(f.fileno())
    try:
        before=gp.read_bytes();g=json.loads(before);zero=exact_zero_qsub(root,g);full=q.collect_queue();view=predispatch(g,full,now_epoch=time.time())
        checked_outside(view['complete_queue_rows'],view['outside'],base64.b64decode(read(root/'MIX_OWNER_CONFIG.private.json')['outside_confirmation_base64']))
        scope_bytes=(control_root/'SCOPE.json').read_bytes();grant(scope_bytes);local=captured_LOCAL(root)
        proposed,proof=propose(g,local,scope_bytes,now_epoch=time.time(),zero_qsub_proof=zero);after=e.raw(proposed)
        need(gp.read_bytes()==before,'RESUME_CURRENT_CANONICAL_BEFORE_ONCE_CONTROL_CAS')
        key='ARX5-EXACT-REGISTERED-UNSUBMITTED-CONTROL-ONCE'
        result=j.call('CONTROL_ADOPT_ONCE','MUTATION',lambda intent,limit:commit(gp,expected_before_sha256=e.sha(before),proposed_bytes=after,request_id=key,intent_path=gp.with_name(key+'.INTENT.json'),receipt_path=gp.with_name(key+'.RECEIPT.json'),lock_path=gp.with_suffix('.owner.lock')))
        put('CONTROL_ADOPTED.json',dict(GLOBAL_sha256=e.sha(after),receipt=result,zero_qsub_proof=zero,all14=proof['dimensions']))
        ib=gp.with_name(key+'.INTENT.json').read_bytes();rb=read(gp.with_name(key+'.RECEIPT.json'));iv=json.loads(ib)
        need(gp.read_bytes()==after and rb['intent_sha256']==e.sha(ib)and rb['after_sha256']==iv['proposed_sha256']==e.sha(after)and not Path(iv['temporary']).exists(),
             'RESUME_INDEPENDENT_NATIVE_CONTROL_CAS_READBACK_NO_REPLAY')
        prepared_CPU=time.process_time();need(prepared_CPU<=120,'RESUME_DISTINCT_METADATA_CPU_BANK')
        base_CPU=math.floor(prepared_CPU);resource.setrlimit(resource.RLIMIT_CPU,(base_CPU+1640,base_CPU+1650))
        o=make_owner(root,proposed,projection=budget_LOCAL(proposed,json.loads(local)))
        put('CONTINUATION_PROCESS.json',dict(pid=os.getpid(),procstart=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],controller_source_sha=plan['controller_source_sha'],science_source_sha=SOURCE,CPU_upper_s=1650,original_role_transferred_not_old_process_restarted=True))
        project=o.auth(secret);quota=o.program('QUOTA');current=gp.read_bytes();g=json.loads(current);exact_zero_qsub(root,g)
        full=q.collect_queue();view=predispatch(g,full,now_epoch=time.time());checked_outside(view['complete_queue_rows'],view['outside'],base64.b64decode(o.CONFIG['outside_confirmation_base64']))
        future=e.scope.first512_forecast(current,o.local)
        need(future['cap_changes_required']=={}and quota['status']=='OFFICIAL_OWN_PROJECT_REPORT_AND_TARGET_FILESYSTEM_STABLE_READ'
            and quota['report_age_s']<=3600 and quota['available_bytes_lower']>=future['dimensions']['Hopper_storage_bytes']['projected']-g['new_storage_current_bytes'],
            'RESUME_FRESH_PROJECT_COMPLETE_RESOURCE_AND_TARGET_QUOTA')
        saved=read(o.BASE/'SOURCE_REQUEST.private.json');request=saved['request'];directory=Path(saved['directory'])
        fresh_queue=collect_effective_queue(walltime_s=request['walltime_s']);mature.once(directory/dispatch.FILE,fresh_queue)
        intent=dict(request_id=REQUEST,once=True,observed_epoch=time.time(),queue_proof_sha256=hashlib.sha256((directory/dispatch.FILE).read_bytes()).hexdigest())
        dispatch.check((directory/dispatch.FILE).read_bytes(),intent,request,now_epoch=time.time())
        body=chr(10).join(k+'='+repr(v)for k,v in dict(TARGET=str(root),DIRECTORY=str(directory),REQUEST=request,RESERVATION=dict(request_sha256=REQUEST_SHA,observed_epoch=time.time(),whole_package_already_reserved=True)).items())+chr(10)+o.CONFIG['programs']['SUBMIT']
        def qsub(intent,limit):
            env={k:v for k,v in os.environ.items()if k not in('CUDA_VISIBLE_DEVICES','PBS_JOBID')}
            p=subprocess.run([sys.executable,'-'],input=body,capture_output=True,text=True,timeout=min(limit,40),env=env)
            need(p.returncode==0,'RESUME_QSUB_UNKNOWN_NO_REPLAY');return json.loads(p.stdout)
        o.record('SOURCE_QSUB_INTENT',intent);reply=o.setup.call('QSUB','MUTATION',qsub);o.record('SOURCE_QSUB_REPLY',reply)
        import re
        need(reply['returncode']==0 and re.fullmatch(r'[0-9]+\.stdct-mgmt-02',reply['stdout']),
             'RESUME_EXACT_NATIVE_QSUB_RETURN_NOT_INFERRED_HANDLE')
        job=reply['stdout'];o.setup.call('BIND','MUTATION',lambda intent,limit:o.write_global('BIND',lambda g:remaining.bind(g,REQUEST,job,now_epoch=time.time())))
        o.record('SOURCE_JOB_BOUND',dict(job_id=job,request_id=REQUEST));o.record('TRAIN_REGISTERED',dict(root=str(root),manifest_sha256=MANIFEST))
        o.running=mature.new_journal(o.BASE,'running-rpc',dict(run_uuid=root.name,job_id=job,source_sha=SOURCE,request_sha256=REQUEST_SHA))
        put('UNIQUE_JOB_BOUND.json',dict(job_id=job,request_id=REQUEST,science_source_sha=SOURCE,GLOBAL_sha256=e.sha(gp.read_bytes()),new_PBS_counter_or_request=0))
        terminal=cold_owner.watch(o,secret,scope_binding=o.successor_scope);o.record('RESIDENT_TERMINAL_RETURNED',terminal);done=o.finish(terminal)
        o.record('RESIDENT_EXIT',dict(returncode=0,observed_epoch=time.time(),endpoint_status=done['status']));put('CONTINUATION_COMPLETE.json',dict(status=done['status'],job_id=job))
        return done
    except BaseException as exc:
        put('CONTINUATION_FAILED.json',dict(error_type=type(exc).__name__,message=str(exc)[-1800:],no_automatic_qsub_or_adopt_replay=True,observed_epoch=time.time()));raise
    finally:secret=None;lock.close()
