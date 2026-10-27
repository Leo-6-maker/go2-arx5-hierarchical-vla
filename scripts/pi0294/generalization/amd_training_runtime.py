"""Registered GEN1 A40 entry, using the original model/Trainer/probe methods."""
import base64,json,os,resource,subprocess,sys,time
from pathlib import Path
from scripts.pi0292.durable import read,OwnerLock
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.pbs_owner import publish_release_record as once
from scripts.pi0294.dec1.a40_training_context import sealed,verify_consumers
from scripts.pi0294.dec1.a40_training_runtime import native_job,compatibility
from scripts.pi0294.dec1.a40_native_ledger import Ledger as OriginalEffects
from scripts.pi0294.pv1.mix_native import prewrite
from . import amd_training_accounting as a,amd_training as training,amd_adoption as authority

STATUS='PASS_GEN1_OWN_NATIVE_A40_ALLOCATION'
need=a.need

class StartupOwnerStillQueued(Exception):
    """Same fresh bound owner, but it has not yet observed native R."""

def bank_gate(root):
    root=Path(root);i=read(root/'plans/GEN1_TRAIN_INPUTS.private.json');bank=Path(i['dataset'])
    need(bank.resolve()==(root/'datasets/FROZEN_GEN1_BANK').resolve()and not bank.is_symlink(),
        'TRAIN_OWN_PREDISPATCH_BANK_OUTPUT_ROOT')
    p=read(root/'evidence/GEN1_PREMODEL_BANK_READY.json');r=read(bank/'READY.json')
    need(p['status']=='PASS_ACTUAL_SAVED_GEN1_PREMODEL_BANK_PACK'and p['ready']==r
        and all(type(p[k])is int and p[k]==0 for k in ('model_loads','queries','updates','constructors','physics_ticks','Teacher_episodes'))
        and r['dataset_manifest_sha256']==i['bank_manifest_sha256']==sha(bank/'DATASET_MANIFEST.json')
        and r['payload_manifest_sha256']==sha(bank/'PAYLOAD_MANIFEST.json')
        and r['cross_parent_qualification_sha256']==sha(bank/'CROSS_PARENT_QUALIFICATION.json'),
        'TRAIN_ACTUAL_COMPLETE_CPU_BANK_BEFORE_ANY_MODEL_OR_OPTIMIZER')
    from .amd_bank import validate
    validate(read(bank/'DATASET_MANIFEST.json'),(root/'plans/GEN1_V5_PREVIEW_V2.private.json').read_bytes())
    for row in read(bank/'PAYLOAD_MANIFEST.json'):
        q=bank/row['path'];need(not q.is_symlink()and q.resolve().is_relative_to(bank.resolve())
            and q.is_file()and q.stat().st_uid==os.getuid()and q.stat().st_mode&0o777==0o400
            and q.stat().st_size==row['bytes']and sha(q)==row['sha256'],'TRAIN_SEALED_ACTUAL_BANK_PAYLOAD')
    return r

def root_context(root,request_path,*,full=False):
    root=Path(root);request_path=Path(request_path);m=sealed(root,'RUN_MANIFEST.json');b=sealed(root,'OUTPUT_BINDING.private.json')
    r=sealed(root,request_path.relative_to(root).as_posix());a.validate_request(r)
    g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json');mark=authority.validate(g);rows=a.rows(g)
    need(m['scope']==a.VERSION and m['platform_training_version']==a.VERSION
        and m['science']['arm']==r['arm']==training.bank.IDENTITY
        and m['execution']['profile']==a.PROFILE and m['execution']['agent_liveness_required']is False
        and m['output_root']==b['run_root']==str(root)and m['run_uuid']==root.name
        and b['uid']==os.getuid()==22182 and b['run_manifest_sha256']==sha(root/'RUN_MANIFEST.json')==r['run_manifest_sha256']
        and m['code']['vla_commit']==r['source_sha']and r['T_CD']==mark['T_CD']
        and r['deadline_epoch']==mark['deadline_epoch'],'TRAIN_OWN_STANDARD_EXPLICIT_GEN_MANIFEST')
    own=[x for x in rows if x['request_id']==r['request_id']]
    need(len(own)==1 and own[0]['request_sha256']==sha(request_path)
        and own[0]['job_id']==os.environ.get('PBS_JOBID')and own[0]['status']==a.STATES[1],
        'TRAIN_ACTUAL_CENTRAL_GEN_RESERVATION_AND_JOB_BIND')
    lease=read(b['lease']);reg=[x for x in read(b['registry'])['entries']if x.get('role')==b['role']]
    need(lease['status']=='ACTIVE'and lease['run_uuid']==root.name and lease['manifest_sha256']==r['run_manifest_sha256']
        and lease['deadline_epoch']==r['deadline_epoch']and len(reg)==1 and reg[0]['path']==str(root)
        and mark['T_CD']<=time.time()<mark['deadline_epoch'],'TRAIN_REAL_OWN_LEASE_REGISTRY_AND_ORIGINAL_CLOCK')
    profiles=[x for x in read(b['environment'])['profiles']if x['name']==a.PROFILE]
    need(len(profiles)==1 and profiles[0]['manifest']==str(root/'RUN_MANIFEST.json')
        and profiles[0]['entry_module']=='scripts.pi0294.generalization.amd_training_runtime'
        and profiles[0]['PBS_required']is True,'TRAIN_REGISTERED_EXPLICIT_GEN_ENVIRONMENT_PROFILE')
    code=root/m['code']['capsule_relative'];seal=sealed(root,'SOURCE_SEAL.json')
    need(sha(root/'SOURCE_SEAL.json')==m['code']['source_seal_sha256']and seal['source_sha']==r['source_sha']
        and Path(__file__).resolve()==(code/'scripts/pi0294/generalization/amd_training_runtime.py').resolve(),
        'TRAIN_EXECUTE_OWN_IMMUTABLE_SOURCE')
    for n,h in m['execution']['sealed_inputs'].items():need(sha(sealed(root,n,parse=False))==h,'TRAIN_SEALED_INPUT:'+n)
    if full:
        for n,h in seal['files'].items():need(sha(sealed(root,str(Path(m['code']['capsule_relative'])/n),parse=False))==h,'TRAIN_SOURCE:'+n)
    env=sealed(root,'plans/A40_ENVIRONMENT_BINDING.private.json');bindings=sealed(root,'NEURAL_BINDINGS.private.json')
    need(Path(sys.executable).resolve()==Path(env['python']).resolve()and Path(sys.prefix).resolve()==Path(env['venv_prefix']).resolve()
        and sha(env['python'])==env['python_binary_sha256']and sha(env['environment_registry'])==env['environment_registry_sha256']
        and env['official_source']==bindings['source']and env['official_commit']=='51d4c89f72fda44cbf77285c6a8114b52676b8a1',
        'TRAIN_EXACT_NATIVE_PYTHON_AND_OFFICIAL_ENVIRONMENT')
    a.need(sha(root/'plans/GEN1_CLOSED_EXPORT_INPUT.private.json')==r['closed_export_sha256'],
        'TRAIN_SEALED_SAVED_INPUT_CLOSURE')
    closed=sealed(root,'plans/GEN1_CLOSED_EXPORT_INPUT.private.json');source=Path(closed['root'])
    source_binding=read(source/'OUTPUT_BINDING.private.json')
    need(not source.is_symlink()and source.resolve().is_relative_to(root.parent.resolve())
        and sha(source/'RUN_MANIFEST.json')==closed['manifest_sha256']
        and sha(source_binding['lease'])==closed['lease_sha256']and read(source_binding['lease'])['status']=='CLOSED'
        and sha(source/'RESOURCE_LEDGER.json')==closed['all_captured_files_sha256']['RESOURCE_LEDGER.json'],
        'TRAIN_FRESH_ORIGINAL_CLOSED_SOURCE_ROLE_AND_PRESERVED_FAILED_COUNTER')
    verify_consumers(root);return m,r

def guard(root,allocation,*,startup=False):
    from scripts.pi0294.mp1.owner_heartbeat import read_live_owner
    root=Path(root);g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json');mark=authority.validate(g);a.rows(g)
    b=read(root/'OUTPUT_BINDING.private.json');lease=read(b['lease'])
    need(lease['status']=='ACTIVE'and time.time()<min(mark['deadline_epoch'],allocation['allocation_end_epoch'])
        and not any(p.exists()for p in (root/'STOP.json',root.parent.parent/'STOP.json'))
        and g['unknown_effects']==g['pending_updates']==[],'TRAIN_LIVE_STOP_LEASE_GLOBAL_AND_ALLOCATED_TIME')
    beat=read_live_owner(root/'control',root.name)['heartbeat']
    need(beat.get('scope')==a.VERSION and beat.get('grant_sha256')==authority.GRANT_SHA
        and beat.get('job_id')==allocation['job_id']and beat.get('request_sha256')==allocation['request_sha256']
        and beat.get('source_sha')==allocation['source_sha']
        and beat.get('global_unknown_effects')==[],'TRAIN_CURRENT_NATIVE_SERVER_OWNER_HEARTBEAT')
    if startup and beat.get('native_job_state')=='Q':raise StartupOwnerStillQueued()
    need(beat.get('native_job_state')=='R','TRAIN_CURRENT_RUNNING_NATIVE_SERVER_OWNER_HEARTBEAT')

def wait_initial_running_owner(root,allocation,*,clock=time.monotonic,sleep=time.sleep):
    # Only a missing first publication or an otherwise exact/fresh Q receipt
    # may wait. Wrong identity, STOP, stale300s heartbeat and unknown effects
    # still fail immediately. No CPU pack/model/query/update is released by Q.
    root=Path(root);until=clock()+180
    while True:
        try:guard(root,allocation,startup=True);return
        except (FileNotFoundError,StartupOwnerStillQueued):
            b=read(root/'OUTPUT_BINDING.private.json')
            need(read(b['lease'])['status']=='ACTIVE'and not any(p.exists()for p in
                (root/'STOP.json',root.parent.parent/'STOP.json'))and time.time()<allocation['allocation_end_epoch'],
                'TRAIN_STOP_WHILE_AWAITING_FIRST_NATIVE_OWNER')
            if clock()>=until:raise TimeoutError('GEN1_NO_FIRST_RUNNING_OWNER_WITHIN_ORIGINAL_180S')
            sleep(.2)

def wait_release(root,directory,allocation,*,first_update):
    name='A40_PRETRAIN_RELEASE.json'if first_update else'START_NEURAL.json'
    ready='A40_PRETRAIN_READY.json'if first_update else'PBS_READY.json';start=time.monotonic();root=Path(root)
    while not(directory/name).is_file():
        guard(root,allocation)
        if time.monotonic()-start>=180:raise TimeoutError('GEN1_NATIVE_OWNER_RELEASE_MISSING_NO_IDLE_MODEL')
        time.sleep(.2)
    r=read(directory/'REQUEST.json');v=read(directory/name);g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json')
    need(v['job_id']==os.environ['PBS_JOBID']and v['request_sha256']==sha(directory/'REQUEST.json')
        and v['ready_sha256']==sha(directory/ready)and v['scope_grant_sha256']==authority.GRANT_SHA
        and 0<=time.time()-v['observed_epoch']<=120,'TRAIN_ACTUAL_FRESH_ONCE_RELEASE')
    p=v['native_proof'];need(p['global_ledger_sha256']==sha(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json')
        and p['request_sha256']==sha(directory/'REQUEST.json')and p['whole_remaining_package_shortfalls']=={}
        and p['native_own_job_id']==os.environ['PBS_JOBID']and p['project_eligible']is True
        and p['actual_project_quota_covers_remaining_output']is True and p['complete_queue_verified']is True,
        'TRAIN_FRESH_NATIVE_FULL_ACCOUNT_PROJECT_AND_WHOLE_FUTURE_PROOF')
    local=base64.b64decode(read(root/'plans/GEN1_OWNER_RESOURCES.private.json')['historical_LOCAL_base64'],validate=True)
    need(sha_bytes(local)==authority.validate(g)['historical_LOCAL_sha256'],'TRAIN_HISTORICAL_LOCAL_NOT_FAKE_FRESH')
    proof=a.scope.full_forecast(a.raw(g),local);need(not proof['cap_changes_required'],'TRAIN_WORKER_FULL_REMAINING_RECOMPUTATION')
    bank_gate(root);native_job(r);return v

def sha_bytes(b):return a.sha(b)

def await_actual_bind(root,request_path):
    root=Path(root);r=sealed(root,Path(request_path).relative_to(root).as_posix());a.validate_request(r)
    job,_,_=native_job(r);until=time.monotonic()+180
    while True:
        g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json');rows=a.rows(g)
        own=[x for x in rows if x['request_id']==r['request_id']]
        need(len(own)==1 and own[0]['request_sha256']==sha(request_path)
            and own[0]['job_id']in(None,job),'TRAIN_SAME_KNOWN_RESERVATION_AWAITING_NATIVE_BIND')
        if own[0]['job_id']==job and own[0]['status']==a.STATES[1]:return
        b=read(root/'OUTPUT_BINDING.private.json')
        need(read(b['lease'])['status']=='ACTIVE'and not any(p.exists()for p in
            (root/'STOP.json',root.parent.parent/'STOP.json'))and time.time()<r['deadline_epoch'],
            'TRAIN_ORIGINAL_STOP_WHILE_AWAITING_BIND')
        if time.monotonic()>=until:raise TimeoutError('GEN1_NATIVE_BIND_MISSING_NO_MODEL')
        time.sleep(.2)

def require_training(root,allocation):
    root=Path(root);need(allocation['status']==STATUS and allocation['job_id']==os.environ.get('PBS_JOBID'),'TRAIN_ACTUAL_OWN_ALLOCATION')
    guard(root,allocation);bank_gate(root);d=root/'pbs'/('alloc-'+str(allocation['submit_count']).zfill(2))
    need(sha(d/'REQUEST.json')==allocation['request_sha256'],'TRAIN_EXACT_CURRENT_REQUEST')
    q=read(d/'A40_PREMODEL_QUALIFICATION.json');v=read(root/'A40_COMPATIBILITY/RESULT.json')
    need(q['job_id']==allocation['job_id']and q['request_sha256']==allocation['request_sha256']and q['models_loaded']==0
        and v['status']=='PASS_ACTUAL_SAME_JOB_17_SAVED_TRAIN_DRAWS_NOT_TASK_SUCCESS'
        and v['job_id']==allocation['job_id']and v['request_sha256']==allocation['request_sha256']
        and v['parent_checkpoint_manifest_sha256']==a.original.PARENT and v['statistics_sha256']==training.STATS
        and [x['index']for x in v['records']]==list(range(17)),'TRAIN_ORIGINAL_ACTUAL_SAME_JOB_SEVENTEEN_COMPATIBILITY')
    for i,row in enumerate(v['records']):need(row['seed']==29400+i and row['finite_shapes_verified']is True
        and sha(root/'A40_COMPATIBILITY'/('query-'+str(i).zfill(2)+'.npz'))==row['arrays_sha256'],'TRAIN_REAL_COMPATIBILITY_ARRAYS')
    return d

def authorize_training(root,directory,allocation,guard_callback):
    guard_callback();once(directory/'A40_PRETRAIN_READY.json',dict(job_id=allocation['job_id'],
        request_sha256=allocation['request_sha256'],observed_epoch=time.time(),updates=0))
    return wait_release(root,directory,allocation,first_update=True)

class Effects(OriginalEffects):
    def initialize(self):
        v=dict(schema='GEN1_LOCAL_NATIVE_EFFECT_LEDGER_V1',platform_training_version=a.VERSION,
            run_uuid=self.root.name,job_id=self.allocation['job_id'],request_sha256=self.allocation['request_sha256'],
            source_sha=self.request['source_sha'],arm=training.bank.IDENTITY,start_native_step=0,optimizer_updates=0,
            generation_queries=0,model_loads=0,CPU_s=0.,pending_updates=[],unknown_effects=[],query_intent_links=[],
            training_runs={},global_budget_is_only_external_reservation=True)
        self.check(v);once(self.path,v);return v
    def check(self,v):
        r=self.request
        need(v.get('schema')=='GEN1_LOCAL_NATIVE_EFFECT_LEDGER_V1'and v.get('platform_training_version')==a.VERSION
            and v['job_id']==self.allocation['job_id']and v['request_sha256']==self.allocation['request_sha256']
            and v['source_sha']==r['source_sha']and v['arm']==training.bank.IDENTITY and v['start_native_step']==0
            and v['unknown_effects']==[]and type(v['optimizer_updates'])is int and 0<=v['optimizer_updates']<=1024
            and type(v['generation_queries'])is int and 0<=v['generation_queries']<=29
            and type(v['model_loads'])is int and 0<=v['model_loads']<=4
            and len(v['query_intent_links'])==v['generation_queries']and type(v['CPU_s'])in(int,float)
            and 0<=v['CPU_s']<=54000 and type(v['pending_updates'])is list and len(v['pending_updates'])<=1
            and set(v['training_runs'])<={training.bank.IDENTITY}
            and v['training_runs'].get(training.bank.IDENTITY,{}).get('last_native_step',0)==v['optimizer_updates'],
            'TRAIN_EXPLICIT_GEN_NATIVE_EFFECT_COUNTERS')
        if v['pending_updates']:need(v['pending_updates']==[dict(arm=training.bank.IDENTITY,run_key=training.bank.IDENTITY,
            local_step_before=v['optimizer_updates'],request_id=r['request_id'])],'TRAIN_SINGLE_EXACT_PENDING_STEP')
        if v['pending_updates']or v['optimizer_updates']:need(v['generation_queries']>=17,'TRAIN_NO_UPDATE_BEFORE_COMPATIBILITY')
        return v
    def guard(self):guard(self.root,self.allocation);return self.check(read(self.path))
    def summary(self):
        v=super().summary();v.update(schema=a.EFFECTS,platform_training_version=a.VERSION);return v

def run(root,request_path):
    root=Path(root);request_path=Path(request_path);await_actual_bind(root,request_path)
    m,r=root_context(root,request_path,full=True);directory=request_path.parent
    lock=OwnerLock(root/'A40_NATIVE_ALLOCATION_OWNER.lock');began=time.process_time();ledger=None;instance=None;allocation=None;result=None;error=None;phase='CPU_BANK_PRELUDE'
    need(not(directory/'PBS_PROCESS_FINAL.json').exists(),'TRAIN_NO_NATIVE_PROCESS_REPLAY')
    try:
        job,host,pbs=native_job(r)
        from scripts.pi0294.e0_scope import pbs_epoch
        start=pbs_epoch(pbs['stime']);allocation=dict(status=STATUS,job_id=job,hostname=host,submit_count=r['submit_count'],
            request_id=r['request_id'],request_sha256=sha(request_path),source_sha=r['source_sha'],
            allocation_start_epoch=start,allocation_end_epoch=min(start+22500,r['deadline_epoch']))
        os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));resource.setrlimit(resource.RLIMIT_CPU,(53160,53160));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        once(directory/'ALLOCATION_ADMISSION.json',allocation);prewrite(root)
        # Begin the effect journal at the real reserved allocation boundary.
        # Zero counters grant no model permission. Even a pre-model failure
        # now leaves an authoritative known-effect summary for the owner.
        ledger=Effects(root,allocation);ledger.initialize()
        # Qsub can begin computation before the login owner has finished bind
        # and its first complete-queue heartbeat. Wait under the original
        # pre-model180s bound; never invent or extend a heartbeat.
        wait_initial_running_owner(root,allocation)
        # The original CPU guard masks CUDA and forbids torch Module.forward
        # process-wide. Keep it in a bounded child; it must never contaminate
        # the later unchanged neural Trainer in this parent process.
        need(time.time()+600<r['CPU_bank_deadline_epoch'],'TRAIN_ORIGINAL_CPU_PACK_WINDOW_NOT_REOPENED')
        once(directory/'CPU_BANK_PRELUDE_INTENT.json',dict(job_id=job,request_sha256=sha(request_path),
            maximum_wall_s=600,maximum_CPU_s=600,neural_permission=False,automatic_retry=False))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (directory/'CPU_BANK_PRELUDE.stdout').open('xb')as out,(directory/'CPU_BANK_PRELUDE.stderr').open('xb')as err:
            p=subprocess.run([sys.executable,'-u','-m','scripts.pi0294.generalization.amd_training_pack',
                '--root',str(root),'--request',str(request_path)],env=env,stdout=out,stderr=err,timeout=600)
        once(directory/'CPU_BANK_PRELUDE_OS_EXIT.json',dict(job_id=job,returncode=p.returncode,request_sha256=sha(request_path)))
        need(p.returncode==0,'TRAIN_CPU_BANK_PRELUDE_FAILED_NO_MODEL_OR_REPLAY');bank_gate(root)
        from .amd_export_repair_runtime import tools
        tools(run_git=True)
        once(directory/'PBS_READY.json',dict(job_id=job,hostname=host,run_uuid=root.name,request_sha256=sha(request_path),
            observed_epoch=time.time(),models_loaded=0,actual_bank_ready_sha256=sha(root/'datasets/FROZEN_GEN1_BANK/READY.json')))
        wait_release(root,directory,allocation,first_update=False);ledger.guard()
        from scripts.pi0294.numerics import prepare_environment,configure_torch
        prepare_environment(r['numerical_mode']);import torch
        need(torch.cuda.device_count()==1 and 'A40'in torch.cuda.get_device_name(0),'TRAIN_ONE_ACTUAL_A40')
        configure_torch(torch,r['numerical_mode']);torch.set_num_threads(4)
        once(directory/'A40_PREMODEL_QUALIFICATION.json',dict(status='PASS_GEN1_A40_PREMODEL_WITH_ACTUAL_BANK',job_id=job,
            request_sha256=sha(request_path),models_loaded=0,queries=0,physics=0,closed_consumers=verify_consumers(root)))
        from scripts.pi0294.e1 import model
        from scripts.pi0294.e1.neural import checkpoint_identity
        identity=read(root/'POLICY_IDENTITY.json');consumer=read(root/'ASSET_CONSUMPTION.json');parent=Path(read(root/'plans/PV1_TRAIN_INPUTS.private.json')['CORR_parent_checkpoint'])
        need(identity['checkpoint_manifest_sha256']==a.original.PARENT and identity['global_step']==2048
            and identity['statistics_sha256']==training.STATS and identity['files']==consumer['inference_files_sha256']
            and consumer['original_checkpoint_manifest_sha256']==a.original.PARENT,'TRAIN_COMMON_CORR2048_NO_C_D_INHERITANCE')
        phase='ACTUAL_COMPATIBILITY_17';ledger.charge('model_loads',1);instance=model.load(root,parent,identity,trainable=False)
        compatibility(root,instance,allocation,ledger,ledger.guard);model.unload(instance);instance=None
        phase='GEN1_INDEPENDENT_1024_TRAINING';ledger.charge('model_loads',1);instance=model.load(root,parent,identity,trainable=True)
        result=training.run(root,instance,allocation,ledger.guard,ledger,arm=r['arm'],request_id=r['request_id'])
        model.unload(instance);instance=None
        if result['global_step']==1024:
            phase='ORIGINAL_FIXED_TWELVE_TRAIN_PROBES';cp=root/'training'/r['arm']/'checkpoint-1024'
            identity=checkpoint_identity(cp,r['arm']+'-1024',r['source_sha']);ledger.charge('model_loads',1);instance=model.load(root,cp,identity,trainable=False)
            # Original fixed benchmark uses the original COV reader. GEN data
            # must not silently replace/reselect any of these twelve probes.
            from scripts.pi0294.pv1.mix_training import dataset as original_probe_reader
            from scripts.pi0294.pv1.probes import run as probes
            data=original_probe_reader(root,'C50',root/'evidence/fixed-probe-reader')
            result['fixed_probes']=probes(root,r['arm']+'-1024',instance,data,ledger,ledger.guard);model.unload(instance);instance=None
        phase='TERMINAL';return result
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc)[-2000:]);raise
    finally:
        if instance is not None and allocation is not None and time.time()<allocation['allocation_end_epoch']:
            try:
                from scripts.pi0294.e1.model import unload
                unload(instance)
            except BaseException:pass
        final=dict(status='GEN1_PROCESS_SUCCESS'if error is None and result and result['global_step']==1024 else'GEN1_FAILED_OR_PARTIAL_PRESERVED',
            phase=phase,error=error,job_id=os.environ.get('PBS_JOBID'),source_sha=r['source_sha'],request_sha256=sha(request_path),
            result=result,CPU_actual_s=time.process_time()-began,new_physics=0,new_episodes=0)
        if ledger is not None:
            try:final['effects']=ledger.summary()
            except BaseException as exc:final['effect_summary_error']=dict(type=type(exc).__name__,message=str(exc))
        once(directory/'PBS_PROCESS_FINAL.json',final);lock.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True);v=p.parse_args();run(v.root,v.request)
