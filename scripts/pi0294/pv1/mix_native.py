"""New typed C/D allocation hook; reuse mature model, reader and Trainer.

No old lease, clock, arm or A/B endpoint is impersonated. CPU validation and
current native release precede torch/model import. Owner remains server-side.
"""
import base64,json,os,resource,subprocess,sys,time
from pathlib import Path
from scripts.pi0292.durable import read,write,OwnerLock
from scripts.pi0294.native_dataset import sha
from . import mix1024 as mix,mix_accounting as accounting,successor_cd as scope
from scripts.pi0294.pv1.pbs_owner import publish_release_record as once

STATUS='PASS_V5_CD_OWN_NATIVE_A40_ALLOCATION'

def fail(m):raise PermissionError('V5_CD_NATIVE_'+m)

def root_context(root,request_path,*,full=False):
    from scripts.pi0294.dec1.a40_training_context import sealed,verify_consumers
    root=Path(root);request_path=Path(request_path);m=sealed(root,'RUN_MANIFEST.json');b=sealed(root,'OUTPUT_BINDING.private.json');r=sealed(root,request_path.relative_to(root).as_posix());accounting.validate_request(r)
    g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json');marker=scope.validated(g);accounting.validate_global(g)
    if (m.get('scope')!=scope.TASK or m.get('platform_training_version')!=accounting.VERSION
        or m.get('run_uuid')!=root.name or m.get('output_root')!=str(root) or m['execution']['profile']!='vanda-neural-training'
        or b['uid']!=os.getuid() or b['run_root']!=str(root) or b['run_manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
        or r['run_manifest_sha256']!=sha(root/'RUN_MANIFEST.json') or r['source_sha']!=m['code']['vla_commit']
        or r['arm']!=m['science']['arm'] or r['T_CD']!=marker['T_CD'] or r['deadline_epoch']!=marker['deadline_epoch']
        or m['authority']['scope_binding_sha256']!=sha(root/'V5_CD_SCOPE_BINDING.json')
        or read(root/'V5_CD_SCOPE_BINDING.json')!=marker):fail('REGISTERED_TYPED_SCOPE_ROOT_AND_REQUEST')
    own=[x for x in g['PBS_submissions'] if x.get('request_id')==r['request_id']]
    if len(own)!=1 or own[0]['request_sha256']!=sha(request_path) or own[0].get('job_id')not in (None,os.environ.get('PBS_JOBID')):fail('ACTUAL_CENTRAL_RESERVATION_AND_BINDING')
    lease=read(b['lease']);rows=[v for v in read(b['registry'])['entries'] if v.get('role')==b['role']]
    if (lease['status']!='ACTIVE' or lease['run_uuid']!=root.name or lease['authority_release']!=scope.TASK_SOURCE
        or lease['deadline_epoch']!=marker['deadline_epoch'] or len(rows)!=1 or rows[0]['path']!=str(root)
        or not marker['T_CD']<=time.time()<marker['deadline_epoch']):fail('ACTUAL_SUCCESSOR_LEASE_REGISTRY_AND_CLOCK')
    for n,h in m['execution']['sealed_inputs'].items():
        if full or n in ('NEURAL_BINDINGS.private.json','plans/PV1_TRAIN_INPUTS.private.json','plans/PRIMARY_LEARNING_RECIPES.json','POLICY_IDENTITY.json','ASSET_CONSUMPTION.json'):
            p=sealed(root,n,parse=False)
            if sha(p)!=h:fail('SEALED_INPUT_CHANGED:'+n)
    seal=sealed(root,'SOURCE_SEAL.json');code=root/m['code']['capsule_relative']
    if sha(root/'SOURCE_SEAL.json')!=m['code']['source_seal_sha256'] or seal['source_sha']!=r['source_sha']:fail('EXACT_SOURCE_SEAL')
    if full:
        for n,h in seal['files'].items():
            p=sealed(root,str(Path(m['code']['capsule_relative'])/n),parse=False)
            if sha(p)!=h:fail('SOURCE_MEMBER_CHANGED:'+n)
    if Path(__file__).resolve()!=(code/'scripts/pi0294/pv1/mix_native.py').resolve():fail('EXECUTE_OWN_COLD_SOURCE_CAPSULE')
    bindings=sealed(root,'NEURAL_BINDINGS.private.json');env=sealed(root,'plans/A40_ENVIRONMENT_BINDING.private.json')
    if (Path(sys.executable).resolve()!=Path(env['python']).resolve() or Path(sys.prefix).resolve()!=Path(env['venv_prefix']).resolve()
        or sha(env['python'])!=env['python_binary_sha256'] or sha(env['environment_registry'])!=env['environment_registry_sha256']
        or env['official_source']!=bindings['source'] or env['official_commit']!='51d4c89f72fda44cbf77285c6a8114b52676b8a1'):fail('FROZEN_NATIVE_PYTHON_OFFICIAL_SOURCE_AND_ENV')
    verify_consumers(root);return m,r


def guard(root,allocation):
    from scripts.pi0294.mp1.owner_heartbeat import read_live_owner
    root=Path(root);binding=read(root/'OUTPUT_BINDING.private.json');lease=read(binding['lease']);g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json');m=scope.validated(g)
    if (lease['status']!='ACTIVE' or time.time()>=min(m['deadline_epoch'],allocation['allocation_end_epoch'])
        or (root/'STOP.json').exists() or (root.parent.parent/'STOP.json').exists()):fail('SUCCESSOR_STOP_LEASE_OR_ALLOCATED_TIME')
    beat=read_live_owner(root/'control',root.name)['heartbeat']
    if (beat.get('scope')!=scope.TASK or beat.get('grant_sha256')!=scope.GRANT_SHA
        or beat.get('job_id')!=allocation['job_id'] or beat.get('request_sha256')!=allocation['request_sha256']
        or beat.get('source_sha')!=allocation['source_sha'] or beat.get('native_job_state')!='R'
        or beat.get('global_unknown_effects')!=[] or g['unknown_effects']):fail('SAME_REAL_OWNER_HEARTBEAT_AND_GLOBAL_EFFECTS')
    accounting.validate_global(g)


def native_job(r):
    # Reuse exact own PBS/node checks and capacity parser. That primitive does
    # not call old scope/clock admission and has no model import.
    from scripts.pi0294.dec1.a40_training_runtime import native_job as original
    return original(r)


def require_training(root,allocation):
    if allocation.get('status')!=STATUS or os.environ.get('PBS_JOBID')!=allocation.get('job_id'):fail('OWN_TYPED_ALLOCATION_NOT_CPU_PREPARATION')
    root=Path(root);directory=root/'pbs'/('alloc-'+str(allocation['submit_count']).zfill(2));r=read(directory/'REQUEST.json')
    if sha(directory/'REQUEST.json')!=allocation['request_sha256'] or r['request_id']!=allocation['request_id']:fail('EXACT_TRAINING_REQUEST')
    guard(root,allocation)
    q=read(directory/'A40_PREMODEL_QUALIFICATION.json');v=read(root/'A40_COMPATIBILITY/RESULT.json')
    if (q.get('status')!='PASS_A40_NATIVE_SOURCE_PREMODEL_NO_CORE_TRANSPORT_USED' or q.get('job_id')!=allocation['job_id']
        or q.get('request_sha256')!=allocation['request_sha256'] or q.get('models_loaded')!=0 or q.get('queries')!=0 or q.get('physics')!=0
        or v.get('status')!='PASS_ACTUAL_SAME_JOB_17_SAVED_TRAIN_DRAWS_NOT_TASK_SUCCESS' or v.get('job_id')!=allocation['job_id']
        or v.get('request_sha256')!=allocation['request_sha256'] or v.get('parent_checkpoint_manifest_sha256')!=accounting.PARENT
        or v.get('statistics_sha256')!=mix.STATS or [x.get('index') for x in v.get('records',[])]!=list(range(17))):fail('ACTUAL_SAME_JOB_NATIVE_QUALIFICATION_AND_SEVENTEEN_DRAWS')
    for i,row in enumerate(v['records']):
        if row.get('seed')!=29400+i or row.get('finite_shapes_verified')is not True or sha(root/'A40_COMPATIBILITY'/('query-'+str(i).zfill(2)+'.npz'))!=row['arrays_sha256']:fail('ACTUAL_SAVED_TRAIN_QUERY_BYTES_BEFORE_OPTIMIZER')
    return directory


def wait_release(root,directory,allocation,*,first_update):
    root=Path(root);name='A40_PRETRAIN_RELEASE.json' if first_update else 'START_NEURAL.json';ready='A40_PRETRAIN_READY.json' if first_update else 'PBS_READY.json';start=time.time()
    while not(directory/name).is_file():
        if first_update:guard(root,allocation)
        else:
            b=read(root/'OUTPUT_BINDING.private.json');m=scope.validated(read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json'))
            if read(b['lease'])['status']!='ACTIVE' or (root/'STOP.json').exists() or time.time()>=m['deadline_epoch']:fail('STOP_BEFORE_FIRST_MODEL_RELEASE')
        if time.time()-start>=180:raise TimeoutError('V5_NATIVE_OWNER_RELEASE_MISSING_NO_IDLE_MODEL')
        time.sleep(.2)
    value=read(directory/name);req=read(directory/'REQUEST.json')
    if (value['request_sha256']!=sha(directory/'REQUEST.json') or value['job_id']!=os.environ['PBS_JOBID']
        or value['ready_sha256']!=sha(directory/ready) or value['scope_grant_sha256']!=scope.GRANT_SHA
        or not 0<=time.time()-value['observed_epoch']<=120):fail('ACTUAL_FRESH_NATIVE_RELEASE')
    proof=value['native_proof'];g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json');accounting.validate_global(g)
    if (proof['global_ledger_sha256']!=sha(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json')
        or proof['request_sha256']!=sha(directory/'REQUEST.json') or proof['whole_remaining_package_shortfalls']!={}
        or proof['native_own_job_id']!=os.environ['PBS_JOBID'] or proof['project_eligible']is not True
        or proof['actual_project_quota_covers_remaining_output']is not True
        or proof['complete_queue_verified']is not True):fail('REAL_FRESH_WHOLE_PACKAGE_SITE_QUEUE_AND_PROJECT_PROOF')
    import base64
    core_snapshot=read(root/'plans/V5_CD_PAIRED_PACKAGE_RESERVATION.private.json')
    local=json.loads(base64.b64decode(core_snapshot['Core_LOCAL_base64'],validate=True))
    if local.get(scope.FIELD)!=g.get(scope.FIELD) or core_snapshot['Core_LOCAL_sha256']!=accounting.sha(base64.b64decode(core_snapshot['Core_LOCAL_base64'],validate=True)):fail('ACTUAL_EXCLUSIVE_CORE_PACKAGE_ESCROW_NOT_A_CPU_FIXTURE')
    fresh=scope.recompute(g,local,Core_storage_upper_bytes=core_snapshot['Core_storage_upper'],local_storage_upper_bytes=core_snapshot['LOCAL_storage_upper'])
    if fresh['shortfalls']:fail('WORKER_INDEPENDENT_WHOLE_PACKAGE_RECOMPUTATION')
    native_job(req);return value


def authorize_training(root,directory,allocation,guard_callback):
    guard_callback();once(directory/'A40_PRETRAIN_READY.json',dict(job_id=allocation['job_id'],request_sha256=allocation['request_sha256'],observed_epoch=time.time(),updates=0))
    return wait_release(root,directory,allocation,first_update=True)


def prewrite(root):
    root=Path(root);used=0
    for p in root.rglob('*'):
        if p.is_file() and not p.is_symlink():used+=p.stat().st_size
    free=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
    if used+28000000000>70000000000 or free<28000000000:fail('FULL_STATE_SAVE_HEADROOM_NO_CHECKPOINT_DELETION')
    return dict(used_logical_bytes=used,next_full_save_upper=28000000000,free_bytes=free)


from scripts.pi0294.dec1.a40_native_ledger import Ledger as OriginalEffects
class Effects(OriginalEffects):
    def initialize(self):
        r=self.request
        if self.path.exists():raise FileExistsError('V5_NATIVE_EFFECT_JOURNAL_NO_RESTART')
        v=dict(schema='V5_CD_LOCAL_NATIVE_EFFECT_LEDGER_V1',platform_training_version=accounting.VERSION,run_uuid=self.root.name,job_id=self.allocation['job_id'],request_sha256=self.allocation['request_sha256'],source_sha=r['source_sha'],arm=r['arm'],start_native_step=0,optimizer_updates=0,generation_queries=0,model_loads=0,CPU_s=0.,pending_updates=[],unknown_effects=[],query_intent_links=[],training_runs={},global_budget_is_only_external_reservation=True)
        self.check(v);once(self.path,v);return v
    def check(self,v):
        r=self.request;arm=r['arm'];p=v.get('pending_updates')
        if (v.get('schema')!='V5_CD_LOCAL_NATIVE_EFFECT_LEDGER_V1' or v.get('platform_training_version')!=accounting.VERSION
            or v.get('arm')!=arm or v.get('source_sha')!=r['source_sha'] or v.get('request_sha256')!=self.allocation['request_sha256']
            or v.get('job_id')!=self.allocation['job_id'] or v.get('unknown_effects')!=[]
            or type(v.get('optimizer_updates'))is not int or not 0<=v['optimizer_updates']<=1024
            or type(v.get('generation_queries'))is not int or not 0<=v['generation_queries']<=29
            or type(v.get('model_loads'))is not int or not 0<=v['model_loads']<=4
            or len(v['query_intent_links'])!=v['generation_queries'] or not 0<=v['CPU_s']<=54000
            or type(p)is not list or len(p)>1 or set(v['training_runs'])-{arm}
            or v['training_runs'].get(arm,{}).get('last_native_step',0)!=v['optimizer_updates']):fail('EXACT_C_D_ALLOCATION_EFFECTS')
        if p and p!=[dict(arm=arm,run_key=arm,local_step_before=v['optimizer_updates'],request_id=r['request_id'])]:fail('EXACT_NATIVE_PENDING_UPDATE')
        if (p or v['optimizer_updates']) and v['generation_queries']<17:fail('SEVENTEEN_NATIVE_COMPATIBILITY_DRAWS_FIRST')
        return v
    def guard(self):guard(self.root,self.allocation);return self.check(read(self.path))
    def summary(self):
        v=super().summary();v.update(schema='V5_CD_ACTUAL_NATIVE_EFFECT_SUMMARY_V1',platform_training_version=accounting.VERSION);return v


def run(root,request_path):
    root=Path(root);request_path=Path(request_path);m,r=root_context(root,request_path,full=True);directory=request_path.parent;lock=OwnerLock(root/'A40_NATIVE_ALLOCATION_OWNER.lock');began=time.process_time();allocation=None;instance=None;ledger=None;result=None;phase='CPU_BOOTSTRAP';error=None
    if (directory/'PBS_PROCESS_FINAL.json').exists():raise FileExistsError('V5_NO_NATIVE_PROCESS_REPLAY')
    try:
        job_id,host,job=native_job(r);once(directory/'PBS_READY.json',dict(job_id=job_id,hostname=host,run_uuid=root.name,request_sha256=sha(request_path),observed_epoch=time.time(),models_loaded=0))
        wait_release(root,directory,{},first_update=False)
        from scripts.pi0294.e0_scope import pbs_epoch
        start=pbs_epoch(job['stime']);end=min(start+22500,r['deadline_epoch'])
        allocation=dict(status=STATUS,job_id=job_id,submit_count=r['submit_count'],request_id=r['request_id'],request_sha256=sha(request_path),source_sha=r['source_sha'],allocation_start_epoch=start,allocation_end_epoch=end,hostname=host,observed_epoch=time.time(),models_loaded=False)
        once(directory/'ALLOCATION_ADMISSION.json',allocation);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));resource.setrlimit(resource.RLIMIT_CPU,(53760,53760));resource.setrlimit(resource.RLIMIT_CORE,(0,0));prewrite(root)
        ledger=Effects(root,allocation);ledger.initialize();ledger.guard()
        from scripts.pi0294.numerics import prepare_environment,configure_torch
        prepare_environment(r['numerical_mode']);import torch
        if torch.cuda.device_count()!=1 or 'A40'not in torch.cuda.get_device_name(0):fail('ONE_ACTUAL_A40')
        configure_torch(torch,r['numerical_mode']);torch.set_num_threads(4)
        from scripts.pi0294.dec1.a40_training_context import verify_consumers
        closed=verify_consumers(root)
        once(directory/'A40_PREMODEL_QUALIFICATION.json',dict(status='PASS_A40_NATIVE_SOURCE_PREMODEL_NO_CORE_TRANSPORT_USED',job_id=job_id,request_sha256=sha(request_path),source_sha=r['source_sha'],native_GPU='NVIDIA_A40',visible_GPU_count=1,closed_consumers=closed,models_loaded=0,queries=0,physics=0))
        from scripts.pi0294.e1 import model
        from scripts.pi0294.e1.neural import checkpoint_identity
        identity=read(root/'POLICY_IDENTITY.json');consumer=read(root/'ASSET_CONSUMPTION.json');parent=Path(read(root/'plans/PV1_TRAIN_INPUTS.private.json')['CORR_parent_checkpoint'])
        if identity['checkpoint_manifest_sha256']!=accounting.PARENT or identity['statistics_sha256']!=mix.STATS or identity['files']!=consumer['inference_files_sha256'] or identity['global_step']!=2048 or consumer['original_checkpoint_manifest_sha256']!=accounting.PARENT:fail('ORIGINAL_CORR_PARENT_COMPLETE_INPUT_IDENTITY')
        phase='ACTUAL_COMPATIBILITY_17'
        ledger.charge('model_loads',1);instance=model.load(root,parent,identity,trainable=False)
        from scripts.pi0294.dec1.a40_training_runtime import compatibility
        compatibility(root,instance,allocation,ledger,ledger.guard);model.unload(instance);instance=None
        phase='MATCHED_1024_TRAINING';ledger.charge('model_loads',1);instance=model.load(root,parent,identity,trainable=True)
        from .mix_training import run as train
        result=train(root,instance,allocation,ledger.guard,ledger,arm=r['arm'],request_id=r['request_id'])
        model.unload(instance);instance=None
        if result['global_step']==1024:
            phase='FIXED_TWELVE_TRAIN_PROBES';cp=root/'training'/r['arm']/'checkpoint-1024';identity=checkpoint_identity(cp,r['arm']+'-1024',r['source_sha']);ledger.charge('model_loads',1);instance=model.load(root,cp,identity,trainable=False)
            from .mix_training import dataset
            from .probes import run as probes
            data=dataset(root,r['arm'],root/'evidence/fixed-probe-reader');result['fixed_probes']=probes(root,r['arm']+'-1024',instance,data,ledger,ledger.guard);model.unload(instance);instance=None
        phase='TERMINAL';return result
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e)[-2000:]);raise
    finally:
        if instance is not None and allocation is not None and time.time()<allocation['allocation_end_epoch']:
            try:
                from scripts.pi0294.e1.model import unload
                unload(instance)
            except BaseException:pass
        out=dict(status='V5_CD_PROCESS_SUCCESS' if error is None and result and result['global_step']==1024 else 'V5_CD_FAILED_OR_PARTIAL_PRESERVED',phase=phase,error=error,job_id=os.environ.get('PBS_JOBID'),source_sha=r['source_sha'],request_sha256=sha(request_path),result=result,CPU_actual_s=time.process_time()-began,new_physics=0,new_episodes=0)
        if ledger is not None:
            try:out['effects']=ledger.summary()
            except BaseException as e:out['effect_summary_error']=dict(type=type(e).__name__,message=str(e))
        once(directory/'PBS_PROCESS_FINAL.json',out);lock.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True);a=p.parse_args();run(a.root,a.request)
