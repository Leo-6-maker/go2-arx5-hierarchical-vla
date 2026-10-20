"""Native A40 saved-TRAIN compatibility, not a task or model service.

The mature A40 probe still loads and generates. This adapter supplies current
DEC1 identity, a same-job release/heartbeat fence and append-only effect logs.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.pbs_owner import publish_release_record
from scripts.pi0294.pv1.visibility import refresh
from scripts.pi0294.e0_scope import pbs_epoch
from .authority import AUTHORITY,PARENT_AUTHORITY,PARENT,R0,HARD_CLOSE
from .alternate_accounting import VERSION,QUERIES,WALL,PAD,CPU_UPPER,digest

CONSUMER='64c7ba476573866dd6fd94e32a5136f1828970eb6ffc9c26c813d0b76761e8f0'
READY='688b41a7e2ceb7e89b61b00f58b4f478cb46af4fae942bcb3c512fe08f482e0d'
PBS='/opt/pbs/bin/'


def fail(message):
    raise PermissionError('DEC1_COMPAT_'+message)


def sealed(path):
    path=Path(path)
    if (path.is_symlink() or not path.is_file() or path.stat().st_uid!=os.getuid()
            or path.stat().st_mode&0o777!=0o400):
        fail('OWN_SEALED_RECORD_REQUIRED')
    return read(path)


def validate(root):
    root=Path(root);m=sealed(root/'RUN_MANIFEST.json');b=sealed(root/'OUTPUT_BINDING.private.json')
    r=sealed(root/'REQUEST.json');lease=read(m['lease'])
    if (m.get('dec1_qualification_version')!=VERSION or m.get('dec1_request_id')!=r.get('request_id')
            or m.get('authority',{}).get('protocol_ref')!=AUTHORITY
            or m.get('parent_authority_release')!=PARENT_AUTHORITY
            or m.get('campaign_run_uuid')!='20261018T182956Z-localpc-pv1'
            or m.get('original_T0_epoch')!=1792348196
            or m.get('output_root')!=str(root) or m.get('run_uuid')!=root.name
            or m.get('owner_uid')!=os.getuid() or m.get('owner_user')!='e1192548'
            or m.get('deadline_epoch')!=HARD_CLOSE or time.time()>=HARD_CLOSE
            or not root.resolve().is_relative_to(Path(m['registered_runs_root']).resolve())
            or root.is_symlink() or root.stat().st_uid!=os.getuid()
            or b.get('run_manifest_sha256')!=sha(root/'RUN_MANIFEST.json')
            or b.get('uid')!=os.getuid() or b.get('lease')!=m['lease']
            or lease.get('status')!='ACTIVE' or lease.get('run_uuid')!=root.name
            or lease.get('authority_release')!=AUTHORITY or lease.get('original_T0_epoch')!=1792348196
            or lease.get('deadline_epoch')!=HARD_CLOSE
            or m.get('limits')!=dict(ngpus=1,walltime_s=WALL,PBS_submissions=1,queries=QUERIES,
                active_cpu=4,robot_physics_steps=0,optimizer_steps=0)
            or r.get('scientific_admission_version')!=VERSION or r.get('source_sha')!=m['code']['vla_commit']
            or r.get('authority_release')!=PARENT_AUTHORITY or r.get('authority_amendment')!=AUTHORITY
            or r.get('original_T0_epoch')!=1792348196 or r.get('hard_close_epoch')!=HARD_CLOSE
            or r.get('ncpus')!=4 or r.get('memory_gb')!=32 or r.get('submit_count')!=12
            or r.get('model_service_or_training_or_physics') is not False
            or r.get('run_uuid')!=root.name or r.get('walltime_s')!=WALL
            or r.get('campaign_run_uuid')!=m['campaign_run_uuid']
            or (root/'STOP.json').exists()):
        fail('FIXED_DEC1_EXISTING_OWNER_SCOPE_REGISTRY_LEASE_AND_CLOCK')
    registry=read(b['registry']);rows=[v for v in registry['entries'] if v.get('role')==b['role']]
    if len(rows)!=1 or Path(rows[0]['path']).resolve()!=root.resolve():fail('ACTUAL_REGISTERED_OUTPUT')
    for name,expected in m['sealed_files'].items():
        p=Path(name)
        if (p.is_symlink() or not p.resolve().is_relative_to(root.resolve())
                or p.stat().st_mode&0o222 or sha(p)!=expected):fail('IMMUTABLE_CAPSULE_AND_SMALL_INPUT_BYTES')
    if not {str(root/n) for n in ('REQUEST.json','POLICY_IDENTITY.json','ASSET_CONSUMPTION.json',
                                  'HF_READY.json','PROBE_OBSERVATION.private.json')}<=set(m['sealed_files']):
        fail('EXPLICIT_REQUEST_AND_POLICY_IDENTITY_SEALS')
    code=root/m['code']['capsule_relative']
    if Path(__file__).resolve()!=(code/'scripts/pi0294/dec1/alternate_runtime.py').resolve():
        fail('EXECUTE_OWN_FIXED_SOURCE_CAPSULE_NOT_MUTABLE_CHECKOUT')
    identity=sealed(root/'POLICY_IDENTITY.json');consumer=sealed(root/'ASSET_CONSUMPTION.json')
    if (sha(root/'ASSET_CONSUMPTION.json')!=CONSUMER or sha(root/'HF_READY.json')!=READY
            or identity.get('checkpoint_manifest_sha256')!=PARENT or identity.get('statistics_sha256')!=R0
            or identity.get('global_step')!=2048 or identity.get('files')!=consumer['inference_files_sha256']
            or consumer.get('status')!='PASS_FULL_CONSUMER_BYTES_AND_REUSED_FROZEN_SHARD_NO_MODEL'
            or m['checkpoint']!=read(m['asset_progress'])['checkpoint']
            or m['identity']!=str(root/'POLICY_IDENTITY.json')
            or m['observation']!=str(root/'PROBE_OBSERVATION.private.json')
            or m['stage0_result']!=str(root/'SAME_JOB_HARDWARE.json')):
        fail('EXACT_CURRENT_CORR_R0_AND_ACTUAL_CLOSED_ASSET_CONSUMER')
    from scripts.pi0292.relay_protocol import validate_identity
    validate_identity(identity['relay_identity'])
    f=identity['files'];canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
    expected=dict(profile='B1',model_revision=canonical(f)[:40],source_sha=r['source_sha'],
        config_sha256=f['config.json'],processor_sha256=canonical({n:h for n,h in f.items() if n.startswith('processor/')}),
        statistics_sha256=R0)
    if identity['relay_identity']!=expected:fail('RELAY_IDENTITY_DERIVED_FROM_EXACT_CURRENT_FILES')
    saved=sealed(root/'PROBE_OBSERVATION.private.json');provenance=saved['provenance'];payload=saved['payload']
    from scripts.pi0292.eval_contract import FRAME,TCP
    from scripts.pi0294.e0_scope import DATA
    if (provenance.get('dataset_manifest_sha256')!=DATA or provenance.get('statistics_sha256')!=R0
            or provenance.get('split')!='TRAIN' or payload.get('frame')!=FRAME or payload.get('tcp')!=TCP
            or [v['camera'] for v in payload['images']]!=['external_rgb','wrist_rgb']
            or not isinstance(payload.get('language'),str)):
        fail('ORIGINAL_SINGLE_TRAIN_INPUT_NOT_DEV_OR_OBJECT_ORACLE')
    import numpy as np
    state=np.asarray(payload['state_eef10'],dtype=np.float64)
    if state.shape!=(10,) or not np.isfinite(state).all():fail('ACTUAL_SAVED_EEF10_FINITE_STATE')
    for image in payload['images']:
        raw=base64.b64decode(image['png_base64'],validate=True)
        if len(raw)>4*1024**2 or hashlib.sha256(raw).hexdigest()!=image['sha256']:
            fail('ACTUAL_TWO_CAMERA_INPUT_BYTES')
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as picture:
            if picture.format!='PNG' or picture.mode!='RGB' or picture.size!=(256,256):
                fail('ORIGINAL_SAVED_DUAL_RGB_256_INPUT')
            picture.verify()
    from scripts.operations.vanda_inference_probe import validate as mature_validate
    mature_validate(root/'RUN_MANIFEST.json',execute=False)
    return m,r


def native_running(m,r, *, query=subprocess.run):
    job=os.environ.get('PBS_JOBID')
    if not job or not socket.gethostname().split('.')[0].lower().startswith('gn-a40-'):
        fail('OWN_ALLOCATED_A40_COMPUTE_ONLY')
    p=query([PBS+'qstat','-f','-F','json',job],capture_output=True,text=True,timeout=25,check=True)
    value=json.loads(p.stdout)['Jobs'][job]
    variables=value.get('Variable_List',{})
    if isinstance(variables,str):variables=dict(v.split('=',1) for v in variables.split(',') if '=' in v)
    limits=value['Resource_List']
    if (value.get('job_state')!='R' or value.get('Job_Owner','').split('@')[0]!=m['owner_user']
            or value.get('project')!='CFP04-CF-007' or value.get('Job_Name')!='pi0294_dec1_a40'
            or int(limits['ngpus'])!=1 or int(limits['ncpus'])!=4
            or limits['walltime']!='00:30:00'
            or variables.get('DEC1_REQUEST_ID')!=r['request_id']
            or variables.get('DEC1_NONCE')!=r['submission_nonce']):
        fail('EXACT_NATIVE_RUNNING_REQUEST_RESOURCES_AND_NONCE')
    start=pbs_epoch(value['stime']);end=start+WALL
    if not start<=time.time()<end-60:fail('ACTUAL_ALLOCATION_WITH_CLOSEOUT_HEADROOM')
    return dict(job_id=job,request_id=r['request_id'],submission_nonce=r['submission_nonce'],
        request_sha256=digest(r),observed_epoch=time.time(),allocation_start_epoch=start,
        allocation_end_epoch=end,physical_GPU='NVIDIA_A40',ngpus=1,project=value['project'],
        native_R_sha256=digest(value))


def cold_check(root):
    """Bound CPU-only input audit in the already registered candidate capsule."""
    root=Path(root)
    if os.environ.get('PBS_JOBID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        fail('COLD_AUDIT_IS_LOGIN_CPU_ONLY_NOT_MODEL_OR_ALLOCATION')
    budget=sealed(root/'COLD_CPU_PREPAYMENT.json')
    if (budget.get('authority_release')!=AUTHORITY or budget.get('CPU_upper_s')!=300
            or budget.get('inside_original_and_DEC1_GPU_site_CPU_caps') is not True
            or budget.get('campaign_run_uuid')!='20261018T182956Z-localpc-pv1'):
        fail('ACTUAL_BOUND_COLD_CONTROL_CPU_PREPAYMENT_REQUIRED')
    if (root/'COLD_INPUTS_READY.json').exists() or (root/'COLD_INPUT_FAILURE.json').exists():
        fail('NO_REPEATED_COLD_INPUT_AUDIT')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));resource.setrlimit(resource.RLIMIT_CPU,(300,300))
    def expire(*unused):raise TimeoutError('DEC1_COLD_INPUT_WALL_BOUND')
    signal.signal(signal.SIGALRM,expire);signal.alarm(600)
    began=time.process_time();result=dict(status='FAILED_CPU_INPUT_AUDIT_PRESERVED',
        new_models=0,new_GPU_PBS=0,new_queries=0,new_updates=0,new_physics=0)
    try:
        m,r=validate(root)
        if str(root/'COLD_CPU_PREPAYMENT.json') not in m['sealed_files']:
            fail('COLD_PREPAYMENT_MUST_BE_SEALED_INPUT')
        result.update(source_sha=r['source_sha'],request_sha256=digest(r))
        identity=sealed(root/'POLICY_IDENTITY.json');checkpoint=Path(m['checkpoint'])
        for name,checksum in identity['files'].items():
            path=checkpoint/name
            if path.is_symlink() or not path.resolve().is_relative_to(checkpoint.resolve()) or sha(path)!=checksum:
                fail('ACTUAL_COLD_CORR_MEMBER_CHANGED')
        for name,checksum in m['cosmos_files'].items():
            snapshot=Path(m['bindings']['cosmos']);path=snapshot/name
            if (path.is_symlink() and not path.resolve().is_relative_to(snapshot.parent.parent/'blobs')
                    or not path.is_file() or path.stat().st_uid!=os.getuid() or sha(path)!=checksum):
                fail('ACTUAL_COLD_COSMOS_MEMBER_CHANGED')
        from scripts.pi0294.local_backbone import local_model_name
        local_model_name(root,m['bindings'])
        result.update(status='PASS_CPU_MODEL_COSMOS_SOURCE_RGB_AND_ALIAS_BYTES_NO_MODEL',
            inference_files_sha256=identity['files'],cosmos_files_sha256=m['cosmos_files'],
            observation_file_sha256=sha(m['observation']),parent_checkpoint_manifest_sha256=PARENT,
            statistics_sha256=R0)
        return result
    finally:
        signal.alarm(0);result.update(CPU_s=time.process_time()-began,observed_epoch=time.time())
        publish_release_record(root/('COLD_INPUTS_READY.json' if result['status'].startswith('PASS_') else 'COLD_INPUT_FAILURE.json'),result)


def control(root,r,allocation, *, now=None):
    root=Path(root);now=time.time() if now is None else now
    refresh(root);release=sealed(root/'START_COMPATIBILITY.json')
    bound=sealed(root/'GLOBAL_ALLOCATION_BOUND.json')
    for v in (release,bound):
        if (v.get('job_id')!=allocation['job_id'] or v.get('request_id')!=r['request_id']
                or v.get('request_sha256')!=digest(r) or v.get('submission_nonce')!=r['submission_nonce']):
            fail('SAME_NATIVE_JOB_GLOBAL_RESERVATION_AND_OWNER_RELEASE')
    cold=sealed(root/'COLD_INPUTS_READY.json')
    if (bound.get('run_manifest_sha256')!=sha(root/'RUN_MANIFEST.json')
            or bound.get('cold_input_audit_sha256')!=sha(root/'COLD_INPUTS_READY.json')
            or cold.get('request_sha256')!=digest(r)
            or cold.get('status')!='PASS_CPU_MODEL_COSMOS_SOURCE_RGB_AND_ALIAS_BYTES_NO_MODEL'):
        fail('SAME_NATIVE_COLD_INPUT_AUDIT_AND_MANIFEST_NOT_JUST_SOURCE_NAME')
    if (bound.get('global_submit_count')!=12 or bound.get('reserved_GPU_s')!=WALL+PAD
            or bound.get('reserved_queries')!=QUERIES or bound.get('other_live_GPU_jobs')!=[]
            or release.get('global_bound_sha256')!=sha(root/'GLOBAL_ALLOCATION_BOUND.json')
            or release.get('fresh_Vanda_project_admission') is not True
            or not allocation['observed_epoch']<=release['observed_epoch']<=allocation['observed_epoch']+240):
        fail('EXCLUSIVE_FRESH_PROJECT_FIRST_RELEASE_ORIGINAL_WINDOW')
    directory=root/'control/heartbeats';refresh(directory)
    records=sorted(directory.glob('[0-9][0-9][0-9][0-9][0-9].json'))
    if not records or len(records)>128:fail('BOUNDED_REAL_OWNER_HEARTBEAT_REQUIRED')
    beat=sealed(records[-1])
    if (beat.get('request_id')!=r['request_id'] or beat.get('job_id')!=allocation['job_id']
            or beat.get('source_sha')!=r['source_sha'] or beat.get('owner_session')!=r['owner_session']
            or beat.get('original_parent_STOP') is not False or beat.get('global_unknown_effects')!=[]
            or beat.get('Hopper_owned_jobs')!=[] or beat.get('Vanda_other_owned_jobs')!=[]
            or not -2<=now-beat['observed_epoch']<=120 or now>=allocation['allocation_end_epoch']-60):
        fail('FRESH_SINGLE_OWNER_AND_SAME_ALLOCATION_FENCE')
    return beat


def event_summary(root,r,job):
    root=Path(root);events=[]
    directory=root/'events'
    if directory.is_dir():
        refresh(directory)
        events=[sealed(p) for p in sorted(directory.glob('*.json'))]
    if len(events)>40:fail('FINITE_SEVENTEEN_QUERY_JOURNAL')
    load_begin=load_end=0;queries=[];pending=None;final=False
    for seq,e in enumerate(events):
        if (e.get('sequence')!=seq or e.get('request_id')!=r['request_id'] or e.get('job_id')!=job
                or e.get('request_sha256')!=digest(r) or final):fail('EXACT_APPEND_ONLY_NATIVE_EVENT_ORDER')
        kind=e['kind'];v=e['payload']
        if kind=='MODEL_LOAD_BEGIN':
            if load_begin or queries:fail('SINGLE_MODEL_BEFORE_QUERIES')
            load_begin=1
        elif kind=='MODEL_LOAD_COMPLETE':
            if load_begin!=1 or load_end or queries:fail('ONE_MODEL_LOAD_COMPLETION')
            load_end=1
        elif kind=='QUERY_BEGIN':
            index=v['index']
            if (not load_end or pending is not None or type(index) is not int or index!=len(queries) or index>=QUERIES
                    or v['seed']!=29400+index):fail('EXACT_SAVED_QUERY_ORDER_SEEDS_AND_BOUNDS')
            queries.append(dict(index=index,status='UNKNOWN_NO_RETRY'));pending=index
        elif kind in ('QUERY_COMPLETE','QUERY_FAILURE'):
            if pending is None or v['index']!=pending or v['seed']!=29400+pending:
                fail('EXACT_PENDING_QUERY_TERMINAL')
            queries[pending]=dict(index=pending,status=kind,record=v);pending=None
        elif kind=='FINAL':final=True
        else:fail('KNOWN_EFFECT_EVENT_ONLY')
    return dict(schema='DEC1_SAVED_TRAIN_EFFECT_SUMMARY_V1',request_id=r['request_id'],job_id=job,
        request_sha256=digest(r),physical_or_optimizer_effects=0,query_attempts=len(queries),
        model_load_attempts=load_begin,model_loads_complete=load_end,
        completed_queries=sum(x['status']=='QUERY_COMPLETE' for x in queries),
        failed_queries=sum(x['status']=='QUERY_FAILURE' for x in queries),
        unknown_query_intents=[x['index'] for x in queries if x['status']=='UNKNOWN_NO_RETRY'],
        journal_final=final,queries=queries,events_sha256=digest(events),
        saved_observation_not_closed_loop=True)


class Events:
    def __init__(self,root,r,allocation):
        self.root=Path(root);self.request=r;self.allocation=allocation;self.sequence=0
        (self.root/'events').mkdir(mode=0o700)

    def __call__(self,kind,payload):
        if kind not in ('QUERY_FAILURE','FINAL'):control(self.root,self.request,self.allocation)
        if kind=='QUERY_COMPLETE':
            index=payload['index'];path=self.root/('query-%02d.npz'%index)
            if sha(path)!=payload['arrays_sha256']:fail('ATOMIC_ARRAY_HASH_BEFORE_TERMINAL_EVENT')
            from scripts.pi0294.mp1.probe_compare import PRECISION
            if any(payload['inference_precision'].get(k)!=v for k,v in PRECISION.items()):
                fail('ORIGINAL_BF16_FP32_TF32_FALSE_PRECISION')
        value=dict(sequence=self.sequence,request_id=self.request['request_id'],job_id=self.allocation['job_id'],
            request_sha256=digest(self.request),source_sha=self.request['source_sha'],kind=kind,payload=payload,
            observed_epoch=time.time())
        path=self.root/'events'/('%03d.json'%self.sequence)
        publish_release_record(path,value);self.sequence+=1
        summary=event_summary(self.root,self.request,self.allocation['job_id'])
        write(self.root/'EFFECT_SUMMARY.json',summary)


def hardware(root,m,allocation):
    """Reuse the mature same-job CUDA fixture, never an old node's PASS."""
    from scripts.operations.vanda_qualify import query
    code=('import torch,json;assert torch.cuda.is_available() and torch.cuda.device_count()==1;'
          'assert "A40" in torch.cuda.get_device_name(0);'
          'x=torch.ones((64,64),device="cuda");y=x@x;torch.cuda.synchronize();'
          'assert y[0,0].item()==64;'
          'print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0))))')
    rows=[dict(check='gpu',**query(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'],timeout=30)),
          dict(check='cuda_torch',**query([m['python'],'-c',code],timeout=90))]
    result=dict(status='HOST_CHECKS_COMPLETED_SOFTWARE_GAPS_RECORDED',job_id=allocation['job_id'],
        request_sha256=digest(sealed(root/'REQUEST.json')),checks=rows,new_policy_queries=0,new_models=0)
    publish_release_record(root/'SAME_JOB_HARDWARE.json',result)
    if any(v['exit_code'] for v in rows):fail('ACTUAL_SAME_JOB_A40_AND_CUDA_FIXTURE_FAILED')
    return result


def launch(root):
    root=Path(root);m,r=validate(root)
    owner=OwnerLock(root/'NATIVE_COMPAT_OWNER.lock')
    if (root/'PROCESS_FINAL.json').exists() or (root/'NATIVE_PROCESS.json').exists():
        owner.close();fail('NEVER_RESTART_AN_EXISTING_NATIVE_QUALIFICATION')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    resource.setrlimit(resource.RLIMIT_CPU,(CPU_UPPER,CPU_UPPER))
    allocation=native_running(m,r)
    allocation.update(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    publish_release_record(root/'NATIVE_PROCESS.json',allocation)
    publish_release_record(root/'ALLOCATION_ADMISSION.json',allocation)
    state=dict(status='ENGINEERING_INCOMPLETE',request_id=r['request_id'],job_id=allocation['job_id'],
        new_optimizer_updates=0,new_physics=0,started_epoch=time.time())
    def stop(*unused):raise TimeoutError('DEC1_ACTUAL_ALLOCATION_CLOSEOUT')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop)
    signal.alarm(max(1,int(allocation['allocation_end_epoch']-time.time()-60)))
    try:
        until=min(allocation['observed_epoch']+240,allocation['allocation_end_epoch']-120)
        while True:
            refresh(root)
            if (root/'START_COMPATIBILITY.json').is_file():break
            if time.time()>=until:raise TimeoutError('DEC1_NO_FRESH_OWNER_RELEASE_NO_MODEL')
            time.sleep(2)
        control(root,r,allocation)
        submission=sealed(root/'SUBMISSION.private.json')
        if submission['job_id']!=allocation['job_id'] or submission['request_sha256']!=digest(r):
            fail('ONE_REGISTERED_NATIVE_SUBMISSION')
        hardware(root,m,allocation)
        events=Events(root,r,allocation)
        from scripts.operations.vanda_inference_probe import probe
        result=probe(root/'RUN_MANIFEST.json',events=events)
        summary=event_summary(root,r,allocation['job_id'])
        if (result['status']!='PASS_SAVED_OBSERVATION_A40_INFERENCE_ONLY'
                or summary['completed_queries']!=QUERIES or summary['failed_queries']
                or summary['unknown_query_intents'] or not summary['journal_final']):
            fail('ONLY_COMPLETE_FINITE_SEVENTEEN_CURRENT_CORR_DRAWS_QUALIFY')
        state.update(status='PASS_CORR_A40_NUMERICAL_COMPATIBILITY_ONLY',effect_summary=summary,
            model_result_sha256=sha(root/'RESULT.private.json'),saved_input_not_closed_loop=True,
            policy_improvement_or_target_transfer=False)
        return state
    except BaseException as error:
        state.update(error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0);state.update(completed_epoch=time.time())
        state['effect_summary']=event_summary(root,r,allocation['job_id'])
        publish_release_record(root/'PROCESS_FINAL.json',state)
        owner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    group=parser.add_mutually_exclusive_group();group.add_argument('--execute',action='store_true');group.add_argument('--cold-check',action='store_true')
    args=parser.parse_args()
    value=(launch(args.root) if args.execute else cold_check(args.root) if args.cold_check
           else dict(status='COLD_SCOPE_VALIDATED_NO_MODEL',run_uuid=validate(args.root)[0]['run_uuid']))
    print(json.dumps(value))
