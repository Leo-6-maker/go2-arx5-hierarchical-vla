"""Use explicit local bindings and the existing registry/Journal, no scheduler."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


def cuda_uuid(value):
    """PyTorch returns a bare UUID; NVML/nvidia-smi require the GPU prefix."""
    from uuid import UUID
    value = str(value)
    if value.startswith('GPU-'):
        value = value[4:]
    try:
        return 'GPU-'+str(UUID(value))
    except (ValueError, AttributeError) as error:
        raise ValueError('Invalid native CUDA device UUID') from error


def validate_core_smoke(spec, observation):
    """The directly authorized GPU1 sharing is a finite model-only diagnostic."""
    share = spec.get('sharing', {})
    if (spec.get('host') != 'corelab-4080s' or spec.get('uid') != 2009
            or spec.get('profile') != 'core-wm01-shared-smoke'
            or spec.get('physics') is not False
            or share.get('direct_user_answer') != '允许GPU1共享'
            or spec.get('GPU_uuid') != 'GPU-845348f1-1511-dc19-0877-cc97658e3edd'
            or share.get('endpoint') != 200
            or share.get('allocator_limit_bytes') != 512*1024**2
            or observation.get('uuid') != spec['GPU_uuid']
            or observation.get('name') != 'NVIDIA GeForce RTX 4080 SUPER'
            or observation.get('total_bytes', 0) < 15*1024**3
            or observation.get('free_bytes', 0) < 2*1024**3):
        raise PermissionError('Exact authorized Core GPU1 model-only smoke and live headroom required')


@contextmanager
def admitted_run(spec_path, run_id, profile, output, *, physics=False, gpu=False):
    from scripts.pi0292.durable import OwnerLock
    from .storage import digest, write_json
    spec = json.loads(Path(spec_path).read_text())
    if not Path(__file__).resolve().is_relative_to(Path(spec['source_root']).resolve()):
        raise PermissionError('Interpreter imported a different WM01 source capsule')
    root = Path(spec['registered_root']).resolve()
    output = Path(output).resolve()
    if (spec['profile'] != profile or spec['host'] != os.uname().nodename or spec['uid'] != os.getuid()
            or output.parent != root or output.name != run_id or time.time() >= spec['deadline_epoch']
            or (root/'STOP.json').exists()):
        raise PermissionError('Exact WM01 host/profile/run and finite endpoint required')
    if physics and spec.get('physics') is not True:
        raise PermissionError('Profile does not admit native source physics')
    if gpu and spec.get('execution_site')=='CORE_AUTO1_TARGET_NEURAL':
        import subprocess
        from .target_neural import validate_campaign,validate_device
        from scripts.pi0294.e1.smoke import node_local
        validate_campaign(spec)
        for key in('source_root','registered_root','python','data','cache','logs','tmp','replay'):node_local(spec[key])
        reply=subprocess.run(['nvidia-smi','--id='+spec['GPU_uuid'],'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        uuid,free=[v.strip()for v in reply.stdout.strip().split(',')]
        validate_device(spec,dict(uuid=uuid,free_bytes=int(free)*1024**2))
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=spec['GPU_uuid']:raise PermissionError('TARGET_SINGLE_EXPLICIT_GPU_REQUIRED')
        gpu_baseline=reply.stdout
    elif gpu and spec.get('execution_site') == 'CORE_SHARED_GPU1':
        import subprocess
        from scripts.pi0294.e1.smoke import node_local
        for name in ('source_root', 'registered_root', 'python', 'data', 'cache', 'logs', 'tmp'):
            node_local(spec[name])
        response = subprocess.run(['nvidia-smi', '--id='+spec['GPU_uuid'],
            '--query-gpu=uuid,name,memory.total,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=10)
        fields = [value.strip() for value in response.stdout.strip().split(',')]
        validate_core_smoke(spec, dict(uuid=fields[0], name=fields[1],
            total_bytes=int(fields[2])*1024**2, free_bytes=int(fields[3])*1024**2))
        if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['GPU_uuid']:
            raise PermissionError('Core smoke must expose only the authorized GPU1 UUID')
    elif gpu and spec.get('execution_site')in('CORE_WM02_GPU1_REAL_S','CORE_WM02_GPU0_REAL_S'):
        import subprocess
        from scripts.pi0294.e1.smoke import node_local
        gpu0=spec['execution_site']=='CORE_WM02_GPU0_REAL_S'
        expected_uuid='GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'if gpu0 else'GPU-845348f1-1511-dc19-0877-cc97658e3edd'
        expected_profile='wm02-gpu0-real-s'if gpu0 else'wm02-gpu1-real-s'
        response=spec.get('authorization_scope')=='USER_WM02_RESPONSE_M1_M2_MODEL_ONLY_20261104'
        temporal=spec.get('authorization_scope')=='USER_WM02_TEMPORAL_M3_MODEL_ONLY_20261104'
        autonomous=spec.get('authorization_scope')=='USER_WM02_AUTONOMOUS_MECHANISM_MODEL_ONLY_20261104'
        if autonomous and(not gpu0 or spec.get('mechanism_candidate')not in('M4_A','M4_B','M4_C')or spec.get('endpoint')!=200):raise PermissionError('Finite delegated GPU0 fixed200 mechanism candidate required')
        if gpu0 and(spec.get('data_domain')!='WM02_DECISION_PAIRED_REPLAY_V5'or spec.get('authorization_scope')not in('USER_WM02_NEW_DECISION_DATA_MODEL_ONLY_20261104','USER_WM02_RESPONSE_M1_M2_MODEL_ONLY_20261104','USER_WM02_TEMPORAL_M3_MODEL_ONLY_20261104','USER_WM02_AUTONOMOUS_MECHANISM_MODEL_ONLY_20261104')):
            raise PermissionError('New separately authorized GPU0 V5 model-only identity required')
        if (spec.get('schema')!='WM02_REAL_S_NEURAL_SPEC_V1'
                or spec.get('host')!='corelab-4080s'or spec.get('uid')!=2009
                or profile!=expected_profile or spec.get('physics')is not False
                or spec.get('GPU_uuid')!=expected_uuid
                or (spec.get('authorization_scope'),spec.get('data_domain'))not in (
                    ('USER_WM02_GPU1_REAL_S_MODEL_ONLY_20261104','REAL_S_NATIVE_DIAGNOSTIC_REPLAY'),
                    ('USER_WM02_CONTACT_IDENTIFICATION_MODEL_ONLY_20261104','WM02_S_CONTACT_PAIRED_REPLAY_V1'),
                    ('USER_WM02_NEW_DECISION_DATA_MODEL_ONLY_20261104','WM02_DECISION_PAIRED_REPLAY_V5'),
                    ('USER_WM02_RESPONSE_M1_M2_MODEL_ONLY_20261104','WM02_DECISION_PAIRED_REPLAY_V5'),
                    ('USER_WM02_TEMPORAL_M3_MODEL_ONLY_20261104','WM02_DECISION_PAIRED_REPLAY_V5'),
                    ('USER_WM02_AUTONOMOUS_MECHANISM_MODEL_ONLY_20261104','WM02_DECISION_PAIRED_REPLAY_V5'))
                or not 2<=spec.get('endpoint',0)<=200
                or not 0<spec.get('allocator_limit_bytes',0)<=4*1024**3
                or os.environ.get('CUDA_VISIBLE_DEVICES')!=spec['GPU_uuid']):
            raise PermissionError('Exact finite real S model-only GPU1 identity required')
        expected_arms=[spec['mechanism_candidate']]if autonomous else['M3']if temporal else['M1','M2']if response else['action_conditioned','no_future_action']
        if spec.get('data_domain')in('WM02_S_CONTACT_PAIRED_REPLAY_V1','WM02_DECISION_PAIRED_REPLAY_V5')and spec.get('arms')!=expected_arms:
            raise PermissionError('Exact matched two-arm finite contact diagnostic required')
        for name in ('source_root','registered_root','python','data','cache','logs','tmp','replay','eval_replay'):
            node_local(spec[name])
        reply=subprocess.run(['nvidia-smi','--id='+spec['GPU_uuid'],
            '--query-gpu=uuid,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        uuid,total,free=[x.strip()for x in reply.stdout.strip().split(',')]
        if (uuid!=spec['GPU_uuid']or int(total)*1024**2<15*1024**3
                or int(free)*1024**2<spec['allocator_limit_bytes']+1024**3):
            raise PermissionError('Actual GPU1 capacity below finite diagnostic allowance')
        gpu_baseline=reply.stdout
    elif gpu and spec.get('execution_site') == 'CORE_WM02_GPU0':
        import subprocess
        from scripts.pi0294.e1.smoke import node_local
        if (spec.get('schema') != 'WM02_LOCAL_NEURAL_SPEC_V1'
                or spec.get('host') != 'corelab-4080s' or spec.get('uid') != 2009
                or spec.get('profile') != 'wm02-gpu0-neural-smoke'
                or spec.get('physics') is not False or spec.get('endpoint') != 200
                or spec.get('GPU_uuid') != 'GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad'
                or spec.get('authorization_scope') != 'USER_GPU0_WM02_ENV_NEURAL_SMOKE_20261104'
                or spec.get('sharing_policy') != 'USER_GPU0_WITH_MEASURED_HEADROOM'
                or spec.get('data_domain') != 'SYNTHETIC_ENGINEERING_ONLY'
                or not 0 < spec.get('allocator_limit_bytes',0) <= 13*1024**3
                or os.environ.get('CUDA_VISIBLE_DEVICES') != spec['GPU_uuid']):
            raise PermissionError('Explicit bounded WM02 GPU0 neural-only identity required')
        for name in ('source_root','registered_root','python','data','cache','logs','tmp'):
            node_local(spec[name])
        if not spec.get('old_owner_leases'):
            raise PermissionError('Old automatic dispatch custody must be independently closed')
        for path in spec['old_owner_leases']:
            if json.loads(Path(path).read_text()).get('status') != 'CLOSED':
                raise PermissionError('An old dispatch or worker lease is still live')
        reply = subprocess.run(['nvidia-smi','--id='+spec['GPU_uuid'],
            '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True,timeout=10)
        uuid,free = [v.strip() for v in reply.stdout.strip().split(',')]
        if uuid != spec['GPU_uuid'] or int(free)*1024**2 < spec['allocator_limit_bytes']+1024**3:
            raise PermissionError('Current GPU0 capacity is below the finite smoke allowance')
        reply = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        # The user permits GPU0 work. Sharing is bounded by measured free memory,
        # the allocator cap and the original project owner lock; never displace a process.
        gpu_baseline = reply.stdout
    elif gpu:
        if spec.get('execution_site') != 'VANDA_A40':
            raise PermissionError('Explicit Vanda allocation or authorized Core smoke required')
        import subprocess
        from scripts.pi0294.multitask.shared_site import validate_vanda, project_quota
        key = os.environ.get('PBS_JOBID')
        if not key:
            raise PermissionError('Actual PBS allocation required before model import')
        response = subprocess.run(['/opt/pbs/bin/qstat', '-f', '-F', 'json', key],
                                  capture_output=True, text=True, check=True, timeout=20)
        jobs = json.loads(response.stdout)['Jobs']
        matches = [v for k,v in jobs.items() if k.split('.')[0] == key.split('.')[0]]
        if len(matches) != 1:
            raise PermissionError('Exact single PBS allocation required')
        context = dict(spec['site_context'])
        quota_path = Path('/home/svu/vanda/quotas/CFP04-CF-007.proj_scratch.txt')
        now = time.time()
        context['quota_receipt'] = project_quota(quota_path.read_text(), user=os.environ['USER'],
            report_epoch=quota_path.stat().st_mtime, observed_epoch=now)
        validate_vanda(context, matches[0], now=now, uid=os.getuid(), user=os.environ['USER'],
                       hostname=os.uname().nodename, job_id=key)
    for name, pin in spec['source_files'].items():
        if digest(Path(spec['source_root'])/name) != pin:
            raise PermissionError('Source identity changed:'+name)
    registry = json.loads(Path(spec['registry_snapshot']).read_text())
    if not any(e.get('role') == spec['role'] and e.get('path') == str(root)
               and e.get('source_sha') == spec['source_sha'] for e in registry['entries']):
        raise PermissionError('Existing host-aware registry binding missing')
    lease = json.loads(Path(spec['lease']).read_text())
    if lease['status'] != 'ACTIVE' or lease['run_uuid'] != root.name or lease['spec_sha256'] != digest(spec_path):
        raise PermissionError('Exact active WM01 lease required')
    cpus = set(spec['cpus'])
    if not cpus or not cpus <= os.sched_getaffinity(0):
        raise PermissionError('Actual CPU affinity not available')
    lock = OwnerLock(root/'OWNER.lock')
    neural_owner = None
    night_campaign_owner = None
    night_physical_owner = None
    try:
        if gpu and spec.get('execution_site') in ('CORE_WM02_GPU0','CORE_WM02_GPU1_REAL_S','CORE_WM02_GPU0_REAL_S','CORE_AUTO1_TARGET_NEURAL'):
            neural_owner = OwnerLock(spec['neural_owner_lock'])
        if gpu and(spec.get('night1_binding')or spec.get('next1_binding')):
            night_campaign_owner=OwnerLock(Path(spec['campaign_root'])/'OWNER.lock')
            night_physical_owner=OwnerLock(spec['physical_owner_lock'])
        os.sched_setaffinity(0, cpus)
        output.mkdir(exist_ok=False)
        write_json(output/'STARTED.json', dict(run_id=run_id, source_sha=spec['source_sha'],
            spec_sha256=digest(spec_path), pid=os.getpid(), epoch=time.time(), profile=profile,
            GPU_compute_baseline=gpu_baseline if gpu and spec.get('execution_site')in('CORE_WM02_GPU0','CORE_WM02_GPU1_REAL_S','CORE_WM02_GPU0_REAL_S','CORE_AUTO1_TARGET_NEURAL') else None))
        yield spec
    finally:
        if night_physical_owner is not None:
            night_physical_owner.close()
        if night_campaign_owner is not None:
            night_campaign_owner.close()
        if neural_owner is not None:
            neural_owner.close()
        lock.close()
