"""Launch one finite, nonintegrating Core GPU1 replay after matched site custody."""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import time

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from go2arx5_vla.world_model.multisource_d4_native import PARENT_SHA
from scripts.pi0292.durable import OwnerLock, write
from .d416_endpoint_transfer import ARM_SET, pin


GIB = 1024**3


def output(*command):
    return subprocess.check_output(command,text=True,timeout=20).strip()


def resources(root, gpu, cpus):
    if (root.parent.stat().st_uid != os.getuid()
            or output('loginctl','show-user',str(os.getuid()),'-p','Linger') != 'Linger=yes'
            or output('findmnt','-n','-o','FSTYPE','-T',str(root.parent)) != 'ext4'
            or not set(cpus).issubset(os.sched_getaffinity(0)) or not cpus
            or shutil.disk_usage(root.parent).free <= 4*GIB):
        raise PermissionError('CORE_GPU_REPLAY_REQUIRES_OWN_LOCAL_DISK_CPU_AND_LINGER')
    uuid,name,total,free = [x.strip() for x in output('nvidia-smi','--id=1',
        '--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits').split(',')]
    active = output('nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader')
    host = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')))*1024
    if (uuid == gpu['GPU_uuid'] or '4080' not in name or int(free) <= int(total)*.70+2048
            or uuid in active or host <= 36*GIB):
        raise PermissionError('FREE_SEPARATE_CORE_GPU1_AND_HOST_HEADROOM_REQUIRED')
    lock_path = Path(gpu['neural_owner_lock']).parent/'sla-pt-gpu1-owner.lock'
    lock = OwnerLock(lock_path)
    lock.close()
    return dict(uuid=uuid,name=name,GPU_free_MiB=int(free),GPU_total_MiB=int(total),
                host_available_bytes=host,gpu_owner_lock=str(lock_path),cpus=cpus,linger=True)


def launch(config):
    from go2arx5_vla.world_model.sla_followup_native_v1 import PLAN

    run = Path(config['root'])
    source = Path(config['source_root']).resolve(strict=True)
    authority = read_pin(config['authority'])
    gpu = read_pin(config['core_gpu_reference'])
    data = read_pin(config['core_C20_reference'])
    deployment = read_pin(config['custody'])
    prior = read_pin(config['active_or_closed_D4_spec'])
    mode = deployment['mode']
    script = source/'scripts/wm02/d4_core_rgb_replay.py'
    now = time.time()
    deadline = min(now + 1800, authority['deadline_epoch'])
    if (os.getuid() != config['uid'] or os.uname().nodename != config['host']
            or gpu['uid'] != config['uid'] or gpu['host'] != config['host']
            or authority['plan_commit'] != PLAN or authority['mode'] != mode
            or authority['new_model_generations_budget'] != 32 or authority['native_requests']
            or authority['source_sha'] != config['source_sha'] or authority['training_updates']
            or deployment['plan_commit'] != PLAN or set(deployment['arms']) != set(ARM_SET[mode])
            or deployment['source_job']['id'] != config['confirmed_training_job']
            or run.exists() or sha(script) != config['replay_worker_sha256']
            or deadline < now + 600 or set(config['cpus']).intersection(prior['cpus'])):
        raise PermissionError('D416_OWN_BOUNDED_CORE_REPLAY_AND_FIXED_SITE_PROOF_REQUIRED')
    if (gpu['parent_manifest']['sha256'] != PARENT_SHA
            or sha(gpu['parent_manifest']['path']) != PARENT_SHA
            or data['dataset'] != gpu['dataset']):
        raise ValueError('ORIGINAL_CORE_PARENT_BYTES_AND_C20_DATA_REQUIRED')
    available = resources(run,gpu,config['cpus'])
    transfer_file = Path(deployment['transfer']['path'])
    archive = read_pin(deployment['transfer'])
    if archive['job'] != deployment['source_job']:
        raise ValueError('SITE_PROOF_CHANGED_AFTER_RECEIPT')
    for path, member in archive['members'].items():
        artifact = transfer_file.parent/path
        if (not artifact.resolve().is_relative_to(transfer_file.parent.resolve())
                or artifact.stat().st_size != member['bytes'] or sha(artifact) != member['sha256']):
            raise ValueError('TRANSFERRED_ASSET_CHANGED_SINCE_CUSTODY:'+path)

    run.mkdir(mode=0o700)
    for part in ('predictions','logs','cache','tmp'):
        (run/part).mkdir()
    original = gpu['gpu_environment']
    flags = original['XLA_FLAGS']
    spec = dict(root=str(run),stage=mode,plan_commit=PLAN,host=config['host'],uid=config['uid'],
        deadline_epoch=deadline,GPU_uuid=available['uuid'],gpu_owner_lock=available['gpu_owner_lock'],
        source_sha=config['source_sha'],XLA_FLAGS=flags,dataset=data['dataset'],
        time_codec_audit=data['time_codec_audit'],parent_params=gpu['parent_params'],
        parent_manifest_path=gpu['parent_manifest']['path'],arms=deployment['arms'],
        portable_evaluation=deployment['portable_evaluation'],
        reference_root=str(transfer_file.parent/'reference'))
    spec_file = run/'SPEC.private.json'
    write(spec_file,spec)
    env = {key:original[key] for key in ('JAX_PLATFORMS','XLA_FLAGS')}
    env.update(PYTHONPATH=':'.join((str(source),str(source/'src'),original['PYTHONPATH'])),
        CUDA_VISIBLE_DEVICES=available['uuid'],XLA_PYTHON_CLIENT_MEM_FRACTION='.70',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TMPDIR=str(run/'tmp'),XDG_CACHE_HOME=str(run/'cache'),JAX_COMPILATION_CACHE_DIR=str(run/'cache'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
        HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',SSH_AUTH_SOCK='',DISPLAY='')
    unit = 'wm02-'+run.name
    command = ['systemd-run','--user','--unit='+unit,'--property=Restart=no',
        '--property=KillMode=control-group','--property=TimeoutStopSec=30','--property=StandardInput=null',
        '--property=RuntimeMaxSec='+str(int(deadline-time.time())+30),'--property=WorkingDirectory='+str(source),
        '--property=CPUAffinity='+' '.join(map(str,config['cpus'])),'--property=MemoryMax=20G',
        '--property=StandardOutput=append:'+str(run/'logs/worker.log'),
        '--property=StandardError=append:'+str(run/'logs/worker.log')]
    command += ['--setenv='+key+'='+value for key,value in env.items()]
    command += [gpu['python'],'-B',str(script),str(spec_file),sha(spec_file)]
    launched = subprocess.run(command,capture_output=True,text=True,timeout=35)
    receipt = dict(unit=unit,source_sha=config['source_sha'],spec=pin(spec_file),deadline_epoch=deadline,
        launch_exit=launched.returncode,stdout=launched.stdout,stderr=launched.stderr,resources=available)
    write(run/'LAUNCH_ACTUAL.private.json',receipt)
    if launched.returncode:
        raise RuntimeError('CORE_GPU_LAUNCH_EFFECT_REQUIRES_READBACK_NO_REPLAY')
    actual = unit_state(unit)
    if not actual['pid'] or actual['pid'] not in actual['cgroup_pids']:
        raise RuntimeError('CORE_GPU_UNIT_START_REQUIRES_INDEPENDENT_RECONCILIATION')
    return dict(receipt,actual=actual)


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(launch(read_pin(dict(path=args.config,sha256=args.sha256))),allow_nan=False))


if __name__ == '__main__':
    main()
