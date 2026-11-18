"""Run the unchanged D4 recipe inside one authorized VANDA PBS allocation."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from scripts.pi0292.durable import read, write
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.multisource_d4 import ARMS, RECIPE, UPDATES, VISITS, locate_input
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin

GIB = 1024**3


def pin(path):
    return dict(path=str(path), sha256=sha(path))


def pbs_job(config):
    job_id = os.environ['PBS_JOBID']
    text = subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',job_id], text=True, timeout=20)
    job = json.loads(text)['Jobs'][job_id]
    hosts = {item.split('/')[0].split('.')[0] for item in job['exec_host'].split('+')}
    user = os.environ['USER']
    if (job_id != config['job_id'] or os.uname().nodename.startswith('stdct-login')
            or os.uname().nodename.split('.')[0] not in hosts or os.getuid() != config['uid']
            or user != config['user'] or job['Job_Owner'].split('@')[0] != user
            or job['job_state'] != 'R' or job['project'] != config['project']
            or int(job['Resource_List']['ngpus']) != 1):
        raise PermissionError('D4_OWN_RUNNING_PBS_JOB_AND_EXACT_PROJECT_REQUIRED')
    return job_id


def actual_gpu():
    # The CUDA driver honors the scheduler's visibility mask. The physical
    # UUID prevents a later NVML index lookup from selecting a sibling card.
    cuda = ctypes.CDLL('libcuda.so.1')
    total, device = ctypes.c_int(), ctypes.c_int()
    raw = (ctypes.c_ubyte*16)()
    if (cuda.cuInit(0) or cuda.cuDeviceGetCount(ctypes.byref(total)) or total.value != 1
            or cuda.cuDeviceGet(ctypes.byref(device),0) or cuda.cuDeviceGetUuid(ctypes.byref(raw),device)):
        raise PermissionError('D4_PBS_ASSIGNED_EXACTLY_ONE_CUDA_DEVICE_REQUIRED')
    identity = 'GPU-'+str(uuid.UUID(bytes=bytes(raw)))
    row = subprocess.check_output(['nvidia-smi','--id='+identity,
        '--query-gpu=name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
    name, total_text, free_text = [part.strip() for part in row.split(',')]
    capacity, free = int(total_text)*1024**2, int(free_text)*1024**2
    if name != 'NVIDIA A40' or free <= 25*GIB or capacity < 40*GIB:
        raise PermissionError('D4_PBS_A40_AND_ARCHIVED_MINIMUM_25_GIB_FREE_REQUIRED')
    return identity, name, capacity, free


def build_cards(config, root, job_id, gpu):
    identity, name, capacity, free = gpu
    portable = read_pin(config['portable_data'])
    base = portable['spec']
    source = Path(config['source_root'])
    foundation = root/'foundation'/portable['model_manifest_filename']
    dataset = (root/'C20').resolve(strict=True)
    sidecar = pin(root/'sealed/INPUTS.private.json')
    contexts = pin(root/'eval/CONTEXTS.private.json')
    bank = pin(root/'eval/diagnostic/WINDOW48_PLAN.private.json')
    if (sidecar['sha256'] != portable['portable_inputs_sha256']
            or contexts['sha256'] != portable['portable_target_audit_sha256']
            or bank['sha256'] != base['fixed_TRAIN48']['sha256']
            or sha(foundation) != base['parent_manifest']['sha256']
            or sha(dataset/'MANIFEST.private.json') != base['dense_manifest_sha256']):
        raise ValueError('D4_FIXED_PORTABLE_ARTIFACTS_REQUIRED')
    cache = Path(os.environ['TMPDIR']).resolve(strict=True)/'sla-d4-compile'
    cache.mkdir()
    deadline = base['deadline_epoch']
    inherited = dict(schema='SLA_D4_VANDA_PBS_OBSERVED_V1', job_id=job_id,
        project=config['project'], uid=config['uid'], host=os.uname().nodename,
        state='R', allocated_GPU_uuids=[identity], GPU_free_bytes=free,
        observed_epoch=time.time())
    allocation = root/'PBS_ALLOCATION_OBSERVED.private.json'
    write(allocation,inherited)
    binding = root/'SITE_BINDING.private.json'
    params = (foundation.parent/'model/params').resolve(strict=True)
    write(binding,dict(schema='SLA_SINGLE_GPU_EXISTING_SITE_V2', host=inherited['host'],
        uid=config['uid'], GPU_uuid=identity,GPU_name=name,allocation_deadline_epoch=deadline+60,
        allocator_limit_bytes=24*GIB,GPU_headroom_bytes=GIB,host_headroom_bytes=32*GIB,
        paths=dict(parent_params=str(params),dataset=str(dataset),compilation_cache_dir=str(cache)),
        allocation=dict(kind='PBS',job_id=job_id,project=config['project'],actual_allocation=pin(allocation))))
    common = dict(base, source_root=str(source), source_sha=config['source_sha'], campaign_root=str(root),
        scope=config['scope'],operation=RECIPE, planned_candidate_updates=UPDATES, new_training_visits_budget=VISITS,
        new_optimizer_updates_budget=UPDATES,microbatch=4,training_matmul_precision='highest',
        aux_samples_per_update=4,aux_sampling_steps=10,physics=False,new_physical_requests_budget=0,
        parent_manifest=pin(foundation),parent_params=str(params),dataset=str(dataset),
        portable_inputs=sidecar,portable_evaluation=contexts,previous_campaign=str(root/'eval'),
        fixed_TRAIN48=bank,compilation_cache_dir=str(cache), site_binding=pin(binding),GPU_uuid=identity,
        JAX_memory_fraction=min(.5,24*GIB/capacity),max_walltime_s=21600,max_gpu_hours=6,
        RPC_timeout_s=90,model_service_timeout_s=90,save_and_exit_on_limit=True)
    for key in ('target_audit','tokenizer','initial_function_rule'):
        common[key] = locate_input(sidecar,base[key])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=identity, JAX_PLATFORMS='cuda',
        JAX_COMPILATION_CACHE_DIR=str(cache), XLA_FLAGS=portable['XLA_FLAGS'],
        XLA_PYTHON_CLIENT_MEM_FRACTION=str(common['JAX_memory_fraction']),
        XLA_PYTHON_CLIENT_PREALLOCATE='false', HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    cards = []
    for arm in ARMS:
        folder = root/arm
        folder.mkdir()
        (folder/'data').mkdir()
        (folder/'logs').mkdir()
        card = folder/'EXECUTION_SPEC.json'
        write(card,dict(common,arm=arm,registered_root=str(folder),
                       S1000_transfer_result=base['S1000_transfer_result'] if arm=='D4-B' else None))
        cards.append(card)
    return cards, env


def run(config):
    root=Path(config['root'])
    journal=Journal(root/'pbs-journal', root.name)
    records, first_failure, code = [], None, 1
    started = time.monotonic()
    try:
        job_id=pbs_job(config)
        for member,digest in config['source_files'].items():
            if sha(Path(config['source_root'])/member)!=digest:
                raise ValueError('D4_PBS_IMMUTABLE_SOURCE_MEMBER_CHANGED:'+member)
        predecessor=read_pin(config['core_released'])
        if predecessor['lease']!='CLOSED' or predecessor['any_dispatched'] or predecessor['verified_updates']:
            raise PermissionError('D4_CORE_MUST_CLOSE_WITH_ZERO_DISPATCH')
        if time.time() >= config['latest_pair_start_epoch'] or (root/'STOP.json').exists():
            raise TimeoutError('D4_PBS_TERMINAL_BEFORE_PAIR_START')
        cards, env = build_cards(config,root,job_id,actual_gpu())
        for card in cards:
            folder=card.parent
            row=dict(arm=folder.name)
            records.append(row)
            for kind,out in (('train','RESULT.private.json'),('restore','RESTORE.private.json')):
                if time.time() >= config['deadline_epoch']-300 or (root/'STOP.json').exists():
                    raise TimeoutError('D4_PBS_FINITE_STOP_OR_DEADLINE')

                def execute():
                    with (folder/'logs'/f'{kind}.log').open('xb') as log:
                        proc=subprocess.Popen([sys.executable,'-B','-m','scripts.wm02.multisource_d4_vanda',
                                               kind,str(card)],env=env,stdout=log,stderr=subprocess.STDOUT)
                        write(root/'ACTIVE_CHILD.private.json',dict(epoch=time.time(),arm=folder.name,kind=kind,pid=proc.pid))
                        result=proc.wait()
                    proof=read(folder/out)
                    return dict(exit_code=result,result=pin(folder/out),new_updates=proof.get('new_updates'),
                        new_visits=proof.get('new_visits'),actual_generations=proof.get('actual_generations'))

                row[kind]=journal.once(folder.name+'-'+kind,pin(card),execute)
                if row[kind]['exit_code']:
                    raise RuntimeError('D4_PBS_CHILD_TERMINAL_NO_REPLAY:'+folder.name+'-'+kind)
        code=0
    except TimeoutError as error:
        first_failure=str(error)
        code=75
    except BaseException:
        first_failure=traceback.format_exc()
    finally:
        write(root/'RESULT.private.json',dict(exit_code=code, records=records, first_failure=first_failure,
            source_sha=config['source_sha'], job_id=config['job_id'], elapsed_s=time.monotonic()-started,
            new_native_requests=0,new_integrations=0))
    return code


if __name__ == '__main__':
    mode, path = sys.argv[1:]
    if mode=='pair':
        raise SystemExit(run(read(path)))
    if mode not in ('train','restore') or not os.environ.get('PBS_JOBID'):
        raise SystemExit('D4_PBS_CHILD_MODE_REQUIRED')
    from scripts.sla.train_multisource_d4 import run as child

    raise SystemExit(child(read(path),restore=mode=='restore'))
