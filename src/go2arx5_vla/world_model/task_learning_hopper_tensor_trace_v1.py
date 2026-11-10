"""One PBS-owned H200 inference trace, with no Trainer or optimizer."""
import json
import os
from pathlib import Path
import shlex
import signal
import time
import traceback
from contextlib import closing

from .target_gr00t_profile_v1 import need, PARENT
from .storage import digest

PROFILE = 'TASK_LEARNING_HOPPER_TENSOR_TRACE_V1'
PROJECT_ROOT = Path('/scratch/Projects/CFP-04/CFP04-CF-007/e1192548/Projects/GR00T-WBC')


def configuration(c):
    """Small sealed metadata only; large model checks run on the compute node."""
    root = Path(c['root']); old = Path(c['producer_root'])
    need(root.resolve() == root and root.parent == PROJECT_ROOT / 'runs/WM02'
         and old.resolve() == old and old.parent == root.parent and old != root,
         'DISTINCT_OWN_PROJECT_TRACE_ROOT')
    need(c.get('profile') == PROFILE and c.get('project') == 'CFP04-CF-007'
         and c.get('pbs_job_name') == 'gr00t_numtrace'
         and type(c.get('pbs_server')) is str and bool(c['pbs_server'].strip())
         and c.get('allocation_gpu_model') == 'H200' and c.get('wall_s') == 3600
         and c.get('numerical_diagnostic') is None
         and 0 < c.get('predicted_work_s', 0) + 1320 < c['wall_s'],
         'FINITE_SINGLE_H200_TRACE_WITH_ORIGINAL_CLOSEOUT_RESERVE')
    cfg = json.loads((old / 'ALLOCATION_CONFIG.json').read_bytes())
    lease = json.loads((old / 'control/LEASE.json').read_bytes())
    worker = json.loads((old / 'WORKER_EXIT.private.json').read_bytes())
    result = json.loads((old / 'RESULT.json').read_bytes())
    need(lease['status'] == 'CLOSED' and lease['worker_exit_code'] == worker['exit_code'] == 0
         and worker['failure'] is None and lease['result_sha256'] == digest(old / 'RESULT.json')
         and lease['config_sha256'] == digest(old / 'ALLOCATION_CONFIG.json')
         and result['status'] == 'CHECKPOINT_SEALED' and result['global_step'] == 2000
         and result['GR00T_updates'] == 2000 and result['sample_visits'] == 64000
         and result['source_sha'] == cfg['source_sha']
         and all(result.get('restore', {}).get(k) is True for k in (
             'model_parameters_restored', 'optimizer_restored', 'scheduler_restored',
             'RNG_restored', 'fixed_batch_generation_restored'))
         and result['restore']['sampler_cursor'] == 64000
         and result['backbone_frozen'] is True and result['real_action_parameter_changed'] is True,
         'ACTUAL_FIRST2000_TERMINAL_PRODUCER_BEFORE_TRACE')
    package = c['candidate_package']
    need(package['path'] == str(old / 'inference-package')
         and package['sha256'] == result['inference_package']['sha256']
         == digest(Path(package['path']) / 'PACKAGE_MANIFEST.json')
         and all(cfg[k] == c[k] for k in ('bindings', 'parent', 'python', 'registry', 'ledger_lock')),
         'SAME_SEALED_PACKAGE_PARENT_PROCESSOR_DATA_AND_RUNTIME')
    data = Path(cfg['bindings']['dataset']); recipe = cfg['recipe']
    reference = c['tensor_trace']['reference_result']; reference_path = Path(reference['path'])
    need(digest(reference_path) == reference['sha256'], 'PINNED_PAIRED_CORE40_METADATA')
    identity = json.loads(reference_path.read_bytes())['model_identity']
    need(digest(data / 'DATASET_MANIFEST.json') == recipe['dataset_sha256'] == identity['dataset_sha256']
         and digest(data / 'statistics.json') == recipe['statistics_sha256'] == identity['statistics_sha256'],
         'SAME_SEALED_TRAIN100_MANIFEST_AND_STATISTICS_FOR_PAIRED_TRACE')
    proof = json.loads((old / 'PACKAGE_RECONSTRUCTION_ACTUAL.json').read_bytes())
    need(proof['status'] == 'PASS_ACTUAL_INDEPENDENT_PARENT_DELTA_RECONSTRUCTION'
         and proof['actual_matched_queries'] == 40
         and all(0 <= v <= 1e-6 for v in proof['max_abs_error'].values()),
         'ACTUAL_HOPPER40_PACKAGE_REFERENCE')
    storage = c['project_storage_admission']; sp = Path(storage['path'])
    need(digest(sp) == storage['sha256'], 'PINNED_TRACE_STORAGE_ADMISSION')
    a = json.loads(sp.read_bytes())
    need(a['project'] == 'CFP04-CF-007' and a['own_root'] == str(PROJECT_ROOT)
         and a['own_uid'] == 22182 and a['project_group_membership_verified'] is True
         and a['quota_remaining_bytes'] > a['predicted_new_peak_bytes'] >= 10 * 1024**3,
         'ACTUAL_PROJECT_STORAGE_FOR_BOUNDED_TRACE')
    s = dict(registered_root=str(root), source_root=c['source_root'], source_sha=c['source_sha'],
        source_files=c['source_files'], profile=PROFILE, kind=PROFILE,
        execution_site='HOPPER_SINGLE_GPU', device='cuda', physics=False, diagnostic_only=True,
        max_queries=40, GR00T_updates=0, WM_updates=0, Actor_updates=0,
        candidate_package=package, tensor_trace=c['tensor_trace'],
        target_dataset_sha256=recipe['dataset_sha256'], target_statistics_sha256=recipe['statistics_sha256'],
        parent_manifest_sha256=PARENT, bindings=c['bindings'], parent=c['parent'])
    from .task_learning_tensor_trace_v1 import validate
    validate(s)
    return s


def pbs_script(root, source, python):
    q = shlex.quote
    return '\n'.join(['#!/bin/bash -l', '#PBS -N gr00t_numtrace',
        '#PBS -P CFP04-CF-007', '#PBS -q small',
        '#PBS -l select=1:ngpus=1:gpu_model=H200:ncpus=12:mem=225GB',
        '#PBS -l walltime=01:00:00', '#PBS -m n', '#PBS -r n', '#PBS -j oe',
        '#PBS -o ' + q(str(Path(root) / 'logs/pbs.log')), 'set -eu',
        'export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'export PYTHONPATH=' + q(str(Path(source) / 'src') + ':' + str(source)),
        'export TMPDIR=' + q(str(Path(root) / 'tmp')),
        'export XDG_CACHE_HOME=' + q(str(Path(root) / 'cache')),
        'exec ' + q(str(python)) + ' -B -m scripts.wm02.task_learning_hopper_tensor_trace_v1 --config '
        + q(str(Path(root) / 'ALLOCATION_CONFIG.json')), ''])


def run(c):
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.multitask.shared_vanda import seconds
    from .task_learning_hopper_v1 import job, allocated, actual_gpu
    root = Path(c['root']); started = time.monotonic(); cpu = time.process_time()
    lease_path = root / 'control/LEASE.json'; jid = os.environ.get('PBS_JOBID')
    code = 1; failure = None; result = {}; handlers = {}
    with closing(OwnerLock(root / 'NEURAL_OWNER.lock')):
        try:
            s = configuration(c)
            need(bool(jid) and os.getuid() == 22182 and root.stat().st_uid == 22182,
                 'ACTUAL_HOPPER_COMPUTE_OWNER')
            actual = job(jid, c['pbs_server'])
            allocated(actual, jid, uid=os.getuid(), hostname=os.uname().nodename,
                      user=os.environ.get('USER'), expected_job_name=c['pbs_job_name'])
            lease = json.loads(lease_path.read_bytes())
            need(lease['status'] == 'PBS_SUBMITTED' and lease['job_id'] == jid
                 and lease['config_sha256'] == digest(root / 'ALLOCATION_CONFIG.json')
                 and not (root / 'ALLOCATION_STARTED.private.json').exists(),
                 'FRESH_TRACE_ALLOCATION_NO_REPLAY')
            remaining = seconds(actual['Resource_List']['walltime']) - seconds(actual.get('resources_used', {}).get('walltime', '00:00:00'))
            deadline = min(c['absolute_deadline_epoch'], time.time() + remaining - 120)
            need(time.time() + c['predicted_work_s'] + 1200 < deadline,
                 'FINITE_TRACE_MODEL_AND_CLOSEOUT_WINDOW')

            def stop(sig, frame): raise TimeoutError('FINITE_HOPPER_TRACE_STOP_' + str(sig))
            handlers = {n:signal.signal(n, stop) for n in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
            signal.alarm(max(1, int(deadline - time.time() - 60)))
            for name, pin in c['source_files'].items():
                need(digest(Path(c['source_root']) / name) == pin, 'SEALED_TRACE_SOURCE_' + name)
            reg = json.loads(Path(c['registry']).read_bytes())
            need(sum(e.get('path') == str(root) and e.get('source_sha') == c['source_sha']
                     and e.get('scope') == PROFILE for e in reg['entries']) == 1,
                 'CANONICAL_HOPPER_TRACE_REGISTRY')
            from scripts.pi0294.numerics import prepare_environment, configure_torch, MODE
            from scripts.pi0294.native_dataset import official_import
            prepare_environment(MODE); official_import(c['bindings']['source'])
            import torch
            import gr00t.model.gr00t_n1d7.gr00t_n1d7
            p = torch.cuda.get_device_properties(0)
            gpu = dict(count=torch.cuda.device_count(), name=p.name, uuid=str(p.uuid),
                       major=p.major, total_memory=p.total_memory)
            actual_gpu(gpu, 'H200'); configure_torch(torch, MODE); torch.set_num_threads(4)
            s.update(GPU_uuid=gpu['uuid'], GPU_name=gpu['name'], deadline_epoch=deadline,
                     job_id=jid, pbs_server=c['pbs_server'])
            journal = Journal(root / 'control/journal', root.name)

            def activate():
                alias = root / 'model_inputs/nvidia/Cosmos-Reason2-2B'
                alias.parent.mkdir(parents=True, exist_ok=True)
                need(not alias.exists() and not alias.is_symlink(), 'FRESH_TRACE_MODEL_ALIAS')
                alias.symlink_to(c['bindings']['cosmos'], target_is_directory=True)
                write(root / 'EXECUTION_SPEC.json', s)
                write(root / 'ACTUAL_GPU.private.json', gpu)
                write(root / 'ALLOCATION_STARTED.private.json', dict(job_id=jid, epoch=time.time(), source_sha=c['source_sha'], automatic_retry=False))
                current = json.loads(lease_path.read_bytes()); need(current == lease, 'TRACE_LEASE_UNCHANGED')
                current.update(status='ACTIVE', spec_sha256=digest(root / 'EXECUTION_SPEC.json'), deadline_epoch=deadline)
                write(lease_path, current)
                return dict(spec_sha256=digest(root / 'EXECUTION_SPEC.json'))

            with closing(OwnerLock(c['ledger_lock'])):
                journal.once('ACTUAL_TRACE_ALLOCATION_ACTIVATE', dict(job_id=jid, config_sha256=digest(root / 'ALLOCATION_CONFIG.json')), activate)
            from .target_gr00t_training_v1 import _load_original_parent
            from .target_gr00t_delivery_v1 import install
            from .task_learning_native_dataset_v1 import TaskLearningNativeDataset
            from scripts.pi0294.local_backbone import local_model_name
            from scripts.pi0294.neural import COSMOS_REVISION
            from scripts.pi0294.e1.model import E1ResidentModel
            from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
            from .target_gr00t_probes_v1 import measure
            from .task_learning_tensor_trace_v1 import trace
            model = _load_original_parent(s, root, 'TRACE_MODEL_LOAD')
            installed = install(model, c['candidate_package']['path'], c['candidate_package']['sha256'], c['parent']['files'])
            need(installed['global_step'] == 2000, 'EXACT_FIRST2000_TRACE_CANDIDATE')
            data = TaskLearningNativeDataset(c['bindings']['dataset'], c['bindings']['source'], c['bindings']['cosmos'])
            data.initialize()
            need(data.manifest_sha == s['target_dataset_sha256'] and data.statistics_sha == s['target_statistics_sha256'],
                 'ACTUAL_TRACE_DATASET_IDENTITIES')
            processor = Gr00tN1d7Processor.from_pretrained(Path(c['candidate_package']['path']) / 'processor',
                model_name=local_model_name(root, c['bindings']), local_files_only=True,
                transformers_loading_kwargs=dict(local_files_only=True, trust_remote_code=True,
                    cache_dir=c['bindings']['cosmos_hub'], revision=COSMOS_REVISION))
            processor.eval(); data.processor = processor; model.to('cuda').eval()
            parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            need(parameter_bytes == 12576064000 and {str(p.dtype) for p in model.parameters()} == {'torch.float32'},
                 'ORIGINAL_FULL_FP32_PARAMETER_STORAGE_NOT_SILENT_CAST')
            need(processor.embodiment_id_mapping['new_embodiment'] == 10, 'ORIGINAL_TRACE_EMBODIMENT_MAPPING')
            instance = E1ResidentModel.__new__(E1ResidentModel)
            instance.root, instance.bindings, instance.model, instance.processor = root, c['bindings'], model, processor
            instance.identity = dict(mode='CONTINUOUS_TASK_PACKAGE_RECONSTRUCTION_TRAIN40',
                global_step=2000, dataset_sha256=data.manifest_sha, statistics_sha256=data.statistics_sha,
                parent_manifest_sha256=PARENT, candidate_package=c['candidate_package'])
            with trace(s, root, model): records = measure(root, 'sealed-TRAIN40', instance, data)
            result = dict(status='ACTUAL_HOPPER_COMPLETE40_FIRST4_TENSOR_TRACE', actual_VLA_queries=len(records),
                Processor_decode_calls=len(records), actual_model_loads=1, model_identity=instance.identity,
                source_sha=c['source_sha'], GR00T_updates=0, WM_updates=0, Actor_updates=0, native_integrations=0,
                diagnostic_only=True, package_qualified=False, precision_changed=False,
                parameter_bytes=parameter_bytes,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            code = 0
        except BaseException:
            failure = traceback.format_exc()
        finally:
            if handlers:
                signal.alarm(0)
                for n, handler in handlers.items(): signal.signal(n, handler)
            result.update(exit_code=code, failure=failure, job_id=jid, wall_s=time.monotonic()-started, CPU_s=time.process_time()-cpu)
            write(root / 'RESULT.private.json', result)
            write(root / 'WORKER_EXIT.private.json', dict(exit_code=code, failure=failure, wall_s=time.monotonic()-started,
                  source_sha=c['source_sha'], job_id=jid))
            lease = json.loads(lease_path.read_bytes())
            if lease.get('job_id') == jid and lease.get('status') in ('PBS_SUBMITTED', 'ACTIVE'):
                lease.update(status='CLOSED', worker_exit_code=code, closed_epoch=time.time(), result_sha256=digest(root / 'RESULT.private.json'))
                write(lease_path, lease)
    return code
