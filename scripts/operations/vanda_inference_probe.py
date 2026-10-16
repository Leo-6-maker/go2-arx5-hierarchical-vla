"""A40 inference qualification on one saved TRAIN observation; never E0.

Uses the unchanged ResidentModel.generate primitive under a separate Vanda
admission contract. It neither dispatches an action nor evaluates a task.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def validate(path, *, execute=False):
    m = read(path)
    root = Path(m['output_root']).resolve()
    assert Path(path).resolve().parent == root, 'OUTPUT_BINDING'
    assert root.is_relative_to(Path(m['registered_runs_root']).resolve()), 'OUTPUT_ESCAPE'
    assert m['owner_uid'] == os.getuid(), 'OWNER'
    assert m['authority']['direct_user_project_gpu_authorization'], 'AUTHORITY'
    assert time.time() < m['deadline_epoch'], 'EXPIRED'
    assert m['scope'] == 'SAVED_TRAIN_INFERENCE_PROBE_NO_ROBOT_OR_OPTIMIZER', 'SCOPE'
    assert m['execution']['profile'] == 'vanda-neural-probe', 'PROFILE'
    assert m['limits']['ngpus'] == 1 and m['limits']['queries'] == 17, 'BOUNDS'
    lease = read(m['lease'])
    assert lease['run_uuid'] == m['run_uuid'] and lease['status'] == 'ACTIVE', 'LEASE'
    for name, expected in m['sealed_files'].items():
        assert sha(name) == expected, 'SEALED_FILE'
    source = Path(m['bindings']['source'])
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == '51d4c89f72fda44cbf77285c6a8114b52676b8a1', 'OFFICIAL_SOURCE'
    assert subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD', '--']).returncode == 0, 'DIRTY_OFFICIAL'
    prep = read(m['asset_progress'])
    assert prep['status'] == 'PASS_FIXED_INFERENCE_BYTES_NO_MODEL_LOAD', 'ASSETS_NOT_READY'
    assert read(m['observation'])['provenance']['split'] == 'TRAIN', 'TRAIN_ONLY'
    assert read(m['observation'])['provenance']['kind'] == 'SAVED_ORIGINAL_OBSERVATION_FOR_INTERFACE_TIMING_ONLY', 'SAVED_INPUT'
    if execute:
        assert socket.gethostname().split('.')[0].lower().startswith('gn-a40-'), 'COMPUTE_HOST_REQUIRED'
        jobid = os.environ.get('PBS_JOBID')
        assert jobid and read(root/'SUBMISSION.private.json')['job_id'] == jobid, 'OWN_JOB'
        result = subprocess.run(['/opt/pbs/bin/qstat', '-f', '-F', 'json', jobid],
                                capture_output=True, text=True, timeout=20, check=True)
        job = json.loads(result.stdout)['Jobs'][jobid]
        assert job['job_state'] == 'R' and job['project'] == m['project'], 'RUNNING_PROJECT'
        assert job['Job_Owner'].split('@')[0] == m['owner_user'], 'PBS_OWNER'
        assert int(job['Resource_List']['ngpus']) == 1, 'ONE_GPU'
        h, mins, secs = map(int, job['Resource_List']['walltime'].split(':'))
        assert h*3600 + mins*60 + secs <= 1800, 'ALLOCATION_BOUND'
        hardware = read(m['stage0_result'])
        assert hardware['status'] == 'HOST_CHECKS_COMPLETED_SOFTWARE_GAPS_RECORDED', 'STAGE0_INCOMPLETE'
        checks = {row['check']: row for row in hardware['checks'] if row['check'] != 'container'}
        assert checks['gpu']['exit_code'] == 0 and checks['cuda_torch']['exit_code'] == 0, 'CUDA_NOT_QUALIFIED'
    return m


def probe(path):
    m = validate(path, execute=True)
    root = Path(m['output_root'])
    lock = (root/'PROBE_OWNER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root/'RESULT.private.json').exists(), 'NO_REPEATED_PROBE'
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))
    resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))
    def stop(signum, frame):
        raise TimeoutError('ALLOCATION_TERMINATION')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(1600)
    started = time.time()
    result = dict(run_uuid=m['run_uuid'], job_id=os.environ['PBS_JOBID'],
                  status='INITIALIZING', model_loads=0, queries=[],
                  task_evaluations=0, physics_steps=0, WBC_launches=0, optimizer_steps=0,
                  input_kind='SAVED_TRAIN_OBSERVATION_NOT_LIVE_CLOSED_LOOP')
    try:
        from scripts.pi0294.numerics import prepare_environment, configure_torch, MODE
        prepare_environment(MODE)
        import numpy as np
        import torch
        assert torch.cuda.device_count() == 1 and 'A40' in torch.cuda.get_device_name(0), 'ONE_A40'
        result['numerics'] = configure_torch(torch, MODE)
        result['torch'] = torch.__version__
        result['gpu'] = torch.cuda.get_device_name(0)
        from scripts.pi0294.resident import ResidentModel
        from scripts.pi0292.train_adapter import official_import, build_processor
        from scripts.pi0294.local_backbone import local_model_name
        from scripts.pi0294.neural import COSMOS_REVISION
        checkpoint = Path(m['checkpoint'])
        identity = read(m['identity'])
        for name, expected in identity['files'].items():
            p = checkpoint/name
            assert p.resolve().is_relative_to(checkpoint.resolve()) and sha(p) == expected, 'MODEL_BYTES'
        for name, expected in m['cosmos_files'].items():
            assert sha(Path(m['bindings']['cosmos'])/name) == expected, 'COSMOS_BYTES'
        official_import(m['bindings']['source'])
        from transformers import AutoModel
        import gr00t.model.gr00t_n1d7.gr00t_n1d7

        class VandaProbeModel(ResidentModel):
            # The parent H200/E0 admission is not invoked by this new platform;
            # validate() above owns this distinct, bounded engineering scope.
            def __init__(self):
                self.model, loading = AutoModel.from_pretrained(
                    checkpoint, local_files_only=True, trust_remote_code=True,
                    output_loading_info=True, model_name=local_model_name(root, m['bindings']),
                    transformers_loading_kwargs=dict(local_files_only=True, trust_remote_code=True,
                        cache_dir=m['bindings']['cosmos_hub'], revision=COSMOS_REVISION),
                    tune_llm=False, tune_visual=False, tune_projector=True,
                    tune_diffusion_model=True, tune_vlln=True, state_dropout_prob=0.,
                    backbone_trainable_params_fp32=True, load_bf16=False)
                assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys')), 'LOADING_MISMATCH'
                self.model.to('cuda').eval()
                self.model.requires_grad_(False)
                assert self.model.action_head.num_inference_timesteps == 4, 'FOUR_STEPS'
                statistics = read(checkpoint/'statistics.json')
                self.processor = build_processor(m['bindings']['source'], m['bindings']['cosmos'], statistics)
                self.identity = identity

        load_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        model = VandaProbeModel()
        torch.cuda.synchronize()
        result.update(model_loads=1, load_wall_s=time.monotonic()-load_start)
        payload = read(m['observation'])['payload']
        for index in range(17):
            torch.cuda.synchronize()
            began = time.monotonic()
            output = model.generate(payload, seed=29400+index)
            torch.cuda.synchronize()
            elapsed = time.monotonic()-began
            h40 = output['physical_H40']
            assert h40.shape == (40, 10) and np.isfinite(h40).all(), 'H40_FINITE'
            np.savez_compressed(root/f'query-{index:02}.npz', physical_H40=h40, normalized=output['normalized'])
            result['queries'].append(dict(index=index, seed=29400+index, wall_s=elapsed,
                warmup=index == 0, physical_H40_sha256=hashlib.sha256(h40.tobytes()).hexdigest(),
                normalized_outside_unit_components=output['normalized_outside_unit_components']))
        times = [row['wall_s'] for row in result['queries'] if not row['warmup']]
        result.update(status='PASS_SAVED_OBSERVATION_A40_INFERENCE_ONLY',
                      timed_queries=16, median_wall_s=float(np.median(times)),
                      p95_wall_s=float(np.percentile(times, 95)),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      latency_sample_limit='16_REPEATED_SAVED_INPUTS_NOT_PRODUCTION_P95')
    except BaseException as error:
        result.update(status='ENGINEERING_INCOMPLETE', error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(completed_epoch=time.time(), elapsed_s=time.time()-started,
                      cpu_s=usage.ru_utime+usage.ru_stime+child.ru_utime+child.ru_stime)
        with (root/'RESULT.private.json').open('x') as stream:
            json.dump(result, stream, indent=2)
        lease = read(m['lease'])
        lease.update(status='CLOSED', result_status=result['status'], completed_epoch=time.time())
        Path(m['lease']).write_text(json.dumps(lease, indent=2)+'\n')
        lock.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = probe(args.manifest) if args.execute else dict(status='PREPARED_NO_NEURAL_EXECUTION', run_uuid=validate(args.manifest)['run_uuid'])
    print(json.dumps(result, indent=2))
