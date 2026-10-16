"""Reproducible local synthetic CPU receipts, never a scientific run launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


def write(root, name, value):
    path = root/name
    if path.exists():raise FileExistsError('PRESERVE_PREVIOUS_PREPARATION_RECEIPT')
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(output):
    repo = Path(__file__).resolve().parents[2]
    output = Path(output).resolve()
    allowed = repo/'reports/PI0294/design/gpu-queue-dagger-prep-v1'
    if output != allowed or output.is_symlink() or os.environ.get('PBS_JOBID'):
        raise PermissionError('LOCAL_REPOSITORY_CPU_PREP_ONLY_NOT_A_PBS_OR_SITE_JOB')
    names = ['CPU_PREP_TEST_RECEIPT.json', 'CPU_ACTION_CONTRACT_TEST_RECEIPT.json',
             'ORACLE_FIXTURE_RECEIPT.json', 'COLLECTOR_FIXTURE_RECEIPT.json', 'MOCK_INVERSION_ROUNDTRIP.json']
    if any((output/name).exists() for name in names):
        raise FileExistsError('ARCHIVE_PRIOR_RECEIPTS_EXPLICITLY_BEFORE_ANOTHER_VERIFICATION')
    cpus = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, cpus)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    begin, cpu0 = time.monotonic(), time.process_time()
    child0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    with tempfile.TemporaryDirectory(prefix='dagger-cpu-fixture-') as temp:
        xml = Path(temp)/'junit.xml'
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/flowdagger',
            'scripts/pi0294/test_base_intent.py', '--junitxml', str(xml)], cwd=repo,
            text=True, capture_output=True, timeout=600)
        print(result.stdout, end='')
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError('SYNTHETIC_CPU_TESTS_FAILED_NO_PASS_RECEIPT')
        cases = []
        for case in ET.parse(xml).iter('testcase'):
            status = 'FAIL' if list(case.iter('failure')) or list(case.iter('error')) else 'SKIP' if list(case.iter('skipped')) else 'PASS'
            cases.append(dict(module=case.attrib['classname'], name=case.attrib['name'],
                              wall_s=float(case.attrib.get('time', 0)), status=status))
    if not cases or any(c['status'] != 'PASS' for c in cases):
        raise RuntimeError('ALL_REQUESTED_CPU_FIXTURES_MUST_ACTUALLY_RUN')
    import numpy as np
    import torch
    from go2arx5_vla.flowdagger.inversion import CountedVelocity, round_trip
    from go2arx5_vla.flowdagger.codec import physical_metrics
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    expert = np.tile([.3, .1, .5, 1, 0, 0, 0, 1, 0, .8], (40, 1))
    target = torch.zeros((1, 40, 12), dtype=torch.float32, device='cpu')
    target[0, :, :10] = torch.from_numpy(expert).float()
    velocity = CountedVelocity(lambda x, t: .2*x+t, limit=48)
    mock = round_trip(target, velocity, steps=4, fp_per_step=10, active_dim=10)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = time.process_time()-cpu0+child.ru_utime+child.ru_stime-child0.ru_utime-child0.ru_stime
    source_files = sorted((repo/'src/go2arx5_vla/flowdagger').glob('*.py'))+sorted((repo/'tests/flowdagger').glob('*.py'))+[Path(__file__).resolve()]
    source_hashes = {str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    receipt = dict(schema='LOCAL_CPU_PREP_TEST_RECEIPT_V1', status='PASS_SYNTHETIC_CPU_FIXTURES_ONLY',
        handoff_ref='4a9bbf38cbde90c2e08c4f4270364d801534cdf1',
        source_commit_at_test=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        tested_working_source_file_sha256=source_hashes, source_changes_not_hot_deployed=True,
        local_CPU_affinity=cpus, thread_env_cap=1, CPU_s_self_and_children=cpu,
        wall_s=time.monotonic()-begin, observed_epoch=time.time(), cases_passed=len(cases), cases=cases,
        actual_GR00T_loads=0, optimizer_steps=0, physics_or_shadow_integrations=0, new_PBS=0,
        cuda_initialized=torch.cuda.is_initialized(), selected='NONE', deployment='NONE')
    output.mkdir(parents=True, exist_ok=True)
    write(output, 'CPU_PREP_TEST_RECEIPT.json', receipt)
    for name, fragments in [
        ('CPU_ACTION_CONTRACT_TEST_RECEIPT.json', ['test_abi', 'test_codec', 'test_base_intent']),
        ('ORACLE_FIXTURE_RECEIPT.json', ['test_oracle']),
        ('COLLECTOR_FIXTURE_RECEIPT.json', ['test_contracts', 'test_aggregation', 'test_transaction'])]:
        selected = [c for c in cases if any(f in c['module'] for f in fragments)]
        write(output, name, dict(status='PASS_CPU_FIXTURE_NOT_PHYSICAL_OR_RECOVERY_QUALIFICATION',
            tests_passed=len(selected), cases=selected, physically_executed=False,
            actual_saved_student_state_queries=0, actual_recovery_qualified=False,
            original_legacy_T1_plan_on_synthetic_snapshot_tested='test_oracle' in fragments))
    write(output, 'MOCK_INVERSION_ROUNDTRIP.json', dict(status='PASS_AFFINE_TENSOR_CPU_FIXTURE_ONLY',
        fixture='v(x,t)=0.2*x+t; structural shape 1x40x12; first10 synthetic physical EEF10',
        processor='SYNTHETIC_IDENTITY_NOT_OFFICIAL_PROCESSOR', inference_steps=4, fp_per_step=10,
        velocity_evaluations=mock['velocity_evaluations'], normalized_active_MSE=mock['active_normalized_MSE'],
        normalized_full_MSE=mock['full_padded_normalized_MSE'],
        synthetic_physical_errors=physical_metrics(mock['reconstructed'][0, :, :10].numpy(), expert, expert[0]),
        real_M2048_sampler_parity='NOT_RUN', real_M2048_inversion='NOT_RUN', real_processor_roundtrip='NOT_RUN',
        no_model_weights_loaded=True, no_optimization=True, no_physics=True))
    return dict(status=receipt['status'], cases_passed=len(cases), CPU_s=cpu, wall_s=receipt['wall_s'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(run(a.output)))
