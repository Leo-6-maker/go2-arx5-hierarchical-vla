"""Finite registered Core frozen-feature qualification and 4-vs-2+2 worker."""
import argparse
import json
import os
import resource
import time
import traceback
from pathlib import Path


def admitted(spec, spec_path):
    from go2arx5_vla.world_model.task_learning_features_v1 import SCOPE, need
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.target_gr00t_core_v1 import gpu
    need(os.getuid() == 2009 and os.uname().nodename == 'corelab-4080s' and not os.environ.get('PBS_JOBID'), 'REAL_CORE_SITE')
    root = Path(spec['root'])
    need(root.resolve() == root and str(root).startswith('/mnt/local/hdd1/UserData/liuyu-nfs/Projects/GR00T-WBC/WM02/'), 'REGISTERED_LOCAL_STORAGE')
    need(spec['scope'] == SCOPE and spec['physics'] is False and spec['WBC_changes'] is False, 'NEURAL_ONLY_SCOPE')
    lease = json.loads((root / 'control/LEASE.json').read_text())
    need(lease['status'] == 'ACTIVE' and lease['spec_sha256'] == digest(spec_path)
         and time.time() + 1200 < spec['deadline_epoch'], 'FRESH_FINITE_LEASE')
    registry = json.loads(Path(spec['registry']).read_text())
    need(any(e.get('path') == str(root) and e.get('scope') == SCOPE
             and e.get('source_sha') == spec['source_sha'] for e in registry['entries']), 'ORIGINAL_REGISTRY')
    for name, pin in spec['source_files'].items():
        need(digest(Path(spec['source_root']) / name) == pin, 'SEALED_SOURCE_' + name)
    for name, pin in spec['dataset_files'].items():
        need(digest(Path(spec['bindings']['dataset']) / name) == pin, 'SEALED_DATA_' + name)
    for name, identity in spec['parent_file_identity'].items():
        stat = (Path(spec['parent']['path']) / name).stat()
        need(dict(size=stat.st_size, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns) == identity,
             'QUALIFIED_PARENT_IDENTITY_' + name)
    actual = gpu(spec)
    need(actual['free_bytes'] >= spec['allocator_limit_bytes'] + 2 * 1024**3, 'ACTUAL_SHARED_GPU_HEADROOM')
    return root


def run_phase(spec, root, phase):
    from go2arx5_vla.world_model import task_learning_features_v1 as features
    from scripts.pi0294.numerics import prepare_environment, configure_torch, MODE
    from scripts.pi0294.native_dataset import official_import
    from scripts.pi0292.durable import write
    prepare_environment(MODE)
    official_import(spec['bindings']['source'])
    import torch
    from transformers import set_seed
    from go2arx5_vla.world_model.target_gr00t_training_v1 import _load_original_parent, action_contract, gpu_identity
    from go2arx5_vla.world_model.target_gr00t_dataset_v1 import TargetNativeDataset
    from go2arx5_vla.world_model.target_gr00t_core_v1 import trim_unused_logits
    configure_torch(torch, MODE)
    torch.set_num_threads(4)
    set_seed(spec['recipe']['seed'])
    gpu_identity(torch, dict(spec, execution_site='CORELAB_4080S'))
    torch.cuda.reset_peak_memory_stats()
    output = root / phase
    output.mkdir(exist_ok=False)
    (output / 'model_inputs/nvidia').mkdir(parents=True)
    (output / 'model_inputs/nvidia/Cosmos-Reason2-2B').symlink_to(spec['bindings']['cosmos'], target_is_directory=True)
    def guard():
        features.need(time.time() + 1200 < spec['deadline_epoch'] and not (root / 'STOP.json').exists(), 'FINITE_WINDOW')
    model = _load_original_parent(spec, output, 'MODEL_LOAD')
    contract = action_contract(model, 'INTERFACES_LAST2_DIT')
    write(output / 'PARAMETER_CONTRACT.json', contract.describe())
    guard()
    if phase == 'prepare':
        trim_unused_logits(model)
        data = TargetNativeDataset(spec['bindings']['dataset'], spec['bindings']['source'], spec['bindings']['cosmos'],
                                 for_optimization=True, optimization_site='CORELAB_4080S')
        data.initialize()
        result = features.qualify_cache(model, contract, data, output, guard)
        write(output / 'RESULT.json', result)
        return result
    endpoint = 2 if phase == 'split2' else 200 if phase == 'fit200' else 4
    resume = root / 'split2/checkpoint-2' if phase == 'resume4' else None
    checkpoint = features.train_cached(model, contract, output, root / 'prepare/feature_cache.pt', spec['recipe'],
                                       endpoint, guard, resume=resume)
    return dict(checkpoint=str(checkpoint), global_step=endpoint)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'continuous4', 'split2', 'resume4', 'fit200'])
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    root = admitted(spec, args.spec)
    from scripts.pi0292.durable import OwnerLock, write
    from go2arx5_vla.world_model.task_learning_features_v1 import need, compare_four
    if args.phase:
        started = time.monotonic()
        result = None
        failure = None
        code = 1
        try:
            result = run_phase(spec, root, args.phase)
            code = 0
            return 0
        except BaseException:
            failure = traceback.format_exc()
            raise
        finally:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            write(root / (args.phase + '_EXIT.json'), dict(exit_code=code, result=result, failure=failure,
                wall_s=time.monotonic() - started, CPU_s=usage.ru_utime + usage.ru_stime,
                peak_RSS_KiB=usage.ru_maxrss))
    import subprocess
    from contextlib import closing
    started = time.monotonic()
    stages = []
    code = 1
    failure = None
    with closing(OwnerLock(Path(spec['owner_root']) / 'NIGHT1_NEURAL_OWNER.lock')):
        need(not (root / 'WORKER_STARTED.json').exists(), 'NEVER_REPLAY_UNKNOWN_WORKER')
        stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        write(root / 'WORKER_STARTED.json', dict(pid=os.getpid(), proc_start=stat[19], epoch=time.time()))
        try:
            for phase in ['prepare', 'continuous4', 'split2', 'resume4']:
                with (root / (phase + '.stdout')).open('xb') as out, (root / (phase + '.stderr')).open('xb') as err:
                    command = [spec['python'], '-B', '-m', 'scripts.wm02.task_learning_core_v1',
                               '--spec', str(args.spec), '--phase', phase]
                    child = subprocess.Popen(command, stdout=out, stderr=err)
                    write(root / 'CURRENT_CHILD.json', dict(pid=child.pid, phase=phase, epoch=time.time()))
                    try:
                        status = child.wait(timeout=max(1, spec['deadline_epoch'] - time.time() - 1200))
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:
                            child.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        status = 124
                stages.append(dict(phase=phase, exit_code=status))
                write(root / 'STAGES_ACTUAL.json', dict(stages=stages))
                need(status == 0, 'PHASE_FAILED_' + phase)
            proof = compare_four(root / 'continuous4/checkpoint-4', root / 'resume4/checkpoint-4', root / 'split2/checkpoint-2')
            write(root / 'FOUR_VERSUS_TWO_PLUS_TWO_ACTUAL.json', proof)
            write(root / 'RESULT.json', dict(status='CORE_MINIMAL_TRAIN_RESTORE_READY', proof=proof,
                stages=stages, task_successes=0, physics_integrations=0, formal_training=False,
                next_gate='FINITE_FIT4_AND_HOPPER_CANARY; NEW_CONTACT_DATA_IN_PARALLEL'))
            code = 0
            return 0
        except BaseException:
            failure = traceback.format_exc()
            write(root / 'RESULT.json', dict(status='CORE_DIAGNOSTIC_FAILURE_PRESERVED', stages=stages, failure=failure))
            raise
        finally:
            write(root / 'WORKER_EXIT.json', dict(exit_code=code, wall_s=time.monotonic() - started, failure=failure))
            lease_path = root / 'control/LEASE.json'
            lease = json.loads(lease_path.read_text())
            need(lease['status'] == 'ACTIVE', 'LEASE_NOT_REOPENED')
            lease.update(status='CLOSED', worker_exit_code=code, closed_epoch=time.time())
            write(lease_path, lease)


if __name__ == '__main__':
    raise SystemExit(main())
