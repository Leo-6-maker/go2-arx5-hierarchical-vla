"""One legal allocation, fresh-process canaries, then both native-loss arms."""
import argparse, hashlib, json, os, resource, subprocess, sys, time
from pathlib import Path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as output:
        json.dump(value, output, indent=2); output.write('\n'); output.flush(); os.fsync(output.fileno())
    temporary.replace(path)


def bind(root, point, checkpoint):
    """Read-only effective native BF16 identity, not a model query."""
    import torch
    from safetensors import safe_open
    from learning import file_digest
    from scope import digest
    reference = json.loads((root/'learning/WEIGHT_LOADING_REFERENCE.json').read_text())
    expected = set(reference['parameter_hashes']); hashes = {}
    shards = sorted(checkpoint.glob('*.safetensors'))
    assert shards and (checkpoint/'PI0282_RECOVERY.json').exists()
    for shard in shards:
        with safe_open(shard, framework='pt', device='cpu') as tensors:
            for name in tensors.keys():
                if name in expected:
                    assert name not in hashes
                    hashes[name] = digest(tensors.get_tensor(name).to(torch.bfloat16))
    assert set(hashes) == expected
    files = {p.name:file_digest(p) for p in shards}
    for name in ('config.json','processor_config.json','statistics.json','embodiment_id.json'):
        files[name] = file_digest(checkpoint/name)
    path = root/'design/MODEL_POINTS.json'
    points = json.loads(path.read_text()) if path.exists() else {}
    binding = dict(model_root=str(checkpoint), files=files, effective_bf16_tensor_sha256=hashes)
    assert point not in points or points[point] == binding
    points[point] = binding; write(path, points)


def call(root, name, script, arguments):
    """Wait for the actual child; a failed receipt requires owner recovery."""
    runtime = root/'runtime'; done = runtime/(name+'.done.json'); start = runtime/(name+'.start.json')
    if done.exists():
        receipt = json.loads(done.read_text()); assert receipt['returncode'] == 0, receipt
        return
    assert not start.exists(), 'Adopt recorded process; do not repeat completed optimizer work'
    command = [sys.executable,'-u',str(root/'code'/script),*arguments]
    before = resource.getrusage(resource.RUSAGE_CHILDREN); began = time.monotonic()
    with (runtime/(name+'.log')).open('ab') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        write(start, dict(pid=child.pid, parent_pid=os.getpid(), PBS_JOBID=os.environ['PBS_JOBID'], argv=command, utc_ns=time.time_ns()))
        result = child.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    write(done, dict(returncode=result, elapsed_s=time.monotonic()-began, pid=child.pid,
        user_CPU_s=after.ru_utime-before.ru_utime, system_CPU_s=after.ru_stime-before.ru_stime))
    assert result == 0, (name, result, 'Original attempt and accounting retained')


def retire_consumed_canary(root, arm):
    """Only disposable checkpoints consumed by a passing exact-restore proof."""
    import shutil
    from learning import file_digest
    base = root/'learning'/arm
    proof = base/'CANARY_COMPARISON.json'
    assert json.loads(proof.read_text())['status'] == 'PASS_FULL_RESUME_CANARY'
    inventory = base/'CANARY_PAYLOAD_RETIREMENT.json'
    if inventory.exists():return
    entries = []
    for relative in ('continuous/checkpoint-8','split/checkpoint-4','split/checkpoint-8'):
        path = (base/relative).resolve()
        assert path.is_relative_to(base.resolve()) and not path.is_symlink()
        recovery = json.loads((path/'PI0282_RECOVERY.json').read_text())
        assert recovery['cursor'] in (32,64)
        files = {str(p.relative_to(path)):dict(bytes=p.stat().st_size,sha256=file_digest(p)) for p in path.rglob('*') if p.is_file()}
        # Keep small recovery/scope records next to the complete comparison.
        for name in ('PI0282_RECOVERY.json','PI0282_SCOPE_PROOF.json','trainer_state.json'):
            shutil.copyfile(path/name,base/(relative.replace('/','_')+'_'+name))
        entries.append(dict(path=str(path), files=files, bytes=sum(v['bytes'] for v in files.values())))
    write(inventory, dict(status='PREVIEW_ONLY_DISPOSABLE_CANARY_AFTER_PASS', comparison_sha256=file_digest(proof),entries=entries))
    for item in entries:shutil.rmtree(item['path'])
    assert all(not Path(item['path']).exists() for item in entries)
    write(inventory, dict(status='DELETED_CONSUMED_CANARY_PAYLOAD_SMALL_PROOFS_RETAINED', comparison_sha256=file_digest(proof),entries=entries))


def run(root, base):
    import fcntl
    root = Path(root); base = Path(base); runtime = root/'runtime'; runtime.mkdir(parents=True,exist_ok=True)
    assert os.environ.get('PBS_JOBID') and os.environ.get('CUDA_VISIBLE_DEVICES'), 'Use scheduler assigned GPU'
    with (runtime/'TRAINING_ALLOCATION.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        start = runtime/'TRAINING_ALLOCATION.json'
        assert not start.exists(), 'No duplicate allocation; inspect existing job before exact recovery'
        write(start, dict(PBS_JOBID=os.environ['PBS_JOBID'],pid=os.getpid(),utc_ns=time.time_ns(),scheduler_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']))
        layout = json.loads((root/'design/TRAINING_LAYOUT.json').read_text())
        common = ['--root',str(root),'--base',str(base),'--data-root',str(root/'data'),'--microbatch',str(layout['microbatch'])]
        # R2 prioritizes the task policy; INTERFACE is explicitly deferred.
        for arm in ('ACTION2048',):
            for phase,resume,stop in (('continuous',0,8),('split',0,4),('split',4,8)):
                call(root,f'{arm}-{phase}-{resume}-{stop}','train.py',common+['--arm',arm,'--phase',phase,'--resume',str(resume),'--stop',str(stop)])
            for phase in ('continuous','split'):
                point=arm+'-CANARY-'+phase; bind(root,point,root/'learning'/arm/phase/'checkpoint-8')
                call(root,point,'inference.py',['--root',str(root),'--point',point,'--canary','--arm',arm,'--phase',phase])
            call(root,arm+'-canary-compare','canary.py',['--root',str(root),'--arm',arm])
            retire_consumed_canary(root,arm)
            call(root,arm+'-formal-2048','train.py',common+['--arm',arm,'--phase','main','--stop','2048'])
            for step in (1024,2048):
                checkpoint=root/'learning'/arm/'main'/f'checkpoint-{step}'
                assert json.loads((checkpoint/'PI0282_RECOVERY.json').read_text())['cursor']==step*8
                bind(root,arm.replace('2048','')+str(step),checkpoint)
        write(runtime/'TRAINING_COMPLETE.json',dict(status='ACTION_BASE_PHASE_COMPLETE_INTERFACE_DEFERRED_R2',PBS_JOBID=os.environ['PBS_JOBID'],utc_ns=time.time_ns(),selected='NONE',deployment='NONE'))


def self_check(root):
    import tempfile
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='allocation-cpu-check-',dir=root) as temporary:
        path=Path(temporary); (path/'code').mkdir(); (path/'runtime').mkdir()
        (path/'code/probe.py').write_text("import sys; assert sys.argv[1:]==['fixed']; print('CPU_CHILD_ONLY')\n")
        previous=os.environ.get('PBS_JOBID'); os.environ['PBS_JOBID']='CPU_CHECK_NOT_AN_ALLOCATION'
        try:
            call(path,'probe','probe.py',['fixed']); call(path,'probe','probe.py',['fixed'])
            assert len(list((path/'runtime').glob('*.start.json')))==1
            assert json.loads((path/'runtime/probe.done.json').read_text())['returncode']==0
        finally:
            if previous is None:os.environ.pop('PBS_JOBID')
            else:os.environ['PBS_JOBID']=previous
    print('PASS_CPU_CHILD_WAIT_AND_NO_DUPLICATE_CALL; model/optimizer/physics/PBS=0')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--base');parser.add_argument('--self-check',action='store_true')
    arguments=parser.parse_args()
    if arguments.self_check:self_check(arguments.root)
    else:assert arguments.base;run(arguments.root,arguments.base)
