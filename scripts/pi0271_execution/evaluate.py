"""Two serialized frozen policy loads, exact registered requests, first-valid reuse."""
import fcntl
import json
import os
import subprocess
from pathlib import Path
from support import BASE, ROOT, read, sha, write
from worker import admitted
from quality_offline import message


def main():
    admitted()
    plan_path = ROOT/'design/QUERY_PLAN.json'; plan = read(plan_path)
    with (ROOT/'evidence/inference.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for point in ('PREFIX4096', 'AGG4096'):
            rows = [r for r in plan['rows'] if r['point'] == point]
            missing = [r for r in rows if not Path(r['request']['output']).exists() or
                       not Path(r['request']['output']).with_suffix('.receipt.json').exists()]
            # ponytail: one existing worker per policy; no competing per-panel model copies.
            if not missing: continue
            env = dict(os.environ); env.pop('LD_PRELOAD', None); env.pop('LD_LIBRARY_PATH', None)
            env.update(PYTHONPATH=str(ROOT/'code'), CUDA_VISIBLE_DEVICES='0')
            with (ROOT/'evidence'/('worker-'+point+'.log')).open('ab') as log:
                worker = subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'), str(ROOT/'code/worker.py'),
                    '--point', point], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
                try:
                    assert message(worker) == dict(ready=True, point=point)
                    for row in rows:
                        request = row['request']
                        worker.stdin.write(json.dumps(request)+'\n'); worker.stdin.flush()
                        answer = message(worker)
                        assert answer['status'] in ('VALID', 'VALID_REUSED') and answer['output'] == request['output']
                        print('REGISTERED_OUTPUT', point, row['panel'], row['input_id'], row['j'], flush=True)
                    worker.stdin.close(); assert worker.wait() == 0
                finally:
                    if worker.poll() is None: worker.terminate(); worker.wait()
        bindings = {p:read(ROOT/'model_bindings'/(p+'.json')) for p in ('PREFIX4096','AGG4096')}
        registries = {name:read(ROOT/'design'/name) for name in ('LATE_SUPPORT_CARD.json','VISITED_QUERY_CARD.json')}
        outputs = []
        for row in plan['rows']:
            path = Path(row['request']['output']); receipt = read(path.with_suffix('.receipt.json'))
            assert receipt['status'] == 'VALID' and receipt['point'] == row['point'] and receipt['binding'] == bindings[row['point']]
            assert receipt['input_sha256'] == row['request']['expected_condition_sha256'] and receipt['output_sha256'] == sha(path)
            request=row['request'];entry=next(r for r in registries[request['noise_registry']]['noise'] if r['file']==request['noise_file'])
            key=next(r for r in entry['keys'] if r['name']==request['noise_name'])
            assert receipt['noise_sha256']==key['noise_sha256'] and sha(request['noise_file'])==entry['sha256']
            assert sha(request['input'])==row['input_file_sha256'] and receipt['native_S4'] and receipt['heads']==4
            outputs.append(dict(point=row['point'], panel=row['panel'], input_id=row['input_id'], j=row['j'],
                path=str(path), sha256=sha(path), receipt_sha256=sha(path.with_suffix('.receipt.json'))))
        write(ROOT/'evidence/QUERY_OUTPUTS_COMPLETE.json', dict(status='ALL_FIXED_PI0271_OUTPUTS_COMPLETE',
            query_plan_sha256=sha(plan_path), rows=outputs, neural_updates=0, teacher_calls_by_this_process=0,
            selected_checkpoint='NONE', deployment_selected_checkpoint='NONE'))


if __name__ == '__main__': main()
