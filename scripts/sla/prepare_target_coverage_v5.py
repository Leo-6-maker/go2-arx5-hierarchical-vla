"""Finite two-CPU static map; produces no trajectories or execution grants."""
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time
import traceback

from go2arx5_vla.world_model import sla_target_coverage_v5 as coverage
from go2arx5_vla.world_model.storage import digest


def check_one(args):
    preparation, output, object_id, pose_id = args
    start = time.time(); cpu = time.process_time()
    try:
        result = coverage.compile_and_check(preparation, output, object_id, pose_id)
        result['result_sha256'] = digest(Path(output) / 'STATIC_RESULT.private.json')
    except BaseException:
        result = dict(object_id=object_id, pose_id=pose_id, passes=False, error=traceback.format_exc(),
            actual_integrations=0, policy_model_loads=0, optimizer_updates=0,
            formal_demonstrations=0, native_execution_authorized=False)
        Path(output).mkdir(exist_ok=True)
        (Path(output) / 'STATIC_ERROR.private.json').write_text(json.dumps(result, indent=2) + '\n')
    return dict(result, wall_s=time.time()-start, CPU_s=time.process_time()-cpu,
        actual_cpu_affinity=sorted(os.sched_getaffinity(0)), worker_pid=os.getpid())


def run(spec_path):
    s = json.loads(Path(spec_path).read_bytes()); root = Path(s['registered_root'])
    start = time.time()
    assert s['operation'] == coverage.VERSION and not s['physics'] and s['model_calls'] == 0
    assert s['cpus'] == [0, 1] and s['workers'] == 2 and not os.environ.get('CUDA_VISIBLE_DEVICES')
    assert digest(s['declaration']['path']) == s['declaration']['sha256']
    assert json.loads(Path(s['declaration']['path']).read_bytes()) == coverage.catalogue()
    assert time.time() < s['deadline_epoch'] - 300
    cases = root / 'cases'; cases.mkdir()
    jobs = [(s['compiler'], str(cases / (o+'-'+p)), o, p) for o in coverage.OBJECTS for p in coverage.POSES]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(check_one, jobs, chunksize=1):
            rows.append(result)
            (root / 'PROGRESS.private.json').write_text(json.dumps(dict(
                completed=len(rows), requested=50, passed=sum(r['passes'] for r in rows),
                actual_integrations=0, model_calls=0, optimizer_updates=0, epoch=time.time())) + '\n')
    result = dict(schema=coverage.VERSION, requested=50, completed=len(rows),
        passed=sum(r['passes'] for r in rows), rows=rows, source_sha=s['source_sha'],
        actual_worker_processes=len({r['worker_pid'] for r in rows}),
        wall_s=time.time()-start, worker_CPU_s=sum(r['CPU_s'] for r in rows),
        checked_saved_startup_states=sum(r.get('saved_startup_checks', 0) for r in rows),
        actual_integrations=0, policy_model_loads=0, optimizer_updates=0,
        full_demonstrations=0, D100_new=False, D300_new=False, native_execution_authorized=False,
        static_only=True, negative_cases_retained=True, automatic_retry=False)
    (root / 'RESULT.private.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    result = run(sys.argv[1])
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))
