"""Compare verified matched benchmark ledgers without generating model outputs."""
import argparse
import json
from pathlib import Path
import subprocess

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, canonical_hash, cursors
from scripts.pi0292.durable import read, write
from scripts.sla.benchmark_scale20k import ARM, SCOPE, START, UPDATES, WARMUP, summarize


def candidate(run, job):
    root = Path(run).resolve(strict=True)
    parent = read(root/'RESULT.private.json')
    if (job['job_state'] != 'F' or job['Exit_status'] != 0 or parent['exit_code']
            or parent['PBS_JOBID'] != job['id'] or Path(parent['candidate']['path']) != root/'candidate/RESULT.private.json'):
        raise ValueError('BENCHMARK_TIMING_REQUIRES_A_REAL_CLOSED_SUCCESSFUL_PBS_WORKER')
    outcome = read_pin(parent['candidate'])
    before, after = cursors(ARM,START), cursors(ARM,START+UPDATES)
    if (outcome['scope'] != SCOPE or not outcome['engineering_only'] or outcome['eligible_as_formal_parent']
            or outcome['actual_optimizer_updates'] != UPDATES or outcome['actual_aux_generations'] != 4*UPDATES
            or outcome['actual_optimization_visits'] != {d:after[d]-before[d] for d in DOMAINS}
            or outcome['scientific_updates'] or outcome['native_requests'] or outcome['exit_code']):
        raise ValueError('BENCHMARK_ACTUAL_UPDATES_AND_VISITS_MUST_MATCH_THE_FIXED_ENGINEERING_ARM')
    ledger = root/'candidate/UPDATES.private.jsonl'
    raw = ledger.read_bytes()
    if not raw.endswith(b'\n'):
        raise ValueError('BENCHMARK_UPDATE_LEDGER_IS_NOT_DURABLE')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != UPDATES or summarize(rows) != outcome['measured_20']:
        raise ValueError('BENCHMARK_TIMER_SUMMARY_DIFFERS_FROM_COMPLETE_LEDGER')
    visits = []
    for offset, row in enumerate(rows, START+1):
        if row['step'] != offset or row['cursors'] != cursors(ARM, offset):
            raise ValueError('BENCHMARK_UPDATE_CURSOR_CHANGED')
        visits.append(dict(step=offset,lr=row['learning_rate'],cursors=row['cursors'],
                           visits={d:row['domains'][d]['visit_slice_sha256'] for d in DOMAINS}))
    cp = outcome['engineering_checkpoint']
    marker = read_pin(dict(path=str(Path(cp['path'])/'COMPLETE.json'),sha256=cp['sha256']))
    if (marker['step'] != START+UPDATES or not marker['binding'].get('engineering_only')
            or marker['binding']['variant'] != outcome['variant'] or len(marker['arrays']) != 140):
        raise ValueError('BENCHMARK_CHECKPOINT_MUST_REMAIN_ENGINEERING_ONLY')
    return dict(outcome=outcome,job=job['id'],ledger_sha256=sha(ledger),visit_prefix=visits,
                array_hashes={r['key']:r['sha256'] for r in marker['arrays']},
                gradient_norm=[row['grad_norm'] for row in rows],
                peak_allocator_bytes=max((row['actual_GPU_memory_stats'] or {}).get('peak_bytes_in_use',0) for row in rows),
                elapsed_30_s=sum(r['wall_s'] for r in rows))


def combine(passes):
    if [p['outcome']['microbatch'] for p in passes] != [1,2,4]:
        raise ValueError('THREE_PREDEFINED_A40_MICROBATCH_RUNS_REQUIRED')
    baseline = passes[0]
    rows = []
    reference = baseline['outcome']
    for p in passes:
        out = p['outcome']
        if (p['visit_prefix'] != baseline['visit_prefix'] or set(p['array_hashes']) != set(baseline['array_hashes'])
                or any(out[k] != reference[k] for k in ('source_sha','parent_sha256','learning_rates_sha256',
                                                        'visit_banks_sha256','device_name'))):
            raise ValueError('BENCHMARK_FIXED_PARENT_AND_EVERY_UPDATE_INPUT_MUST_MATCH')
        timings = out['measured_20']
        speed = reference['measured_20']['update_wall']['p50']/timings['update_wall']['p50']
        exact = sum(digest == baseline['array_hashes'][key] for key,digest in p['array_hashes'].items())
        rows.append(dict(variant=out['variant'],PBS=p['job'],checkpoint=out['engineering_checkpoint'],
            durable_update_ledger_sha256=p['ledger_sha256'],actual_updates=out['actual_optimizer_updates'],
            visits=out['actual_optimization_visits'],H40_generations=out['actual_aux_generations'],
            exact_checkpoint_array_hashes_vs_mb1=exact,checkpoint_state_array_denominator=140,
            measured_20=timings,wall_s_for_30_updates=p['elapsed_30_s'],save_s=out['save_s'],
            parent_restore_s=out['restore_parent_s'],allocator_peak_bytes=p['peak_allocator_bytes'],
            max_grad_norm_delta=max(abs(a-b) for a,b in zip(p['gradient_norm'],baseline['gradient_norm'],strict=True)),
            warm_p50_speedup_vs_mb1=speed,reached_15pct_wall_speedup=speed>=1.15,eligible_for_adoption=False))
    return dict(schema='SLA_A40_MATCHED_BENCHMARK_LEDGER_ACTUAL_V1',frozen_parent=reference['parent_sha256'],
        verified_update_input_prefix_sha256=canonical_hash(baseline['visit_prefix']),
        warmup_steps=WARMUP,timed_steps=UPDATES-WARMUP,rows=rows,
        actual_engineering_updates=sum(r['actual_updates'] for r in rows),
        output_replay_generations=0,new_optimizer_updates_for_science=0,new_native_requests=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    spec=read_pin(dict(path=args.spec,sha256=args.sha256))
    if args.output.exists():
        raise FileExistsError(args.output)
    ids=[row['job_id'] for row in spec['variants']]
    check=subprocess.check_output(['/opt/pbs/bin/qstat','-x','-f','-F','json',*ids],text=True,timeout=30)
    jobs=json.loads(check)['Jobs']
    passes=[candidate(row['root'],dict(jobs[row['job_id']],id=row['job_id'])) for row in spec['variants']]
    write(args.output,combine(passes))


if __name__ == '__main__':
    main()
