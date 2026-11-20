"""Run the matched ADAPT engineering probes sequentially in one existing PBS allocation."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from go2arx5_vla.sla_adapter.adapt5k_contract import PLAN
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.adapt5k_vanda import pbs_history, pbs_site
from scripts.sla.benchmark_adapt5k import SCOPE, VARIANTS, compare, exposure_fingerprint, phase_window, timing_summary
from scripts.sla.scale20k_vanda_offline import read_pbs_with_auth_retry
from scripts.wm02.multisource_d4_vanda import pin


def archived_serial(plan, spec, config):
    source = plan['archived_serial']
    previous = read_pin(source['terminal'])
    baseline_site = read_pin(source['site'])
    original_contract = read_pin(source['contract'])
    original = read_pbs_with_auth_retry(pbs_history,previous['job_id'])
    ledger = Path(source['ledger']['path'])
    if (pin(ledger) != source['ledger'] or ledger.parent.name != 'serial'
            or ledger.parent.parent != Path(source['terminal']['path']).parent
            or previous['scope'] != SCOPE or previous['status'] != 'ADAPT_PROFILE_REQUIRES_RECONCILIATION'
            or not previous['engineering_only'] or previous['completed'] or previous['unresolved_role'] != 'serial'
            or previous['exit_code'] != 1 or original['job_state'] != 'F' or original.get('Exit_status') != 1
            or original['project'] != config['project'] or original['Job_Owner'].split('@')[0] != config['user']
            or original_contract['parent_checkpoint_sha256'] != plan['parent_checkpoint_sha256']
            or original_contract['training_card_sha256'] != spec['training_card']['sha256']
            or any((ledger.parent.parent/name).exists() for name in VARIANTS[1:])):
        raise PermissionError('ADAPT_REMAINING_ROLES_NEED_COLD_EXACT_SERIAL_TRACE_FAILURE')
    raw = ledger.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    if (not raw.endswith(b'\n') or [row['measurement'] for row in rows] != ['warmup']*3+['timed']*5
            or [row['phase_step'] for row in rows] != list(range(plan['phase_start']+1,plan['phase_start']+9))):
        raise ValueError('ADAPT_ARCHIVED_SERIAL_MUST_PRESERVE_ONLY_ITS_EIGHT_DURABLE_UPDATES')
    return dict(
        prior_job=previous['job_id'],GPU_uuid=baseline_site['GPU_uuid'],host=baseline_site['host'],
        matched_exposure_sha256=exposure_fingerprint(rows),throughput=timing_summary(rows[3:]),
        committed_updates=8,completed_H40=32,serial_state_fingerprint_available=False)


def compare_remaining(baseline, effects):
    left,right = effects
    for name,effect in zip(VARIANTS[1:],effects,strict=True):
        if (effect['variant'] != name or effect['status'] != 'ADAPT_ENGINEERING_PROBE_COMPLETE'
                or effect['actual_optimizer_updates'] != 8 or effect['committed_ledger_updates'] != 8
                or effect['completed_H40'] != 32 or len(effect['matched_end_state']) != 140
                or effect['matched_exposure_sha256'] != baseline['matched_exposure_sha256']
                or effect['device']['GPU_uuid'] != baseline['GPU_uuid']
                or effect['device']['host'] != baseline['host'] or effect['native_requests']
                or effect['scientific_updates'] or not effect.get('engineering_checkpoint')):
            raise ValueError('ADAPT_REMAINING_ROLE_EFFECT_NOT_MATCHED')
    if any(left[key] != right[key] for key in ('source_sha','parent_checkpoint_sha256',
            'source_training_card_sha256','engineering_contract','device')):
        raise ValueError('ADAPT_REMAINING_ROLES_DO_NOT_SHARE_EXACT_ENGINEERING_IDENTITY')
    return dict(baseline=baseline,same_measured_GPU=True,serial_state_equivalence=None,
        candidate_state_identical=left['matched_end_state']==right['matched_end_state'],
        candidate={row['variant']:dict(throughput=row['throughput'],
            speedup_vs_archived_serial=row['throughput']['FM_windows_per_second']/baseline['throughput']['FM_windows_per_second'],
            checkpoint=row['engineering_checkpoint']) for row in effects},
        engineering_only=True,eligible_as_formal_parent=False,native_requests=0)


def run(spec):
    root = Path(spec['root']).resolve(strict=True)
    source = Path(__file__).resolve().parents[2]
    if (root.stat().st_uid!=os.getuid() or (root/'RESULT.private.json').exists()
            or (root/'ACTIVATED.private.json').exists() or spec['scope']!=SCOPE or spec['plan']!=PLAN
            or spec['job_id']!=os.environ.get('PBS_JOBID')
            or spec['source_sha']!=(source/'SOURCE_SHA.private').read_text().strip()
            or spec['hardware_allowed'] is not False or spec['physics'] is not False):
        raise PermissionError('ADAPT_PROFILE_REQUIRES_OWN_EXISTING_PBS_ALLOCATION')
    plan = read_pin(spec['engineering_contract'])
    cold = read_pin(spec['sealed_predecessor'])
    previous = read_pin(spec['cold_PBS_scope'])
    config = read_pin(spec['phase_zero_config'])
    receipt = read_pbs_with_auth_retry(pbs_history,previous['PBS_JOBID'])
    if (previous['role']!='train' or previous['exit_code'] or previous['completed_roles']['train']!=spec['sealed_predecessor']
            or previous['data_contract']!=config['data_contract'] or previous['parent_checkpoint']!=config['parent_checkpoint']
            or previous['PBS_JOBID']==spec['job_id'] or receipt['job_state']!='F' or receipt.get('Exit_status')!=0
            or receipt['project']!=config['project'] or receipt['Job_Owner'].split('@')[0]!=config['user']
            or plan['update_limit']!={name:phase_window(name) for name in VARIANTS} or plan['scope']!=SCOPE
            or plan['parent_checkpoint_sha256']!=cold['checkpoint']['sha256']):
        raise PermissionError('ADAPT_PROFILE_REQUIRES_COLD_EXISTING_CHECKPOINT_AND_FIXED_ENGINEERING_PANEL')
    stop = root/'STOP.json'
    lock = OwnerLock(root/'OWNER.lock')
    actual = dict(scope=SCOPE,engineering_only=True,job_id=spec['job_id'],source_sha=spec['source_sha'],
                  optimizer_updates=0,native_requests=0,native_ticks=0,completed={},exit_code=1)

    def receive_stop(*_):
        if not stop.exists():
            write(stop,dict(reason='OPERATOR_PBS_STOP'))

    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,receive_stop)

    try:
        baseline = archived_serial(plan,spec,config) if 'archived_serial' in plan else None
        common,env,_ = pbs_site(dict(config,job_id=spec['job_id'],deadline_epoch=spec['deadline_epoch']),root,wall=5400)
        if baseline is not None and (common['GPU_uuid'] != baseline['GPU_uuid']
                                     or os.uname().nodename != baseline['host']):
            raise PermissionError('ADAPT_REMAINING_ROLES_REQUIRE_THE_ORIGINAL_PHYSICAL_A40')
        write(root/'ACTIVATED.private.json',dict(engineering_only=True,job=spec['job_id'],checkpoint=cold['checkpoint']))
        variants = VARIANTS[1:] if baseline is not None else VARIANTS
        for name in variants:
            remaining = common['deadline_epoch']-time.time()
            if stop.exists() or remaining<=360:
                raise TimeoutError('ADAPT_PROFILE_STOP_BEFORE_NEXT_ROLE')
            work = root/name
            work.mkdir()
            card = root/(name+'.SPEC.private.json')
            write(card,dict(common,scope=SCOPE,source_sha=spec['source_sha'],registered_root=str(work),variant=name,
                            training_card=spec['training_card'],sealed_predecessor=spec['sealed_predecessor'],
                            engineering_contract=spec['engineering_contract']))
            actual['unresolved_role'] = name
            with (root/(name+'.log')).open('x') as log:
                worker = subprocess.run([sys.executable,'-B','-m','scripts.sla.benchmark_adapt5k',str(card),pin(card)['sha256']],
                                        timeout=remaining,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=source)
            terminal = work/'RESULT.private.json'
            if not terminal.is_file():
                ledger = work/'UPDATES.private.jsonl'
                if ledger.is_file():
                    raw = ledger.read_bytes()
                    durable = [json.loads(row) for row in raw.splitlines(keepends=True) if row.endswith(b'\n')]
                    exposure = exposure_fingerprint(durable)
                    actual['optimizer_updates'] += len(durable)
                    actual['unsealed_ledger_prefix'] = dict(ledger=pin(ledger),updates=len(durable),
                        visit_slice_sha256=exposure,completed_H40=4*len(durable),
                        child_exit=worker.returncode,role=name)
                    interval = work/'MATCHED_INTERVAL.private.json'
                    if interval.is_file():
                        actual['unsealed_ledger_prefix']['matched_interval'] = pin(interval)
                raise RuntimeError(f'ADAPT_PROFILE_{name}_EXIT_{worker.returncode}_WITHOUT_TERMINAL')
            effect_pin = pin(terminal)
            effect = read_pin(effect_pin)
            if worker.returncode!=effect['exit_code']:
                raise ValueError('ADAPT_PROFILE_WORKER_EXIT_DOES_NOT_MATCH_ITS_EFFECT_LEDGER')
            actual['completed'][name] = effect_pin
            actual['optimizer_updates'] += effect['actual_optimizer_updates']
            actual['unresolved_role'] = None
            if effect['exit_code']:
                actual.update(status=effect['status'],first_failure=effect.get('failure'),exit_code=effect['exit_code'])
                return actual['exit_code']
        comparison = root/'COMPARISON.private.json'
        effects = [read_pin(actual['completed'][name]) for name in variants]
        write(comparison,compare(effects) if baseline is None else compare_remaining(baseline,effects))
        status = ('ADAPT_PROFILE_MATCHED_ENGINEERING_COMPLETE' if baseline is None else
                  'ADAPT_PROFILE_REMAINING_ROLES_COMPLETE_SERIAL_FINGERPRINT_UNAVAILABLE')
        actual.update(status=status,comparison=pin(comparison),exit_code=0)
        return 0
    except (Exception,KeyboardInterrupt) as error:
        actual.update(status='ADAPT_PROFILE_REQUIRES_RECONCILIATION',first_failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            write(root/'RESULT.private.json',actual)
        finally:
            lock.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
