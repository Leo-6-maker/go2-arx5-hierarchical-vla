"""Run the matched ADAPT engineering probes sequentially in one existing PBS allocation."""
import argparse
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
from scripts.sla.benchmark_adapt5k import SCOPE, VARIANTS, compare, exposure_fingerprint, phase_window
from scripts.sla.scale20k_vanda_offline import read_pbs_with_auth_retry
from scripts.wm02.multisource_d4_vanda import pin


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
        common,env,_ = pbs_site(dict(config,job_id=spec['job_id'],deadline_epoch=spec['deadline_epoch']),root,wall=5400)
        write(root/'ACTIVATED.private.json',dict(engineering_only=True,job=spec['job_id'],checkpoint=cold['checkpoint']))
        for name in VARIANTS:
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
                    from json import loads

                    raw = ledger.read_bytes()
                    durable = [loads(row) for row in raw.splitlines(keepends=True) if row.endswith(b'\n')]
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
        write(comparison,compare([read_pin(actual['completed'][name]) for name in VARIANTS]))
        actual.update(status='ADAPT_PROFILE_MATCHED_ENGINEERING_COMPLETE',comparison=pin(comparison),exit_code=0)
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
