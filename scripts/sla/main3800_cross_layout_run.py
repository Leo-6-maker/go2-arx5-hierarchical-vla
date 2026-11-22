"""Run one reconciled two-arm DEV comparison with the existing Corelab owner and guards."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import subprocess
import time

from scripts.pi0292.durable import OwnerLock, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.core_hopper_native_campaign import audit_block, prepare
from scripts.sla.core_hopper_native_queue import capacity, handoff
from scripts.sla.hopper_parent_custody import write_once
from scripts.sla.hopper_site import pinned
from scripts.sla.main3800_cross_layout_proof import COHORTS, pin, publish, roster_rows
from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases


def coordinate(root,cfg):
    base = pinned(cfg['base_native_config'])
    expert = pinned(cfg['expert'])
    source = Path(__file__).resolve().parents[2]
    arms = COHORTS[cfg['cohort']]
    if (cfg['root']!=str(root) or cfg['source_root']!=str(source) or not os.environ.get('INVOCATION_ID')
            or os.uname().nodename!=expert['host'] or os.getuid()!=expert['uid']
            or any(cfg[k]!=base[k] for k in ('expert','expert_owner','deadline','population','T0',
                                           'historic_run','numeric_service','gpu_template','receiver'))):
        raise PermissionError('Expected a finite cross-layout cohort on the original Corelab service')
    deadline = min(cfg['goal_deadline'],base['deadline'])
    cfg['dispatch_deadline'] = deadline
    journal = Journal(root/'journal','CORE-MAIN3800-CROSS-LAYOUT-'+cfg['cohort'])

    def state(value, **detail):
        write(root/'HEARTBEAT.json',dict(state=value,epoch=time.time(),pid=os.getpid(),**detail))

    def stopped(remaining):
        return time.time()+remaining*(6*1980+600)>=deadline or (root/'STOP').exists()

    def dispatch(arm, proof, cap):
        folder = root/'blocks'/f'{arm}-DEV-cross-layout'
        task = prepare(cfg,proof,arm,'DEV',0,folder,cap)
        cmd = [task['python'],'-B',str(source/'scripts/wm02/v6_bootstrap.py'),
               '--core-hopper-students',str(folder/'EXECUTION_SPEC.json')]
        with (folder/'logs/block.log').open('xb') as log:
            child = subprocess.Popen(cmd,env={**os.environ,**task['native_environment']},cwd=source,
                                     stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                state('CROSS_LAYOUT_STUDENT_BLOCK',child=child.pid,arm=arm)
                if (root/'STOP').exists() and not (folder/'STOP.json').exists():
                    write(folder/'STOP.json',dict(reason='Operator stop'))
                time.sleep(10)
        if child.returncode or not (folder/'RESULT.private.json').is_file():
            raise RuntimeError(f'Native block {arm} requires terminal reconciliation')
        actual = audit_block(folder)
        write_once(folder/'RECONCILED.json',actual)
        return actual

    with closing(OwnerLock(root/'owner.lock')):
        cases = {row['id'] for arm in arms for row in roster_rows(arm,cfg['roster'],cfg['population'])}
        require_unstarted_cases(expert['registry'],cases)
        write_once(root/'OWNER.json',dict(pid=os.getpid(),invocation=os.environ['INVOCATION_ID'],deadline=deadline,
                   arms=arms,request_cap=len(cases),updates=0))
        pending = cfg.get('offline_wait')
        if pending is None:
            offline = pinned(cfg['offline_config'])
            offline_root = Path(cfg['offline_config']['path']).parent
        else:
            wait = pinned(pending)
            offline_root = Path(pending['path']).parent
            if cfg['cohort'] != 'auxiliary' or Path(wait['root']) != offline_root:
                raise ValueError('Expected the registered Core A/B evaluation service')
            offline = pinned(wait['event_config'])
        if offline['native_config'] != cfg['base_native_config']:
            raise ValueError('The offline prerequisite must use the original Corelab config')
        offline_terminal = offline_root/'TERMINAL.json'
        while not offline_terminal.is_file():
            if (offline_root/'REVIEW_REQUIRED.json').is_file():
                raise RuntimeError('The fixed offline evaluation needs review')
            if stopped(len(arms)) or time.time() >= offline['deadline']:
                write_once(root/'TERMINAL.json',dict(status='OFFLINE_PREREQUISITE_UNAVAILABLE',native_requests=0,
                           updates=0,visits=0,epoch=time.time()))
                return
            state('WAIT_OFFLINE_TERMINAL',offline_config=(pending or cfg['offline_config'])['sha256'])
            time.sleep(20)
        terminal_pin = pin(offline_terminal)
        if cfg.get('offline_terminal',terminal_pin) != terminal_pin:
            raise ValueError('The fixed offline result changed')
        if pinned(terminal_pin)['status'] != 'COMPLETE':
            write_once(root/'TERMINAL.json',dict(status='OFFLINE_PREREQUISITE_UNAVAILABLE',offline=terminal_pin,
                       native_requests=0,updates=0,visits=0,epoch=time.time()))
            return
        if pending is not None:
            evaluated = pin(offline_root/'CONFIG.json')
            verdict = pinned(evaluated)
            export = verdict.pop('auxiliary_export')
            if verdict != offline or Path(export['path']) != Path(wait['delivery'])/'EXPORT.json':
                raise ValueError('The released A/B evaluation changed its fixed input or delivery')
            cfg['offline_config'] = evaluated
        cfg['offline_terminal'] = terminal_pin
        proofs = {arm:publish(root/f'{arm}.proof.json',cfg,arm) for arm in arms}
        totals = {}
        for index,arm in enumerate(arms):
            remaining = len(arms)-index
            while not stopped(remaining):
                if capacity(cfg,expert) is None:
                    state('WAIT_ORIGINAL_RESOURCES',arm=arm)
                    time.sleep(20)
                    continue
                with handoff(expert,root):
                    try:
                        with closing(OwnerLock(expert['mobile_world_lock'])):
                            pass
                    except BlockingIOError:
                        state('WAIT_ACTIVE_WORLD',arm=arm)
                        time.sleep(15)
                        continue
                    cap = capacity(cfg,expert)
                    if cap is None or stopped(remaining):
                        continue
                    proof = proofs[arm]
                    totals[arm] = journal.once(arm,dict(arm=arm,proof=proof,requests=6),
                                               lambda: dispatch(arm,proof,cap))
                    break
            else:
                break
        write_once(root/'TERMINAL.json',dict(status='COMPLETE' if len(totals)==2 else 'STOP_OR_INSUFFICIENT_ALLOCATION',
                  arms=totals,native_requests=sum(r['requests'] for r in totals.values()),updates=0,visits=0,epoch=time.time()))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('config_sha')
    args = cli.parse_args()
    os.umask(0o077)
    try:
        coordinate(args.root,pinned(dict(path=str(args.root/'CONFIG.json'),sha256=args.config_sha)))
    except Exception as exc:
        records = []
        for path in sorted(args.root.glob('blocks/*/cases/*/RESULT.private.json')):
            result = pinned(pin(path))
            records.append(dict(result=pin(path),requests=result['physical_requests'],
                                ticks=result['actual_native_integrations'],generations=result['model_generations']))
        starts = sorted(str(path) for path in args.root.glob('blocks/*/cases/*/native-01/BEFORE_FIRST_INTEGRATION.json'))
        write_once(args.root/'REVIEW_REQUIRED.json',dict(reason=f'{type(exc).__name__}: {exc}',
                   starts=starts,terminal_cases=records,epoch=time.time()))
        raise
