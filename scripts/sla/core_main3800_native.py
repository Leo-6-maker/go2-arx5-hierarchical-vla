"""Run the independently bounded MAIN3800 matched diagnostic on Corelab."""
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
from scripts.sla.core_main3800_native_proof import ARM, pin, publish
from scripts.sla.hopper_parent_custody import write_once
from scripts.sla.hopper_site import pinned


def coordinate(root, cfg):
    expert = pinned(cfg['expert'])
    original = pinned(cfg['original'])
    source = Path(__file__).resolve().parents[2]
    if (cfg['root'] != str(root) or cfg['source_root'] != str(source)
            or cfg['historic_run'] != original['root'] or not os.environ.get('INVOCATION_ID')
            or os.getuid() != expert['uid'] or os.uname().nodename != expert['host']
            or any(cfg[key] != original[key] for key in ('expert','expert_owner','deadline','population','T0'))
            or cfg['population'] != expert['compiled_population']):
        raise PermissionError('Expected the original Corelab world and independent six-condition scope')
    deadline = min(cfg['goal_deadline'],original['goal_deadline'],cfg['deadline'])
    cfg['dispatch_deadline'] = deadline
    journal = Journal(root/'journal','CORE-MAIN3800-MATCHED-DEV6-V1')
    folder = root/'blocks'/f'{ARM}-DEV-00'

    def state(value, **detail):
        write(root/'HEARTBEAT.json',dict(state=value,epoch=time.time(),pid=os.getpid(),**detail))

    def stopped():
        return time.time()+6*1980+600 >= deadline or (root/'STOP').exists()

    def dispatch(proof, cap):
        task = prepare(cfg, proof, ARM, 'DEV', 0, folder, cap)
        cmd = [task['python'],'-B',str(source/'scripts/wm02/v6_bootstrap.py'),
               '--core-hopper-students',str(folder/'EXECUTION_SPEC.json')]
        with (folder/'logs/block.log').open('xb') as log:
            child = subprocess.Popen(cmd,env={**os.environ,**task['native_environment']},cwd=source,
                                     stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                state('STUDENT_BLOCK',child=child.pid,arm=ARM)
                if (root/'STOP').exists() and not (folder/'STOP.json').exists():
                    write(folder/'STOP.json',dict(reason='Operator stop'))
                time.sleep(10)
        if not (folder/'RESULT.private.json').is_file():
            raise RuntimeError(f'Native block exit {child.returncode} before terminal reconciliation')
        result = audit_block(folder)
        write_once(folder/'RECONCILED.json',result)
        return result

    with closing(OwnerLock(root/'owner.lock')):
        write_once(root/'OWNER.json',dict(pid=os.getpid(),invocation=os.environ['INVOCATION_ID'],
                   deadline=deadline,arm=ARM,maximum_requests=6,updates=0))
        proof = publish(root/'EXPLORATORY_PROOF.json',cfg)
        while not stopped():
            if capacity(cfg,expert) is None:
                state('WAIT_ORIGINAL_RESOURCES')
                time.sleep(20)
                continue
            with handoff(expert,root):
                while not stopped():
                    try:
                        with closing(OwnerLock(expert['mobile_world_lock'])):
                            break
                    except BlockingIOError:
                        state('WAIT_ACTIVE_WORLD')
                        time.sleep(15)
                if stopped():
                    break
                cap = capacity(cfg,expert)
                if cap is None:
                    continue
                actual = journal.once(folder.name,dict(proof=proof,arm=ARM,requests=6),
                                      lambda: dispatch(proof,cap))
                write_once(root/'TERMINAL.json',dict(status='SIX_MATCHED_CONDITIONS_RECONCILED',
                           actual=actual,updates=0,visits=0,epoch=time.time()))
                return
        write_once(root/'TERMINAL.json',dict(status='STOP_OR_INSUFFICIENT_REMAINING_ALLOCATION',
                   native_requests=0,updates=0,visits=0,epoch=time.time()))


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
