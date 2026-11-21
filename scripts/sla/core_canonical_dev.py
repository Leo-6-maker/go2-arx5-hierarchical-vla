"""Dispatch the two original first DEV6 blocks after independent Core qualification."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.pi0292.durable import OwnerLock, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.core_canonical_cold import INITIAL, pin
from scripts.sla.core_canonical_package import CANONICAL_PLAN
from scripts.sla.core_canonical_proof import publish
from scripts.sla.core_hopper_native_campaign import audit_block, prepare
from scripts.sla.core_hopper_native_queue import capacity, handoff
from scripts.sla.hopper_parent_custody import PLAN, write_once
from scripts.sla.hopper_site import pinned


def coordinate(root,cfg):
    cfg['dispatch_deadline'] = min(cfg['deadline'],cfg['goal_deadline'])
    end = cfg['dispatch_deadline']
    expert=pinned(cfg['expert'])
    prior=Path(expert['registered_root'])
    old_queue=Path(cfg['legacy_queue'])
    if (cfg['canonical_plan']!=CANONICAL_PLAN or cfg['plan']!=PLAN or cfg['root']!=str(root)
            or not os.environ.get('INVOCATION_ID') or os.getuid()!=expert['uid']
            or os.uname().nodename!=expert['host'] or cfg['source_root']!=str(Path(__file__).resolve().parents[2])
            or cfg['population']!=expert['compiled_population'] or cfg['deadline']!=expert['campaign_deadline_epoch']):
        raise PermissionError('Expected the finite Core student owner and original DEV population')
    journal=Journal(root/'journal','CORE-CANONICAL-DEV6-V1')
    ipc=Path(cfg['ipc_root'])
    sealed={}
    def state(value,**detail):
        write(root/'HEARTBEAT.json',dict(state=value,epoch=time.time(),pid=os.getpid(),**detail))
    def stopped():
        return time.time()+2100>=end or (root/'STOP').exists()
    with closing(OwnerLock(root/'owner.lock')):
        write_once(root/'OWNER.json',dict(pid=os.getpid(),invocation=os.environ['INVOCATION_ID'],deadline=end))
        for arm in INITIAL:
            folder=root/'blocks'/f'{arm}-DEV-00'
            proof_path=root/'endpoints'/f'{arm}.json'
            worker=ipc/arm/'worker'
            old_block=old_queue/'blocks'/folder.name
            while not (worker/'IPC_COMPARISON.json').is_file() and not stopped():
                if (ipc/'TERMINAL.json').is_file():
                    break
                state('WAIT_IPC',arm=arm)
                time.sleep(15)
            if not (worker/'IPC_COMPARISON.json').is_file() or stopped():
                sealed[arm]='NOT_RUN'
                continue
            if old_block.exists():
                raise RuntimeError('Original DEV block was already reserved')
            # IPC outputs precede the child exit and final lease closure.
            while not stopped() and pinned(pin(worker/'control/LEASE.json'))['status']!='CLOSED':
                state('WAIT_IPC_LEASE_CLOSED',arm=arm)
                time.sleep(5)
            if stopped():
                sealed[arm]='NOT_RUN'
                continue
            proof_path.parent.mkdir(exist_ok=True)
            proof=publish(proof_path,Path(cfg['cold_root']),worker,cfg['receiver'],
                          cfg['population'],arm,cfg['source'])
            since=proof['check_epoch']
            baseline=len(pinned(pin(prior/'RECORDS.private.json')))
            def teacher_turn():
                return ((prior/'RESULT.private.json').is_file() or time.time()-since>=1800 or
                        len(pinned(pin(prior/'RECORDS.private.json')))-baseline>=12)
            while not stopped():
                if not teacher_turn():
                    state('EXPERT_TURN',arm=arm)
                    time.sleep(20)
                    continue
                if capacity(cfg,expert) is None:
                    state('WAIT_RESOURCES',arm=arm)
                    time.sleep(20)
                    continue
                with handoff(expert,root):
                    while not stopped():
                        try:
                            with closing(OwnerLock(expert['mobile_world_lock'])):
                                break
                        except BlockingIOError:
                            state('WAIT_ACTIVE_WORLD',arm=arm)
                            time.sleep(15)
                    if stopped():
                        sealed[arm]='NOT_RUN'
                        break
                    cap=capacity(cfg,expert)
                    if cap is None:
                        continue
                    def launch():
                        if old_block.exists():
                            raise RuntimeError('Original DEV block was registered by another owner')
                        task=prepare(cfg,pin(proof_path),arm,'DEV',0,folder,cap)
                        cmd=[task['python'],'-B',str(Path(cfg['source_root'])/'scripts/wm02/v6_bootstrap.py'),
                             '--core-hopper-students',str(folder/'EXECUTION_SPEC.json')]
                        log_path=folder/'logs/block.log'
                        with log_path.open('xb') as log:
                            p=subprocess.Popen(cmd,env={**os.environ,**task['native_environment']},cwd=cfg['source_root'],
                                               stdout=log,stderr=subprocess.STDOUT)
                            while p.poll() is None:
                                state('STUDENT_BLOCK',arm=arm,child=p.pid)
                                if (root/'STOP').exists() and not (folder/'STOP.json').exists():
                                    write(folder/'STOP.json',dict(reason='Core canonical operator stop'))
                                time.sleep(10)
                        if not (folder/'RESULT.private.json').is_file():
                            evidence=pin(log_path)
                            raise RuntimeError(f'Native child exited {p.returncode} before a terminal result. '
                                               f'Launch log SHA256 {evidence["sha256"]}')
                        actual=audit_block(folder)
                        write_once(folder/'RECONCILED.json',actual)
                        return actual
                    sealed[arm]=journal.once(folder.name,dict(proof=pin(proof_path),arm=arm,start=0,split='DEV'),launch)
                    break
            sealed.setdefault(arm,'NOT_RUN')
        write_once(root/'TERMINAL.json',dict(outcomes=sealed,native_requests=sum(
            v['requests'] for v in sealed.values() if isinstance(v,dict)),epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('config_sha')
    args=p.parse_args()
    os.umask(0o077)
    try:
        coordinate(args.root,pinned(dict(path=str(args.root/'CONFIG.json'),sha256=args.config_sha)))
    except Exception as exc:
        records=[]
        for f in sorted(args.root.glob('blocks/*/cases/*/RESULT.private.json')):
            row=json.loads(f.read_bytes())
            records.append(dict(result=pin(f),requests=row['physical_requests'],
                                ticks=row['actual_native_integrations'],model_generations=row['model_generations']))
        started=sorted(str(f) for f in args.root.glob('blocks/*/cases/*/native-01/BEFORE_FIRST_INTEGRATION.json'))
        write_once(args.root/'REVIEW_REQUIRED.json',dict(reason=f'{type(exc).__name__}: {exc}',
                   epoch=time.time(),starts=started,terminal_cases=records))
        raise
