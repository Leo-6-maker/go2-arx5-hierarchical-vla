"""Schedule finite DEV/CONF blocks only from a verified Core replay and an available world."""
import argparse
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.core_hopper_expert import PLAN, POP
from scripts.sla.core_hopper_native_campaign import prepare, audit_block
from scripts.sla.core_hopper_native_scope import ENDPOINTS, MAIN, SCOPE
from scripts.wm02.multitask_stage_cards_v6 import pin, read_pin

GIB=1024**3


def capacity(cfg, expert):
    receiver=read_pin(cfg['receiver'])
    gpu=read_pin(cfg['gpu_template'])
    device=receiver['gpu']
    line=subprocess.check_output(['nvidia-smi','--id='+device,'--query-gpu=uuid,name,memory.total,memory.free',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    uid,name,total,free=(x.strip() for x in line.split(','))
    total,free=int(total)*1024**2,int(free)*1024**2
    host=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    psi=max(float(s.split()[1].split('=')[1]) for s in Path('/proc/pressure/memory').read_text().splitlines())
    disk=shutil.disk_usage(cfg['root'])
    if (uid!=device or device!=gpu['GPU_uuid'] or receiver['gpu_lock']!=gpu['neural_owner_lock']):
        raise PermissionError('Unexpected Core inference GPU or resource owner')
    if (free<=total*receiver['memory_fraction']+2*GIB or host<=max(36*GIB,gpu['minimum_host_available_bytes'])
            or psi>.1 or disk.free<max(100*GIB,.15*disk.total+8*GIB)):
        return None
    try:
        with closing(OwnerLock(receiver['gpu_lock'])):
            pass
    except BlockingIOError:
        return None
    return dict(name=name,gpu_uuid=uid,total_bytes=total,free_bytes=free,host_bytes=host,disk_free=disk.free,psi=psi)


@contextmanager
def handoff(expert, root):
    # systemd removes RuntimeDirectory on exit. A stale symlink cannot pause collection.
    runtime=Path(os.environ['RUNTIME_DIRECTORY'])
    if runtime.parent != Path('/run/user')/str(os.getuid()) or runtime.name!=root.name:
        raise PermissionError('World handoff requires the registered process lifetime')
    token=runtime/'student-turn'
    marker=Path(expert['registered_root'])/'STUDENT_TURN_READY.private.json'
    if marker.is_symlink():
        if marker.readlink()!=token or marker.exists():
            raise RuntimeError('Another world handoff exists')
        marker.unlink()
    elif marker.exists():
        raise RuntimeError('Unknown world handoff owner')
    with token.open('x') as f:
        json.dump(dict(pid=os.getpid(),unit=os.environ['INVOCATION_ID'],epoch=time.time()),f)
    try:
        marker.symlink_to(token)
        yield
    finally:
        token.unlink(missing_ok=True)
        if marker.is_symlink() and marker.readlink()==token:
            marker.unlink()


def choose(root,cfg):
    gates=Path(cfg['gate']['path']).parent/'endpoints'
    proofs=sorted(gates.glob('*.json'),key=lambda p:(int(p.stem[-5:]),p.stem))
    if len(proofs)>7:
        raise RuntimeError('Maximum number of registered endpoints exceeded')
    pairs=set()
    for proof in proofs:
        key=proof.stem
        if key not in ENDPOINTS:
            raise PermissionError('Unexpected endpoint')
        if int(key[-5:]) not in MAIN:
            pairs.add(int(key[-5:]))
    if len(pairs)>1:
        raise PermissionError('Only one matched course pair is allowed')
    for proof in proofs:
        endpoint=read(proof)
        if endpoint['gate']['status']=='ENDPOINT_NATIVE_HELD':
            continue
        if endpoint['gate']['status']!='ENDPOINT_RGB16_VERIFIED':
            raise ValueError('Unrecognized replay outcome')
        for split in ('DEV','CONF') if proof.stem=='MAIN-phase20000' else ('DEV',):
            if len(endpoint['panels'][split])!=24:
                raise ValueError('Missing one or more registered conditions')
            for start in range(0,24,6):
                key=f'{proof.stem}-{split}-{start:02d}'
                prior=root/'blocks'/key
                if (prior/'RECONCILED.json').exists():
                    actual=read(prior/'RECONCILED.json')
                    read_pin(actual['block'])
                    if actual['requests']!=6:
                        raise ValueError('Incomplete prior student block')
                    continue
                if prior.exists():
                    raise RuntimeError('A previous block needs reconciliation: '+key)
                return proof, split, start, prior
    return None


def watch(root,cfg):
    expert=read_pin(cfg['expert'])
    original=Path(expert['registered_root'])
    gate=read_pin(cfg['gate'])
    receiver=read_pin(cfg['receiver'])
    end=cfg['deadline']
    if (gate['receiver']!=cfg['receiver'] or gate['expert']!=cfg['expert'] or gate['population']['sha256']!=POP
            or not cfg['T0']+14*86400==end==receiver['deadline']==expert['campaign_deadline_epoch']
            or not cfg['scope']==SCOPE or expert['owner']!=cfg['expert_owner'] or os.uname().nodename!=expert['host']
            or os.getuid()!=expert['uid'] or set(expert['cpus'])-os.sched_getaffinity(0)):
        raise PermissionError('Coordinator is outside its registered site, source, population or deadline')
    journal=Journal(root/'journal','CORE-HOPPER-V1-STUDENTS')
    def stopped():
        return (root/'STOP').exists() or time.time()+2100>=end
    def heartbeat(state,**values):
        write(root/'HEARTBEAT.json',dict(status=state,pid=os.getpid(),epoch=time.time(),**values))
    with closing(OwnerLock(root/'coordinator.lock')):
        while not stopped():
            next_block=choose(root,cfg)
            experts=read(original/'RECORDS.private.json')
            served=root/'LAST_TURN.json'
            completed_at=read(served)['expert_terminals'] if served.is_file() else 0
            if next_block is None:
                heartbeat('WAIT_QUALIFIED_ENDPOINT_OR_NEXT_SEAL')
            elif len(experts)-completed_at<12 and not (original/'RESULT.private.json').exists():
                heartbeat('EXPERT_TURN',completed=len(experts),since_handoff=len(experts)-completed_at)
            else:
                proof,split,start,folder=next_block
                cap=capacity(cfg,expert)
                if cap is None:
                    heartbeat('WAIT_GPU_OR_HOST',endpoint=proof.stem)
                else:
                    with handoff(expert,root):
                        # The expert that already owns this world finishes normally.
                        while not stopped():
                            try:
                                with closing(OwnerLock(expert['mobile_world_lock'])):
                                    break
                            except BlockingIOError:
                                heartbeat('WAIT_ACTIVE_WORLD',endpoint=proof.stem)
                                time.sleep(15)
                        if stopped():
                            break
                        cap=capacity(cfg,expert)
                        if cap is not None:
                            def launch():
                                spec=prepare(cfg,pin(proof),proof.stem,split,start,folder,cap)
                                path=folder/'EXECUTION_SPEC.json'
                                cmd=[spec['python'],'-B',str(Path(cfg['source_root'])/'scripts/wm02/v6_bootstrap.py'),
                                     '--core-hopper-students',str(path)]
                                with (folder/'logs/block.log').open('xb') as log:
                                    task=subprocess.Popen(cmd,env=dict(os.environ,**spec['native_environment']),cwd=cfg['source_root'],
                                                          stdout=log,stderr=subprocess.STDOUT)
                                    while task.poll() is None:
                                        heartbeat('STUDENT_BLOCK',block=folder.name,child=task.pid)
                                        if (root/'STOP').exists() and not (folder/'STOP.json').exists():
                                            write(folder/'STOP.json',dict(reason='operator requested finite stop'))
                                        time.sleep(10)
                                actual=audit_block(folder)
                                write(folder/'RECONCILED.json',actual)
                                return actual
                            journal.once(folder.name,dict(proof=pin(proof),start=start,split=split,source=cfg['source']),launch)
                            write(served,dict(block=folder.name,expert_terminals=len(read(original/'RECORDS.private.json'))))
            time.sleep(40)
    heartbeat('STOP_OR_DEADLINE')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('config_sha')
    a=p.parse_args()
    cfg=read_pin(dict(path=str(a.root/'CONFIG.json'),sha256=a.config_sha))
    if (not os.environ.get('INVOCATION_ID') or cfg['plan']!=PLAN or cfg['root']!=str(a.root)
            or Path(cfg['source_root'])!=Path(__file__).resolve().parents[2]):
        raise PermissionError('Run from the registered student coordinator')
    try:
        watch(a.root,cfg)
    except Exception as err:
        write(a.root/'REVIEW_REQUIRED.json',dict(error=repr(err),epoch=time.time(),pid=os.getpid()))
        raise
