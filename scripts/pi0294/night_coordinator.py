"""Window-scoped persistent coordinator; immutable source, no external monitor."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .night_runtime import read,sha,validate_entry,cpu_scope,CUTOFF,DEADLINE
from scripts.pi0292.durable import OwnerLock,write


def run(root,native,startup):
    root=Path(root);manifest,_=validate_entry(root);owner=OwnerLock(root/'NIGHT_COORDINATOR.lock')
    all_cpus=sorted(os.sched_getaffinity(0));child=None;active=None;log=None;last_summary=0;stop=False
    def stop_requested(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,stop_requested);signal.signal(signal.SIGTERM,stop_requested)
    identity=dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],source=manifest['code']['vla_commit'],run_uuid=root.name)
    write(root/'COORDINATOR_IDENTITY.json',identity)
    sequence=len(list((root/'worker_processes').glob('COORDINATOR_*-START.json')))+1
    stage='NIGHT_COORDINATOR_'+str(sequence)
    write(root/'worker_processes'/('COORDINATOR_'+str(sequence)+'-START.json'),identity)
    try:
        with cpu_scope(root,stage,cores=1,cpu_s_cap=32400,include_children=False):
            while time.time()<DEADLINE:
                queuefile=root/'factory/NIGHT_WORK_QUEUE.json'
                queue=read(queuefile) if queuefile.exists() else dict(items=[])
                ledger=read(root/'NIGHT_RESOURCE_LEDGER.json');epoch=time.time()
                if child is not None and child.poll() is not None:
                    result_path=root/'factory/episodes'/active/'RESULT.json'
                    result=read(result_path) if result_path.exists() else dict(status='ENGINEERING_PROCESS_INCOMPLETE',returncode=child.returncode)
                    receipt=root/'worker_processes'/(active+'-TERMINAL.json');write(receipt,dict(episode=active,returncode=child.returncode,result=result,epoch=epoch))
                    child=None;active=None
                    if log:log.close();log=None
                pending=[x for x in queue['items'] if not (root/'worker_processes'/(x['episode']+'-TERMINAL.json')).exists()]
                stop=stop or (root/'STOP_COORDINATOR_AT_BOUNDARY').exists()
                if child is None and (stop or epoch>=CUTOFF or ledger['native_safety_failures']>=2):break
                if child is None and pending:
                    item=pending[0];episode=item['episode']
                    if (root/'factory/episodes'/episode).exists():
                        raise RuntimeError('NIGHT_EXISTING_EPISODE_WITHOUT_LIVE_HANDLE_NO_REEXECUTION')
                    log=(root/'worker_logs'/(episode+'.log')).open('xb')
                    env=dict(os.environ,PYTHONPATH=str(root/manifest['code']['capsule_relative'])+':'+str(root/manifest['code']['capsule_relative']/'src')+':'+os.environ.get('PYTHONPATH',''))
                    child=subprocess.Popen([sys.executable,'-m','scripts.pi0294.night_worker','--root',str(root),'--native',str(native),'--startup',str(startup),'--episode',episode],
                        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,all_cpus))
                    active=episode;write(root/'worker_processes'/(episode+'-PID.json'),dict(pid=child.pid,proc_start=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],episode=episode,source=manifest['code']['vla_commit'],started_epoch=epoch))
                state=dict(identity,state='SEED_DEVELOP' if child is not None else 'WAIT_NEXT_ADMITTED_WORK',heartbeat_epoch=epoch,
                    child_pid=None if child is None else child.pid,active_episode=active,actual_child_running=child is not None and child.poll() is None,
                    main_starts=ledger['main_starts'],main_integrations=ledger['main_integrations'],RGB_pairs=ledger['RGB_pairs'],
                    native_safety_failures=ledger['native_safety_failures'],queue_pending=len(pending),new_work_cutoff_epoch=CUTOFF,hard_deadline_epoch=DEADLINE)
                write(root/'COORDINATOR_STATE.json',state)
                if epoch-last_summary>=300:
                    with (root/'COORDINATOR_SUMMARIES.jsonl').open('a') as f:f.write(__import__('json').dumps(state)+'\n')
                    last_summary=epoch
                for _ in range(30):
                    if child is not None and child.poll() is not None or stop:break
                    time.sleep(1)
            if child is not None and child.poll() is None:
                # Only the hard deadline terminates this goal's own process group.
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            write(root/'COORDINATOR_BOUNDARY.json',dict(identity,status='STOPPED_AT_OWN_BOUNDARY',epoch=time.time(),child_running=False))
            write(root/'worker_processes'/('COORDINATOR_'+str(sequence)+'-END.json'),dict(identity,status='STOPPED_AT_OWN_BOUNDARY',epoch=time.time(),child_running=False))
    finally:
        if log:log.close()
        owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','native','startup']:p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.root,a.native,a.startup)
