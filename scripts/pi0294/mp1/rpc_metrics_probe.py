"""Finite real OS-lock probe of reply telemetry; not a model/physics test."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import resource
import select
import subprocess
import sys
import time

from scripts.pi0292.durable import read,write
from scripts.pi0294.rpc_metrics import SessionMetrics
from .scope import validate,charge_cpu


class SyntheticLedger:
    def __init__(self,path):self.path=Path(path);self.role='hopper';self.transactions=0

    @contextmanager
    def transaction(self):
        self.transactions+=1
        with self.path.with_suffix('.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);value=read(self.path);yield value;write(self.path,value)


def probe(folder):
    folder=Path(folder);folder.mkdir(mode=0o700);path=folder/'SYNTHETIC_LEDGER.json'
    initial=dict(synthetic=True,generation_queries=3,optimizer_updates=0,CPU_s=1.)
    write(path,initial);ledger=SyntheticLedger(path);metrics=SessionMetrics();responses=[]
    code='import fcntl,select,sys;h=open(sys.argv[1],"a");fcntl.flock(h,fcntl.LOCK_EX);print("LOCKED",flush=True);ready=select.select([sys.stdin],[],[],30)[0];sys.stdin.readline() if ready else None'
    child=subprocess.Popen([sys.executable,'-u','-c',code,str(path.with_suffix('.lock'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    write(folder/'SYNTHETIC_CHILD.private.json',dict(pid=child.pid,proc_start=Path('/proc',str(child.pid),'stat').read_text().split()[21],maximum_wall_s=30,model=False,physics=False))
    try:
        if not select.select([child.stdout],[],[],10)[0] or child.stdout.readline().strip()!='LOCKED':raise TimeoutError('SYNTHETIC_LOCK_CHILD_NOT_READY')
        with path.with_suffix('.lock').open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:locked=True
            else:raise AssertionError('INDEPENDENT_PROCESS_MUST_HOLD_SYNTHETIC_LOCK')
        began=time.monotonic()
        for n in range(80):metrics.reply(dict(synthetic=True,sequence=n),responses.append)
        reply_s=time.monotonic()-began
        if child.poll() is not None or ledger.transactions or read(path)!=initial or len(responses)!=80:
            raise AssertionError('REPLIES_MUST_NOT_TOUCH_OR_WAIT_FOR_HELD_LEDGER')
        child.communicate('\n',timeout=10)
        end=time.monotonic();settled=metrics.settle(ledger,.125);settle_s=time.monotonic()-end
        if read(path)!=dict(initial,CPU_s=1.125,RPC_calls=80) or ledger.transactions!=1:raise AssertionError('ONE_EXACT_SYNTHETIC_TERMINAL_SETTLEMENT')
        try:metrics.settle(ledger,.125)
        except PermissionError:repeat_rejected=True
        else:raise AssertionError('UNKNOWN_SETTLEMENT_RETRY_FORBIDDEN')
        return dict(status='PASS_ACTUAL_SAME_HOST_OS_LOCK_REPLY_TELEMETRY_ISOLATION',
            independent_lock_holder_confirmed=locked,replies_while_lock_held=len(responses),reply_loop_wall_s=reply_s,
            per_reply_shared_ledger_transactions=0,terminal_shared_ledger_transactions=1,terminal_settlement_wall_s=settle_s,
            repeated_settlement_rejected=repeat_rejected,synthetic_scientific_counters_unchanged=True,
            actual_scientific_counter_or_data_modified=False,new_model_queries=0,new_physical_integrations=0,
            compute_to_login_NFS_qualified=False,actual_SSH_RPC_qualified=False,full_32s_model_loop_qualified=False,
            CPU_charge_point125_is_synthetic_not_actual_host_charge=True)
    finally:
        if child.poll() is None:child.kill();child.communicate(timeout=10)


def run(root,role,attempt=1):
    root=Path(root)
    if role not in ['core','hopper'] or attempt not in [1,2]:raise PermissionError('FIXED_TWO_CPU_ROLES_AND_TWO_BOUNDED_ATTEMPTS_ONLY')
    m=validate(root,'4080-cpu-data' if role=='core' else 'hopper-neural')
    binding=root/'plans'/('RPC_METRICS_CPU_BINDING_'+role+'_'+str(attempt)+'.json')
    from scripts.pi0294.native_dataset import sha
    if m['execution'].get('rpc_metrics_binding_sha256')!=sha(binding):raise PermissionError('IMMUTABLE_RPC_METRICS_PROBE_BINDING')
    if read(binding)!=dict(run_uuid=root.name,role=role,attempt=attempt,source_sha=m['code']['vla_commit'],maximum_CPU_s=600,maximum_wall_s=120,new_model_queries=0,new_physics=0):
        raise PermissionError('FIXED_FINITE_CPU_METRICS_PROBE')
    output=root/'evidence'/('RPC_METRICS_CPU_'+str(attempt).zfill(2)+'_'+role);output.mkdir(mode=0o700);started=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        value=probe(output/'synthetic');value.update(role=role,source_sha=m['code']['vla_commit'])
        write(output/'RESULT.json',value);return value
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN);charge_cpu(root,time.process_time()-started+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--role',required=True);p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    def expire(signum,frame):raise TimeoutError('FINITE_RPC_METRICS_CPU_120S_BOUNDARY')
    import signal
    signal.signal(signal.SIGALRM,expire);signal.alarm(120)
    try:print(json.dumps(run(a.root,a.role,a.attempt)),flush=True)
    finally:signal.alarm(0)
