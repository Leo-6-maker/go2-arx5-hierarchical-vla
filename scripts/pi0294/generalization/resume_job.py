"""Attach the original owner/watch to an existing, exactly bound PBS job.

The immutable training root and PBS payload are read only during attachment.
The new metadata role has its own process records; it cannot submit a job.
Original release, heartbeat, Trainer, audit and settlement methods are reused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def raw(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def dead(pid,start):
    path=Path('/proc',str(pid),'stat')
    try:fields=path.read_text().rsplit(')',1)[1].split()
    except(FileNotFoundError,ProcessLookupError):return True
    return fields[0]in('Z','X')or int(fields[19])!=int(start)


def cached_pure_validation(function):
    """Only full JSON identity is reused; queue/permission reads stay live."""
    last=[None]
    def check(value,*args,**kwargs):
        key=hashlib.sha256(raw([value,args,kwargs])).digest()
        if key!=last[0]:
            function(value,*args,**kwargs);last[0]=key
        # Both existing callers treat validation as returning their input.
        return value
    return check


def metadata_CPU_limits(remaining, hard):
    """Respect the site's inherited hard limit; never attempt to raise it."""
    if remaining<=0 or hard==0:raise PermissionError('RESUME_NO_REMAINING_CONTROL_TIME')
    return (remaining if hard==resource.RLIM_INFINITY else min(remaining,hard),hard)


def attach(control):
    control=Path(control);h=read(control/'RUN_MANIFEST.json');root=Path(h['science_root'])
    if os.getuid()!=22182 or not os.uname().nodename.startswith('stdct-login-')or os.environ.get('PBS_JOBID'):
        raise PermissionError('RESUME_METADATA_LOGIN_ONLY')
    if h['kind']!='EXISTING_E2_JOB_MANAGEMENT_HANDOFF'or time.time()>=h['deadline_epoch']:
        raise PermissionError('RESUME_EXPLICIT_FINITE_METADATA_ROLE')
    binding=read(control/'OUTPUT_BINDING.private.json');lease=read(binding['lease'])
    reg=read(binding['registry'])
    if lease['status']!='ACTIVE'or lease['manifest_sha256']!=sha(control/'RUN_MANIFEST.json')or not any(
        x.get('path')==str(control)and x.get('role')==binding['role']for x in reg['entries']):
        raise PermissionError('RESUME_REGISTERED_ACTIVE_METADATA_ROLE')
    if sha(root/'RUN_MANIFEST.json')!=h['science_manifest_sha256']:
        raise PermissionError('RESUME_ORIGINAL_SCIENCE_MANIFEST')
    m=read(root/'RUN_MANIFEST.json');code=root/m['code']['capsule_relative']
    sys.path[:0]=[str(code),str(code/'src')]
    from scripts.pi0292.durable import OwnerLock
    from scripts.pi0294.generalization import minimal_native_owner as original,minimal_execution_adapter as adapters
    from scripts.pi0294.generalization.minimal_native_context import captured_LOCAL,registered_identity
    from scripts.pi0294.generalization.minimal_e2only_physical import Guard
    from scripts.pi0294.pv1 import mix_owner as mature,successor_cd,neural_ledger,mix_accounting
    from scripts.pi0294.dec1 import cold_owner
    from scripts.pi0294.native_dataset import sha as file_sha
    lock=OwnerLock(root/'control/STAGE_SERVER_OWNER.lock')
    try:
        old=root/'control/resident-owner'
        actor=read(old/'SERVER_OWNER_PROCESS.private.json');wrapper=read(root/'control/RESIDENT_WRAPPER_PROCESS.private.json')
        if not dead(actor['pid'],actor['proc_start_ticks'])or not dead(wrapper['wrapper_pid'],wrapper['wrapper_proc_start_ticks']):
            raise PermissionError('RESUME_OLD_OWNER_OR_WRAPPER_STILL_LIVE')
        for name,pin in h['sealed_inputs'].items():
            if file_sha(control/name)!=pin:raise PermissionError('RESUME_METADATA_INPUT_CHANGED:'+name)
        for name in('setup-rpc','running-rpc'):
            p=old/name/'RPC_EFFECT_STATE.json'
            if p.exists()and read(p)['status']!='CONFIRMED':
                raise PermissionError('RESUME_ORIGINAL_RPC_EFFECT_UNRESOLVED')
        saved=read(old/'SOURCE_REQUEST.private.json');d=Path(saved['directory']);r=saved['request'];request_bytes=(d/'REQUEST.json').read_bytes()
        if (sha(d/'REQUEST.json')!=saved['request_sha256']
            or saved['request_sha256']!=h['request_sha256']or json.loads(request_bytes)!=r
            or read(old/'SOURCE_JOB_BOUND.private.json')['job_id']!=h['job_id']):
            raise PermissionError('RESUME_EXACT_EXISTING_BOUND_REQUEST_JOB')
        g=read(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json');neural_ledger.check(g)
        sb=read(root/'OUTPUT_BINDING.private.json')
        # Pure join receives the observed bound job explicitly, never fake PBS_JOBID.
        registered_identity(root,d/'REQUEST.json',m,sb,r,g,uid=os.getuid(),pbs_job=h['job_id'],now=time.time(),require_bound=True)
        if read(sb['lease'])['status']!='ACTIVE':raise PermissionError('RESUME_SCIENCE_LEASE_NOT_ACTIVE')
        if (code/'scripts/pi0294/generalization/minimal_native_owner.py').resolve()!=Path(original.__file__).resolve():
            raise PermissionError('RESUME_ORIGINAL_OWNER_IMPLEMENTATION')
        source_seal=read(root/'SOURCE_SEAL.json')
        if sha(root/'SOURCE_SEAL.json')!=m['code']['source_seal_sha256']or source_seal['source_sha']!=h['science_source_sha']:
            raise PermissionError('RESUME_ORIGINAL_SOURCE_SEAL')
        for name,pin in source_seal['files'].items():
            if file_sha(code/name)!=pin:raise PermissionError('RESUME_SCIENCE_SOURCE_CHANGED:'+name)
        # Check sealed metadata once at this handoff; no weights/dataset stream.
        for name,pin in m['execution']['sealed_inputs'].items():
            if file_sha(root/name)!=pin:raise PermissionError('RESUME_SCIENCE_INPUT_CHANGED:'+name)
        class Owner(original.Owner):
            def submit(self,*args,**kwargs):raise PermissionError('RESUME_NEVER_QSUB')
        o=Owner.__new__(Owner);o.root=root;o.BASE=control;o.manifest=m;o.binding=sb
        o.CONFIG=read(root/'MIX_OWNER_CONFIG.private.json');o.adapter=adapters.manifest(m)
        if o.adapter is not adapters.e2only:raise PermissionError('RESUME_EXISTING_E2_ONLY')
        o.code=code;o.global_path=root.parent.parent/'HOPPER_RESOURCE_LEDGER.json';o.successor_scope=successor_cd.validated(g)
        o.request_template=o.CONFIG['request_template'];o.local=captured_LOCAL(root)
        o.retry_physical=o.remaining_physical=o.saved512_physical=None
        o.e2only_physical=Guard(root,o.request_template,m,lambda **_:None)
        o.last=o.current=None;o.cold_rpc_bound=o.preserve_readonly_until_deadline=True
        o.previous=SimpleNamespace(rpc=o.heartbeat_rpc);o.HEARTBEAT=original.HEARTBEAT
        identity=dict(run_uuid=root.name,job_id=h['job_id'],source_sha=r['source_sha'],request_sha256=h['request_sha256'])
        o.setup=mature.new_journal(control,'setup-rpc',identity);o.running=mature.new_journal(control,'running-rpc',identity)
        for name in('SOURCE_REQUEST','SOURCE_JOB_BOUND','TRAIN_REGISTERED'):
            o.record(name,read(old/(name+'.private.json')))
        o.record('SERVER_OWNER_PROCESS',dict(pid=os.getpid(),proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            source_sha=r['source_sha'],metadata_source_sha=h['metadata_source_sha'],job_id=h['job_id'],agent_liveness_required=False))
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        # A finite wall-bound metadata role replaces the exhausted project CPU
        # allowance. Platform watchdogs and the original job deadline remain.
        from scripts.pi0294.pv1 import mix_accounting as accounting
        neural_ledger.check=cached_pure_validation(neural_ledger.check)
        accounting.validate_global=cached_pure_validation(accounting.validate_global)
        # The original watch obtains the complete fresh queue before publishing
        # a heartbeat or release and already handles bounded read-only outages.
        o.record('HANDOFF_ATTACHED',dict(job_id=h['job_id'],native_state='REQUIRES_ORIGINAL_WATCH_FRESH_READ',request_sha256=h['request_sha256'],
            observed_epoch=time.time(),qsub_calls=0,old_owner_records_unchanged=True))
        return o,lock,cold_owner
    except BaseException:
        lock.close();raise


def main(control):
    control=Path(control);secret=sys.stdin.readline(257).rstrip('\n')or None
    o=lock=None
    try:
        # Do not inherit the metadata RPC's short CPU soft limit. The role is
        # bounded by the original deadline and the detached launcher's wall
        # timer, with one CPU. This does not alter a PBS limit or science run.
        remaining=max(1,int(read(control/'RUN_MANIFEST.json')['deadline_epoch']-time.time()))
        hard=resource.getrlimit(resource.RLIMIT_CPU)[1]
        resource.setrlimit(resource.RLIMIT_CPU,metadata_CPU_limits(remaining,hard))
        o,lock,cold=attach(control)
        terminal=cold.watch(o,secret,scope_binding=o.successor_scope)
        o.record('RESIDENT_TERMINAL_RETURNED',terminal)
        result=o.finish(terminal);o.record('RESIDENT_EXIT',dict(observed_epoch=time.time(),status=result['status']))
    except BaseException as error:
        value=dict(type=type(error).__name__,error=str(error)[-3000:],observed_epoch=time.time(),qsub_calls=0)
        if o is not None:o.record('PRESERVED_STOP',value)
        else:
            with(control/'ATTACH_FAILED.json').open('x')as f:json.dump(value,f)
        raise
    finally:
        secret=None
        if lock is not None:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--control',required=True);main(parser.parse_args().control)
