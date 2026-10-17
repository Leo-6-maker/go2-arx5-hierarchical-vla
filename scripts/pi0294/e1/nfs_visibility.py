"""Finite read-only directory refresh; no action, inference or deadline changes."""
import argparse
import os
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import write
from scripts.pi0294.native_dataset import sha
from .control_read import read
from .scope import validate,Ledger


def refresh(directory):
    count=0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.endswith('.json') and not entry.is_symlink():
                try:entry.stat(follow_symlinks=False);count+=1
                except FileNotFoundError:pass
    return count


def run(root,label):
    root=Path(root);m=validate(root,'hopper-neural');ledger=Ledger(root,'hopper');ledger.guard()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or label not in ['M2-BC-REMAINDER-03','M2-CORR-REMAINDER-03']:
        raise PermissionError('READONLY_CPU_VISIBILITY_ONLY_FIXED_LAST_E1_PANEL')
    registry=root/'plans'/label/'REGISTRY.json';digest=sha(registry);reg=read(registry)
    resident=root/'evidence'/label/'resident';state=read(resident/'STATE.json')
    if reg['run_uuid']!=root.name or state['label']!=label or state['job_id']!='671655.hopper-m-02':
        raise PermissionError('EXACT_EXISTING_LAST_NATIVE_MODEL_SESSION')
    old=(root/'RUN_MANIFEST.json').read_bytes();started=time.process_time();scans=0;seen=0
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    out=root/'evidence/NFS_VISIBILITY_REFRESH'/label;out.mkdir(parents=True,exist_ok=False)
    write(out/'STARTED.json',dict(status='READONLY_EXISTING_OUTBOX_METADATA_REFRESH',source_sha=m['code']['vla_commit'],
        job_id=state['job_id'],registry_sha256=digest,original_deadline_epoch=reg['deadline_epoch'],started_epoch=time.time(),
        new_model_queries=0,new_physical_integrations=0,original_watchdogs_unchanged=True))
    status='FAILED_PRESERVED'
    try:
        while time.time()<reg['deadline_epoch'] and not (resident/'STAGE_DONE.json').exists():
            ledger.guard()
            if (root/'RUN_MANIFEST.json').read_bytes()!=old or sha(registry)!=digest:raise PermissionError('ORIGINAL_SESSION_BINDING_CHANGED')
            if read(root/'OUTPUT_BINDING.private.json')['run_uuid']!=root.name:raise PermissionError('ORIGINAL_OUTPUT_BINDING_CHANGED')
            seen=max(seen,refresh(resident/'outbox'));scans+=1;time.sleep(.25)
        status='FINITE_METADATA_REFRESH_TERMINAL';return dict(status=status,scans=scans,maximum_responses_observed=seen)
    finally:
        cpu=time.process_time()-started;ledger.cpu(cpu)
        write(out/'TERMINAL.json',dict(status=status,scans=scans,maximum_responses_observed=seen,CPU_s=cpu,
            new_model_queries=0,new_physical_integrations=0,unknown_intents_retried=False,original_watchdogs_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--label',required=True);a=p.parse_args();print(run(a.root,a.label))
