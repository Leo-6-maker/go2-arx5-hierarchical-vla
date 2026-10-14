"""Original dispatcher/worker/audit with the adopted R4 collection budget."""
import argparse
import os
from pathlib import Path
import socket
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.iteration.queue import digest,LIVE_KEYS
from .dispatcher import dispatch
from .r4 import validate_entry,T0,CUTOFF,DEADLINE


def run(root,*,pilot,workers):
    root=Path(root);m=validate_entry(root,profile='4080-simulation')
    freeze=root/'plans/CONTACT48_PLAN_FROZEN.json';f=read(freeze)
    if len(f['cards'])!=48 or len(f['pilot_cards'])!=4 or f['parameter_search_after_rollout'] or f['failed_seed_replacement']:
        raise PermissionError('R4_EXACT_PRE_ROLLOUT_DESIGN_REQUIRED')
    name='CONTACT_PILOT' if pilot else 'CONTACT_TRAIN48'
    if not pilot:
        qualification=read(root/'evidence/CONTACT_PILOT_QUALIFICATION.json')
        if qualification['successes']!=4 or qualification['native_failures'] or qualification['rejected']:
            raise PermissionError('R4_FOUR_ORIGINAL_PILOT_SUCCESSES_REQUIRED')
    owner=OwnerLock(root/'R4_RESOURCE_OWNER.lock');ledger=read(root/'R4_RESOURCE_LEDGER.json')
    if ledger['CPU_reservations']:raise PermissionError('R4_COLLECTION_START_ONLY_WITH_DISJOINT_RESERVED_CAPACITY')
    cores=3*workers+2;cpus=sorted(os.sched_getaffinity(0))[-16:][:cores]
    if workers not in (1,2,4) or pilot and workers!=2 or len(cpus)!=cores:raise PermissionError('R4_ACTUAL_COLLECTION_CORE_BINDING')
    previous=read(root/'LIVE_QUEUE.json') if (root/'LIVE_QUEUE.json').exists() else None
    prior=previous['actual'] if previous else {k:0 for k in LIVE_KEYS}
    # CPU and disk are total collection-lane ceilings; already charged pilot
    # costs remain in the queue while the global ledger includes them once.
    collection_charged=ledger.get('collection_CPU_s',0.)
    cap=min(80*3600,96*3600-ledger['Core_CPU_s']+collection_charged)
    ticket=dict(kind='COLLECTION',stage=name,pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
        cores=cores,cpus=cpus,cpu_s_cap=cap-prior['cpu_s'],started_epoch=time.time())
    ledger['CPU_reservations'][name]=ticket;write(root/'R4_RESOURCE_LEDGER.json',ledger);owner.close()
    cards=f['pilot_cards'] if pilot else f['cards']
    plan=dict(fixture=False,authorizes_execution=True,kind='R4_CONTACT_PILOT' if pilot else 'R4_CONTACT_TRAIN',
        frozen_design_sha256=digest(freeze),episodes=[dict(id=c['id'],card=c) for c in cards])
    plan_path=root/'plans'/f'{name}.json';admission_path=root/'evidence'/f'{name}-ADMISSION.private.json'
    if plan_path.exists() or admission_path.exists():raise FileExistsError('R4_COLLECTION_FIRST_ATTEMPT_PLAN_EXISTS')
    write(plan_path,plan)
    old=root.parent/'20261013T142457Z-localpc-pi0294';parallel=old/'evidence/PARALLEL_THROUGHPUT_requal-f1fdf3ac.json'
    if not parallel.is_file():raise PermissionError('EXISTING_FOUR_WORKER_PROOF_MISSING')
    admission=dict(schema='PI0294_LIVE_ADMISSION_V1',fixture=False,status='LIVE_ADMITTED_ACTUAL_RUNTIME',
        hostname=socket.gethostname(),profile='4080-simulation',root=str(root),run_uuid=root.name,
        source_sha=m['code']['vla_commit'],source_root=str(root/m['code']['capsule_relative']),
        source_seal_sha256=m['code']['source_seal_sha256'],run_manifest_sha256=digest(root/'RUN_MANIFEST.json'),
        plan_sha256=digest(plan_path),asset_root=read(root/'INPUT_BINDINGS.private.json')['asset_root'],
        T0_epoch=T0,cutoff_epoch=CUTOFF,deadline_epoch=DEADLINE,lease_expires_epoch=DEADLINE,
        qualified_workers=4,allowed_cpus=cpus,parallel4_receipt_sha256=digest(parallel),
        coordinator_cpu_count=1,auxiliary_cpu_count=1,memory_limit_bytes=24*1024**3,
        free_space_margin_bytes=20*1024**3,limits=dict(physics_starts=160,physics_steps=2560000,rgb_pairs=102560,
            cpu_s=cap,output_bytes=72*1024**3),prior_actual=prior)
    write(admission_path,admission)
    try:
        result=dispatch(root,plan_path,admission_path,workers=workers,timeout_s=600,stage_boundary=previous is not None)
        write(root/'evidence'/f'{name}-QUEUE_RETURN.json',result)
        if pilot:
            rows=[]
            for c in cards:
                folder=root/'episodes'/c['id'];r=read(folder/'WORKER_RECEIPT.json');a=read(folder/'AUDIT.json')
                rows.append(dict(id=c['id'],status=r['status'],admitted=a['admitted'],task_success=r.get('score',{}).get('persistent_success',False),
                    source_sha=r['source_sha'],worker_receipt_sha256=digest(folder/'WORKER_RECEIPT.json'),audit_sha256=digest(folder/'AUDIT.json')))
            successes=sum(r['admitted'] and r['task_success'] and r['status']=='VALID_COMPLETE' for r in rows)
            qualification=dict(successes=successes,rejected=4-successes,native_failures=sum(r['status']=='SAFETY_STOP_PRESERVED' for r in rows),rows=rows,
                plan_sha256=digest(freeze),status='PASS_FOUR_ACTUAL_PILOT' if successes==4 else 'VALID_PILOT_FAIL_NO_NEW_DATA_RETRY')
            write(root/'evidence/CONTACT_PILOT_QUALIFICATION.json',qualification)
            print(qualification['status'],flush=True)
    finally:
        queue=read(root/'LIVE_QUEUE.json') if (root/'LIVE_QUEUE.json').exists() else None
        owner=OwnerLock(root/'R4_RESOURCE_OWNER.lock');ledger=read(root/'R4_RESOURCE_LEDGER.json')
        if queue and not queue['reservations'] and not queue['auxiliary']:
            ledger['Core_CPU_s']+=queue['actual']['cpu_s']-ledger.get('collection_CPU_s',0.)
            ledger['collection_CPU_s']=queue['actual']['cpu_s']
            ledger['main_starts']=queue['actual']['physics_starts'];ledger['main_integrations']=queue['actual']['physics_steps'];ledger['RGB_pairs']=queue['actual']['rgb_pairs']
            ledger['CPU_reservations'].pop(name)
        write(root/'R4_RESOURCE_LEDGER.json',ledger);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,choices=(1,2,4),required=True)
    a=p.parse_args();run(a.root,pilot=a.pilot,workers=a.workers)
