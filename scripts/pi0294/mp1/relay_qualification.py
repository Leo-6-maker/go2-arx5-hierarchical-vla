"""Finite same-host filesystem qualification, not model or closed-loop validation."""
import argparse
import json
from pathlib import Path
import statistics
import signal
import time

from scripts.pi0292.durable import read,write
from scripts.pi0294.relay_io import read_available
from .scope import validate,charge_cpu


def filesystem_probe(folder,guard,*,cycles=96):
    if type(cycles) is not int or not 1<=cycles<=96:raise ValueError('FINITE_FILESYSTEM_PROBE_BOUND')
    folder=Path(folder);folder.mkdir(mode=0o700,exist_ok=False);records=[]
    for sequence in range(cycles):
        guard();row=dict(sequence=sequence);packet=folder/(f'packet-{sequence:03d}.json')
        payload=dict(synthetic=True,sequence=sequence,padding='x'*(96*1024))
        def measure(name,operation):
            start=time.monotonic()
            try:return operation()
            finally:row[name]=time.monotonic()-start
        if measure('absent_direct_open_s',lambda:read_available(packet)) is not None:raise ValueError('FRESH_SYNTHETIC_PACKET_ONLY')
        measure('atomic_packet_fsync_replace_and_directory_fsync_s',lambda:write(packet,payload))
        actual=measure('direct_read_s',lambda:read_available(packet))
        if actual!=payload:raise ValueError('SYNTHETIC_DIRECT_FILE_READ_MISMATCH')
        actual=measure('stat_then_read_s',lambda:read(packet) if packet.is_file() else None)
        if actual!=payload:raise ValueError('SYNTHETIC_STAT_FILE_READ_MISMATCH')
        measure('heartbeat_fsync_s',lambda:write(folder/'SYNTHETIC_HEARTBEAT.json',dict(synthetic=True,sequence=sequence)))
        measure('accounting_fsync_s',lambda:write(folder/'SYNTHETIC_LEDGER.json',dict(synthetic=True,RPC_calls=sequence+1)))
        records.append(row)
    summary={}
    for key in records[0]:
        if key=='sequence':continue
        values=sorted(r[key] for r in records);summary[key]=dict(samples=len(values),median_s=statistics.median(values),
            p95_s=values[max(0,(95*len(values)+99)//100-1)],maximum_s=max(values),at_least_20s=sum(x>=20 for x in values))
    return dict(cycles=cycles,timings=records,summary=summary,
        all_operations_same_host_and_kernel=True,cross_compute_node_NFS_visibility_tested=False,
        SSH_roundtrip_or_full_RPC_measured=False,original_20s_RPC_watchdog_changed=False,
        full_32s_closed_loop_qualification=False,new_model_queries=0,new_physical_integrations=0)


def run(root,role):
    root=Path(root)
    if role not in ['core','hopper']:raise PermissionError('EXACT_MP1_CPU_FILESYSTEM_ROLE')
    m=validate(root,'4080-cpu-data' if role=='core' else 'hopper-neural');a=read(root/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json')
    request=root/'plans'/('FILE_RPC_CPU_QUALIFICATION_'+role+'.json');spec=read(request)
    if spec!=dict(run_uuid=root.name,role=role,purpose='SYNTHETIC_FILESYSTEM_TIMING_NO_MODEL_OR_PHYSICS',
            source_sha=m['code']['vla_commit'],maximum_CPU_s=600,maximum_wall_s=180,cycles=96):
        raise PermissionError('EXACT_BOUND_MP1_CPU_PROBE_REQUEST')
    binding=read(root/'OUTPUT_BINDING.private.json');out=root/'evidence'/('file-rpc-cpu-qualification-'+role)
    started=time.process_time();wall=time.monotonic();out.mkdir(mode=0o700,exist_ok=False)
    value=dict(status='FAILED_OR_UNKNOWN_PRESERVED_NO_AUTOMATIC_RETRY',role=role,source_sha=m['code']['vla_commit'])
    old_handler=signal.getsignal(signal.SIGALRM)
    def expire(signum,frame):raise TimeoutError('FINITE_CPU_FILESYSTEM_PROBE_WALL_TIMEOUT')
    try:
        signal.signal(signal.SIGALRM,expire);signal.alarm(180)
        def guard():
            lease=read(binding['lease'])
            if lease['status']!='ACTIVE_CONTROL_PLANE_ONLY' or lease['run_uuid']!=root.name:raise PermissionError('ORIGINAL_MP1_CONTROL_LEASE')
            if time.monotonic()-wall>=180 or time.time()>=a['hard_closeout_epoch']:raise TimeoutError('FINITE_ORIGINAL_MP1_CPU_PROBE_DEADLINE')
        value.update(filesystem_probe(out/'synthetic',guard,cycles=96),status='PASS_SAME_HOST_FILESYSTEM_PROBE_NOT_CLOSED_LOOP')
        return value
    except BaseException as exc:value['error_type']=type(exc).__name__;raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        value.update(CPU_s=time.process_time()-started,wall_s=time.monotonic()-wall,
            synthetic_only=True,scientific_source_or_dataset_modified=False)
        write(out/'RESULT.json',value);charge_cpu(root,time.process_time()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--role',required=True);a=p.parse_args();v=run(a.root,a.role)
    print(json.dumps({k:v[k] for k in ['status','role','CPU_s','wall_s','summary']}),flush=True)
