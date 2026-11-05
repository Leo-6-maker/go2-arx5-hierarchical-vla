"""Close one fresh finite native batch using the existing inherited-owner runner."""
import argparse,json,os,resource,signal,time
from pathlib import Path


def close_proven_unstarted_children(spec_path,spec,root):
    """run_batch writes this marker before Popen; never infer an active exit."""
    from scripts.pi0292.durable import write
    from go2arx5_vla.world_model.storage import digest
    if (root/'batch-01/INHERITED_OWNER.private.json').exists():return []
    rows=[]
    for slot in spec['slots']:
        p=Path(slot['spec']);child=json.loads(p.read_text());r=Path(child['registered_root'])
        if (r/'native-01').exists():return []
        if digest(p)!=slot['spec_sha256'] or child.get('parallel_batch_spec')!=str(spec_path):
            raise PermissionError('UNSTARTED_CHILD_IDENTITY_NOT_PROVEN')
        lease=Path(child['lease']);v=json.loads(lease.read_text())
        if v.get('run_uuid')!=r.name or v.get('spec_sha256')!=digest(p):
            raise PermissionError('UNSTARTED_CHILD_LEASE_IDENTITY_DIFFERS')
        rows.append((r,lease,v))
    closed=[]
    for r,lease,v in rows:
        if v['status']=='ACTIVE':
            v.update(status='CLOSED',closed_epoch=time.time(),worker_started=False,worker_exit_code=None,
                reason='PARENT_REJECTED_BEFORE_INHERITED_OWNER_AND_NATIVE_LAUNCH')
            write(lease,v);closed.append(r.name)
    return closed


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);a=p.parse_args(argv)
    from scripts.pi0292.durable import write
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model.target_parallel import run_batch
    s=json.loads(a.spec.read_text());root=Path(s['registered_root']);lease=Path(s['lease']);v=json.loads(lease.read_text())
    if v.get('status')!='ACTIVE'or v.get('run_uuid')!=root.name or v.get('spec_sha256')!=digest(a.spec):raise PermissionError('FRESH_ACTUAL_TWO_SLOT_BATCH_LEASE_REQUIRED')
    if(root/'SUPERVISOR.private.json').exists():raise FileExistsError('RECONCILE_EXISTING_OR_UNKNOWN_BATCH')
    write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),source_sha=s['source_sha'],started_epoch=time.time()))
    def stop(signum,frame):raise KeyboardInterrupt('OWNED_BATCH_SIGNAL_'+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();cpu=time.process_time();code=1;failure=None
    try:
        r=run_batch(a.spec,root/'batch-01','batch-01');code=0 if r['status']=='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED'else 1
    except BaseException as error:failure=type(error).__name__+': '+str(error)
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        unstarted=close_proven_unstarted_children(a.spec,s,root)
        write(root/'WORKER_EXIT.private.json',dict(exit_code=code,failure=failure,CPU_s=time.process_time()-cpu,
              child_CPU_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,wall_s=time.monotonic()-start,
              proven_unstarted_children_closed=unstarted,
              cost_scope='BATCH_TOTAL_INCLUDING_DESCENDANTS_NOT_PER_CHILD_CPU'))
        v=json.loads(lease.read_text())
        if v.get('status')!='ACTIVE'or v.get('spec_sha256')!=digest(a.spec):raise PermissionError('ACTUAL_BATCH_LEASE_CHANGED')
        v.update(status='CLOSED',closed_epoch=time.time(),worker_exit_code=code);write(lease,v)
    return code


if __name__=='__main__':raise SystemExit(main())
