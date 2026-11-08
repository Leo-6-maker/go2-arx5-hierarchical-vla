"""Execute one already registered PBS A40 allocation, with durable closeout."""
import argparse,json,os,time,traceback
from contextlib import closing
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args(argv);c=json.loads(a.config.read_bytes());root=Path(c['root'])
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.night10.native_relay import snapshot,once
    from go2arx5_vla.world_model.target_gr00t_allocation_v1 import run
    from go2arx5_vla.world_model.storage import digest
    original=json.loads((root/'control/LEASE.json').read_bytes())
    if (os.getuid()!=22182 or not os.environ.get('PBS_JOBID')
            or a.config.resolve()!=root/'ALLOCATION_CONFIG.json'
            or original.get('status')!='PBS_SUBMITTED' or original.get('job_id')!=os.environ['PBS_JOBID']
            or original.get('config_sha256')!=digest(a.config)
            or (root/'ALLOCATION_NATIVE_FINAL.json').exists()):
        raise PermissionError('ACTUAL_NEW_TARGET_ALLOCATION_ENVELOPE_NEVER_REPLAY')
    lock=OwnerLock(root/'OWNER.lock');result=None;error=None;exit_code=1
    try:
        result=run(c);exit_code=0;return 0
    except BaseException:
        error=traceback.format_exc();raise
    finally:
        write(root/'ALLOCATION_NATIVE_FINAL.json',result or dict(status='FAILED_PRESERVED',failure=error,
            children=json.loads((root/'CHILDREN_ACTUAL.json').read_bytes())if(root/'CHILDREN_ACTUAL.json').exists()else [],
            actual_PBS_terminal='PENDING_INDEPENDENT_READBACK'))
        lease,pin=snapshot(root/'control/LEASE.json')
        if lease['status']!='PBS_SUBMITTED':raise PermissionError('UNKNOWN_ALLOCATION_LEASE_RECONCILE')
        lease.update(status='CLOSED',worker_exit_code=exit_code,closed_epoch=time.time(),
            result_sha256=digest(root/'ALLOCATION_NATIVE_FINAL.json'),actual_PBS_terminal='PENDING_INDEPENDENT_READBACK')
        once(root/'control/LEASE.json',lease,root.name+'-NATIVE-CLOSE',expected_before_sha256=pin)
        with closing(OwnerLock(c['ledger_lock'])):
            ledger,pin=snapshot(c['ledger']);owner=ledger['TARGET_GR00T_V1_owner']
            if owner['root']!=str(root) or owner['job_id']!=os.environ['PBS_JOBID']:
                raise PermissionError('ACTUAL_TARGET_CANONICAL_OWNER_CHANGED')
            owner.update(status='NATIVE_CLOSED_PENDING_PBS_TERMINAL',native_exit_code=exit_code,
                native_final_sha256=digest(root/'ALLOCATION_NATIVE_FINAL.json'))
            once(Path(c['ledger']),ledger,root.name+'-NATIVE-OWNER-CLOSE',expected_before_sha256=pin)
        lock.close()

if __name__=='__main__':raise SystemExit(main())
