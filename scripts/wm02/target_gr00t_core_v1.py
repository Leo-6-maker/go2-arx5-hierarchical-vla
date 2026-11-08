"""One registered Core diagnostic chain; no PBS submission or physics."""
import argparse,json,time,traceback,resource
from pathlib import Path

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(argv)
    from scripts.pi0292.durable import write
    from go2arx5_vla.world_model import target_gr00t_core_v1 as core
    from go2arx5_vla.world_model.storage import digest
    c=json.loads(args.config.read_bytes());root=Path(c['root']);core.scope(c)
    start=time.monotonic();result=None;failure=None;code=1
    try:result=core.run(c);code=0
    except BaseException:failure=traceback.format_exc();raise
    finally:
        children=json.loads((root/'CHILDREN_ACTUAL.json').read_bytes()).get('children',[])if(root/'CHILDREN_ACTUAL.json').exists()else[]
        write(root/'RESULT.json',result or dict(status='CORE_TRAINING_FAILURE_PRESERVED',failure=failure,
            children=children,actual_GR00T_updates=sum(r['GR00T_updates']for r in children)))
        usage=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        write(root/'WORKER_EXIT.private.json',dict(exit_code=code,wall_s=time.monotonic()-start,
            CPU_s=usage.ru_utime+usage.ru_stime,child_CPU_s=child.ru_utime+child.ru_stime,failure=failure))
        lease=json.loads((root/'control/LEASE.json').read_bytes())
        if lease.get('status')!='ACTIVE':raise PermissionError('CORE_MASTER_LEASE_CHANGED')
        lease.update(status='CLOSED',worker_exit_code=code,closed_epoch=time.time(),result_sha256=digest(root/'RESULT.json'))
        write(root/'control/LEASE.json',lease)
    return code

if __name__=='__main__':raise SystemExit(main())
