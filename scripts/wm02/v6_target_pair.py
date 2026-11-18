"""Finite existing-Journal sequence of the two registered V6 target arms."""
import json,subprocess,time,traceback
from pathlib import Path
from scripts.pi0292.durable import write
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close


def run(spec):
    root=Path(spec['registered_root']);journal=Journal(root/'control/journal',root.name);records=[]
    try:
        for child in spec['children']:
            path=Path(child['spec_path']);raw=path.read_bytes()
            if __import__('hashlib').sha256(raw).hexdigest()!=child['spec_sha256']:raise ValueError('IMMUTABLE_CHILD_SPEC_CHANGED')
            c=json.loads(raw);role=c['arm'];write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='REGISTERED_TARGET_CHILD',arm=role,completed=records))
            def start_and_wait():
                command=[c['python'],'-B',str(Path(c['source_root'])/'scripts/wm02/v6_bootstrap.py'),'--target-train',str(path)]
                log=Path(c['registered_root'])/'logs/supervisor.log'
                with log.open('ab')as out:
                    process=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,env={**__import__('os').environ,**c['native_environment']},start_new_session=True)
                    write(root/'CURRENT_CHILD.private.json',dict(pid=process.pid,spec=str(path),arm=role,epoch=time.time()))
                    while process.poll()is None:
                        if time.time()>=spec['deadline_epoch']-300 or(root/'STOP.json').exists():
                            write(Path(c['registered_root'])/'STOP.json',dict(reason='FINITE_PARENT_LIMIT_OR_STOP',epoch=time.time()))
                        time.sleep(2)
                result_path=Path(c['registered_root'])/'RESULT.private.json'
                if not result_path.exists():raise RuntimeError('CHILD_TERMINAL_WITHOUT_RESULT_REQUIRES_RECONCILIATION')
                result=json.loads(result_path.read_bytes());return dict(arm=role,process_exit=process.returncode,result_exit=result['exit_code'],
                    new_updates=result.get('new_updates'),new_visits=result.get('new_visits'),result_path=str(result_path),result_sha256=__import__('hashlib').sha256(result_path.read_bytes()).hexdigest())
            actual=journal.once('TARGET-'+role,child,start_and_wait);records.append(actual)
            if actual['result_exit']or actual['process_exit']:break
        code=0 if len(records)==2 and all(r['result_exit']==r['process_exit']==0 for r in records)else 1
        result=dict(exit_code=code,status='TWO_TARGET_ENDPOINTS_SEALED'if code==0 else'FINITE_SUFFIX_NOT_STARTED_NO_REPLAY',records=records,source_sha=spec['source_sha'],new_native_requests=0)
    except BaseException:code=1;result=dict(exit_code=code,status='PRESERVED_PAIR_TECHNICAL_FAILURE',failure=traceback.format_exc(),records=records,new_native_requests=0)
    write(root/'RESULT.private.json',result);close(spec,code);return code
