"""Finite source1000 and two independent reloads using existing Journal."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from scripts.pi0292.durable import write
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close


def run(spec):
    root=Path(spec['registered_root']);journal=Journal(root/'control/journal',root.name);records=[]
    try:
        # A capacity boundary only: source learning does not require target
        # scientific success. Leave the healthy pair's inter-arm gap alone.
        predecessor=spec['wait_for_existing_pair_terminal'];path=Path(predecessor['spec_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=predecessor['spec_sha256']:raise ValueError('EXISTING_TARGET_PAIR_IDENTITY_CHANGED')
        previous=path.parent
        while True:
            lease=json.loads((previous/'control/LEASE.json').read_bytes())
            if lease['status']=='CLOSED'and(previous/'RESULT.private.json').exists():break
            if time.time()>=spec['deadline_epoch']-600 or(root/'STOP.json').exists():raise TimeoutError('FINITE_SOURCE_CAPACITY_QUEUE_LIMIT_NO_TARGET_PREEMPTION')
            write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='WAIT_HEALTHY_TARGET_PAIR_COLD_BOUNDARY',predecessor_spec_sha256=predecessor['spec_sha256'],
                predecessor_status=lease['status'],new_updates=0,new_generations=0,new_native_requests=0,deadline_epoch=spec['deadline_epoch']))
            time.sleep(30)
        for child in spec['children']:
            path=Path(child['spec_path']);raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=child['spec_sha256']:raise ValueError('IMMUTABLE_SOURCE_CHILD_CHANGED')
            c=json.loads(raw);role=child['role'];write(root/'PROGRESS.private.json',dict(epoch=time.time(),phase='REGISTERED_SOURCE_CHILD',role=role,completed=records))
            def execute():
                command=[c['python'],'-B',str(Path(c['source_root'])/'scripts/wm02/v6_bootstrap.py'),child['route'],str(path)]
                with(Path(c['registered_root'])/'logs/supervisor.log').open('ab')as log:
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,**c['native_environment']},start_new_session=True)
                    write(root/'CURRENT_CHILD.private.json',dict(pid=process.pid,role=role,spec=str(path),epoch=time.time()))
                    while process.poll()is None:
                        if time.time()>=spec['deadline_epoch']-600 or(root/'STOP.json').exists():write(Path(c['registered_root'])/'STOP.json',dict(reason='FINITE_PARENT_LIMIT_OR_STOP',epoch=time.time()))
                        time.sleep(2)
                result_path=Path(c['registered_root'])/'RESULT.private.json'
                if not result_path.exists():raise RuntimeError('SOURCE_CHILD_TERMINAL_WITHOUT_RESULT_RECONCILE')
                result=json.loads(result_path.read_bytes())
                return dict(role=role,process_exit=process.returncode,result_exit=result['exit_code'],result_path=str(result_path),result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
            actual=journal.once('SOURCE-'+role,child,execute);records.append(actual)
            if actual['result_exit']or actual['process_exit']:break
        code=0 if len(records)==3 and all(r['result_exit']==r['process_exit']==0 for r in records)else 1
        result=dict(exit_code=code,status='SOURCE1000_AND_INDEPENDENT_RESTORES_SEALED_TEST_PENDING'if code==0 else'FINITE_SOURCE_SUFFIX_NOT_STARTED_NO_REPLAY',records=records,
            source_sha=spec['source_sha'],new_native_requests=0)
        if code==0:
            from go2arx5_vla.sla_adapter.source_evaluation_v6 import transfer_gate
            values=[json.loads(Path(r['result_path']).read_bytes())for r in records];train,a,b=values
            result['transfer_gate']=transfer_gate(train['S0']['SOURCE_DEV'],train['S1000']['SOURCE_DEV'],reloadable=a['passed']and b['passed'])
            result['new_updates']=train['new_updates'];result['new_visits']=train['new_visits']
            result['actual_offline_generations']=sum(v['actual']['ordinary_RGB_generations']for v in values)
    except BaseException:code=1;result=dict(exit_code=code,status='PRESERVED_SOURCE_CAMPAIGN_TECHNICAL_FAILURE',failure=traceback.format_exc(),records=records,new_native_requests=0)
    write(root/'RESULT.private.json',result);close(spec,code);return code
