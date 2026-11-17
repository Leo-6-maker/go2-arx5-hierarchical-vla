"""One finite serial world, optional frozen GPU worker, no scientific replay."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.hybrid_nonphysical_v3 import read
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close


def compact_case(spec,result):
    detail=result.get('detail')or{};metrics=result.get('metrics')or{}
    trace=result.get('trace_audit')or{}
    return dict(id=Path(spec['registered_root']).name,role=spec['sim_only_role'],
        arm=spec.get('arm'),panel_entry_id=spec.get('panel_entry_id'),
        family=spec['mobile_condition']['family'],condition=spec['mobile_condition'],
        actual_native_requests=result.get('physical_requests'),
        native_ticks=result.get('actual_native_integrations'),
        initialization_ticks=min(result.get('actual_native_integrations',0),5703),
        task_ticks=max(0,result.get('actual_native_integrations',0)-5703),
        technical_failure=result['technical_failure'],hard_STOP=result['unsafe'],
        completed=result['completed'],native_grasp_ever=detail.get('ever_grasp'),
        task_success=metrics.get('task_success',False),
        task_and_mobile_success=metrics.get('task_and_required_motion_success',False),
        directional_base_motion_m=metrics.get('directional_motion'),
        maximum_lift_m=detail.get('maximum_lift_m'),hold_s=detail.get('longest_grasp_lift_hold_s'),
        model_queries_confirmed=detail.get('actual_SLA_queries'),
        interface_protocol_pass=detail.get('interface_protocol_pass'),
        valid_policy_rejection=detail.get('valid_policy_rejection'),
        first_failure=result.get('failure')or detail.get('terminal_reason'),
        trace_status=trace.get('status'),integrations_after_STOP=trace.get('integrations_after_first_STOP'),
        trace_errors=trace.get('errors'),wall_s=result.get('wall_s'),CPU_s=result.get('CPU_s'),
        source_sha=spec['source_sha'],scene_sha256=spec['frozen_files'][spec['model_path']],
        result_sha256=digest(Path(spec['registered_root'])/'RESULT.private.json'))


def interface_gate(rows):
    items=[r for r in rows if r['role']=='INTERFACE']
    return len(items)==2 and {r['family']for r in items}=={'F1','F2'} and all(r['completed']and not r['hard_STOP']
        and r['interface_protocol_pass'] for r in items)


def fatal_control_violation(row):
    return (bool(row.get('integrations_after_STOP'))
        or any(e in (row.get('trace_errors')or[]) for e in (
            'NATIVE_INTEGRATIONS_AFTER_FIRST_STOP','DECLARED_NATIVE_TICK_LIMIT_EXCEEDED')))


def pilot_gate(rows):
    items=[r for r in rows if r['role']=='STUDENT']
    return len(items)==8 and not any(r['hard_STOP']for r in items)and all(
        any(r['family']==family and r['task_and_mobile_success']for r in items)for family in ('F1','F2'))


def panel_preflight(spec):
    """Normalize ownership before any case runs; original per-case guard stays."""
    prior=spec.get('prior_case_results',[])
    if len(spec['cases'])+len(prior)!=18:
        raise ValueError('FINITE_ORIGINAL_PANEL_INCLUDES_CARRIED_RESULTS')
    for path in spec['cases']:
        case=read(path)
        if (case['mobile_world_lock']!=spec['mobile_world_lock']
                or case['ROS_DOMAIN_ID']!=spec['ROS_DOMAIN_ID']
                or case['mobile_group']!=spec['registered_root']):
            raise ValueError('ALL_SERIAL_CASES_REQUIRE_THE_ACTUAL_PARENT_DOMAIN_AND_LOCK')
    return prior


def carried_rows(prior):
    rows=[]
    for p in prior:
        if digest(p['spec']['path'])!=p['spec']['sha256']or digest(p['result']['path'])!=p['result']['sha256']:
            raise ValueError('CARRIED_RESULT_IDENTITY_CHANGED')
        case=read(p['spec']['path']);result=read(p['result']['path'])
        lease=read(Path(case['registered_root'])/'control/LEASE.json')
        if (lease['status']!='CLOSED'
                or case.get('scope')!='SIM_ONLY_PROJECT_AUTHORIZED'
                or case.get('sim_only_role')not in('INTERFACE','EXPERT','STUDENT')
                or result.get('physical_requests')!=1):
            raise ValueError('ONLY_SEALED_KNOWN_SCOPE_RESULTS_MAY_BE_CARRIED')
        if case['sim_only_role']=='INTERFACE'and(not result['completed']or result['unsafe']
                or not(result.get('detail')or{}).get('interface_protocol_pass')):
            raise ValueError('CARRIED_INTERFACE_MUST_HAVE_PASSED')
        # A closed failed student/expert still occupies its original panel
        # condition. Carrying preserves it; it never authorizes a replay.
        rows.append(compact_case(case,result))
    return rows


def run(spec_path,*,preflight=panel_preflight,bootstrap='sim_only_bootstrap_v51.py',expected_count=18,result_schema='MFREE_SIM_ONLY_V51_ACTUAL_TERMINAL',pilot_evaluate=pilot_gate):
    from scripts.wm02.mobile_boundary_runner_v1 import registered
    spec=read(spec_path);root=Path(spec['registered_root']);registered(spec,spec_path)
    prior=preflight(spec)
    if (root/'SUPERVISOR.private.json').exists():
        raise PermissionError('FINITE_18_CASE_NEW_CONTROLLER_ONLY')
    write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),epoch=time.time(),source_sha=spec['source_sha']))
    journal=Journal(root/'control/journal',root.name)
    rows=carried_rows(prior);prior_count=len(rows)
    failure=None;code=1;gpu=None;start=time.monotonic();launches=[]
    server_pin=spec['resident_server_spec'];server=read(server_pin['path']);server_root=Path(server['registered_root'])
    if digest(server_pin['path'])!=server_pin['sha256']:
        raise ValueError('IMMUTABLE_MODEL_SERVICE_SPEC')
    def progress(phase,**extra):
        write(root/'PROGRESS.private.json',dict(phase=phase,epoch=time.time(),source_sha=spec['source_sha'],
            deadline_epoch=spec['deadline_epoch'],completed_cases=len(rows),records=rows,**extra))
    def start_gpu():
        command=[server['python'],'-B',str(Path(spec['source_root'])/'scripts/wm02'/bootstrap),
                 '--worker',server_pin['path']]
        with(server_root/'logs/worker.log').open('xb')as log:
            child=subprocess.Popen(command,cwd=spec['source_root'],env=dict(os.environ,**server['gpu_environment']),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=child.pid,proc_start=Path('/proc')/str(child.pid)/'stat')
        write(server_root/'SUPERVISOR.private.json',dict(pid=child.pid,
            proc_start=record['proc_start'].read_text().rsplit(')',1)[1].split()[19],epoch=time.time()))
        return child
    try:
        with _world(spec['mobile_world_lock']) as world:
            for index,path in enumerate(spec['cases']):
                if time.time()>=spec['deadline_epoch']-900 or (root/'STOP.json').exists():
                    progress('FINITE_STOP_OR_DEADLINE');break
                case=read(path)
                if case['sim_only_role']!='INTERFACE'and not interface_gate(rows):
                    progress('INTERFACE_FAILURE_NO_SCIENCE_DISPATCH');break
                if case['sim_only_role']=='STUDENT' and gpu is None:
                    progress('WAIT_SHARED_GPU_RESOURCE')
                    gpu=start_gpu()
                    until=min(time.monotonic()+900,time.monotonic()+spec['deadline_epoch']-time.time()-300)
                    while not(server_root/'READY.private.json').exists():
                        if gpu.poll()is not None:raise RuntimeError('MODEL_SERVICE_FAILED_BEFORE_STUDENT')
                        if (root/'STOP.json').exists()or time.monotonic()>=until:
                            raise TimeoutError('FINITE_MODEL_READY_WAIT')
                        progress('WAIT_SHARED_GPU_RESOURCE',worker_PID=gpu.pid)
                        time.sleep(10)
                progress('NATIVE_CASE',current_case=case['mobile_condition'],case_index=index)
                log=root/'logs'/('case-%02d.log'%index)
                command=[case['python'],'-B',str(Path(spec['source_root'])/'scripts/wm02'/bootstrap),
                    '--case',path,'--world-lock-fd',str(world.handle.fileno())]
                def execute():
                    with log.open('xb')as stream:
                        child=subprocess.Popen(command,cwd=spec['source_root'],
                            env=dict(os.environ,**case['native_environment']),stdout=stream,stderr=subprocess.STDOUT,
                            pass_fds=(world.handle.fileno(),))
                    launches.append(dict(case=path,pid=child.pid))
                    write(root/'CURRENT_CASE.private.json',dict(case=path,pid=child.pid,epoch=time.time()))
                    while child.poll()is None:
                        progress('NATIVE_CASE',current_case=case['mobile_condition'],case_index=index,child_PID=child.pid)
                        if time.time()>=spec['deadline_epoch']-180:
                            write(root/'STOP.json',dict(reason='FINITE_PARENT_DEADLINE'))
                        time.sleep(10)
                    result=Path(case['registered_root'])/'RESULT.private.json'
                    if not result.exists():raise RuntimeError('CASE_TERMINAL_WITHOUT_RESULT_NO_REPLAY')
                    return dict(path=str(result),sha256=digest(result),exit_code=child.returncode)
                pin=journal.once('CASE-%02d'%index,dict(spec_sha256=digest(path)),execute)
                result=read(pin['path']);row=compact_case(case,result);rows.append(row)
                write(root/'RECORDS.private.json',rows);progress('CASE_SEALED',last=row)
                if fatal_control_violation(row):raise RuntimeError('CONTROL_STOP_OR_TICK_BOUND_VIOLATION')
                if row['technical_failure']:
                    progress('PRESERVED_TECHNICAL_FAILURE_NO_AUTO_REPLAY');break
            code=0 if len(rows)==expected_count and not any(r['technical_failure']for r in rows) else 1
    except BaseException:failure=traceback.format_exc()
    finally:
        if gpu is not None:
            write(Path(server['ipc_root'])/'FINISH.json',dict(reason='FINITE_SIMULATION_PANEL_TERMINAL'))
            try:gpu.wait(timeout=60)
            except subprocess.TimeoutExpired:
                write(server_root/'STOP.json',dict(reason='BOUNDED_MODEL_EXIT'))
                try:gpu.wait(timeout=30)
                except subprocess.TimeoutExpired:gpu.terminate();gpu.wait(timeout=30)
        elif read(server_root/'control/LEASE.json')['status']=='ACTIVE':
            close(server,0,not_started=True,reason='NO_STUDENT_DISPATCH')
        for path in spec['cases']:
            case=read(path);case_root=Path(case['registered_root'])
            if (not any(r['case']==path for r in launches)
                    and not(case_root/'SUPERVISOR.private.json').exists()
                    and read(case_root/'control/LEASE.json')['status']=='ACTIVE'):
                close(case,0,not_started=True,reason='UNSTARTED_FINITE_SUCCESSOR')
        neural=read(server_root/'RESULT.private.json')if(server_root/'RESULT.private.json').exists()else None
        value=dict(schema=result_schema,exit_code=code,failure=failure,
            source_sha=spec['source_sha'],records=rows,attempted_case_launches=len(launches),
            verified_native_requests=sum(r['actual_native_requests']for r in rows),
            carried_native_requests=sum(r['actual_native_requests']for r in rows[:prior_count]),
            new_verified_native_requests=sum(r['actual_native_requests']for r in rows[prior_count:]),
            unresolved_case_launches=len(launches)-(len(rows)-prior_count),
            native_ticks=sum(r['native_ticks']for r in rows),
            expert_task_success=sum(r['role']=='EXPERT'and r['task_and_mobile_success']for r in rows),
            student_task_success=sum(r['role']=='STUDENT'and r['task_and_mobile_success']for r in rows),
            student_attempts=sum(r['role']=='STUDENT'for r in rows),unstarted_cases=expected_count-len(rows),
            pilot_GO=pilot_evaluate(rows),DEV_TEST_auto_started=False,model_service=neural,
            new_updates=0,new_optimization_visits=0,real_robot_requests=0,wall_s=time.monotonic()-start)
        write(root/'RESULT.private.json',value);close(spec,code)
    return code


def _world(path):
    from contextlib import closing
    return closing(OwnerLock(path))
