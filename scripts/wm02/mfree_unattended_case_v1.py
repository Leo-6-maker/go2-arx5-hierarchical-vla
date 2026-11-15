"""New72h expert wrapper, reusing the original native lifecycle and scoring."""
import argparse,json,os,time,traceback
from pathlib import Path
from scripts.pi0292.durable import write
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.mobile_wbc_reference_v1 import PHASE,M0_END
from .mobile_boundary_runner_v1 import registered,read

def run_case(spec_path,lock_fd):
    from go2arx5_vla.world_model.mfree_unattended_adapter_v1 import IsolatedAdapter as TargetSequentialAdapter
    from go2arx5_vla.world_model.native_replay import packets
    from go2arx5_vla.world_model.init_trace_reader import audit_trace
    spec=read(spec_path);root=Path(spec['registered_root']);group=Path(spec['mobile_group'])
    lease=registered(spec,spec_path);parent=read(group/'SUPERVISOR.private.json')
    if (spec['phase']!=PHASE or os.uname().nodename!='corelab-4080s'or os.getuid()!=2009
            or lease['status']!='ACTIVE'or lease['spec_sha256']!=digest(spec_path)
            or os.getppid()!=parent['pid']or time.time()>=spec['deadline_epoch']):
        raise PermissionError('REGISTERED_CURRENT_FINITE_MOBILE_CHILD_REQUIRED')
    expected=os.stat(spec['mobile_world_lock']);held=os.fstat(lock_fd)
    if(expected.st_dev,expected.st_ino)!=(held.st_dev,held.st_ino):raise PermissionError('INHERITED_ACTUAL_WORLD_OWNER')
    qualification=read(spec['mobile_guard_qualification']['path'])
    if (digest(spec['mobile_guard_qualification']['path'])!=spec['mobile_guard_qualification']['sha256']
            or not qualification['passed']or qualification['library_sha256']!=spec['mobile_safety_library_sha256']):
        raise PermissionError('QUALIFIED_ACTUAL_MOBILE_GUARD_REQUIRED_BEFORE_PHYSICS')
    for name,pin in spec['frozen_files'].items():
        if digest(name)!=pin:raise PermissionError('FROZEN_MOBILE_EXECUTOR_CHANGED:'+name)
    for name,pin in spec['source_files'].items():
        if digest(Path(spec['source_root'])/name)!=pin:raise PermissionError('SEALED_MOBILE_SOURCE_CHANGED:'+name)
    out=root/'native-01';out.mkdir();adapter=TargetSequentialAdapter(spec,out)
    write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),epoch=time.time(),source_sha=spec['source_sha']))
    start=time.monotonic();cpu=time.process_time();failure=None;detail=None
    try:
        adapter.start()
        if spec.get('mobile_stage')=='L1':
            if spec.get('mobile_experiment_version')=='MFREE_C20_FINAL_PLAN_V1':
                from go2arx5_vla.world_model.mfree_expert_capture_v1 import episode as l1_episode
            elif spec.get('mobile_experiment_version')=='MFREE_STUDENT_60S_V1':
                from go2arx5_vla.world_model.mfree_student_v1 import episode as l1_episode
            elif spec.get('mobile_experiment_version')=='MOBILE_CONTINUOUS_TEACHER_V2':
                from go2arx5_vla.world_model.mobile_expert_episode_v2 import episode as l1_episode
            elif spec.get('mobile_experiment_version')=='MOBILE_GRASP_BUDGET_V3':
                from go2arx5_vla.world_model.mobile_expert_episode_v3 import episode as l1_episode
            elif spec.get('mobile_experiment_version')=='MOBILE_OBSERVED_AFTER_GRASP_V2':
                from go2arx5_vla.world_model.mobile_task_episode_v2 import episode as l1_episode
            elif spec['mobile_condition']['family']=='F1':
                from go2arx5_vla.world_model.mobile_l1_episode_v1 import episode as l1_episode
            else:
                from go2arx5_vla.world_model.mobile_task_episode_v1 import episode as l1_episode
            detail=l1_episode(adapter)
        else:detail=episode(adapter)
    except BaseException:failure=traceback.format_exc()
    finally:
        adapter.isolation_teardown=True
        if adapter.node is not None:
            try:
                adapter.capture_loaded_identity()
                if not adapter.trace_sealed:adapter.seal_trace(adapter.observe()['step'])
            except BaseException:write(out/'FINAL_EVIDENCE_ERROR.private.json',dict(error=traceback.format_exc()))
        adapter.close()
    journal=out/'state.bin.journal';count=len(packets(journal))if journal.exists()and journal.stat().st_size else 0
    stop=read(out/'MOBILE_SAFETY_STOP.private.json')if(out/'MOBILE_SAFETY_STOP.private.json').exists()else None
    physics=[json.loads(x)for x in(out/'physics.jsonl').read_text().splitlines()]if(out/'physics.jsonl').exists()else[]
    stop_events=[r for r in physics if r['kind']=='stop']
    try:trace=audit_trace(out/'init.trace',max_native_ticks=spec['max_ticks'])if(out/'init.trace').exists()else None
    except BaseException:trace=dict(error=traceback.format_exc())
    clean=bool(adapter.exit_records)and all(r['exit_code']==0 for r in adapter.exit_records)
    expected=detail.get('terminal_native_step')if detail is not None and spec.get('mobile_stage')=='L1'else M0_END
    passed=detail is not None and not failure and not stop and not stop_events and clean and count==expected
    from go2arx5_vla.world_model.mobile_boundary_metrics_v1 import metrics
    measured_metrics=metrics(out,spec['mobile_condition'])
    if spec.get('mobile_stage')=='L1':
        measured_metrics['task_success']=detail['task_success']if detail is not None else False
        measured_metrics['task_and_required_motion_success']=bool(measured_metrics['task_success']and measured_metrics.get('directional_motion',0)>=.08)
    response_count=len(list((out/'queries').glob('*/RESPONSE.private.json')))if(out/'queries').exists()else 0
    value=dict(schema='MOBILE_L1_ACTUAL_NATIVE_RESULT_V1'if spec.get('mobile_stage')=='L1'else 'MOBILE_M0_ACTUAL_NATIVE_RESULT_V1',condition=spec['mobile_condition'],
        completed=passed,quarantine_abort=bool(failure and 'SHARED_EXPERT_FAMILY_SAFETY_ISOLATION_ABORT'in failure),technical_failure=bool(failure and not stop and not stop_events and 'SHARED_EXPERT_FAMILY_SAFETY_ISOLATION_ABORT'not in failure),
        unsafe=bool(stop or stop_events),failure=failure,mobile_safety_stop=stop,physics_stops=stop_events,
        actual_native_integrations=count,detail=detail,shutdown_clean=clean,owned_process_exits=adapter.exit_records,
        trace_audit=trace,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,
        optimizer_updates=0,training_visits=0,model_generations=response_count,physical_requests=1,source_sha=spec['source_sha'],
        WBC_changes=False,raw_fixed_root_IK_is_shadow=True,metrics=measured_metrics)
    if value['unsafe']:adapter._mark_own_stop()
    write(root/'RESULT.private.json',value)
    code=0 if passed else 2 if value['unsafe']else 1
    write(root/'WORKER_EXIT.private.json',dict(exit_code=code,wall_s=value['wall_s']))
    write(root/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
    return code


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--world-lock-fd',type=int);a=p.parse_args();raise SystemExit(run_case(a.case,a.world_lock_fd))
