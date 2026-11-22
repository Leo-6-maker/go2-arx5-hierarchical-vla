"""Finite native mobile M0 runner; original JOIN10 and frozen WBC stay separate."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import numpy as np

from scripts.pi0292.durable import OwnerLock,write
from go2arx5_vla.world_model.storage import digest
from go2arx5_vla.world_model.native_collection_performance_v1 import measure, profiled
from go2arx5_vla.world_model.mobile_wbc_reference_v1 import PHASE,START,M0_END,m0_reference


def read(path):return json.loads(Path(path).read_bytes())


def registered(spec,spec_path):
    root=Path(spec['registered_root']);lease=read(root/'control/LEASE.json')
    entry=next((r for r in read(spec['registry'])['entries']if r.get('path')==str(root)),None)
    if (os.uname().nodename!='corelab-4080s'or os.getuid()!=2009
            or lease['status']!='ACTIVE'or lease['spec_sha256']!=digest(spec_path)
            or not entry or entry['source_sha']!=spec['source_sha']or entry['owner']!=spec['owner']
            or entry['role']!=spec['role']or not set(spec['cpus'])<=os.sched_getaffinity(0)
            or spec['ROS_DOMAIN_ID']not in range(192,200)or time.time()>=spec['deadline_epoch']):
        raise PermissionError('REGISTERED_CURRENT_INDEPENDENT_FINITE_MOBILE_OWNER')
    return lease


def measured_mount(adapter):
    from go2_x5_planning.ee_trajectory import Pose
    from go2arx5_vla.trajectory.so3 import quat_to_matrix
    pose=Pose.from_base_state(adapter.latest['observation'].state.value[:6])
    out=np.eye(4);out[:3,3]=pose.position;out[:3,:3]=quat_to_matrix(np.asarray(pose.quaternion_xyzw))
    return out


def write_mobile_gripper(adapter,before,sequence,row):
    from scripts.pi0294.task_state import write_gripper
    from go2arx5_vla.world_model.native_live_control_v3 import live_path
    # Original gripper receiver requires a fresh source for EACH new sequence;
    # the still-live EEF plan source is a separate clock and may be 350ms old.
    source=float(before['time_s'])
    write_gripper(live_path(adapter,'gripper.bin'),sequence,source,float(row[9]))
    return dict(sequence=sequence,source_native_s=source,EEF_plan_source_s=adapter.active_reference['source_mpc_s'])


def arm(adapter):
    from go2arx5_vla.world_model.target_sequential import owned_simulator_pid
    wrapper=next(p for name,p in adapter.processes if name=='sim')
    pid=owned_simulator_pid(wrapper.pid,adapter.spec['native_binary'])
    maps=(Path('/proc')/str(pid)/'maps').read_text()
    if adapter.spec['mobile_safety_library']not in maps:raise PermissionError('MOBILE_GUARD_NOT_LOADED_IN_ACTUAL_SIMULATOR')
    adapter.wait_source()
    mount=measured_mount(adapter)
    # Marker stores the measured mount origin, not an arbitrary zero frame.
    path=adapter.output/'MOBILE_SAFETY_ARMED'
    with path.open('x')as f:f.write(f'MOBILE_ARMED_V1 {mount[0,3]:.17g} {mount[1,3]:.17g}\n')
    adapter.mobile_motion_armed=True;adapter.mobile_mode='M_ASSIST'
    adapter.mobile_base_preview=np.repeat(mount[None],41,axis=0)
    return mount


@profiled('mobile.consume_atom')
def consume(adapter,action,*,sequence):
    from scripts.pi0292.eval_contract import matrix10
    from go2arx5_vla.world_model.native_live_control_v3 import live_path
    from go2arx5_vla.world_model.target_reference import require_live_window
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import require_armed
    require_armed(adapter)
    before=adapter.observe();step=before['step'];source=adapter.active_reference['source_mpc_s']
    require_live_window(source,before['time_s'],25)
    row=np.asarray(adapter.active_reference['action_EEF10'])[(step-adapter.mobile_installed_step)//25]
    gripper=write_mobile_gripper(adapter,before,sequence,row)
    event_start=len(adapter.ros_evidence)
    after=adapter.advance_to(step+25)
    with measure(adapter,'mobile.post_atom_callbacks_150ms'):
        adapter.spin(.15)
    require_armed(adapter)
    from go2arx5_vla.world_model.target_reference import trajectory_identity,installed_target_identities
    sent=installed_target_identities(adapter.reference_events(),adapter.active_reference['wire_sha256'])
    matched=[];arm_requests=0;base_commands=0
    for event in adapter.ros_evidence[event_start:]:
        message=event['message']
        if event['topic_key']=='policy':
            target=message['plan_target_trajectories']
            key=trajectory_identity(target['time_trajectory'],[s['value']for s in target['state_trajectory']],
                                    [s['value']for s in target['input_trajectory']])
            if key in sent and sent[key]['monotonic_s']<=event['received_monotonic_s']:matched.append(key)
        arm_requests+=event['topic_key']=='arm_request';base_commands+=event['topic_key']=='base_command'
    if adapter.latest['mode'].data!='WBC':raise RuntimeError('MOBILE_NATIVE_WBC_MODE_LOST')
    from go2arx5_vla.world_model.close_reference_input_v1 import remember
    remember(adapter,row,gripper,after)
    adapter.reference_history.append(dict(end_native_step=after['step'],pose=matrix10(row)))
    sample=adapter.measured_task_state()
    # Keep the exact native high-frequency state journal; these are metric samples.
    adapter.wait_source();mount=measured_mount(adapter)
    measured_mpc_mount=mount.copy()
    body=adapter.actual_state.model.body('x5_base_link').id;d=adapter.actual_state.data
    mount=np.eye(4);mount[:3,:3]=d.xmat[body].reshape(3,3);mount[:3,3]=d.xpos[body]
    record=dict(native_start_step=step,native_end_step=after['step'],ticks=25,
        installed_reference_sha256=adapter.active_reference['wire_sha256'],
        segment=getattr(adapter,'mobile_segment','WARMUP'if step<START else'MOBILE_TASK'),
        gripper_command=gripper,
        installed_EEF10=row.tolist(),target_endpoint_EEF10=adapter.active_reference['action_EEF10'][(step-adapter.mobile_installed_step)//25+1],
        actual_TCP=sample['tcp'].tolist(),actual_mount=mount.tolist(),measured_MPC_mount=measured_mpc_mount.tolist(),
        solver_exact_installed_targets=matched,arm_requests=arm_requests,base_commands=base_commands,
        controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',
        safety_valid=sample['safety_valid'],native_grasp=sample['native_grasp'],
        object_pose=sample['object_pose'].tolist(),qpos=sample['qpos'].tolist(),qvel=sample['qvel'].tolist())
    with(adapter.output/'MOBILE_SAMPLES.private.jsonl').open('a')as f:f.write(json.dumps(record)+'\n')
    return sample,mount


@profiled('mobile.install_full_H40')
def install(adapter,action,base):
    adapter.mobile_base_preview=base;adapter.active_reference=adapter.install_reference(action)
    adapter.mobile_installed_step=adapter.observe()['step']
    received=[v for v in adapter.active_reference['receiver_events']if v['kind']=='installed']
    expected=7 if adapter.mobile_mode=='M_FREE'else 14
    if not received or any(len(p)!=expected for event in received for p in event['poses']):
        raise RuntimeError('MOBILE_REAL_RECEIVER_DIMENSION_MISMATCH')


def episode(adapter):
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from go2arx5_vla.world_model.fusion_trajectory_adapter import WORKING_OPEN_M,MECHANICAL_TRAVEL_M
    adapter.prepare_dynamic();mount=arm(adapter);sample=adapter.measured_task_state()
    hold=np.repeat(pose10(sample['tcp'],WORKING_OPEN_M/MECHANICAL_TRAVEL_M)[None],40,axis=0)
    # Reuse the existing measured stable startup duration, not an old state restore.
    for i in range(128):
        if i%7==0:install(adapter,hold,np.repeat(mount[None],41,axis=0))
        sample,mount_now=consume(adapter,hold,sequence=i)
    if adapter.observe()['step']!=START:raise RuntimeError('MOBILE_EXACT_INITIALIZATION_CLOCK')
    initial_tcp=sample['tcp'].copy();initial_mount=mount_now.copy();condition=adapter.spec['mobile_condition']
    adapter.command_event(dict(phase='mobile-M0-motion-start',native_step=START,
        initial_TCP=initial_tcp.tolist(),initial_mount=initial_mount.tolist(),condition=condition))
    for i in range(120):
        action,base=m0_reference(initial_tcp,initial_mount,condition['axis'],condition['eef_mode'],i)
        if i%7==0:install(adapter,action,base)
        sample,mount_now=consume(adapter,action,sequence=128+i)
    if adapter.observe()['step']!=M0_END:raise RuntimeError('MOBILE_EXACT_SIX_SECOND_ENDPOINT')
    # Demonstrate the 14D -> 7D omission transaction before final cancel/HOLD.
    adapter.mobile_mode='M_FREE';adapter.mobile_base_preview=None
    hold=np.repeat(pose10(sample['tcp'],WORKING_OPEN_M/MECHANICAL_TRAVEL_M)[None],40,axis=0)
    install(adapter,hold,None)
    cleared=dict(old_BASE_cleared=True,new_target_dimension=7,world_reference=True,
        receiver_events=adapter.active_reference['receiver_events'])
    adapter.command_event(dict(phase='mobile-base-preference-cleared',**cleared))
    cancelled=adapter.cancel_reference()
    return dict(initial_TCP=initial_tcp.tolist(),initial_mount=initial_mount.tolist(),
        final_TCP=sample['tcp'].tolist(),final_mount=mount_now.tolist(),
        base_clear=cleared,cancel=cancelled)


def prepare_native_output(spec, output):
    if 'multienv_payload_storage' in spec:
        from scripts.sla.core_multienv_payload import create_output

        create_output(spec,output)
    else:
        output.mkdir()


def run_case(spec_path,lock_fd):
    from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
    from go2arx5_vla.world_model.native_replay import packets
    from go2arx5_vla.world_model.init_trace_reader import audit_trace
    spec=read(spec_path);root=Path(spec['registered_root']);group=Path(spec['mobile_group'])
    multitask=spec.get('mobile_experiment_version')=='MFREE_MULTITASK_NATIVE_V6'
    d2=spec.get('scope')=='SLA_D2_MFREE_NATIVE_MUJOCO'
    n1=spec.get('scope')=='SLA_CORE_MULTIENV_V1_N1_ENGINEERING'
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
    if multitask:
        if d2:
            from go2arx5_vla.world_model.multitask_scope_d2 import validate
        else:
            from go2arx5_vla.world_model.multitask_scope_v6 import validate
        from go2arx5_vla.world_model.multitask_adapter_v6 import MultitaskAdapter
        validate(spec)
        TargetSequentialAdapter=MultitaskAdapter
    elif spec.get('mobile_experiment_version')=='MFREE_SHIFT7_SIM_ONLY_V51':
        from go2arx5_vla.world_model.sim_only_scope_v51 import validate
        validate(spec)
    elif spec.get('mobile_experiment_version')=='MFREE_B1_FULL_H40_SIM_ONLY_V6':
        from go2arx5_vla.world_model.target_scope_v6 import validate
        validate(spec)
    elif spec.get('mobile_experiment_version')=='MFREE_SHIFT7_SIM_ONLY_V52':
        from go2arx5_vla.world_model.mobile_scope_v52 import validate
        validate(spec)
    out=root/'native-01'
    prepare_native_output(spec,out)
    adapter=TargetSequentialAdapter(spec,out)
    profile=None
    if d2 or n1:
        from go2arx5_vla.world_model.multitask_profile_d2 import MobileProfile
        profile=MobileProfile(adapter)
    write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),epoch=time.time(),source_sha=spec['source_sha']))
    start=time.monotonic();cpu=time.process_time();failure=None;detail=None
    try:
        with measure(adapter,'native.startup'):
            adapter.start()
        if spec.get('mobile_stage')=='L1':
            if d2:
                from go2arx5_vla.world_model.multitask_scope_d2 import episode as l1_episode
            elif multitask:
                from go2arx5_vla.world_model.multitask_scope_v6 import episode as l1_episode
            elif spec.get('mobile_experiment_version')=='MFREE_B1_FULL_H40_SIM_ONLY_V6':
                from go2arx5_vla.world_model.target_scope_v6 import episode as l1_episode
            elif spec.get('mobile_experiment_version')=='MFREE_SHIFT7_SIM_ONLY_V52':
                from go2arx5_vla.world_model.mobile_scope_v52 import episode as l1_episode
            elif spec.get('mobile_experiment_version')=='MFREE_SHIFT7_SIM_ONLY_V51':
                from go2arx5_vla.world_model.sim_only_mobile_v51 import episode as l1_episode
            elif spec.get('mobile_experiment_version')=='MFREE_C20_FINAL_PLAN_V1':
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
        if adapter.node is not None:
            try:
                adapter.capture_loaded_identity()
                if not adapter.trace_sealed:adapter.seal_trace(adapter.observe()['step'])
            except BaseException:write(out/'FINAL_EVIDENCE_ERROR.private.json',dict(error=traceback.format_exc()))
        try:
            with measure(adapter,'native.shutdown_and_archive'):
                adapter.close()
        except BaseException:
            if not d2:
                raise
            failure = (failure or '') + traceback.format_exc()
    journal=out/'state.bin.journal'
    if multitask:
        from go2arx5_vla.world_model.multitask_state_v6 import packets as multitask_packets
        count=0
        try:
            if journal.exists():
                for _ in multitask_packets(journal):count+=1
        except ValueError:
            failure=(failure or '')+traceback.format_exc()
    else:count=len(packets(journal))if journal.exists()and journal.stat().st_size else 0
    stop=read(out/'MOBILE_SAFETY_STOP.private.json')if(out/'MOBILE_SAFETY_STOP.private.json').exists()else None
    physics=[json.loads(x)for x in(out/'physics.jsonl').read_text().splitlines()]if(out/'physics.jsonl').exists()else[]
    stop_events=[r for r in physics if r['kind']=='stop']
    trace_options={'state_abi':(41,38,20)} if multitask else {}
    try:trace=audit_trace(out/'init.trace',max_native_ticks=spec['max_ticks'],**trace_options)if(out/'init.trace').exists()else None
    except BaseException:trace=dict(error=traceback.format_exc())
    clean=bool(adapter.exit_records)and all(r['exit_code']==0 for r in adapter.exit_records)
    expected=detail.get('terminal_native_step')if detail is not None and spec.get('mobile_stage')=='L1'else M0_END
    passed=detail is not None and not failure and not stop and not stop_events and clean and count==expected
    if multitask and passed and ((trace or {}).get('status')!='RECORDED_TRACE_CHECKS_PASS'
                                or trace['native_step_records']!=count):
        passed=False
        failure='MULTITASK_ACTUAL_NATIVE_JOURNAL_AND_TRACE_MUST_AGREE'
    from go2arx5_vla.world_model.mobile_boundary_metrics_v1 import metrics
    measured_metrics=metrics(out,spec['mobile_condition'])
    if spec.get('mobile_stage')=='L1':
        measured_metrics['task_success']=detail['task_success']if detail is not None else False
        measured_metrics['task_and_required_motion_success']=bool(measured_metrics['task_success']and measured_metrics.get('directional_motion',0)>=.08)
    response_count=len(list((out/'queries').glob('*/RESPONSE.private.json')))if(out/'queries').exists()else 0
    if multitask and spec['sim_only_role']=='STUDENT':
        from go2arx5_vla.world_model.shift7_native_input_v5 import G7_QUERY_DIRECTORY
        responses=(out/G7_QUERY_DIRECTORY).glob('*/model-response/RESPONSE.json')
        response_count=sum(read(path)['new_generations_this_request'] for path in responses)
    value=dict(schema='MOBILE_L1_ACTUAL_NATIVE_RESULT_V1'if spec.get('mobile_stage')=='L1'else 'MOBILE_M0_ACTUAL_NATIVE_RESULT_V1',condition=spec['mobile_condition'],
        completed=passed,technical_failure=bool(failure and not stop and not stop_events),
        unsafe=bool(stop or stop_events),failure=failure,mobile_safety_stop=stop,physics_stops=stop_events,
        actual_native_integrations=count,detail=detail,shutdown_clean=clean,owned_process_exits=adapter.exit_records,
        trace_audit=trace,wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,
        optimizer_updates=0,training_visits=0,model_generations=response_count,physical_requests=1,source_sha=spec['source_sha'],
        WBC_changes=False,raw_fixed_root_IK_is_shadow=True,metrics=measured_metrics)
    if profile is not None:
        report_path=root/'PERFORMANCE.private.json'
        write(report_path,profile.close())
        value['performance']=dict(file=report_path.name,sha256=digest(report_path))
    write(root/'RESULT.private.json',value)
    code=0 if passed else 2 if value['unsafe']else 1
    write(root/'WORKER_EXIT.private.json',dict(exit_code=code,wall_s=value['wall_s']))
    write(root/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
    return code


def run_batch(spec_path):
    from scripts.pi0294.e1.journal import Journal
    spec=read(spec_path);group=Path(spec['registered_root']);lease=registered(spec,spec_path)
    if lease['status']!='ACTIVE'or lease['spec_sha256']!=digest(spec_path):raise PermissionError('FINITE_REGISTERED_MOBILE_BATCH')
    if (group/'SUPERVISOR.private.json').exists():raise PermissionError('MOBILE_BATCH_ALREADY_LAUNCHED_NO_REPLAY')
    write(group/'SUPERVISOR.private.json',dict(pid=os.getpid(),epoch=time.time()))
    journal=Journal(group/'control/journal',group.name)
    outcomes=[];quarantine=set();code=1;failure=None
    try:
        with closing(OwnerLock(spec['mobile_world_lock']))as world:
            for path in spec['cases']:
                if time.time()>=spec['deadline_epoch']-900:break
                case=read(path);axis=case['mobile_condition']['axis'].lstrip('+-')
                if case.get('mobile_stage')=='L1':axis=case['mobile_condition']['family']+':'+case['mobile_condition']['mode']
                if spec.get('mobile_experiment_version')=='MFREE_STUDENT_60S_V1':axis=case['mobile_condition']['family']+':'+case['mobile_condition']['method']
                if spec.get('mobile_experiment_version')in('MOBILE_CONTINUOUS_TEACHER_V2','MOBILE_GRASP_BUDGET_V3'):axis=case['mobile_condition']['mode']
                if axis in quarantine:continue
                if(group/'STOP.json').exists():break
                if case.get('mobile_stage')=='L1'and case['mobile_condition']['method']!='EXPERT':
                    server=Path(read(case['resident_server_spec']['path'])['registered_root']);until=time.monotonic()+900
                    while not(server/'READY.private.json').exists():
                        if(server/'RESULT.private.json').exists():raise RuntimeError('SLA_RESIDENT_FAILED_BEFORE_STUDENT')
                        if time.monotonic()>until:raise TimeoutError('FINITE_OWN_MODEL_READINESS')
                        time.sleep(.2)
                write(group/'PROGRESS.private.json',dict(current=case['mobile_condition'],terminal=len(outcomes),epoch=time.time()))
                argv=[spec['python'],'-B',str(Path(spec['source_root'])/'scripts/wm02/mobile_boundary_bootstrap_v1.py'),'--case',path,
                      '--world-lock-fd',str(world.handle.fileno())]
                log=group/'logs'/(Path(path).parent.name+'.log')
                children=[]
                def launch():
                    with log.open('xb')as f:
                        children.append(subprocess.Popen(argv,cwd=spec['source_root'],env=dict(os.environ,**spec['native_environment']),
                            stdout=f,stderr=subprocess.STDOUT,pass_fds=(world.handle.fileno(),)))
                    return dict(pid=children[0].pid)
                journal.once(Path(path).parent.name,dict(spec_sha256=digest(path)),launch)
                if not children:raise PermissionError('PRIOR_CHILD_IDENTITY_MUST_BE_RECONCILED_NO_REPLAY')
                child=children[0]
                try:result_code=child.wait(timeout=case['wall_limit_s']+180)
                except BaseException:
                    child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise
                root=Path(path).parent
                if not(root/'RESULT.private.json').exists():raise RuntimeError('MOBILE_CHILD_EXIT_WITHOUT_RESULT:'+str(result_code))
                result=read(root/'RESULT.private.json');outcomes.append(dict(root=root.name,result=result,result_sha256=digest(root/'RESULT.private.json')))
                write(group/'OUTCOMES.private.json',outcomes)
                if result['unsafe']:quarantine.add(axis)
                elif result['technical_failure']:raise RuntimeError('MOBILE_TECHNICAL_FAILURE_RETAINED_NO_RETRY')
            code=0
    except BaseException:failure=traceback.format_exc()
    if spec.get('mobile_experiment_version')in('MOBILE_CONTINUOUS_TEACHER_V2','MOBILE_GRASP_BUDGET_V3','MFREE_STUDENT_60S_V1'):
        for path in spec['cases']:
            root=Path(path).parent
            if not(root/'SUPERVISOR.private.json').exists():
                pending=read(root/'control/LEASE.json')
                c=read(path)['mobile_condition'];key=c['family']+':'+c['method']if spec.get('mobile_experiment_version')=='MFREE_STUDENT_60S_V1'else c['mode']
                reason='NATIVE_SAFETY_QUARANTINE'if key in quarantine else'BATCH_STOP_OR_DEADLINE'
                write(root/'control/LEASE.json',dict(pending,status='CLOSED_UNSTARTED',physical_requests=0,native_integrations=0,reason=reason,closed_epoch=time.time()))
    value=dict(schema='MOBILE_L1_FINITE_BATCH_ACTUAL_V1'if spec.get('mobile_stage')=='L1'else 'MOBILE_M0_FINITE_BATCH_ACTUAL_V1',exit_code=code,failure=failure,
        requested=len(spec['cases']),terminal=len(outcomes),outcomes=outcomes,quarantined_axes=sorted(quarantine),
        physical_requests=sum(v['result']['physical_requests']for v in outcomes),
        native_integrations=sum(v['result']['actual_native_integrations']for v in outcomes),
        new_training_updates=0,new_model_generations=sum(v['result']['model_generations']for v in outcomes),automatic_retry=False,goal_complete=False)
    if spec.get('mobile_experiment_version')=='MOBILE_GRASP_BUDGET_V3':
        from go2arx5_vla.world_model.mobile_grasp_budget_v3 import qualification_record
        qualifications=[qualification_record(v['result'])for v in outcomes]
        value.update(qualifications=qualifications,
            qualified_complete_experts=sum(v['qualified']for v in qualifications),
            all_four_qualified=len(qualifications)==4 and all(v['qualified']for v in qualifications),
            automatic_collection_started=False,automatic_training_started=False)
    write(group/'RESULT.private.json',value);write(group/'WORKER_EXIT.private.json',dict(exit_code=code))
    write(group/'control/LEASE.json',dict(lease,status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch');parser.add_argument('--case');parser.add_argument('--world-lock-fd',type=int)
    args=parser.parse_args()
    return run_case(args.case,args.world_lock_fd)if args.case else run_batch(args.batch)


if __name__=='__main__':raise SystemExit(main())
