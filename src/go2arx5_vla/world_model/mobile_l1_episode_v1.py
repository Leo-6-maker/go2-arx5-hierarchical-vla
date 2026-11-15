"""L1 grasp episodes through the original raw H40 transport, frozen WBC."""
import json
from pathlib import Path
import time
import numpy as np
from .mobile_wbc_reference_v1 import START,m0_reference
from .storage import digest
from scripts.pi0292.durable import write


def read(path):return json.loads(Path(path).read_bytes())


def predict(adapter,sample,session,frame,query):
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from scripts.pi0294.e1.artifacts import atomic_npz
    from .fusion_paused_observation import _pause,_after_pause
    before=_pause(adapter);images,body,_,cal=session.render_memory(before['packet'])
    root=adapter.output/'queries'/f'{query:02d}';root.mkdir(parents=True)
    state=pose10(np.asarray(adapter.spec['policy_from_world'])@sample['tcp'],sample['gripper_fraction'])
    path=root/'INPUT.private.npz'
    atomic_npz(path,state_eef10=state,body_rgb=body,wrist_rgb=images['wrist_rgb'],T_policy_from_mount=frame)
    # The server serializes all predeclared episodes, so no per-episode reload.
    server_spec=adapter.spec['resident_server_spec'];server=Path(read(server_spec['path'])['registered_root'])
    ready=read(server/'READY.private.json');index=len(list((server/'requests').glob('*/REQUEST.private.json')))
    folder=server/'requests'/f'{index:04d}';folder.mkdir()
    request=dict(sequence=index,method=adapter.spec['mobile_condition']['method'],native_spec=str(Path(adapter.spec['registered_root'])/'EXECUTION_SPEC.json'),
        native_spec_sha256=digest(Path(adapter.spec['registered_root'])/'EXECUTION_SPEC.json'),
        input=str(path),input_sha256=digest(path),language=adapter.spec['language'],
        seed=2026111500+query,task_frame=frame.tolist(),native_step=sample['native_step'],
        provenance='CURRENT_NATIVE_BODY_AND_WRIST_RGB_AND_MEASURED_STATE_ONLY')
    write(folder/'REQUEST.private.json',request);deadline=time.monotonic()+600
    while not(folder/'RESULT.private.json').exists():
        if(server/'RESULT.private.json').exists():raise RuntimeError('RESIDENT_SLA_SERVER_TERMINAL_BEFORE_RESPONSE')
        adapter.spin(.05)
        if time.monotonic()>deadline:raise TimeoutError('FINITE_SLA_REQUEST')
    response=read(folder/'RESULT.private.json')
    if response['request_sha256']!=digest(folder/'REQUEST.private.json')or digest(response['action'])!=response['action_sha256']:
        raise PermissionError('EXACT_ORIGINAL_SLA_RESPONSE')
    _after_pause(adapter,before)
    write(root/'RESPONSE.private.json',response)
    return response,state


def episode(adapter):
    from scripts.wm02.mobile_boundary_runner_v1 import arm,install,consume,measured_mount
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from .fusion_trajectory_adapter import WORKING_OPEN_M,MECHANICAL_TRAVEL_M
    from .task_learning_grasp_v1 import Teacher
    from .native_body_camera_capture_v1 import BodyBaseCameraSession
    from .sla_dense_policy_v5 import compile_proposal
    adapter.prepare_dynamic();mount=arm(adapter);sample=adapter.measured_task_state()
    hold=np.repeat(pose10(sample['tcp'],WORKING_OPEN_M/MECHANICAL_TRAVEL_M)[None],40,axis=0)
    for i in range(128):
        if i%7==0:install(adapter,hold,np.repeat(mount[None],41,axis=0))
        sample,_=consume(adapter,hold,sequence=i)
    initial=sample;initial_mount=measured_mount(adapter);reference=adapter.reference_history[-1]['pose']
    if adapter.observe()['step']!=START:raise RuntimeError('L1_EXACT_POST_HOLD_START')
    # Task frame is actual initial mount, fixed for this entire episode.
    body=adapter.actual_state.model.body('x5_base_link').id;d=adapter.actual_state.data
    transform=np.eye(4);transform[:3,:3]=d.xmat[body].reshape(3,3);transform[:3,3]=d.xpos[body]
    frame=np.asarray(adapter.spec['policy_from_world'])@transform
    write(adapter.output/'SLA_TASK_FRAME.private.json',dict(T_policy_from_mount=frame.tolist(),native_step=START,
        fixed_for_episode=True,future_base_truth_used=False,object_truth_used=False))
    condition=adapter.spec['mobile_condition'];adapter.mobile_mode=condition['mode']
    method=condition['method'];teacher=Teacher(initial)if method=='EXPERT'else None
    session=BodyBaseCameraSession(adapter.spec['model_path'])if teacher is None else None
    samples=[];queries=0;valid_rejection=None;grasp_since=None;success=False
    try:
        for query in range(adapter.spec['policy_queries']):
            sample=adapter.measured_task_state()
            if teacher is not None:
                program,evidence=teacher.decide(sample,reference,adapter.base_pose_hold)
                adapter.mobile_contact_stage={'DESCEND':'approach','CLOSE':'close','LIFT':'lift','HOLD':'hold'}.get(teacher.stage,'approach')
                adapter.command_event(dict(phase='mobile-expert-query',query=query,**evidence))
            else:
                response,state=predict(adapter,sample,session,frame,query);queries+=1
                try:
                    program,evidence=compile_proposal(response['action'],response['action_sha256'],
                        source_from_target=adapter.spec['policy_from_world'],query_state10=state,
                        actual_TCP=sample['tcp'],consumed_reference_TCP=reference,base_hold=adapter.base_pose_hold)
                except ValueError as error:
                    valid_rejection=dict(query=query,reason=str(error),full_H40_checked=True,installed=False);break
                adapter.command_event(dict(phase='mobile-student-query',query=query,**evidence))
            for atom in range(10):
                action=program.preview(atom)
                _,base=m0_reference(initial['tcp'],initial_mount,'+x','WORLD_HOLD',query*10+atom)
                if atom in(0,7):install(adapter,action,base if adapter.mobile_mode=='M_ASSIST'else None)
                sample,_=consume(adapter,action,sequence=128+query*10+atom);samples.append(sample)
                reference=adapter.reference_history[-1]['pose']
                lift=float(sample['object_pose'][2,3]-initial['object_pose'][2,3])
                if sample['native_grasp']and lift>=.020:
                    if grasp_since is None:grasp_since=sample['time_s']
                    if sample['time_s']-grasp_since>=1.:success=True
                else:grasp_since=None
            if success:break
    finally:
        if session is not None:session.close()
    cancel=adapter.cancel_reference();end=adapter.observe()['step']
    return dict(schema='MOBILE_L1_ACTUAL_GRASP_ENDPOINT_V1',condition=condition,task_success=success,
        valid_policy_rejection=valid_rejection,actual_SLA_queries=queries,consumed_task_atoms=len(samples),
        terminal_native_step=end,maximum_lift_m=max((float(s['object_pose'][2,3]-initial['object_pose'][2,3])for s in samples),default=0.),
        grasp_atoms=sum(s['native_grasp']for s in samples),cancel=cancel,initial_mount=initial_mount.tolist(),
        initial_TCP=initial['tcp'].tolist(),fixed_frame=frame.tolist(),expert_takeover=False,
        task_predicate='ACTUAL_NATIVE_GRASP_AND_LIFT_20MM_CONTINUOUS_ONE_SECOND')
