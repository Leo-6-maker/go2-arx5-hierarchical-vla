"""Real MuJoCo episode caller, using the original frozen mobile consumer.

No hardware backend or fallback Teacher is provided to the student. Model
dispatch is independent of native install/consumption and has a finite wall
deadline while the existing simulator pause ABI stays unchanged.
"""
from pathlib import Path
import time

import numpy as np

from scripts.pi0292.durable import write
from .hybrid_nonphysical_v3 import Receiver, read
from .storage import digest
from .sim_only_scope_v51 import validate, STARTUP_TICKS


def checked_response(query, response):
    if (response.get('request_id') != query['request_id']
            or response.get('bindings') != query['bindings']
            or response.get('status') != 'GENERATED_NOT_INSTALLED'
            or response.get('new_generations_this_request') != 1
            or digest(response['action']['path']) != response['action']['sha256']):
        raise ValueError('REAL_SHIFT7_RESPONSE_IDENTITY_STATUS_OR_BYTES')
    return response


def request_policy(adapter, prepared, query_index):
    from go2arx5_vla.sla_adapter.multitask_contract_v6 import NATIVE_VERSION as G7_VERSION
    from .shift7_native_input_v5 import verify_query_return
    from .fusion_paused_observation import _pause, _same_pause
    spec = adapter.spec
    server_pin = spec['resident_server_spec']
    if digest(server_pin['path']) != server_pin['sha256']:
        raise ValueError('SEALED_SHIFT7_SERVER_SPEC_CHANGED')
    server = read(server_pin['path']); root = Path(server['registered_root'])
    version=spec.get('mobile_experiment_version')
    multi=version in ('MFREE_SHIFT7_SIM_ONLY_V52','MFREE_B1_FULL_H40_SIM_ONLY_V6',G7_VERSION)
    package=server['packages'][spec['arm']] if multi else server['SHIFT7_package']
    bindings=server['request_bindings_by_arm'][spec['arm']] if multi else server['request_bindings']
    expected=spec['G7_package'] if version==G7_VERSION else spec['SHIFT7_package']
    if (package['sha256'] != expected['sha256']
            or read(root/'control/LEASE.json')['status'] != 'ACTIVE'):
        raise RuntimeError('OWN_SINGLE_SHIFT7_SERVICE_NOT_ACTIVE')
    meta = prepared['metadata']; folder = Path(meta['input']['path']).parent
    control = Receiver(folder/'inference-control',
        Path(spec['registered_root']).name+'-q%03d'%query_index, meta['native_step'],
        prepared['previous_reference'],
        dict(kind='ACTUAL_CONFIRMED_NATIVE_COMMAND',native_state_sha256=meta['native_state_sha256']),
        bindings, operator_stop=Path(spec['registered_root'])/'STOP.json',
        observation=prepared['observation'])
    response_root = folder/'model-response'; response_root.mkdir()
    until = meta['observation_available_monotonic'] + spec['model_service_timeout_s']
    remaining = until-time.monotonic()
    if remaining <= 0:
        control.cancel('OBSERVATION_ALREADY_TOO_OLD')
        raise TimeoutError('BOUNDED_SIM_PAUSED_MODEL_SERVICE')
    noise = (int(spec['mobile_condition']['seed']) + query_index) % 2**31
    request = control.request(meta['input'],spec['language'],noise,
                              timeout_s=remaining,clock_mode='SIM_PAUSED_ONLY')
    write(folder/'MODEL_REQUEST.private.json',request)
    message = dict(request=request,live_path=str(control.root/'live.json'),
                   response_root=str(response_root),branch_barriers=False)
    try:
        route = None
        if server.get('d2_world_queue') is not None:
            from .multitask_queue_d2 import WorldQueue

            queue = WorldQueue(server['ipc_root'],server['d2_world_queue'])
            route = queue.publish(spec['d2_world_id'],message)
        else:
            write(Path(server['ipc_root'])/'REQUEST.json',message)
        last_pause_check = 0.
        while True:
            adapter.guard()
            wall = time.monotonic()
            if wall >= until:
                raise TimeoutError('BOUNDED_SIM_PAUSED_MODEL_SERVICE')
            try:
                response = read(response_root/'RESPONSE.json')
                break
            except FileNotFoundError:
                if (root/'RESULT.private.json').exists():
                    raise RuntimeError('SHIFT7_SERVICE_TERMINAL_NO_ACTION_NO_REPLAY')
                if wall-last_pause_check >= .2:
                    _same_pause(prepared['pause_snapshot'],_pause(adapter))
                    last_pause_check = wall
                adapter.spin(.01)
        checked_response(request,response)
        if route is not None and response.get('d2_route')!=route:
            raise ValueError('D2_QUEUE_WRONG_WORLD_EPISODE_CAPTURE_OR_PACKAGE_RESPONSE')
        pause = verify_query_return(adapter,prepared)
        write(folder/'MODEL_RETURN.private.json',dict(response=response,
            pause=pause,wall_end_monotonic=time.monotonic(),native_install_ACK=False))
        return response
    finally:
        # This is the inference-control file receiver only, never a native
        # cancellation or a mock-emission claim. Native history lives in the
        # actual adapter and advances exclusively in original consume().
        control.cancel('MODEL_REQUEST_FINISHED_OR_FAILED_NO_MOCK_CONSUMPTION')


def interface_episode(adapter):
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from scripts.wm02.mobile_boundary_runner_v1 import install, consume
    from .mfree_expert_capture_v1 import initialize
    from .task_learning_grasp_v1 import FiftyMsProgram
    from .fusion_paused_observation import _pause, _after_pause
    initial,mount,reference,_,_=initialize(adapter)
    before=_pause(adapter)
    wall=time.monotonic();time.sleep(.1)
    after=_after_pause(adapter,before)
    nominal=np.repeat(pose10(initial['tcp'],1.)[None],40,axis=0)
    program=FiftyMsProgram(nominal,initial['tcp'],reference,adapter.base_pose_hold)
    adapter.mobile_program=program;adapter.mobile_program_start_step=initial['native_step']
    installs=[];atoms=[]
    for atom in range(10):
        action=program.preview(atom)
        if atom in(0,7):
            install(adapter,action,None)
            installs.append(dict(atom=atom,source_native_s=adapter.active_reference['source_mpc_s'],
                wire_sha256=adapter.active_reference['wire_sha256'],receiver_events=adapter.active_reference['receiver_events']))
        sample,_=consume(adapter,action,sequence=128+atom)
        atoms.append(sample['native_step'])
    cancel=adapter.cancel_reference()
    final=adapter.observe()
    return dict(schema='MFREE_SIM_ONLY_INTERFACE_ACTUAL_V51',task_success=False,
        interface_protocol_pass=len(atoms)==10 and final['step']==STARTUP_TICKS+250,
        actual_SLA_queries=0,expert_queries=0,consumed_task_atoms=len(atoms),
        terminal_native_step=final['step'],terminal_reason='NONCONTACT_INTERFACE_COMPLETE',
        original_full_H40_compiled=True,M_FREE_reference_dimension=7,BASE_reference_present=False,
        installs=installs,native_atom_end_steps=atoms,cancel=cancel,
        pause_wall_s=time.monotonic()-wall,controlled_pause_wait_s=.1,
        frozen_pause_fields_before=before['pause'],frozen_pause_fields_after=after['pause'],
        initial_mount=mount.tolist(),initial_TCP=initial['tcp'].tolist(),
        grasp_or_task_evaluation=False,dynamic_STOP_latency_s=None)


def student_episode(adapter):
    from .mfree_expert_capture_v1 import initialize
    from .mfree_capture_v1 import LiveCamera
    from .shift7_native_input_v5 import capture,verify_query_return
    from .sla_dense_policy_v5 import compile_proposal
    from .mobile_boundary_clock_v1 import boundary
    from .mobile_expert_episode_v2 import funnel
    from scripts.wm02.mobile_boundary_runner_v1 import install,consume
    initial,mount,reference,_,frame=initialize(adapter)
    adapter.shift7_fixed_task_frame=frame.copy()
    session=LiveCamera(adapter.spec['model_path'])
    samples=[];queries=0;rejection=None;grasp_since=None;longest=0.;success=False;ever=False
    try:
        for query in range(120):
            sample=adapter.measured_task_state()
            prepared=capture(adapter,sample,session,frame,reference,query)
            response=request_policy(adapter,prepared,query);queries+=1
            state=prepared['observation'].values_for(prepared['metadata']['input'])['state_eef10']
            try:
                program,evidence=compile_proposal(response['action']['path'],response['action']['sha256'],
                    source_from_target=adapter.spec['policy_from_world'],query_state10=state,
                    actual_TCP=sample['tcp'],consumed_reference_TCP=reference,base_hold=adapter.base_pose_hold)
            except ValueError as error:
                rejection=dict(query=query,reason=str(error),full_H40_checked=True,installed=False)
                write(Path(prepared['metadata']['input']['path']).parent/'QUERY_ACTUAL.private.json',
                    dict(query=query,action=response['action'],full_H40_compiled=False,
                         install_ACKs=[],confirmed_atoms=0,actual_model_generations=1,
                         native_start_step=sample['native_step'],rejection=rejection))
                break
            verify_query_return(adapter,prepared)
            adapter.mobile_program=program;adapter.mobile_program_start_step=sample['native_step']
            adapter.command_event(dict(phase='sim-only-SHIFT7-full40-compile',query=query,**evidence))
            confirmed=[];acks=[];start_step=sample['native_step']
            try:
                for atom in range(prepared['metadata']['available_prefix_atoms']):
                    adapter.guard()
                    action=program.preview(atom)
                    if atom in(0,7):
                        if atom==0:verify_query_return(adapter,prepared)
                        install(adapter,action,None)
                        acks.append(dict(atom=atom,wire_sha256=adapter.active_reference['wire_sha256'],
                            source_native_s=adapter.active_reference['source_mpc_s']))
                    sample,_=consume(adapter,action,sequence=128+query*10+atom)
                    samples.append(sample);confirmed.append(sample['native_step'])
                    reference,witness=boundary(adapter);adapter.reference_history[-1]['pose']=reference.copy()
                    ever|=bool(sample['native_grasp'])
                    lift=float(sample['object_pose'][2,3]-initial['object_pose'][2,3])
                    if sample['native_grasp']and lift>=.020:
                        if grasp_since is None:grasp_since=sample['time_s']
                        longest=max(longest,sample['time_s']-grasp_since)
                        if longest>=1.:success=True
                    else:grasp_since=None
                    funnel(adapter,sample,'STUDENT_NO_EXPERT_STAGE',query,None)
            finally:
                try:
                    terminal=adapter.observe()['step'];partial_ticks=(terminal-start_step)%25
                except Exception:terminal=partial_ticks=None
                folder=Path(prepared['metadata']['input']['path']).parent
                write(folder/'QUERY_ACTUAL.private.json',dict(query=query,
                    action=response['action'],full_H40_compiled=True,install_ACKs=acks,
                    confirmed_atom_end_steps=confirmed,confirmed_atoms=len(confirmed),
                    full_P10=len(confirmed)==10,terminal_native_step=terminal,
                    partial_native_ticks=partial_ticks,actual_model_generations=1,
                    mock_consumption=False,controller_selected_reference='UNKNOWN'))
            if success:break
    finally:session.close()
    reason='TASK_SUCCESS'if success else'ORIGINAL_FULL_H40_POLICY_REJECTED'if rejection else'TASK_60S_BUDGET_EXHAUSTED'
    funnel(adapter,sample,'STUDENT_NO_EXPERT_STAGE',queries,None,terminal_reason=reason)
    cancel=adapter.cancel_reference()
    return dict(schema='MFREE_SIM_ONLY_SHIFT7_ACTUAL_V51',task_success=success,
        actual_SLA_queries=queries,expert_queries=0,consumed_task_atoms=len(samples),
        valid_policy_rejection=rejection,terminal_native_step=adapter.observe()['step'],
        terminal_reason=reason,ever_grasp=ever,grasp_atoms=sum(s['native_grasp']for s in samples),
        longest_grasp_lift_hold_s=longest,
        maximum_lift_m=max((float(s['object_pose'][2,3]-initial['object_pose'][2,3])for s in samples),default=0.),
        initialization_ticks=STARTUP_TICKS,task_ticks=adapter.observe()['step']-STARTUP_TICKS,
        task_budget_s=60.,total_episode_tick_cap=35703,initial_mount=mount.tolist(),
        initial_TCP=initial['tcp'].tolist(),fixed_frame=frame.tolist(),cancel=cancel,
        teacher_phase_input=False,teacher_takeover=False,BASE_reference_present=False,
        controller_selected_reference='UNKNOWN',strict_wall_realtime_qualified=False,
        task_predicate='ACTUAL_NATIVE_GRASP_AND_LIFT_20MM_CONTINUOUS_ONE_SECOND')


def episode(adapter):
    validate(adapter.spec)
    role=adapter.spec['sim_only_role']
    if role=='INTERFACE':return interface_episode(adapter)
    if role=='EXPERT':
        from .mobile_expert_episode_v3 import episode as expert
        return expert(adapter)
    return student_episode(adapter)
