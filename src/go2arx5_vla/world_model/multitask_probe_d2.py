"""Short stable/moving engineering cards, with original mobile transport."""

from collections import Counter
import hashlib
import time
from pathlib import Path

import numpy as np

from scripts.pi0292.durable import write
from .fusion_paused_observation import _pause, _after_pause
from .mfree_expert_capture_v1 import initialize
from .mobile_boundary_clock_v1 import boundary
from .mobile_continuous_teacher_v2 import ContinuousTeacher
from .multitask_camera_v6 import TwoObjectCameraSession
from .native_collection_performance_v1 import measure


class Probe(ContinuousTeacher):
    def __init__(self, initial, offset):
        super().__init__(initial)
        self.endpoint = np.asarray(initial['tcp'])[:3,3]+offset

    def goal(self, sample):
        return self.endpoint

    def grip_trajectory(self, start):
        return np.ones(40)


def project_stop(adapter, step):
    snapshot = _pause(adapter)
    if snapshot['packet']['step'] != step:
        raise RuntimeError('D2_PROBE_EXPECTED_STOP_EXACT_PAUSE_REQUIRED')
    # Close the normal paused trace before latching an operator marker. The
    # STOP guard stays latched throughout subsequent evidence and teardown.
    adapter.seal_trace(step)
    path = Path(adapter.spec['registered_root'])/'STOP.json'
    write(path, dict(reason='D2_REGISTERED_WORLD_LOCAL_STOP',native_step=step,epoch=time.time()))
    try:
        adapter.guard()
    except TimeoutError as error:
        if str(error) != 'AUTO1_FINITE_ENDPOINT_OR_OPERATOR_STOP':
            raise
    else:
        raise RuntimeError('D2_WORLD_LOCAL_STOP_NOT_OBSERVED')
    time.sleep(.25)
    final = adapter.observe()
    if final['step'] != step or final['time_s'] != snapshot['packet']['time_s']:
        raise RuntimeError('D2_INTEGRATION_AFTER_WORLD_PROJECT_STOP')
    return dict(native_step=step,integrations_after_project_STOP=0,
                wall_epoch=time.time(),terminal_reason='EXPECTED_WORLD_LOCAL_PROJECT_STOP')


def paired_stop_barrier(adapter, step):
    """Prove the peer is paused at the same tick before latching one STOP."""
    spec = adapter.spec
    barrier = spec['d3_stop_barrier']
    slot = spec['d2_world_id']
    group = Path(spec['campaign_root']) / 'stop-barriers' / str(spec['d2_stop_pair'])
    group.mkdir(parents=True, exist_ok=True)
    if _pause(adapter)['packet']['step'] != step or step != 7203:
        raise RuntimeError('D3_STOP_PAIR_BARRIER_REQUIRES_EXACT_7203')
    identity = dict(source_sha=spec['source_sha'], pair=spec['d2_stop_pair'], native_step=step)
    write(group / f'{slot}.ready.json', dict(identity, world=slot))
    until = time.monotonic() + barrier['maximum_wait_s']

    def wait_for(path):
        import json
        while time.monotonic() < until:
            adapter.guard()
            if path.is_file():
                value = json.loads(path.read_bytes())
                if any(value.get(k) != v for k, v in identity.items()):
                    raise RuntimeError('D3_STOP_PAIR_IDENTITY_MISMATCH')
                return value
            time.sleep(.025)
        raise TimeoutError('D3_STOP_PAIR_BARRIER_TIMEOUT_NO_RESUME')

    peer = 'W2' if slot == 'W1' else 'W1'
    if wait_for(group / f'{peer}.ready.json')['world'] != peer:
        raise RuntimeError('D3_STOP_PAIR_PEER_WORLD_CHANGED')
    stopped = barrier['stop_world']
    proof_path = group / 'stop-proved.json'
    if slot == stopped:
        proof = project_stop(adapter, step)
        write(proof_path, dict(identity, **{k:v for k,v in proof.items() if k!='native_step'}, world=slot))
        return proof, None
    proof = wait_for(proof_path)
    if (proof['world'] != stopped or proof['integrations_after_project_STOP'] != 0
            or _pause(adapter)['packet']['step'] != step):
        raise RuntimeError('D3_STOP_PAIR_PROOF_AND_PAUSED_PEER_REQUIRED')
    return None, dict(barrier_tick=step, stopped_world=stopped, stopped_wall_epoch=proof['wall_epoch'],
                      peer_resume_wall_epoch=time.time())


def episode(adapter):
    from scripts.wm02.mobile_boundary_runner_v1 import consume, install

    with measure(adapter, 'episode.warmup'):
        initial, mount, reference, _, _ = initialize(adapter)
    motion = adapter.spec['d2_probe_motion']
    offset = np.array([0.,0.,.008]) if motion=='SMALL_EEF' else np.zeros(3)
    controller = Probe(initial, offset)
    capture = TwoObjectCameraSession(adapter.spec['model_path'])
    frames = []
    motion_counts = Counter()
    sample = initial
    expected_stop = None
    peer_continuation = None
    task_origin = np.asarray(initial['tcp'])[:3,3]
    largest_distance = 0.
    try:
        for query in range(adapter.spec['policy_queries']):
            local_stop = adapter.spec.get('d2_stop_at_task_tick')
            if adapter.spec.get('d3_stop_barrier') and sample['native_step']==7203:
                expected_stop, peer_continuation = paired_stop_barrier(adapter, sample['native_step'])
                if expected_stop:
                    break
            elif local_stop is not None and sample['native_step']-initial['native_step']==local_stop:
                expected_stop = project_stop(adapter,sample['native_step'])
                break
            with measure(adapter, 'capture.native_RGB'):
                current = _pause(adapter)
                rgb, body, _, calibration = capture.render_memory(current['packet'])
                _after_pause(adapter,current)
                if current['packet']['step'] != sample['native_step']:
                    raise RuntimeError('D2_PROBE_RGB_NATIVE_CLOCK')
                frames.append(dict(native_step=sample['native_step'],
                    rgb_time_s=calibration['wrist_rgb']['timestamp_s'],
                    wrist=hashlib.sha256(rgb['wrist_rgb'].tobytes()).hexdigest(),
                    body=hashlib.sha256(body.tobytes()).hexdigest()))
            with measure(adapter,'teacher.H40_proposal'):
                program,_ = controller.decide(sample,reference,adapter.base_pose_hold)
            adapter.mobile_contact_stage='approach'
            adapter.mobile_program=program
            adapter.mobile_program_start_step=sample['native_step']
            for atom in range(10):
                action=program.preview(atom)
                if atom in (0,7):
                    install(adapter,action,None)
                    motion_counts['installs']+=1
                sample,_=consume(adapter,action,sequence=128+10*query+atom)
                with measure(adapter,'reference.accepted_boundary'):
                    reference,_=boundary(adapter)
                adapter.reference_history[-1]['pose']=reference.copy()
                motion_counts['atoms']+=1
                distance = float(np.linalg.norm(np.asarray(sample['tcp'])[:3,3]-task_origin))
                largest_distance = max(largest_distance, distance)
            controller.commit(program,sample['native_step'])
    finally:
        capture.close()
    if peer_continuation is not None:
        peer_continuation['integrations_after_peer_STOP'] = sample['native_step'] - peer_continuation['barrier_tick']
    cancel = adapter.cancel_reference() if expected_stop is None else None
    return dict(schema='SLA_MFREE_D2_NONCONTACT_PROBE_ACTUAL',task_success=False,
                interface_protocol_pass=sample['native_step']==adapter.spec['max_ticks'] or expected_stop is not None,
                native_step_start=initial['native_step'],terminal_native_step=sample['native_step'],
                actual_SLA_queries=0,expert_queries=0,model_generations=0,
                consumed_task_atoms=motion_counts['atoms'],install_ACKs=motion_counts['installs'],
                RGB_pairs=frames,protocol_probe=motion,maximum_actual_TCP_displacement_m=largest_distance,
                initial_mount=mount.tolist(),controller_selected_reference='UNKNOWN',
                expected_project_STOP=expected_stop,peer_project_STOP_continuation=peer_continuation,cancel=cancel)
