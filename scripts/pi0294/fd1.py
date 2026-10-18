"""Bounded, read-side FD1 diagnostics. Import never starts physics or a model."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

AUTHORITY = '6ab80b40e2036c2c29a8d2eb648987f2af7447a2'
BASELINE = '658b432fa1ee37182945e84b7d4ec4d830bee67e'
CANDIDATE = '94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'
PLAN = '6e5a45e121aa493e1de32977c00d3f20ecc3c1f9cb7a6b9698438cbb16028e97'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o400)


def update(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    put(temp, value)
    os.replace(temp, path)


def seal_or_verify(path, value):
    """Reuse an already sealed identical plan; never overwrite a changed plan."""
    path=Path(path)
    if path.exists():
        if path.is_symlink() or path.stat().st_mode & 0o222 or read(path)!=value:
            raise ValueError('FD1_EXISTING_SEALED_PLAN_IDENTITY_CHANGED')
    else:
        put(path,value)


def validate(root):
    root = Path(root)
    m = read(root / 'RUN_MANIFEST.json')
    bpath = root / 'OUTPUT_BINDING.private.json'
    ipath = root / 'INPUT_BINDINGS.private.json'
    b, inputs = read(bpath), read(ipath)
    if (m['schema'] != 'PI0294_FD1_RUN_MANIFEST_V1' or m['authority']['protocol_ref'] != AUTHORITY
            or m['scope'] != 'FD1_CPU_DIAG_NOT_N1_REOPEN' or m['run_uuid'] != root.name
            or m['owner'] != 'EXISTING_LOCAL_PC' or m['execution']['profile'] != '4080-cpu-data'
            or m['execution']['host'] != os.uname().nodename or m['execution']['uid'] != os.getuid()
            or sha(bpath) != m['execution']['output_binding_sha256']
            or sha(ipath) != m['execution']['input_binding_sha256']):
        raise PermissionError('FD1_EXACT_MANIFEST_OWNER_HOST_INPUT_OUTPUT')
    if (root.is_symlink() or root.resolve() != Path(b['run_root']).resolve()
            or root.parent.resolve() != Path(b['registered_parent']).resolve()
            or b['uid'] != os.getuid() or b['role'] != 'PI0294_FD1_CPU_DIAGNOSTIC'):
        raise PermissionError('FD1_REGISTERED_OUTPUT_ONLY')
    lease = read(b['lease'])
    limits = m['limits']
    expected_caps = dict(Core_CPU_s=172800,local_CPU_s=28800,shadow_ticks=500000,engine_starts=64,
                         scratch_bytes=12*1024**3,network_bytes=8*1024**3,max_concurrent_cores=4,
                         live_episodes=0,neural_queries=0,optimizer_updates=0,GPU_PBS=0)
    if (any(limits.get(k) != v for k,v in expected_caps.items())
            or limits['hard_close_epoch'] != limits['T0_epoch']+48*3600
            or limits['stop_new_shadow_epoch'] != limits['T0_epoch']+40*3600):
        raise PermissionError('FD1_EXACT_FINITE_AUTHORITY_CAPS')
    if (lease['status'] != 'ACTIVE' or lease['run_uuid'] != root.name
            or lease['authority_release'] != AUTHORITY or lease['deadline_epoch'] != limits['hard_close_epoch']
            or not limits['T0_epoch'] <= time.time() < limits['hard_close_epoch']):
        raise PermissionError('FD1_FRESH_LEASE_AND_ORIGINAL_CLOCK')
    reg = read(b['registry'])
    entries = [x for x in reg['entries'] if x.get('role') == b['role']]
    if len(entries) != 1 or Path(entries[0]['path']).resolve() != root.resolve():
        raise PermissionError('FD1_UNIQUE_REGISTERED_OWNER')
    seal = root / m['code']['source_seal_relative']
    code = root / m['code']['capsule_relative']
    if sha(seal) != m['code']['source_seal_sha256']:
        raise ValueError('FD1_SOURCE_SEAL')
    sealed = read(seal)
    if sealed['source_sha'] != m['code']['vla_commit']:
        raise ValueError('FD1_SOURCE_SHA')
    if any(sha(code / n) != h for n,h in sealed['files'].items()):
        raise ValueError('FD1_IMMUTABLE_SOURCE_MEMBER')
    if Path(__file__).resolve() != (code / 'scripts/pi0294/fd1.py').resolve():
        raise ValueError('FD1_REAL_IMPORTED_CAPSULE')
    infra = Path(b['registry']).parent
    if (sha(infra / 'enter.sh') != m['execution']['entry_sha256']
            or sha(infra / 'doctor.py') != m['execution']['doctor_sha256']):
        raise ValueError('FD1_DEPLOYED_ENTRY_IDENTITY')
    old = Path(inputs['original_source_root'])
    if (sha(old / 'RUN_MANIFEST.json') != inputs['original_source_manifest_sha256']
            or read(read(old / 'OUTPUT_BINDING.private.json')['lease'])['status'] != 'CLOSED'
            or sha(read(old / 'OUTPUT_BINDING.private.json')['lease']) != inputs['source_lease_sha256']
            or sha(old.parent.parent / 'plans/SOURCE_CONFIRM_FINAL_EFFECT_CLOSURE.private.json') != inputs['original_closure_sha256']):
        raise ValueError('FD1_CLOSED_ORIGINAL_INPUTS_CHANGED')
    original = read(old / 'RUN_MANIFEST.json')
    old_code = old / original['code']['capsule_relative']
    for name,h in m['code']['frozen_numerical_files'].items():
        if sha(code / name) != h or sha(old_code / name) != h:
            raise ValueError('FD1_FROZEN_NUMERICAL_BYTE_IDENTITY')
    baseline = root / 'plans/BASELINE_RESULT.json'
    if sha(baseline) != m['data']['manifest_sha256']:
        raise ValueError('FD1_FIXED_BASELINE_REPORT')
    v = read(baseline)
    if v['candidate_manifest_sha256'] != CANDIDATE or v['plan_sha256'] != PLAN:
        raise ValueError('FD1_FIXED_MODEL_PLAN')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'PBS_JOBID' in os.environ:
        raise PermissionError('FD1_NO_GPU_NO_PBS')
    return m, inputs, old, v


class Meter:
    def __init__(self, root, operation):
        self.root = Path(root)
        self.path = self.root / 'RESOURCE_LEDGER.json'
        self.m, self.inputs, self.source, self.baseline = validate(root)
        self.operation = operation
        self.started = time.process_time()
        self.ticks = self.starts = 0
        self.ledger = read(self.path)
        if self.ledger['status'] != 'ACTIVE' or self.ledger['unknown_effects']:
            raise PermissionError('FD1_NO_UNSETTLED_EFFECTS')
        rows = [x for x in self.ledger['operations'] if x['operation'] == operation]
        if len(rows) != 1 or rows[0]['status'] != 'RESERVED':
            raise PermissionError('FD1_SINGLE_DISPATCH_RESERVED_OPERATION')
        self.reservation = rows[0]
        self.cap = self.reservation['CPU_reserved_s']
        self.deadline = min(self.reservation['deadline_epoch'], self.m['limits']['hard_close_epoch'])
        resource.setrlimit(resource.RLIMIT_CPU, (int(self.cap), int(self.cap)))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    def guard(self):
        if time.time() >= self.deadline or time.process_time() - self.started >= self.cap:
            raise TimeoutError('FD1_CPU_OR_WALL_BOUND')
        if (self.operation.startswith('shadow') and time.time() >= self.m['limits']['stop_new_shadow_epoch']):
            raise TimeoutError('FD1_SHADOW_CLOCK')
        if self.ledger['shadow_ticks'] + self.ticks > self.m['limits']['shadow_ticks']:
            raise RuntimeError('FD1_SHADOW_TICK_BOUND')

    def start(self):
        self.guard()
        if self.ledger['engine_starts'] + self.starts >= self.m['limits']['engine_starts']:
            raise RuntimeError('FD1_ENGINE_START_BOUND')
        self.starts += 1

    @contextmanager
    def steps(self, field, amount, identifier):
        if field != 'shadow_integrations' or self.ledger['shadow_ticks'] + self.ticks + amount > self.m['limits']['shadow_ticks']:
            raise PermissionError('FD1_ONLY_BOUNDED_SHADOW_INTEGRATIONS')
        meter = dict(actual=0)
        try:
            yield meter
        finally:
            if not 0 <= meter['actual'] <= amount:
                raise RuntimeError('FD1_UNKNOWN_TICK_EFFECT')
            self.ticks += meter['actual']

    def finish(self, status, error=None):
        row = self.reservation
        cpu = time.process_time() - self.started
        row.update(status=status,CPU_reserved_s=0,actual_worker_CPU_s=cpu,shadow_ticks=self.ticks,
                   engine_starts=self.starts,error=error,finished_epoch=time.time())
        self.ledger['CPU_reserved_s'] = 0
        self.ledger['Core_CPU_s'] += cpu
        self.ledger['shadow_ticks'] += self.ticks
        self.ledger['engine_starts'] += self.starts
        update(self.path, self.ledger)


def first_time(times, flags):
    indices = np.flatnonzero(flags)
    return float(times[indices[0]]) if len(indices) else None


def actual_phase(task, contact, moved_1mm, moved_10cm, grasp, released, separated, goal):
    if task == 'T2':
        if moved_1mm:
            if separated and goal:
                return 'POST_PUSH_SEPARATED_IN_ZONE'
            if separated:
                return 'POST_CONTACT_DISPLACED_NOW_SEPARATED'
            return 'OBJECT_TRANSPORT_CONTACT_ACTIVE' if contact else 'OBJECT_DISPLACED_CONTACT_HISTORY_UNRESOLVED'
        return 'CONTACT_WITHOUT_1MM_DISPLACEMENT' if contact else 'NO_RECORDED_OBJECT_CONTACT_OR_MOTION'
    if released and separated and moved_10cm:
        return 'POST_TRANSPORT_RELEASED_SEPARATED'
    if grasp:
        return 'GRASP_OR_CARRY_NOT_RELEASED'
    if moved_1mm:
        return 'OBJECT_DISPLACED_NO_CURRENT_GRASP'
    return 'CONTACT_WITHOUT_TRANSPORT' if contact else 'NO_RECORDED_OBJECT_CONTACT_OR_MOTION'


def event_summary(data, samples, card):
    from scripts.pi0294.score import score
    from scripts.pi0291_arx5_data.config import PARAMETERS as P
    t = data['time_s']
    xyz = data['object_pose'][:, :3, 3]
    tcp = data['tcp_world'][:, :3, 3]
    micro, mt = data['contact_audit'], data['contact_time_s']
    disp = np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1)
    distance = np.linalg.norm(xyz[:, :2] - np.asarray(card['goal_xy']), axis=1)
    corners = np.asarray([s['privileged']['object_points'] for s in samples])
    in_zone = (np.abs(corners[:, :, :2] - np.asarray(card['goal_xy'])) <= .07).all(axis=(1,2))
    if len(micro):
        contact = micro[:, 0] > 0
        grasp = micro[:, 1] > 0
        lifted = micro[:, 3] - xyz[0,2]
        motion = micro[:, 2]
    else:
        contact = grasp = np.array([], dtype=bool)
        lifted = motion = np.array([])
    score_value = score(data,card,complete=len(t)==641)
    any_contact = bool(contact.any())
    current_contact = bool(data['robot_object_contact'][-1])
    current_grasp = bool(data['native_grasp'][-1])
    now_separated = not current_contact
    phase = actual_phase(card['task'],any_contact,bool(len(motion) and motion.max()>=.001),
        bool(disp.max()>=.10),current_grasp,not current_grasp,now_separated,bool(in_zone[-1]))
    first_contact = first_time(mt,contact)
    last_contact = float(mt[np.flatnonzero(contact)[-1]]) if any_contact else None
    return dict(actual_phase=phase,diagnostic_phase_not_teacher_control=True,
        diagnostic_early_thresholds_m=[.001,.01],original_task_move_threshold_m=.10,original_T2_no_lift_m=.005,
        first_robot_object_contact_s=first_contact,last_robot_object_contact_s=last_contact,
        first_1mm_displacement_s=first_time(mt,motion>=.001),first_10mm_displacement_s=first_time(mt,motion>=.01),
        first_10cm_displacement_s=first_time(mt,motion>=.10),
        first_T2_5mm_lift_violation_s=first_time(mt,lifted>.005+1e-8) if card['task']=='T2' else None,
        first_bilateral_grasp_s=first_time(mt,grasp),first_zone_entry_20Hz_s=first_time(t,in_zone),
        first_TCP_clear_20Hz_s=first_time(t,np.linalg.norm(tcp-xyz,axis=1)>=.08),
        current_robot_contact=current_contact,current_native_grasp=current_grasp,current_object_in_zone=bool(in_zone[-1]),
        highrate_contact_ticks=int(contact.sum()),highrate_bilateral_grasp_ticks=int(grasp.sum()),
        highrate_maximum_displacement_m=float(motion.max()) if len(motion) else 0.,
        highrate_maximum_lift_m=float(lifted.max()) if len(lifted) else 0.,
        maximum_displacement_20Hz_m=float(disp.max()),final_displacement_m=float(disp[-1]),
        minimum_goal_center_error_20Hz_m=float(distance.min()),final_goal_center_error_m=float(distance[-1]),
        final_gripper_gap_m=float(samples[-1]['execution']['actual_gripper_gap_m']),
        final_TCP_object_distance_m=float(np.linalg.norm(tcp[-1]-xyz[-1])),
        goal_center=list(card['goal_xy']),source_center=list(card['source_xy']),
        score=score_value,phase_evidence_resolution='500HZ_CONTACT_DISPLACEMENT_LIFT_AND_20HZ_POSE_ENDPOINT',
        first_release_event_definition='NO_NEW_SUSTAINED_RELEASE_THRESHOLD_OR_SCORING_CHANGE')


def audit_attempt(meter, record):
    from scripts.pi0292.relay_protocol import unseal
    identifier = record['adjudication']['attempt_id']
    folder = meter.source / 'episodes' / identifier
    names = ['CARD.json','RESULT.json','MP1_ADJUDICATION.json','FILE_MANIFEST.json','SCORE.json',
             'S1_PREFLIGHTS.json','S1_RESULT.json','physics.npz','highrate.npz','samples.json','references.json','scene.xml.json']
    hashes = {name:sha(folder/name) for name in names}
    raw = read(folder/'RESULT.json')
    decision = read(folder/'MP1_ADJUDICATION.json')
    if decision != record['adjudication'] or raw['source_sha'] != record['actual_execution_source_sha']:
        raise ValueError('FD1_ORIGINAL_ADJUDICATION_AND_RUNTIME_IDENTITY')
    if sha(folder/'RESULT.json') != decision['result_sha256']:
        raise ValueError('FD1_ORIGINAL_RESULT_HASH')
    manifest = read(folder/'FILE_MANIFEST.json')
    for f in manifest:
        meter.guard()
        p = folder/f['path']
        if p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()) or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:
            raise ValueError('FD1_ORIGINAL_ALL_RAW_MEMBER_HASHES')
    card, samples, forecasts = [read(folder/n) for n in ['CARD.json','samples.json','S1_PREFLIGHTS.json']]
    with np.load(folder/'physics.npz',allow_pickle=False) as z:
        data = {n:z[n].copy() for n in z.files}
    with np.load(folder/'highrate.npz',allow_pickle=False) as z:
        high = z['rows'].copy()
    steps = raw['physics_steps']
    if high.shape[0] != steps or data['contact_audit'].shape != (steps,5):
        raise ValueError('FD1_COMPLETE_NATIVE_TICK_GRAIN')
    if not np.allclose(high[:,0],np.arange(1,steps+1)*.002,atol=1e-8,rtol=0) or np.any(high[:,-1]):
        raise ValueError('FD1_ORIGINAL_NATIVE_TIME_OR_SAFETY')
    summary = event_summary(data,samples,card)
    expected = dict(summary.pop('score'),physics_sha256=hashes['physics.npz'],raw_sample_count=len(data['time_s']))
    if expected != read(folder/'SCORE.json') or expected != record['score']:
        raise ValueError('FD1_ORIGINAL_TASK_SCORE_UNCHANGED')
    reject = [f for f in forecasts if f['decision']=='REJECT']
    if len(reject) != decision['S1_rejections']:
        raise ValueError('FD1_ORIGINAL_REJECTION_COUNT')
    commands=[]
    for p in sorted((folder/'relay/commands').glob('*.json')):
        v=read(p);a=unseal(v['action']);request=unseal(v['request']);obs=read(folder/'relay/observations'/p.name)
        commands.append(dict(sequence=int(p.stem),tick=a.get('tick'),action_sha256=v['action']['sha256'],
            request_sha256=v['request']['sha256'],observation_sha256=obs['sha256'],
            input_state_eef10=unseal(obs)['payload']['state_eef10'],
            camera_hashes={x['camera']:x['sha256'] for x in unseal(obs)['payload']['images']},
            trajectory_sha256=hashlib.sha256(np.asarray(a['payload']['trajectory_eef10'],dtype=np.float64).tobytes()).hexdigest(),
            previous_ack_sha256=None if request.get('previous_ack') is None else request['previous_ack']['sha256'],
            acknowledged=(folder/'relay/acks'/p.name).is_file()))
    references=read(folder/'references.json')
    last_stage=references[-1]['privileged_teacher_stage'] if references else None
    if reject:
        f=reject[0]
        if (folder/'relay/acks'/f'{steps//200:03d}.json').exists() or steps%200:
            raise ValueError('FD1_REJECT_ZERO_LIVE_PREFIX')
        rejection=dict(original=f,decision_time_s=steps*.002,actual_executed_prefixes=steps//200,
            predicted_offset_tick=f['first_predicted_violation_offset_tick'],
            predicted_absolute_time_s=steps*.002+f['first_predicted_violation_offset_tick']*.002,
            exact_geom_pair='FIRST_GEOMETRY_NOT_IN_ARCHIVE',signed_distance_m=None,
            rejected_prefix_actual_live_ticks=0)
    else:
        rejection=None
    stages=[r['privileged_teacher_stage'] for r in references]
    teacher_approach=last_stage=='AUDIT_ONLY_approach'
    mismatch=bool(teacher_approach and (summary['first_robot_object_contact_s'] is not None or summary['highrate_maximum_displacement_m']>=.001))
    return dict(attempt_id=identifier,adjudication=decision,card={k:card.get(k) for k in ['task','family','scene_group','split','source_xy','goal_xy','visual','cube_yaw','confirmation_noise_bank']},
        original_source_sha=raw['source_sha'],original_files_sha256=hashes,all_raw_manifest_members_verified=len(manifest),
        model_manifest_sha256=CANDIDATE,plan_sha256=PLAN,actual_final_time_s=steps*.002,
        original_teacher_final_label=last_stage,original_teacher_labels=dict(Counter(stages)),
        teacher_approach_vs_actual_event_mismatch=mismatch,actual_events=summary,rejection=rejection,
        unchanged_original_score=expected,commands=commands,
        original_PUSH_GEOMETRY=read(folder/'PUSH_GEOMETRY.json') if (folder/'PUSH_GEOMETRY.json').is_file() else None,
        scene_xml_sha256=read(folder/'scene.xml.json')['sha256'],video=record['video'])


def atlas(meter):
    results=[]
    for record in meter.baseline['actual_attempts']:
        meter.guard()
        results.append(audit_attempt(meter,record))
        update(meter.root/'evidence/ATLAS_PROGRESS.private.json',dict(completed=len(results),planned=42))
    qualified=[r for r in results if r['adjudication']['status']!='ENGINEERING_INCOMPLETE']
    rejects=[r for r in qualified if r['rejection']]
    if len(results)!=42 or len(qualified)!=40 or len(rejects)!=23:
        raise ValueError('FD1_42_ATTEMPTS_40_CONDITIONS_23_REJECTIONS')
    cost=sum(int(round(r['actual_final_time_s']/.002))+r['rejection']['predicted_offset_tick'] for r in rejects)
    return dict(status='CPU_LOG_AUDIT_PASS_NO_NEW_PHYSICS',original_attempts=42,qualified_conditions=40,
        actual_rows=results,first_geometry_not_in_archive=23,all_original_results_preserved=True,
        planned_all23_reconstruction_tick_upper=cost,planned_all23_engine_starts=46,
        geometry_reconstruction_needed=True,new_model_queries=0,new_physics=0)


def run(root,operation):
    from scripts.pi0292.durable import OwnerLock
    root=Path(root)
    lock=OwnerLock(root/'FD1_WORKER_OWNER.lock')
    meter=None
    try:
        meter=Meter(root,operation)
        if operation=='atlas':
            value=atlas(meter)
        elif operation in ['shadow','shadow-recovery-01','shadow-recovery-02']:
            from scripts.pi0294.fd1_shadow import run_shadow
            value=run_shadow(meter)
        else:
            raise ValueError('FD1_APPROVED_OPERATION_ONLY')
        value.update(worker_CPU_s=time.process_time()-meter.started,actual_shadow_ticks=meter.ticks,actual_engine_starts=meter.starts)
        put(root/'evidence'/(operation.upper()+'_RESULT.private.json'),value)
        meter.finish('COMPLETE')
        return dict(status=value['status'],operation=operation,worker_CPU_s=value['worker_CPU_s'],shadow_ticks=meter.ticks,engine_starts=meter.starts)
    except BaseException as exc:
        if meter is not None:
            put(root/'evidence'/(operation.upper()+'_FAILURE.private.json'),dict(error_type=type(exc).__name__,error=str(exc)))
            meter.finish('FAILED_PRESERVED',type(exc).__name__+': '+str(exc))
        raise
    finally:
        lock.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--operation',required=True,choices=['atlas','shadow','shadow-recovery-01','shadow-recovery-02'])
    a=p.parse_args()
    print(json.dumps(run(a.root,a.operation)),flush=True)
