"""One native process: qualified reachable HOLD, then the legal +Y1mm exit.

This is a new touch/withdraw curriculum, not the old grasp task. Commands
remain in the previously qualified four-action set; measured contact is never
replaced by the counterfactual geometry used to design the scene.
"""
from pathlib import Path
import json
import struct
import numpy as np
from .storage import digest
from .next1_episode import RULE

PHASE = 'collect-target-pilot1-contact'


def validate(spec, previous):
    path = Path(spec['episode_manifest']); e = json.loads(path.read_text())
    schemas = {'WM02_PILOT1_TOUCH_WITHDRAW_V1': 96, 'WM02_PILOT1_TOUCH_WITHDRAW_HOLD64_V2': 128,
               'WM02_PILOT1_CONTACT_TWO_ACTION_V3': 128,
               'WM02_PILOT1_CONTACT_TWO_SLOT_V4': 128}
    if e.get('schema') not in schemas: raise PermissionError('DECLARED_FINITE_CONTACT_PROTOCOL_REQUIRED')
    parallel = e['schema'] == 'WM02_PILOT1_CONTACT_TWO_SLOT_V4'
    coverage = e['schema'] in ('WM02_PILOT1_CONTACT_TWO_ACTION_V3','WM02_PILOT1_CONTACT_TWO_SLOT_V4')
    offset = e.get('withdrawal_xyz_m') if coverage else [0., .001, 0.]
    if offset not in ([0., 0., 0.], [0., .001, 0.]):
        raise PermissionError('ONLY_NATIVE_QUALIFIED_HOLD_OR_PLUS_Y_1MM_CONTACT_ACTIONS')
    expected = dict(schema=e['schema'], phase=PHASE,
                    reference_rule=RULE, prehold_atoms=40, screen_atom=48,
                    touch_end_atom=64, task_atoms=schemas[e['schema']], withdrawal_xyz_m=offset,
                    contact_force_threshold_N=.05, required_exit_s=.2,
                    minimum_away_Y_m=.0002, purpose='CONTACT_ACTION_COVERAGE' if coverage else 'CONTACT_EXIT_COMMISSION')
    if (digest(path) != spec['episode_manifest_sha256']
            or any(e.get(k) != v for k, v in expected.items())
            or e.get('run') != spec['registered_root'] or e.get('split') not in (('TRAIN', 'DEV', 'TEST') if coverage else ('TRAIN',))
            or not spec.get('pilot1_binding') or bool(spec.get('parallel_batch_spec')) != parallel):
        raise PermissionError('FRESH_SINGLE_PROCESS_QUALIFIED_TOUCH_WITHDRAW_REQUIRED')
    assets = json.loads(Path(spec['scene_allowlist']).read_text())
    scenes = [s for s in assets['scenes'] if s['scene_id'] == e['scene_id']]
    if (digest(spec['scene_allowlist']) != spec['scene_allowlist_sha256'] or len(scenes) != 1
            or scenes[0]['parent'] != e['parent'] or scenes[0]['split'] != e['split']
            or scenes[0]['model_path'] != spec['model_path']
            or scenes[0]['model_sha256'] != digest(spec['model_path'])
            or assets['native_binary_sha256'] != digest(spec['native_binary'])):
        raise PermissionError('EXACT_SEPARATE_REACHABLE_SCENE_REQUIRED')
    proof = Path(spec['scene_qualification']); native = json.loads(proof.read_text())
    prefix = json.loads((proof.parent/'STEADY_PREFIX_RESULT.private.json').read_text())
    contact = Path(spec['contact_entry_evidence']); event = json.loads(contact.read_text())
    if (digest(proof) != spec['scene_qualification_sha256']
            or native.get('status') != 'NATIVE_STAGE_PASSED' or not native.get('shutdown_clean')
            or native.get('physics_stops') or not prefix.get('qualified')
            or native.get('scene_sha256') != scenes[0]['model_sha256']
            or digest(contact) != spec['contact_entry_evidence_sha256']
            or event.get('native_result_sha256') != digest(proof)
            or event.get('scene_sha256') != scenes[0]['model_sha256']
            or event.get('contact_positive_native_ticks', 0) <= 0
            or event.get('contact_force_threshold_N') != .05):
        raise PermissionError('REAL_SAFE_NATIVE_CONTACT_ENTRY_BEFORE_WITHDRAW_REQUIRED')
    if (not previous or previous.get('status') != 'NATIVE_STAGE_PASSED'
            or not previous.get('shutdown_clean')
            or previous.get('native_binary_sha256') != digest(spec['native_binary'])):
        raise PermissionError('EXISTING_UNCHANGED_C08_PREDECESSOR_REQUIRED')
    return 2503+25*e['task_atoms']


class ContactReader:
    """Incremental original journal; every measured 2ms interval is retained."""
    def __init__(self, path, pad_ids, cube_id):
        self.path = Path(path); self.offset = 0; self.last_step = 0
        self.pads = set(pad_ids); self.cube = cube_id

    def interval(self, start, end):
        from scripts.pi0294.task_state import parse_state
        events = []
        with self.path.open('rb') as f:
            f.seek(self.offset)
            while self.last_step < end:
                header = f.read(4)
                if len(header) != 4: raise ValueError('PAUSED_CONTACT_JOURNAL_INCOMPLETE')
                size, = struct.unpack('<I', header)
                if not 40 <= size <= 2*1024**2: raise ValueError('CONTACT_PACKET_LENGTH_INVALID')
                raw = f.read(size)
                if len(raw) != size: raise ValueError('PAUSED_CONTACT_PACKET_INCOMPLETE')
                row = parse_state(raw)
                if row['step'] != self.last_step+1 or abs(row['time_s']-row['step']*.002) > 1e-8:
                    raise ValueError('EVERY_ORIGINAL_NATIVE_CONTACT_TICK_REQUIRED')
                self.last_step = row['step']; self.offset = f.tell()
                if row['step'] <= start: continue
                force = [float(c['wrench'][0]) for c in row['contacts']
                         if self.cube in (c['geom0'], c['geom1'])
                         and any(p in (c['geom0'], c['geom1']) for p in self.pads)]
                if not np.isfinite(force).all(): raise ValueError('NONFINITE_NATIVE_CONTACT_FORCE')
                maximum = max([0.] + force)
                events.append(dict(native_step=row['step'], contact_time_s=row['contact_time_s'],
                                   max_pad_cube_normal_force_N=maximum, contact_positive=maximum >= .05))
        if [r['native_step'] for r in events] != list(range(start+1, end+1)):
            raise ValueError('COMPLETE25_TICK_CONTACT_INTERVAL_REQUIRED')
        return events


def exit_score(events, departure, endpoint, table_supported):
    before = [e for e in events if e['native_step'] <= 4103]
    after = [e for e in events if e['native_step'] > 4103]
    suffix = 0
    for e in reversed(after):
        if e['contact_positive']: break
        suffix += 1
    away = float(np.asarray(endpoint)[1]-np.asarray(departure)[1])
    entry = any(e['contact_positive'] for e in before)
    separated = bool(suffix >= 100 and away >= .0002 and table_supported)
    return dict(actual_contact_before_withdraw=entry, terminal_noncontact_native_ticks=suffix,
                terminal_noncontact_s=suffix*.002, actual_TCP_away_Y_m=away,
                actual_table_support_at_end=bool(table_supported),
                qualified_contact_exit=bool(entry and separated), full_manipulation_success=False)


def run(a):
    import mujoco
    from scripts.pi0292.durable import write
    from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
    from go2arx5_vla.task_data.codec import pose10
    from .target_reference import fixed_goal_poses
    from .next1_episode import assess
    from .next2_contact import advance_atom, gripper_memory
    e = json.loads(Path(a.spec['episode_manifest']).read_text())
    a.prepare_dynamic(); initial = a.measured_task_state(); a.next1_samples = [initial]
    reader = ContactReader(a.output/'state.bin.journal', a.actual_state.pad, a.actual_state.cube_geom)
    goal = initial['tcp'].copy(); installed_step = 2503; windows = []; samples = []; events = []; departure = None

    def install(first=False):
        poses = fixed_goal_poses(initial['tcp'], goal, renewed=hasattr(a, 'active_reference') and not first)
        a.install_reference(np.stack([pose10(p, float(teacher_physical_fraction(1.))) for p in poses]))

    install()
    for atom in range(e['task_atoms']):
        step = a.observe()['step']; force = atom == 40
        if atom == 48 and not assess(a, e)['qualified']:
            raise RuntimeError('UNSTABLE_REAL_CONTACT_DECISION_PREFIX')
        if atom == 64:
            departure = a.measured_task_state()['tcp'][:3, 3].copy()
            goal = initial['tcp'].copy(); goal[:3, 3] += e['withdrawal_xyz_m']
            a.command_event(dict(phase='contact-withdraw-before-install', native_step=step,
                                 offset_xyz_m=e['withdrawal_xyz_m'], future_response_observed=False,
                                 actual_prior_positive_ticks=sum(r['contact_positive'] for r in events)))
            install(first=True); installed_step = step
        elif force or step-installed_step >= 175:
            install(); installed_step = step
        window = advance_atom(a, installed_step); windows.append(window)
        state = a.measured_task_state(); a.next1_samples.append(state)
        tick_events = reader.interval(step, step+25); events.extend(tick_events)
        memory = gripper_memory(a)
        if memory['values']['steps'] != state['native_step'] or memory['values']['sequence'] != a.active_reference['sequence']:
            raise RuntimeError('ACTUAL_GRIPPER_MEMORY_NOT_SAME_NATIVE_PAUSE')
        row = dict(native_start_step=step, native_end_step=step+25,
                   stage='HOLD_TOUCH' if atom < 64 else ('WITHDRAW_PLUS_Y_1MM' if any(e['withdrawal_xyz_m']) else 'HOLD_CONTROL'),
                   contact_positive_native_ticks=sum(r['contact_positive'] for r in tick_events),
                   contact_any=any(r['contact_positive'] for r in tick_events), native_contact_events=tick_events,
                   TCP_m=state['tcp'][:3, 3].tolist(), object_pose=state['object_pose'].tolist(),
                   qpos=state['qpos'].tolist(), qvel=state['qvel'].tolist(), ctrl=state['ctrl'].tolist(),
                   finger_q=state['finger_q'].tolist(), actual_gripper_internal_memory=memory,
                   reference_sha256=window['reference_sha256'],
                   executed_reference_index=(step-installed_step)//25,
                   executed_reference_EEF10=np.asarray(a.active_reference['action_EEF10'])[(step-installed_step)//25].tolist())
        samples.append(row); a.command_event(dict(phase='contact-actual-atom', **row))
    table_id = mujoco.mj_name2id(a.actual_state.model, mujoco.mjtObj.mjOBJ_GEOM, 'task_table')
    support = any({c['geom0'], c['geom1']} == {table_id, a.actual_state.cube_geom} and c['wrench'][0] >= .05 for c in a.observe()['contacts'])
    score = exit_score(events, departure, state['tcp'][:3, 3], support)
    score.update(schema='WM02_PILOT1_ACTUAL_TOUCH_WITHDRAW_SCORE_V1', same_native_process=True,
                 physics_resets=0, state_teleports=0, original_contact_threshold_N=.05,
                 actual_task_atoms=len(windows), contact_positive_native_ticks=sum(r['contact_positive'] for r in events),
                 contact_positive_atoms=sum(r['contact_any'] for r in samples),
                 contact_negative_atoms=sum(not r['contact_any'] for r in samples),
                 history_conditional_not_complete_MPC_WBC_state=True, samples=samples)
    write(a.output/'CONTACT_WITHDRAW_RESULT.private.json', score)
    a.capture_loaded_identity(); cancel = a.cancel_reference(); a.seal_trace(a.spec['max_ticks'])
    value = dict(schema='WM02_TARGET_DYNAMIC_DIAGNOSTIC_V1', phase=PHASE,
                 actual_collection_start_step=2503, actual_collection_start_s=initial['time_s'],
                 task_native_ticks=25*e['task_atoms'], complete_task_atoms=len(windows), windows=windows,
                 operator_cancel=cancel, reference_rule=RULE, recorded_gripper_memory=True,
                 controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED', full_task_success=False)
    write(a.output/'DYNAMIC_DIAGNOSTIC.json', value); a.reference_qualification = value
    if not all(w['solver_policies_with_exact_installed_target'] for w in windows):
        raise RuntimeError('SOLVER_TARGET_NOT_CONFIRMED_FOR_EVERY_CONTACT_ATOM')
    return value
