"""Independent-condition reset contract for the existing book evaluator.

Each registered worker executes one original condition and exits. The existing
owner must collect/audit/close it before submitting the next worker. This module
does not schedule jobs, clear STOPs, step physics or change task scoring.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

PROTOCOL = 'BOOK_RESET_DEV10_V1'
KIND = 'V7_BOOK_RESET_DEV10_CONDITION'
LIMITS = dict(model_loads=1, optimizer_updates=0, physics_integrations=18500,
    generation_queries=80, RGB_pairs=642, source_constructors=1)
SOURCES = ('scripts/pi0294/multitask/book_reset.py',
    'scripts/pi0294/multitask/book_policy.py', 'scripts/pi0294/multitask/source_engine.py',
    'scripts/pi0291_arx5_data/engine.py', 'scripts/pi0294/multitask/capture_core.py',
    'scripts/pi0294/multitask/night_book_audit.py')
NATIVE_CHECKS = ('two_processes_rebuilt_common_home', 'controller_and_IK_clear',
    'original_dual_camera_qualification', 'unchanged_condition_noise',
    'live_predecessor_rejected', 'residual_writer_rejected',
    'unknown_close_rejected', 'dirty_initial_state_rejected')


def digest(value):
    return hashlib.sha256((json.dumps(value, indent=2, allow_nan=False)+'\n').encode()).hexdigest()


def plan(checkpoints):
    from .seed_set_export import DEV_OFFSETS
    if (not isinstance(checkpoints, dict) or not checkpoints
            or any(not isinstance(k, str) or not k for k in checkpoints)
            or any(not isinstance(v, str) or len(v) != 64
                or any(c not in '0123456789abcdef' for c in v) for v in checkpoints.values())):
        raise ValueError('FIXED_NAMED_CHECKPOINT_MANIFESTS_BEFORE_ANY_NEW_PANEL')
    return dict(schema=PROTOCOL, checkpoints=dict(checkpoints),
        conditions=[dict(id=f'BOOK_DEV{i:03}', xy_offset_m=list(xy)) for i, xy in enumerate(DEV_OFFSETS)],
        seed_namespace='BOOK5_DEV10', task_duration_s=32, warmup_s=5,
        H40=40, execution_prefix=8, generation_steps=4,
        one_registered_process_per_condition=True, same_condition_retry=False,
        original_results_and_STOP_preserved=True, exposed_DEV_not_blind_TEST=True,
        teacher_metadata_student_input=False, safety_or_scoring_changes=False)


def validate_manifest(manifest, panel, qualification):
    if (panel != plan(panel.get('checkpoints')) or manifest.get('kind') != KIND
            or manifest.get('reset_model') not in panel['checkpoints']
            or type(manifest.get('reset_condition_index')) is not int
            or not 0 <= manifest['reset_condition_index'] < 10
            or manifest.get('delta_checkpoint_manifest_sha256') != panel['checkpoints'][manifest['reset_model']]
            or manifest.get('limits') != LIMITS
            or manifest.get('sealed_inputs', {}).get('BOOK_RESET_DEV10_PLAN.json') != digest(panel)
            or manifest.get('sealed_inputs', {}).get('RESET_NATIVE_QUALIFICATION.json') != digest(qualification)):
        raise PermissionError('SEALED_FIXED_MODEL_BY_ORIGINAL_DEV10_SINGLE_CONDITION')
    predecessor_pin = manifest.get('sealed_inputs', {}).get('RESET_PREDECESSOR.json')
    if (manifest.get('run_uuid') != condition_run_name(panel, manifest['reset_model'], manifest['reset_condition_index'])
            or not isinstance(predecessor_pin, str) or len(predecessor_pin) != 64
            or any(c not in '0123456789abcdef' for c in predecessor_pin)):
        raise PermissionError('ONE_DETERMINISTIC_MODEL_CONDITION_ID_AND_SEALED_PREDECESSOR')
    if (qualification.get('schema') != 'BOOK_RESET_NATIVE_QUALIFICATION_V1'
            or qualification.get('passed') is not True
            or qualification.get('host') != os.uname().nodename
            or qualification.get('uid') != os.getuid()
            or qualification.get('numpy_version') != np.__version__
            or qualification.get('MuJoCo_version') != '3.2.7'
            or qualification.get('checks') != dict.fromkeys(NATIVE_CHECKS, True)
            or qualification.get('template_manifest_sha256') != manifest.get('source_template_manifest_sha256')
            or qualification.get('source_files') != manifest.get('reset_source_files')
            or set(qualification.get('source_files', {})) != set(SOURCES)):
        raise PermissionError('ACTUAL_RESET_NATIVE_AND_FAULT_INJECTION_QUALIFICATION_REQUIRED')
    # Cold qualification is reusable only for these exact implementation bytes.
    source = Path(__file__).resolve().parents[3]
    for name, pin in qualification['source_files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != pin:
            raise PermissionError('RESET_QUALIFIED_IMPLEMENTATION_CHANGED:'+name)
    return panel['conditions'][manifest['reset_condition_index']]


def condition_run_name(panel, model, index):
    return f"v7-book-reset-{digest(panel)[:16]}-{panel['checkpoints'][model][:12]}-dev{index:03}"


def source_files():
    root = Path(__file__).resolve().parents[3]
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCES}


def unit_state(name):
    if not isinstance(name, str) or '/' in name or not name.endswith('.service'):
        raise PermissionError('EXACT_PREDECESSOR_USER_UNIT')
    p = subprocess.run(['systemctl', '--user', 'show', name, '-p', 'MainPID',
        '-p', 'ControlGroup', '-p', 'SubState', '-p', 'InvocationID'],
        capture_output=True, text=True, check=True, timeout=5)
    return dict(line.split('=', 1) for line in p.stdout.splitlines() if '=' in line)


def process_identity(pid, *, proc=Path('/proc')):
    """Linux start ticks disambiguate a registered PID, including names with ')'."""
    path = proc/str(pid)
    if path.stat().st_uid != os.getuid():
        raise PermissionError('REGISTERED_PROCESS_OWNER_CHANGED')
    fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(PID=int(pid), start_ticks=int(fields[19]), PPID=int(fields[1]))


def registered_writer_scope(root, unit, state, *, extra_pids=()):
    """Capture the live launch identity for sealing in RESET_PREDECESSOR.json.

    Every external project consumer must be registered here before custody is
    closed. Ordinary children remain covered by ancestry and the unit cgroup.
    No name-based daemon exemption or same-UID descriptor census is used.
    """
    group = state.get('ControlGroup', '')
    invocation = state.get('InvocationID', '')
    main = int(state.get('MainPID', '0'))
    if (unit != Path(root).name+'.service' or not group.startswith('/')
            or '..' in Path(group).parts or Path(group).name != unit or main <= 0
            or len(invocation) != 32 or any(c not in '0123456789abcdef' for c in invocation)):
        raise PermissionError('LIVE_REGISTERED_UNIT_IDENTITY_REQUIRED')
    processes = [process_identity(pid) for pid in sorted({main, *extra_pids})]
    membership = (Path('/proc')/str(main)/'cgroup').read_text().splitlines()
    if '0::'+group not in membership:
        raise PermissionError('REGISTERED_MAIN_PID_CGROUP_MISMATCH')
    return dict(schema='REGISTERED_PROJECT_WRITER_SCOPE_V1', root=str(Path(root).resolve()),
        unit=unit, InvocationID=invocation, ControlGroup=group,
        host=os.uname().nodename, uid=os.getuid(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), processes=processes)


def validate_writer_scope(root, scope):
    if (not isinstance(scope, dict) or scope.get('schema') != 'REGISTERED_PROJECT_WRITER_SCOPE_V1'
            or scope.get('root') != str(Path(root).resolve())
            or scope.get('unit') != Path(root).name+'.service'
            or scope.get('host') != os.uname().nodename or scope.get('uid') != os.getuid()
            or scope.get('boot_id') != Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or not isinstance(scope.get('processes'), list) or not scope['processes']):
        raise PermissionError('REGISTERED_PROJECT_WRITER_SCOPE_REQUIRED')
    group = scope.get('ControlGroup', '')
    invocation = scope.get('InvocationID', '')
    if (not isinstance(group, str) or not group.startswith('/') or '..' in Path(group).parts
            or Path(group).name != scope['unit'] or not isinstance(invocation, str)
            or len(invocation) != 32 or any(c not in '0123456789abcdef' for c in invocation)
            or any(not isinstance(p, dict) or type(p.get('PID')) is not int or p['PID'] <= 0
                or type(p.get('start_ticks')) is not int or p['start_ticks'] <= 0 for p in scope['processes'])
            or len({p['PID'] for p in scope['processes']}) != len(scope['processes'])):
        raise PermissionError('AMBIGUOUS_REGISTERED_PROCESS_SCOPE')


def scoped_processes(scope, proc):
    """Read ancestry metadata; open FDs only for registered processes/children."""
    registered = {p['PID']:p['start_ticks'] for p in scope['processes']}
    rows = {}
    for process in proc.iterdir():
        if not process.name.isdecimal():
            continue
        pid = int(process.name)
        try:
            rows[pid] = process_identity(pid, proc=proc)
        except FileNotFoundError:
            if process.exists():
                raise PermissionError('PROCESS_IDENTITY_UNKNOWN')
        except PermissionError:
            if pid in registered:
                raise  # Relevant identity denied is UNKNOWN, never a waiver.
    for pid, start in registered.items():
        if pid in rows and rows[pid]['start_ticks'] != start:
            raise PermissionError('REGISTERED_PID_REUSED_OR_AMBIGUOUS')
    selected = set(registered) & rows.keys()
    while True:
        children = {pid for pid, row in rows.items() if row['PPID'] in selected}
        if children <= selected:
            return {pid:rows[pid] for pid in selected}
        selected |= children


def live_writers(root, *, scope, proc=Path('/proc')):
    """Inspect sealed project identities and descendants, not unrelated user FDs."""
    root = Path(root).resolve(); validate_writer_scope(root, scope); found = []
    selected = scoped_processes(scope, proc)
    for pid, identity in selected.items():
        process = proc/str(pid); handle = None
        try:
            # Pin the actual live process while inspecting its descriptors.
            if proc == Path('/proc') and hasattr(os, 'pidfd_open'):
                handle = os.pidfd_open(pid)
            if process_identity(pid, proc=proc) != identity:
                raise PermissionError('REGISTERED_PROCESS_CHANGED_DURING_INSPECTION')
            for fd in (process/'fd').iterdir():
                try:
                    target = Path(os.readlink(fd).removesuffix(' (deleted)'))
                    if not target.is_absolute() or not target.is_relative_to(root):
                        continue
                    info = (process/'fdinfo'/fd.name).read_text()
                    flags = int(next(x.split()[1] for x in info.splitlines() if x.startswith('flags:')), 8)
                    if flags & os.O_ACCMODE:
                        found.append(dict(PID=int(process.name), fd=int(fd.name), relative_path=str(target.relative_to(root))))
                except FileNotFoundError:
                    continue  # That process/descriptor has actually gone away.
            if process_identity(pid, proc=proc) != identity:
                raise PermissionError('REGISTERED_PROCESS_CHANGED_DURING_INSPECTION')
        except (FileNotFoundError, ProcessLookupError):
            if process.exists():
                raise PermissionError('RELEVANT_PROCESS_DESCRIPTORS_UNKNOWN')
        finally:
            if handle is not None:
                os.close(handle)
    if scoped_processes(scope, proc) != selected:
        raise PermissionError('PROJECT_PROCESS_SET_CHANGED_DURING_INSPECTION')
    return found


def verify_predecessor(predecessor, *, current_root, expected_previous=None,
        observe_unit=unit_state, writers=live_writers, cgroup_root=Path('/sys/fs/cgroup')):
    """A stale state file is insufficient: require CLOSED plus no live writer."""
    current_root = Path(current_root).resolve(); root = Path(predecessor['root'])
    if (not root.is_absolute() or root.is_symlink() or root.resolve().parent != current_root.parent
            or root.resolve() == current_root
            or predecessor.get('unit') != root.name+'.service'
            or set(predecessor.get('files', {})) != {'RUN_MANIFEST.json', 'RESULT.json', 'control/LEASE.json'}):
        raise PermissionError('EXACT_SIBLING_CLOSED_PREDECESSOR')
    values = {}
    for name, pin in predecessor['files'].items():
        p = root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise PermissionError('PREDECESSOR_CLOSURE_PATH')
        body = p.read_bytes()
        if hashlib.sha256(body).hexdigest() != pin:
            raise PermissionError('PREDECESSOR_CLOSURE_CHANGED:'+name)
        values[name] = json.loads(body)
    m, result, lease = (values[n] for n in ('RUN_MANIFEST.json', 'RESULT.json', 'control/LEASE.json'))
    if (lease.get('status') != 'CLOSED' or lease.get('run_uuid') != root.name
            or lease.get('manifest_sha256') != predecessor['files']['RUN_MANIFEST.json']
            or lease.get('result_sha256') != predecessor['files']['RESULT.json']):
        raise PermissionError('INDEPENDENT_NATIVE_CLOSE_NOT_ESTABLISHED')
    if expected_previous is not None:
        model, index, panel_digest = expected_previous
        if (m.get('kind') != KIND or m.get('reset_model') != model or m.get('reset_condition_index') != index
                or m.get('sealed_inputs', {}).get('BOOK_RESET_DEV10_PLAN.json') != panel_digest
                or result.get('status') != 'COMPLETE_BOOK_RESET_DEV_CONDITION'
                or result.get('error') is not None or len(result.get('conditions', [])) != 1
                or result['conditions'][0].get('valid_model_condition') is not True
                or result['conditions'][0].get('reset_close_confirmed') is not True):
            raise PermissionError('PREVIOUS_CONDITION_OR_RESET_FAILED_NO_ADVANCE')
        row=result['conditions'][0]
        names={'RESET_INITIAL.json','RESET_CAMERA.json','RESET_CLOSE.json'}
        if (row.get('id')!=f'BOOK_DEV{index:03}' or row.get('reset_protocol')!=PROTOCOL
                or set(row.get('reset_witness_files',{}))!=names):
            raise PermissionError('PREVIOUS_RESET_WITNESS_REQUIRED')
        witness={}
        for name,pin in row['reset_witness_files'].items():
            path=root/'conditions'/row['id']/name
            body=path.read_bytes()
            if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                    or hashlib.sha256(body).hexdigest()!=pin):
                raise PermissionError('PREVIOUS_RESET_WITNESS_CHANGED:'+name)
            witness[name]=json.loads(body)
        if (witness['RESET_INITIAL.json'].get('controller_and_IK_clear')is not True
                or witness['RESET_CAMERA.json'].get('original_dual_camera_qualified')is not True
                or witness['RESET_CLOSE.json'].get('engine_close_returned')is not True
                or witness['RESET_CLOSE.json'].get('raw_writer_closed')is not True):
            raise PermissionError('PREVIOUS_RESET_OR_CLOSE_NOT_QUALIFIED')
    state = observe_unit(predecessor['unit'])
    if state.get('MainPID') != '0' or state.get('SubState') not in ('dead', 'failed', 'exited'):
        raise PermissionError('PREDECESSOR_PROCESS_STILL_LIVE_OR_UNKNOWN')
    scope = predecessor.get('process_scope')
    if writers is live_writers:
        validate_writer_scope(root, scope)
        if (state.get('InvocationID') != scope['InvocationID']
                or state.get('ControlGroup') not in ('', scope['ControlGroup'])):
            raise PermissionError('PREDECESSOR_UNIT_INVOCATION_OR_CGROUP_CHANGED')
    group = scope['ControlGroup'] if writers is live_writers else state.get('ControlGroup')
    if group:
        relative = Path(group.lstrip('/'))
        if '..' in relative.parts:
            raise PermissionError('PREDECESSOR_CGROUP_PATH')
        directory = cgroup_root/relative
        # Child cgroups can retain writers even when the unit's MainPID is zero.
        if directory.exists():
            for f in directory.rglob('cgroup.procs'):
                if f.read_text().strip():
                    raise PermissionError('PREDECESSOR_CHILD_PROCESS_STILL_LIVE')
    if (writers(root, scope=scope) if writers is live_writers else writers(root)):
        raise PermissionError('PREDECESSOR_RESIDUAL_WRITER')
    if observe_unit(predecessor['unit']) != state:
        raise PermissionError('PREDECESSOR_UNIT_CHANGED_DURING_WRITER_INSPECTION')
    return dict(protocol=PROTOCOL, predecessor=root.name, process_terminal=True,
        cgroup_empty=True, no_residual_writers=True, CLOSED_lease_verified=True)


def pristine_warmstart(engine):
    """Reproduce the constructor + snapshot forwards in independent data.

    MuJoCo computes qacc_warmstart during mj_forward, even with zero integration.
    It must match pristine physics, not be forced to zero or silently reused.
    """
    import mujoco
    fresh=mujoco.MjData(engine.model)
    fresh.qpos[:]=engine.data.qpos
    mujoco.mj_forward(engine.model,fresh)
    mujoco.mj_forward(engine.model,fresh)
    return fresh.qacc_warmstart.copy()


def initial_witness(engine, student, card):
    """Read, never repair, the newly constructed original engine state."""
    from .object_relations import pose
    d, ik = engine.data, engine.ik_data
    initial = engine.snapshot()
    expected = pose(card['initial_tcp']); actual = pose(engine.pose('tool'))
    object_origin = np.asarray(initial['native_book_body_origin'])
    angle = np.arccos(np.clip((np.trace(actual[:3, :3] @ expected[:3, :3].T)-1)/2, -1., 1.))
    zero = ('qvel', 'ctrl', 'act', 'qfrc_applied', 'xfrc_applied')
    if (d is ik or float(d.time) != 0 or float(ik.time) != 0 or engine.steps != 0
            or engine.renderer is not None or engine.safety
            or any(np.any(np.asarray(getattr(d, name)) != 0) for name in zero)
            or np.any(np.asarray(ik.qvel) != 0)
            or any(getattr(engine, name) is not None for name in ('_ik_target', '_cached_ik_solution', '_cached_ik_error'))
            or engine.ik_solve_calls != 0 or engine.ik_cache_hits != 0
            or not np.array_equal(engine.arm_targets, d.qpos[engine.arm_q])
            or not np.array_equal(engine.finger_targets, np.full(2, .042))
            or not np.array_equal(d.qpos[engine.finger_q], np.full(2, .042))
            or initial['safety_valid'] is not True or initial['native_grasp']
            or student.seed_namespace != 'BOOK5_DEV10' or student.identifier != card['id']
            or student.index != -1 or student.last_plan != -1
            or student.queries != 0 or student.completed_generations != 0
            or student.state is not None or student.images is not None
            or not np.allclose(object_origin,pose(card['initial_book_native_world'])[:3,3],atol=1e-12,rtol=0)
            or np.linalg.norm(actual[:3, 3]-expected[:3, 3]) > .0002
            or angle > .002):
        raise PermissionError('RESET_NEW_HOME_PHYSICS_CONTROLLER_IK_AND_STUDENT_STATE')
    if not np.array_equal(d.qacc_warmstart,pristine_warmstart(engine)):
        raise PermissionError('RESET_ACCELERATION_CACHE_MUST_MATCH_PRISTINE_NATIVE_FORWARD')
    return dict(protocol=PROTOCOL, steps=0, simulation_time_s=0., new_data_and_IK_data=True,
        qpos=np.asarray(d.qpos).tolist(), arm_targets=np.asarray(engine.arm_targets).tolist(),
        finger_targets=np.asarray(engine.finger_targets).tolist(), controller_and_IK_clear=True,
        student_history_and_query_cursor_clear=True,
        first_generation_seed=student.generation_seed(0), seed_namespace=student.seed_namespace,
        native_scene_sha256=hashlib.sha256(engine.xml.encode()).hexdigest(),
        initial_native_book_body_origin=object_origin.tolist(),
        acceleration_cache_matches_pristine_native_forwards=True,
        extra_forward_calls=2,extra_physics_integrations=0,
        original_common_home_qualified=True, initial_safety_valid=True, native_grasp=False)
