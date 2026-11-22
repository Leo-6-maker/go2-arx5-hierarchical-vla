"""Run one repaired F2 HOLD in an original slot surrendered by Wave A."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback

from go2arx5_vla.world_model import multisource_d4_native as d4
from go2arx5_vla.world_model.mfree_recovery_v2 import launch_service, register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close, dirs, unit_state
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.core_hopper_collection import reason
from scripts.sla.core_multienv_payload import LOCAL_JOURNAL, capacity
from scripts.sla.core_multienv_scope import AUTHORITY, OWNER, PLAN, SCOPE as ORIGINAL_SCOPE, order
from scripts.wm02.multitask_stage_cards_v6 import case_card, cold_native_owner, panel_cards, pin, require_unstarted_cases
from scripts.wm02.multitask_teacher_campaign_v6 import interface_records, teacher_row
from scripts.wm02.sim_only_campaign_v51 import fatal_control_violation

SCOPE = 'SLA_CORE_MULTIENV_V1_EXPERT_REPAIR'
SLOT = 'MEV1-TRAIN-F2-NEAR_POS-00-HOLD-red'
VERSION = 'F2_RED_HOLD_PITCH_DETOUR_V1'
WAVE_SHA = '45d949a883c7c10ec31c0caef9a40bdd0a4c51225d5417dac68c7b05432e870b'
PREFLIGHT_SHA = '444e49359b86fc4328278946799891444e1fe2c8acbdd6c73ba55d88a9fc2a23'
SCENE_SHA = '4d5e050e945f5a533bf43044a704b8d98504fa9d05f938442b8952cd0e9e00ff'
STOP_SIGNATURE = ['ILLEGAL_CONTACT', 'DESCEND', [['task_blue_cube', 'x5_link6']]]
SLOT_TICKS = 35703
MAX_RUNTIME = 4 * 3600
GIB = 1024**3
PRIOR_PILOT = 'b7ad7e5771093ebe1813b42e080c0c7509b8f62732ad5058ce6166fbcec4bde5'
SLOT25_SPEC = '761d75760d972dc1f6890be87aa84f0848ea82a7b83afdf67cd7cc1008315f0c'
SLOT25 = 'MEV1-TRAIN-F1-NEAR_POS-00-HOLD-blue'


def bind(spec):
    wave_pin, preflight_pin = spec['original_wave'], spec['teacher_preflight']
    if wave_pin['sha256'] != WAVE_SHA or preflight_pin['sha256'] != PREFLIGHT_SHA:
        raise PermissionError('Pilot needs the original Wave A owner and cold rehearsal')
    wave = read_pin(wave_pin)
    proof = read_pin(preflight_pin)
    population = read_pin(wave['compiled_population'])
    rows = order(population)
    row = next(v for v in rows if v['id'] == SLOT)
    if (wave['scope'] != ORIGINAL_SCOPE or wave['owner'] != OWNER or wave['compiled_population'] != spec['compiled_population']
            or proof['status'] != 'F2_RED_HOLD_DETOUR_COLD_CHECK' or proof['teacher'] != VERSION
            or proof['unopened'] != row or proof['scene']['sha256'] != SCENE_SHA
            or row['native_max_ticks'] != SLOT_TICKS or row['split'] != 'TRAIN'
            or any(proof[k] for k in ('native_requests','native_ticks','model_generations','optimizer_updates','VAL_inputs'))):
        raise PermissionError('Pilot changed its original slot, scene or frozen controls')
    if len(proof['outcomes']) != 2 or any(len(case['samples']) != 2 for case in proof['outcomes']):
        raise ValueError('Expected the four cold-rehearsal boundaries')
    for case in proof['outcomes']:
        for sample in case['samples']:
            nominal, shape = sample['H40'], sample['next_scene']
            lift = shape['lift']
            if (not nominal['original_exact'] or not nominal['candidate_H40_gate_pass'] or
                    not shape['approach']['passed'] or lift['first_problem'] is not None or
                    lift['planned_lift_waypoints'] != lift['matched_lift_waypoints'] or
                    any(v['active_contact'] or set(v['geoms']) != {'task_right_pad','task_blue_cube_geom'}
                        or v['compiled_box_separation_mm'] <= v['combined_geom_margin_mm']
                        for v in lift['distance_diagnostics'])):
                raise ValueError('Cold rehearsal lacks the retained H40 and contact checks')
    return wave, proof, population, rows, row


def validate_member(case, authority):
    wave, proof, _, _, row = bind(authority)
    condition = case['mobile_condition']
    if (case['scope'] != SCOPE or case['registry_scope'] != SCOPE or case['sim_only_role'] != 'EXPERT'
            or authority['native_request_cap'] != 1 or authority['native_tick_cap'] != SLOT_TICKS
            or authority['owner'] != OWNER or case['owner'] != OWNER
            or authority['deadline_epoch'] > wave['deadline_epoch']
            or not case['expert_training_eligible'] or case['collection_split'] != 'TRAIN'
            or case['teacher_version'] != VERSION or case['panel_entry_id'] != SLOT
            or case['max_ticks'] != SLOT_TICKS or case['task_instruction_id'] != row['instruction']
            or case['mobile_world_lock'] != wave['mobile_world_lock'] or case['registry'] != wave['registry']
            or case['resident_server_spec'] is not None or row['reset'] != condition['reset']
            or row['scene_seed'] != condition['seed'] or row['family'] != condition['family']
            or row['cluster'] != condition['layout']
            or proof['scene'] != dict(path=case['model_path'], sha256=case['frozen_files'][case['model_path']])):
        raise PermissionError('Expected the one original F2 HOLD slot and shared single-world owner')


def handoff(wave, proof, rows, *, drained=False):
    root = Path(wave['registered_root'])
    records = read(root/'RECORDS.private.json')
    by_id = {r['condition']: r for r in records}
    if SLOT in by_id or (root/'cases'/SLOT).exists():
        raise PermissionError('Original condition was dispatched by Wave A')
    prior = [by_id[x['condition']] for x in proof['outcomes']]
    for record, sample in zip(prior, proof['outcomes'], strict=True):
        if (record['result']['sha256'] != sample['terminal_sha256'] or record['post_STOP_ticks'] != 0
                or record['first_failure_signature'] != STOP_SIGNATURE or record['requests'] != 1):
            raise PermissionError('The original cell pause no longer has both native contact failures')
        read_pin(record['result'])
    terminal = root/'RESULT.private.json'
    if terminal.is_file():
        result = read(terminal)
        paused = result['not_started'].get(SLOT) == 'PAUSED_REPEAT_CONTACT_MECHANISM'
        if drained and not paused:
            stop = root/'STOP.json'
            # This clean disk cutover stopped before the loop could annotate slot 26.
            paused = (digest(terminal) == '55bb24644b237fdb7f2118ff9e9809f97a910063cd34ef9e0d31eccbf0967c16'
                      and result['not_started'].get(SLOT) == 'UNSTARTED_AT_STOP_OR_DEADLINE'
                      and ['F2','HOLD-red'] in result['paused_train_cells'] and stop.is_file()
                      and read(stop)['reason'] == 'PR425_WAVE_A_DISK_COLD_CATALOG_CUTOVER')
        if (result['source_sha'] != wave['source_sha'] or result['exit_code'] or result['unreconciled']
                or not paused):
            raise PermissionError('Closed Wave A did not explicitly surrender this slot')
        return dict(kind='ORIGINAL_WAVE_TERMINAL_PAUSE', receipt=pin(terminal))
    current_path = root/'CURRENT_CASE.private.json'
    current = read(current_path)
    roster = {r['id']: i+1 for i,r in enumerate(rows)}
    here, candidate = roster[current['id']], roster[SLOT]
    if here <= candidate:
        return None
    # A later registered case demonstrates the single sequential producer already
    # passed the reserved slot. Recheck the current upstream role and permit.
    case = read_pin(current['spec'])
    lease = read(root/'control/LEASE.json')
    observed = unit_state('wm02-'+root.name)
    if (case['source_sha'] != wave['source_sha'] or case['panel_entry_id'] != current['id']
            or case['campaign_root'] != str(root) or lease['status'] != 'ACTIVE'
            or lease['spec_sha256'] != WAVE_SHA
            or observed['MainPID'] == '0' or observed['ActiveState'] != 'active'):
        return None
    return dict(kind='ORIGINAL_WAVE_PASSED_PAUSED_SLOT', later_case=current['spec'],
                later_slot=here, pilot_slot=candidate, wave_unit=observed['unit'])


def reopen_binding(reopen, wave):
    """Verify the closed unpaid pilot and keep its original expiry."""
    old, prefix = read_pin(reopen['pilot']), read_pin(reopen['slot25'])
    root = Path(old['registered_root'])
    prior = read_pin(reopen['pilot_terminal'])
    lease = read(root/'control/LEASE.json')
    status = unit_state('wm02-'+root.name)
    if (reopen['pilot']['sha256'] != PRIOR_PILOT or reopen['slot25']['sha256'] != SLOT25_SPEC
            or Path(reopen['pilot']['path']) != root/'EXECUTION_SPEC.json'
            or Path(reopen['pilot_terminal']['path']) != root/'RESULT.private.json'
            or lease['status'] != 'CLOSED' or lease['spec_sha256'] != PRIOR_PILOT
            or status['ActiveState'] != 'inactive' or status['MainPID'] != '0'
            or prior['source_sha'] != old['source_sha'] or prior['condition'] != SLOT
            or prior['native_state'] != 'NOT_RUN' or prior['native_requests'] != 0 or prior['exit_code']
            or (root/'cases'/SLOT).exists() or (root/'control'/'journal'/(SLOT+'.json')).exists()
            or old['original_wave']['sha256'] != WAVE_SHA or old['teacher_preflight']['sha256'] != PREFLIGHT_SHA
            or prefix['multienv_cold_handoff']['producer'] != old['original_wave']
            or prefix['multienv_cold_handoff']['conditions'] != [SLOT25]
            or any(prefix[key] != wave[key] for key in ('mobile_world_lock','registry','compiled_population'))):
        raise PermissionError('The original F2 pilot must be closed and unpaid under the same Wave A authority')
    return old, prefix


def after_slot25(prefix, prefix_pin):
    root = Path(prefix['registered_root'])
    terminal = root/'RESULT.private.json'
    status = unit_state('wm02-'+root.name)
    if status['MainPID'] != '0' or not terminal.is_file():
        return None
    ledger = read(terminal)
    rec, = ledger['records']
    journal = Journal(root/'control/journal',root.name).completed(SLOT25,dict(
        condition=SLOT25,population=prefix['compiled_population']['sha256'],source=prefix['source_sha']))
    folder = root/'cases'/SLOT25
    case, native = read_pin(journal['spec']), read_pin(journal['result'])
    parent_lease, lease = (read(p/'control/LEASE.json') for p in (root,folder))
    score = teacher_row(case,native)
    if (status['ActiveState'] not in ('inactive','failed') or ledger['source_sha'] != prefix['source_sha']
            or Path(prefix_pin['path']) != root/'EXECUTION_SPEC.json'
            or parent_lease['status'] != 'CLOSED' or parent_lease['spec_sha256'] != prefix_pin['sha256']
            or lease['status'] != 'CLOSED' or lease['spec_sha256'] != journal['spec']['sha256']
            or Path(journal['spec']['path']) != folder/'EXECUTION_SPEC.json'
            or Path(journal['result']['path']) != folder/'RESULT.private.json' or journal['result'] != rec['result']
            or ledger['records'] != read(root/'RECORDS.private.json') or rec['condition'] != SLOT25
            or case['panel_entry_id'] != SLOT25 or native['source_sha'] != prefix['source_sha']
            or native['physical_requests'] != 1 or rec['requests'] != 1 or ledger['paid_requests'] != 1
            or not 0 < rec['ticks'] <= case['max_ticks'] or native['actual_native_integrations'] != rec['ticks']
            or rec['ticks'] != ledger['verified_ticks'] or fatal_control_violation(score)
            or not native['shutdown_clean'] or native['trace_audit']['status'] != 'RECORDED_TRACE_CHECKS_PASS'):
        raise PermissionError('Preserve slot 25 for terminal reconciliation before admitting a new world owner')
    if score['technical_failure']:
        if (ledger['unreconciled'] != SLOT25 or ledger['exit_code'] != 1 or native['unsafe']
                or native['wall_s'] < case['wall_limit_s']
                or not native['failure'].strip().endswith('TimeoutError: AUTO1_FINITE_ENDPOINT_OR_OPERATOR_STOP')
                or any((p/'STOP.json').exists() for p in (root,folder,folder/'native-01'))):
            raise PermissionError('Only the verified original wall limit can close a technical predecessor')
    elif ledger['exit_code'] or ledger['unreconciled']:
        raise PermissionError('The preceding native card must have settled before entering the F2 pilot')
    return dict(spec=prefix_pin,terminal=pin(terminal),paid=1,ticks=rec['ticks'],admitted=bool(rec['admitted']))


def available(spec):
    root = Path(spec['registered_root'])
    disk = shutil.disk_usage(root)
    memory = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                      if x.startswith('MemAvailable:'))) * 1024
    limits = spec['multienv']
    if 'multienv_payload_storage' in spec:
        volumes = capacity(root,spec['multienv_payload_storage'],limits)
        write(root/'CAPACITY.private.json',dict(epoch=time.time(),volumes=volumes,available_memory=memory))
        return volumes['passed'] and memory >= 13 * GIB
    reserve = sum(limits[k] for k in ('trajectory_reserve_bytes','snapshot_reserve_bytes','transfer_reserve_bytes'))
    floor = max(50 * GIB, int(.15 * disk.total)) + 8 * GIB
    write(root/'CAPACITY.private.json', dict(epoch=time.time(), free=disk.free, floor=floor,
                                           reserve=reserve, available_memory=memory))
    return memory >= 13 * GIB and disk.free >= floor + reserve


def make_case(spec, template, item, lease_end):
    folder, case = case_card(spec, template, item, spec['registered_panel'])
    case.update(registry_scope=SCOPE, collection_split='TRAIN', compiled_population=spec['compiled_population'],
                expert_training_eligible=True, resident_server_spec=None, teacher_version=VERSION,
                allocation_deadline_epoch=lease_end, deadline_epoch=lease_end-60)
    if 'multienv_payload_storage' in spec:
        case.update(multienv_payload_storage=spec['multienv_payload_storage'],multienv=spec['multienv'])
    return folder, case


def prepare(config):
    source, root = Path(config['source_root']), Path(config['root'])
    if (os.getuid() != 2009 or os.uname().nodename != 'corelab-4080s' or root.exists()
            or not root.name.startswith('sla-core-multienv-red-hold-pilot-') or root.parent.stat().st_uid != os.getuid()
            or source != Path(__file__).resolve().parents[2] or not re.fullmatch('[0-9a-f]{40}', config['source_sha'])):
        raise PermissionError('Expected a new independent Core-only pilot')
    wave = read_pin(config['original_wave'])
    spec = dict(wave, original_wave=config['original_wave'], teacher_preflight=config['teacher_preflight'])
    _, proof, population, _, row = bind(spec)
    interface = cold_native_owner(wave['interface_predecessor'])
    interface_records(interface, wave['mobile_world_lock'], identity=(d4.E0_SCOPE,d4.E0_PLAN))
    templates = {s['mobile_condition']['family']:s for s in map(read, interface['cases'])}
    require_unstarted_cases(wave['registry'], {SLOT})
    if (Path(wave['registered_root'])/'cases'/SLOT).exists():
        raise PermissionError('Original Wave A has already used the slot')
    if subprocess.check_output(['loginctl','show-user','2009','-p','Linger'], text=True).strip() != 'Linger=yes':
        raise PermissionError('Core systemd user service is not persistent')
    closure = read_pin(config['source_members'])
    if not closure or any(digest(source/name) != d for name,d in closure.items()):
        raise ValueError('Pilot source closure changed')
    end = min(wave['deadline_epoch'], time.time()+MAX_RUNTIME)
    reopen = config.get('reopen')
    if reopen:
        from scripts.wm02.mobile_boundary_runner_v1 import prepare_native_output

        if Path(prepare_native_output.__code__.co_filename).resolve() != source/'scripts/wm02/mobile_boundary_runner_v1.py':
            raise ImportError('Expected the sealed runner with the payload directory entry point')
        old, _ = reopen_binding(reopen,wave)
        storage = config['payload_storage']
        if (storage['live_control'] != LOCAL_JOURNAL or not capacity(root.parent,storage,wave['multienv'])['passed']
                or not handoff(wave,proof,order(population),drained=True)):
            raise PermissionError('The reopened pilot needs the original handoff and durable storage reserves')
        spec.update(reopen=reopen,multienv_payload_storage=storage)
        end = min(end,old['deadline_epoch'])
    if end < time.time()+2160:
        raise PermissionError('Original finite window has ended')
    dirs(root)
    (root/'cases').mkdir()
    authority = root/'AUTHORITY.private.json'
    write(authority, dict(source=AUTHORITY, plan_commit=PLAN, source_sha=config['source_sha'], owner=OWNER,
        hardware_allowed=False, native_request_cap=1, native_tick_cap=SLOT_TICKS, deadline_epoch=end,
        original_wave=config['original_wave'], teacher_preflight=config['teacher_preflight'],
        compiled_population=wave['compiled_population']))
    item, = panel_cards(population, [(dict(row, noise_seed=row['scene_seed']), None)], templates, role='EXPERT')
    panel = root/'REGISTERED_PANEL.private.json'
    write(panel, [item])
    env = dict(wave['native_environment'], PYTHONPATH=os.pathsep.join(
        [str(source), str(source/'src'), wave['native_environment']['PYTHONPATH']]),
        TMPDIR=str(root/'tmp'), XDG_CACHE_HOME=str(root/'cache'), ROS_LOG_DIR=str(root/'logs/ros'))
    spec.update(source_root=str(source), source_sha=config['source_sha'], source_files=closure,
        source_seal=hashlib.sha256(json.dumps(closure,sort_keys=True).encode()).hexdigest(), native_environment=env,
        scope=SCOPE, role=SCOPE, registry_scope=SCOPE, registered_root=str(root), campaign_root=str(root),
        mobile_group=str(root), collection_root=str(root/'admission'), unattended_control=str(root/'control'),
        deadline_epoch=end, campaign_deadline_epoch=end, sim_only_project_authorization=pin(authority),
        registered_panel=pin(panel), new_physical_requests_budget=1, cases=[str(root/'cases'/SLOT/'EXECUTION_SPEC.json')],
        data_publication='PILOT_RESULT_PENDING_COLD_CATALOG_ADOPTION')
    spec.update({name:str(root/name) for name in ('tmp','cache','logs','data')})
    _, case = make_case(spec, templates['F2'], item, min(end, time.time()+2160))
    validate(case)
    register(spec, [(root,spec)])
    write(root/'PREFLIGHT.private.json', dict(card=SLOT, native_requests=0, teacher_preflight=config['teacher_preflight'],
        geometry=proof['scene'], source_sha=spec['source_sha'], maximum_ticks=SLOT_TICKS, WBC_changes=False, safety_changes=False))
    return launch_service(spec, 'wm02-'+root.name, '--core-multienv-red-hold', bootstrap='v6_bootstrap.py')


def run(spec):
    root = Path(spec['registered_root'])
    wave, proof, _, rows, row = bind(spec)
    original = Path(wave['registered_root'])
    student = Path(read_pin(wave['multienv']['handoff_owner'])['registered_root'])
    template = next(c for p in read_pin(wave['interface_predecessor'])['cases'] if (c:=read(p))['mobile_condition']['family']=='F2')
    item, = read_pin(spec['registered_panel'])
    journal = Journal(root/'control/journal', root.name)
    reopen = spec.get('reopen')
    prefix = reopen_binding(reopen,wave)[1] if reopen else None

    def permission():
        source = handoff(wave,proof,rows,drained=prefix is not None)
        if source is None or prefix is None:
            return source
        terminal = after_slot25(prefix,reopen['slot25'])
        return dict(source, preceding_card=terminal) if terminal else None

    def progress(status, **extra):
        write(root/'PROGRESS.private.json', dict(epoch=time.time(), pid=os.getpid(), stage=status,
                                              condition=SLOT, **extra))

    def stopped():
        return (time.time()+2160 >= spec['deadline_epoch'] or (root/'STOP.json').exists()
                or (prefix is None and (original/'STOP.json').exists()) or (student/'STOP.json').exists())

    def permitted():
        return not (student/'STUDENT_TURN_READY.private.json').exists() and available(spec) and not stopped()

    def launch(lock, permission):
        acquired = time.time()
        end = min(spec['deadline_epoch'], acquired+2160)
        folder, case = make_case(spec, template, item, end)
        lease = root/'control'/'WORLD.private.json'
        inode = os.fstat(lock.handle.fileno())
        write(lease, dict(kind='CORELAB_LIVE_SINGLE_WORLD_LEASE', owner_pid=os.getpid(), uid=os.getuid(),
            acquired_epoch=acquired, deadline_epoch=end, lock=spec['mobile_world_lock'],
            lock_inode=inode.st_ino, lock_device=inode.st_dev, condition=SLOT))
        case['site_world_lease'] = pin(lease)
        require_unstarted_cases(spec['registry'], {SLOT})
        validate(case)
        dirs(folder)
        register(spec, [(folder,case)])
        spec_path = folder/'EXECUTION_SPEC.json'
        command = [case['python'],'-B',str(Path(spec['source_root'])/'scripts/wm02/v6_bootstrap.py'),
                   '--case',str(spec_path),'--world-lock-fd',str(lock.handle.fileno())]
        with (folder/'logs/case.log').open('xb') as stream:
            child = subprocess.Popen(command,cwd=spec['source_root'],env=dict(os.environ,**case['native_environment']),
                stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lock.handle.fileno(),))
        progress('NATIVE_RUNNING', child=child.pid, spec=pin(spec_path), handoff=permission)
        child.wait()
        return dict(case=pin(spec_path), native=pin(folder/'RESULT.private.json'), exit=child.returncode)

    code, outcome, error = 1, None, None
    write(root/'SUPERVISOR.private.json', dict(pid=os.getpid(), epoch=time.time(), source_sha=spec['source_sha']))
    try:
        while not stopped():
            ready = permission()
            if ready is None:
                progress('WAIT_ORIGINAL_SLOT_HANDOFF')
            elif not permitted():
                progress('WAIT_CAPACITY_OR_STUDENT_HANDOFF', handoff=ready)
            else:
                try:
                    lock = OwnerLock(spec['mobile_world_lock'])
                except BlockingIOError:
                    progress('WAIT_SHARED_WORLD', handoff=ready)
                else:
                    with closing(lock):
                        if permitted() and (ready:=permission()):
                            receipt = journal.once(SLOT, dict(handoff=ready, source=spec['source_sha']),
                                                   lambda: launch(lock, ready))
                            break
            time.sleep(20)
        else:
            code = 0
            return code
        case, native = read_pin(receipt['case']), read_pin(receipt['native'])
        lease = read(Path(case['registered_root'])/'control/LEASE.json')
        score = teacher_row(case,native)
        outcome = dict(condition=SLOT, requests=native['physical_requests'], ticks=native['actual_native_integrations'],
                       admitted=False, first_failure=reason(score,native), native=receipt['native'], score=score)
        if (lease['status'] != 'CLOSED' or lease['spec_sha256'] != receipt['case']['sha256']
                or native['source_sha'] != spec['source_sha'] or outcome['requests'] != 1
                or outcome['ticks'] > SLOT_TICKS or native['optimizer_updates'] or native['training_visits']
                or native['model_generations'] or score['technical_failure'] or fatal_control_violation(score)
                or (score['hard_STOP'] and score['integrations_after_STOP'] != 0)):
            raise RuntimeError('Preserve unresolved pilot for terminal reconciliation')
        if score['task_and_mobile_success'] and score['completed'] and not score['hard_STOP']:
            path = root/'G7_REQUEST.private.json'
            write(path, dict(case=receipt['case'], native=receipt['native'], condition=row,
                             statistics=spec['C20_statistics_path']))
            proc = subprocess.run([spec['admission_python'],'-B','-m','scripts.sla.core_hopper_admission',
                str(path),digest(path)],cwd=spec['source_root'],env=dict(os.environ,**spec['native_environment']),
                text=True,capture_output=True,timeout=180,check=True)
            outcome.update(json.loads(proc.stdout))
        code = 0
    except Exception:
        error = traceback.format_exc()
    finally:
        intent = journal.path(SLOT).is_file()
        write(root/'RESULT.private.json', dict(source_sha=spec['source_sha'], condition=SLOT, result=outcome,
            native_state='RECONCILED' if outcome and not error else 'UNKNOWN_RECONCILE' if intent or error else 'NOT_RUN',
            native_requests=outcome['requests'] if outcome else None if intent else 0,
            automatic_retry=False, unmodified_training_catalog=True, exit_code=code, error=error))
        close(spec, code)
    return code


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('sha256')
    args = p.parse_args()
    os.umask(0o077)
    print(json.dumps(prepare(read_pin(dict(path=args.config, sha256=args.sha256)))))
