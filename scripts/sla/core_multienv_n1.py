"""Run PR425's 12 engineering cards through the existing single-world lease."""

import copy
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close, dirs
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.core_hopper_expert import OWNER
from scripts.sla.core_multienv_geometry import load_population
from scripts.sla.core_multienv_scope import PLAN, POPULATION_SHA
from scripts.wm02.multitask_stage_cards_v6 import pin, require_unstarted_cases
from scripts.wm02.sim_only_campaign_v51 import compact_case, fatal_control_violation

SCOPE = 'SLA_CORE_MULTIENV_V1_N1_ENGINEERING'
AUTHORITY = 'CURRENT_USER_SLA_CORE_MULTIENV_DATA_V1'
TOLERANCE_SHA = '6da63587addd8d9e35d36ce09191593de9dba46b2cee9b47266beeb71df0fc0a'
ORIGINAL_SPEC = 'a76b564c936a5a22058b08a17b9fc1d1b6e4c010b7cbff9fd8100484b68261ea'


def fixed_cards(authority):
    logical = authority['logical']
    if logical['sha256'] != POPULATION_SHA or authority['tolerance']['sha256'] != TOLERANCE_SHA:
        raise ValueError('Wrong PR425 population or frozen tolerance')
    tolerance = read_pin(authority['tolerance'])
    slots = [x for x in load_population(Path(logical['path']))['engineering']
             if x['group'] in ('N1-SHORT', 'N1-FULL')]
    rows = read_pin(authority['fixtures'])
    if ([row['slot'] for row in rows] != slots or len(rows) != 12
            or sum(row['slot']['max_native_ticks'] for row in rows) != 258436
            or authority['native_request_cap'] != 12 or authority['native_tick_cap'] != 258436):
        raise ValueError('Only the fixed eight short and four full N1 cards are authorized')
    short = [row for row in rows if row['slot']['kind'] == 'SHORT']
    if ({row['old_id'] for row in short} != set(tolerance['old_recordings_sha256'])
            or any(row['recording']['sha256'] != tolerance['old_recordings_sha256'][row['old_id']]
                   for row in short)):
        raise ValueError('Short cards must retain their original eight recordings')
    return {row['slot']['id']: row for row in rows}


def retained_records(previous, cards, *, closed):
    old = read_pin(previous['spec'])
    root = Path(old['registered_root'])
    records = read_pin(previous['records'])
    ids = [r['id'] for r in records]
    if (previous['spec']['sha256'] != ORIGINAL_SPEC or Path(previous['spec']['path']) != root/'EXECUTION_SPEC.json'
            or Path(previous['records']['path']) != root/'RECORDS.private.json' or old['scope'] != SCOPE
            or not 0 < len(ids) < len(cards) or ids != list(cards)[:len(ids)]):
        raise PermissionError('Expected the exact paid prefix of the registered N1 matrix')
    for rec in records:
        cid = rec['id']
        folder = root/'cases'/cid
        lease = read(folder/'control/LEASE.json')
        native = read_pin(rec['terminal'])
        case_path = folder/'EXECUTION_SPEC.json'
        member = read(case_path)
        if (lease['status'] != 'CLOSED' or lease['spec_sha256'] != digest(case_path)
                or member['panel_entry_id'] != cid or member['site_world_spec']['fixture'] != cards[cid]['reference']
                or Path(rec['terminal']['path']) != folder/'RESULT.private.json'
                or rec['source'] != old['source_sha'] or native['source_sha'] != old['source_sha']
                or rec['requests'] != 1 or native['physical_requests'] != 1 or rec['ticks'] != native['actual_native_integrations']
                or rec['ticks'] > cards[cid]['slot']['max_native_ticks'] or rec['technical_failure']
                or rec['post_STOP_ticks'] or 'measurement' not in rec or any(native[k] for k in
                   ('optimizer_updates','training_visits','model_generations'))):
            raise ValueError('An earlier N1 condition lacks its original paid terminal: '+cid)
    remaining = list(cards)[len(ids):]
    require_unstarted_cases(old['registry'],set(remaining))
    if closed:
        from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state

        terminal = read_pin(previous['terminal'])
        lease = read(root/'control/LEASE.json')
        owner = unit_state('wm02-'+root.name)
        if (Path(previous['terminal']['path']) != root/'RESULT.private.json'
                or owner['MainPID'] != '0' or owner['ActiveState'] != 'inactive'
                or lease['status'] != 'CLOSED' or lease['spec_sha256'] != ORIGINAL_SPEC
                or terminal['exit_code'] or terminal['unresolved'] or terminal['records'] != records
                or terminal['not_started'] != remaining or terminal['physical_requests'] != len(records)
                or terminal['ticks'] != sum(r['ticks'] for r in records)):
            raise PermissionError('The disk-waiting N1 owner must have closed before transferring its remaining slots')
    return records


def validate_member(case, authority):
    row = fixed_cards(authority)[case['panel_entry_id']]
    template = read_pin(row['reference'])
    condition = dict(template['mobile_condition'], collection_episode=case['panel_entry_id'])
    subset = ('mobile_guard_qualification', 'mobile_safety_binding', 'frozen_files', 'task_instruction_id',
              'multitask_geometry', 'policy_from_world', 'model_path', 'd2_probe_motion', 'carry_repair',
              'collection_contract', 'language', 'sim_only_role', 'untouched_legacy_fixed_wait_s')
    if (case['owner'] != OWNER or authority['owner'] != OWNER
            or case['registry_scope'] != SCOPE or case['mobile_world_lock'] != authority['world_lock']
            or case['ROS_DOMAIN_ID'] != 198 or case['cpus'] != authority['cpus']
            or case['site_world_spec']['fixture'] != row['reference']
            or case['max_ticks'] != row['slot']['max_native_ticks']
            or case['mobile_condition'] != condition or case['collection_split'] != 'ENGINEERING'
            or case['expert_training_eligible'] is not False or case['resident_server_spec'] is not None
            or case.get('d2_isolated_live_control') is not None
            or not all(case.get(key) == template.get(key) for key in subset)
            or authority['deadline_epoch'] > authority['goal_start'] + 7 * 86400):
        raise PermissionError('Exact archived N1 reference with the current single-world lease required')


def make_case(spec, row, world_lease, lease_end):
    root = Path(spec['registered_root'])
    identifier = row['slot']['id']
    folder = root / 'cases' / identifier
    case = copy.deepcopy(read_pin(row['reference']))
    for key in ('plan_commit', 'goal_plan_sha', 'source_sha', 'source_root', 'source_files',
                'trusted_source_parent', 'source_seal', 'registry', 'owner', 'host', 'uid', 'world_control',
                'mobile_world_lock', 'ROS_DOMAIN_ID', 'cpus', 'sim_only_project_authorization',
                'campaign_deadline_epoch', 'registered_panel'):
        case[key] = spec[key]
    case.pop('d2_isolated_live_control', None)
    case.pop('d2_world_binding', None)
    case.update(scope=SCOPE, registry_scope=SCOPE, registered_root=str(folder),
                campaign_root=str(root), mobile_group=str(root), unattended_control=str(root / 'control'),
                collection_split='ENGINEERING', collection_root=None, expert_training_eligible=False,
                allocation_deadline_epoch=lease_end, deadline_epoch=lease_end-60, panel_entry_id=identifier,
                site_world_lease=world_lease, site_world_spec=dict(fixture=row['reference']))
    case['mobile_condition']['collection_episode'] = identifier
    for name in ('tmp', 'cache', 'logs', 'data'):
        case[name] = str(folder / name)
    case['native_environment'] = dict(spec['native_environment'], TMPDIR=case['tmp'],
                                      XDG_CACHE_HOME=case['cache'], ROS_LOG_DIR=str(folder/'logs/ros'))
    return folder, case


def measurement(spec, row, case, native, result):
    kind = row['slot']['kind']
    root = Path(case['registered_root'])
    detail = native.get('detail') or {}
    if kind == 'SHORT':
        from scripts.sla.core_multienv_n1_compare import compare
        return compare(spec, row, root, detail)
    if not native['metrics']['task_and_required_motion_success']:
        return dict(status='NOT_ELIGIBLE', first_failure=detail.get('terminal_reason'))
    request = root/'G7_INPUT.private.json'
    write(request, dict(case=pin(root/'EXECUTION_SPEC.json'), native=result, kind='engineering',
                        statistics=spec['C20_statistics_path']))
    launched = subprocess.run(
        [spec['admission_python'], '-B', '-m', 'scripts.sla.core_multienv_n1_compare', str(request), digest(request)],
        cwd=spec['source_root'], env=dict(os.environ, **spec['native_environment']),
        text=True, capture_output=True, check=True, timeout=180)
    from json import loads
    return loads(launched.stdout)


def run(spec):
    root = Path(spec['registered_root'])
    authority = read_pin(spec['sim_only_project_authorization'])
    cards = fixed_cards(authority)
    wave = Path(read_pin(spec['wave'])['registered_root'])
    handoff = Path(read_pin(spec['student_handoff_owner'])['registered_root'])
    world_path = Path(spec['mobile_world_lock'])
    journal = Journal(root/'control/journal', root.name)
    storage = spec.get('n1_archive_storage')
    records = retained_records(spec['retained_n1'],cards,closed=True) if storage else []
    retained_count = len(records)
    pending, failure, code = None, None, 1
    write(root/'SUPERVISOR.private.json', dict(pid=os.getpid(), epoch=time.time(), source_sha=spec['source_sha']))

    def progress(status, **detail):
        write(root/'PROGRESS.private.json', dict(status=status, pid=os.getpid(), epoch=time.time(),
            paid_requests=sum(r['requests'] for r in records), verified_ticks=sum(r['ticks'] for r in records),
            current=pending, **detail))

    def native_launch(row, world):
        acquired = time.time()
        end = min(spec['deadline_epoch'], acquired+2160)
        lease_path = root/'control'/('world-'+pending+'.json')
        inode = os.fstat(world.handle.fileno())
        write(lease_path, dict(kind='CORELAB_LIVE_SINGLE_WORLD_LEASE', owner_pid=os.getpid(), uid=os.getuid(),
            acquired_epoch=acquired, deadline_epoch=end, lock=str(world_path), lock_inode=inode.st_ino,
            lock_device=inode.st_dev, condition=pending))
        folder, case = make_case(spec, row, pin(lease_path), end)
        require_unstarted_cases(spec['registry'], {pending})
        validate(case)
        dirs(folder)
        register(spec, [(folder, case)])
        path = folder/'EXECUTION_SPEC.json'
        command = [case['python'], '-B', str(Path(spec['source_root'])/'scripts/wm02/v6_bootstrap.py'),
                   '--case', str(path), '--world-lock-fd', str(world.handle.fileno())]
        with (folder/'logs/case.log').open('xb') as stream:
            child = subprocess.Popen(command, cwd=spec['source_root'], env=dict(os.environ, **case['native_environment']),
                stdout=stream, stderr=subprocess.STDOUT, pass_fds=(world.handle.fileno(),))
        progress('NATIVE_RUNNING', child=child.pid, spec=pin(path))
        if storage:
            from scripts.sla.core_multienv_payload import measure

            while child.poll() is None:
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if not measure(folder,spec['multienv']['snapshot_reserve_bytes'])['passed'] and not (folder/'STOP.json').exists():
                        write(folder/'STOP.json',dict(reason='N1_LOCAL_RAW_RESERVE_CONSUMED',epoch=time.time()))
        else:
            child.wait()
        return dict(spec=pin(path), result=pin(folder/'RESULT.private.json'), exit_code=child.returncode)

    def stopped():
        return ((root/'STOP.json').exists() or (handoff/'STOP.json').exists()
                or time.time()+2160 >= spec['deadline_epoch'])

    last_wave_completed = -1
    try:
        for pending, row in list(cards.items())[retained_count:]:
            while not stopped():
                wave_completed = read(wave/'PROGRESS.private.json')['completed']
                disk = shutil.disk_usage(root)
                floor = max(50 * 1024**3, int(.15 * disk.total)) + 8 * 1024**3
                have_space = disk.free >= floor+spec['combined_inflight_reserve_bytes']
                if storage:
                    from scripts.sla.core_multienv_n1_storage import capacity

                    have_space = capacity(root,storage,spec['multienv'])['passed']
                mem = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                               if x.startswith('MemAvailable:'))) * 1024
                if (handoff/'STUDENT_TURN_READY.private.json').exists():
                    progress('WAIT_STUDENT_HANDOFF')
                elif not (wave/'RESULT.private.json').exists() and wave_completed <= last_wave_completed:
                    progress('YIELD_TO_WAVE_A', production_completed=wave_completed)
                elif not have_space or mem < 13*1024**3:
                    progress('WAIT_CAPACITY', free_disk=disk.free, available_memory=mem)
                else:
                    try:
                        world = OwnerLock(world_path)
                    except BlockingIOError:
                        progress('WAIT_WORLD', production_completed=wave_completed)
                    else:
                        held_space = not storage or capacity(root,storage,spec['multienv'])['passed']
                        if held_space and not stopped() and not (handoff/'STUDENT_TURN_READY.private.json').exists():
                            break
                        world.close()
                time.sleep(2)
            else:
                pending = None
                break
            try:
                last_wave_completed = wave_completed
                progress('DISPATCH')
                receipt = journal.once(pending, dict(fixture=row['reference'], source=spec['source_sha']),
                                       lambda: native_launch(row, world))
                archive_error = None
                if storage:
                    from scripts.sla.core_multienv_n1_storage import archive

                    folder = root/'cases'/pending
                    try:
                        archive(folder,storage,spec['multienv'])
                    except (OSError,ValueError) as error:
                        archive_error = error
            finally:
                world.close()
            case, native = read_pin(receipt['spec']), read_pin(receipt['result'])
            lease = read(Path(case['registered_root'])/'control/LEASE.json')
            outcome = compact_case(case, native)
            record = dict(id=pending, kind=row['slot']['kind'], requests=native['physical_requests'],
                          ticks=native['actual_native_integrations'], source=spec['source_sha'],
                          terminal=receipt['result'], technical_failure=outcome['technical_failure'],
                          first_failure=outcome['first_failure'], hard_STOP=outcome['hard_STOP'],
                          post_STOP_ticks=outcome['integrations_after_STOP'], profile=native.get('performance'),
                          task_and_mobile_success=native['metrics']['task_and_required_motion_success'])
            records.append(record)
            if storage and archive_error is None:
                record['archive'] = pin(Path(case['registered_root'])/'ARCHIVE.private.json')
            write(root/'RECORDS.private.json', records)
            if archive_error is not None:
                raise RuntimeError('Retaining the paid native condition after an archive failure: '+pending) from archive_error
            if (lease['status'] != 'CLOSED' or lease['spec_sha256'] != receipt['spec']['sha256']
                    or native['source_sha'] != spec['source_sha'] or record['requests'] != 1
                    or record['ticks'] > case['max_ticks'] or fatal_control_violation(outcome)
                    or outcome['technical_failure'] or not record['profile']
                    or native['optimizer_updates'] or native['training_visits'] or native['model_generations']):
                raise RuntimeError('Stop and preserve the affected engineering card: '+pending)
            record['measurement'] = measurement(spec, row, case, native, receipt['result'])
            write(root/'RECORDS.private.json', records)
            pending = None
            progress('CASE_SEALED')
        code = 0
    except Exception:
        failure = traceback.format_exc()
    finally:
        recorded = {r['id'] for r in records}
        write(root/'RESULT.private.json', dict(exit_code=code, failure=failure, unresolved=pending,
              not_started=[cid for cid in cards if cid not in recorded and cid != pending],
              records=records, physical_requests=sum(r['requests'] for r in records),
              ticks=sum(r['ticks'] for r in records), optimizer_updates=0, training_visits=0,
              model_generations=0, training_admissions=0, source=spec['source_sha'],
              carried_requests=retained_count, new_physical_requests=sum(r['requests'] for r in records[retained_count:])))
        close(spec, code)
    return code
