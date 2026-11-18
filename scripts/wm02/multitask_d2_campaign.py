"""Bounded PR 411 mobile world dispatch over the existing durable result journal."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from scripts.pi0292.durable import OwnerLock, write
from scripts.pi0294.e1.journal import Journal
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
from go2arx5_vla.world_model.multitask_scope_d2 import PLAN, SCOPE, validate
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest
from .mobile_boundary_runner_v1 import registered
from .multitask_teacher_campaign_v6 import teacher_row
from .sim_only_campaign_v51 import compact_case, fatal_control_violation


def read(path):
    return json.loads(Path(path).read_bytes())


def matrix_still_possible(cases, sealed, minimum=2):
    """Pending and undispatched count as remaining; every sealed failure is final."""
    remaining = {}
    successes = {}
    for case in cases:
        cell = case['mobile_condition']['family'], case['task_instruction_id']
        row = sealed.get(case['panel_entry_id'])
        remaining[cell] = remaining.get(cell,0) + (row is None)
        # An eligible episode has an independent G7 receipt, not just task
        # score; scientific success cannot compensate for invalid capture.
        successes[cell] = successes.get(cell,0) + bool(row and row.get('task_and_mobile_success')
                                                        and row.get('G7_pass'))
    return all(successes[cell]+n >= minimum for cell,n in remaining.items())


def paired_dispatch_allowed(case, active_cases):
    """Start each registered STOP pair after the previous owned pair exits."""
    pair = case.get('d2_stop_pair')
    return all(c.get('d2_stop_pair')==pair for c in active_cases)


def verify_g7_capture(case, result):
    from scripts.sla.multitask_g7_capture_v6 import copy_shards, read_capture

    started = time.monotonic()
    try:
        condition = {'instruction':case['task_instruction_id']}
        source, manifest, _, _, arrays = read_capture(condition,case,result)
        shards = copy_shards(source,manifest,arrays)
        exclusion = result['detail']['label_margin_exclusions']
        audit = dict(passed=exclusion==0, full_H40_rows=int(arrays['supervision_mask'].sum()),
                     actual_P10_atoms=int(arrays['consumed_mask'].sum()),
                     current_RGB_pairs=sum(s['frames'] for s in shards),
                     training_margin_exclusions=exclusion,
                     first_failure='RAW_G7_TRAINING_MARGIN' if exclusion else None)
    except (ValueError, KeyError, OSError, TypeError) as error:
        audit = dict(passed=False, first_failure=repr(error))
    audit.update(wall_s=time.monotonic()-started, statistics_fitted=False,
                 new_native_integrations=0, new_model_generations=0)
    return audit


def g7_receipt(case, case_path, result_sha):
    """Use the independently sealed SLA CPU interpreter, after native exit."""
    runtime = case['d2_g7_runtime']
    if digest(runtime['python']) != runtime['python_sha256']:
        return dict(passed=False,first_failure='G7_PINNED_CPU_RUNTIME_CHANGED')
    source = Path(case['source_root'])/'scripts/wm02/multitask_d2_bootstrap.py'
    until = min(120., case['deadline_epoch']-time.time()-10.)
    if until <= 0.:
        return dict(passed=False,first_failure='G7_FINITE_AUDIT_WINDOW_EXHAUSTED')
    command = ['taskset','-c',','.join(map(str,case['cpus'])), runtime['python'],
               '-B',str(source),'--audit-g7',case_path,result_sha]
    environment = dict(os.environ,**runtime['environment'])
    try:
        result = subprocess.run(command,env=environment,text=True,capture_output=True,timeout=until)
        if result.returncode:
            audit = dict(passed=False,first_failure='G7_CPU_AUDIT_FAILED',error=result.stderr[-2500:])
        else:
            audit = json.loads(result.stdout)
    except (OSError,subprocess.TimeoutExpired,ValueError) as error:
        audit = dict(passed=False,first_failure=repr(error))
    receipt = Path(case['registered_root'])/'D2_G7_ACTUAL.private.json'
    write(receipt,dict(audit,native_result_sha256=result_sha))
    return dict(audit,sha256=digest(receipt))


def preflight(spec):
    root = Path(spec['registered_root'])
    binding = read_pin(spec['d2_world_binding'])
    if (spec['scope'] != SCOPE or spec['plan_commit'] != PLAN
            or spec['campaign_root'] != str(root) or spec['new_model_generations_budget']
            or spec['new_training_visits_budget'] or spec['new_optimizer_updates_budget']
            or not time.time() < spec['deadline_epoch'] <= min(spec['allocation_deadline_epoch'],
                                                              time.time()+43200)
            or binding['original_global_lock'] != spec['mobile_world_lock']):
        raise PermissionError('D2_FIXED_FINITE_PARENT')
    cards = [(str(Path(path)),read(path)) for path in spec['cases']]
    if len(cards) != spec['new_physical_requests_budget'] or not cards:
        raise PermissionError('D2_ACTUAL_CASE_CAP')
    if [c['panel_entry_id'] for _,c in cards] != [c['id'] for c in read_pin(spec['registered_panel'])]:
        raise PermissionError('D2_EXACT_REGISTERED_CASE_ORDER')
    stage = spec['d2_stage']
    limits = {'E':64,'D':24,'Q':48}
    if len(cards) > limits[stage]:
        raise PermissionError('D2_PREREGISTERED_STAGE_BOUND')
    if stage != 'E':
        # A scientific manifest needs its own interface, stage advancement,
        # and independent exporter entry. An engineering probe is never a
        # scientific admission.
        gate = read_pin(spec['d2_science_admission'])
        if gate['stage'] != stage or not gate['interface_and_stage_entry_pass'] or gate['source_sha'] != spec['source_sha']:
            raise PermissionError('D2_REAL_STAGE_AND_ACTUAL_INTERFACE_ADMISSION')
    for _,case in cards:
        validate(case,before_spawn=True)
        for key in ('source_sha','scope','campaign_root','d2_world_binding','d2_stage','registered_panel'):
            if case[key] != spec[key]:
                raise PermissionError('D2_CASE_OUTSIDE_BATCH_BINDING')
    return root,binding,cards


def run(spec_path, spec):
    registered(spec,spec_path)
    root,binding,cards = preflight(spec)
    wall = time.monotonic()
    journal = Journal(root/'control/journal',root.name)
    rows, launched = {}, {}
    fatal = None
    active = {}
    pending = iter(enumerate(cards))
    head = next(pending,None)

    def progress(phase):
        write(root/'PROGRESS.private.json',dict(phase=phase,epoch=time.time(),source_sha=spec['source_sha'],
            deadline_epoch=spec['deadline_epoch'],launched=dict(launched),sealed=list(rows.values()),
            current=[case['panel_entry_id'] for _,case in active.values()]))

    def finish_case(case_index, path, case, fd):
        destination = Path(case['registered_root'])
        command = ['taskset','-c',','.join(map(str,case['cpus'])),case['python'],'-B',
                   str(Path(spec['source_root'])/'scripts/wm02/multitask_d2_bootstrap.py'),
                   '--case',path,str(fd)]

        def execute():
            with (destination/'logs/worker.log').open('xb') as log:
                worker = subprocess.Popen(command,cwd=spec['source_root'],
                    env=dict(os.environ,**case['native_environment']), stdout=log,stderr=subprocess.STDOUT,
                    pass_fds=(fd,))
            launched[case['panel_entry_id']] = worker.pid
            write(destination/'DISPATCH_ACTUAL.private.json',dict(pid=worker.pid,epoch=time.time(),
                  world=case['d2_world_id'],source_sha=case['source_sha'],spec_sha256=digest(path)))
            code = worker.wait()
            result = destination/'RESULT.private.json'
            if not result.is_file():
                raise RuntimeError('D2_NATIVE_TERMINAL_WITHOUT_RESULT_NO_REPLAY')
            return dict(exit_code=code,path=str(result),sha256=digest(result))

        pin = journal.once(f'CASE-{case_index:03d}',dict(spec_sha256=digest(path)),execute)
        result = read_pin({key:pin[key] for key in ('path','sha256')})
        reducer = compact_case if case['sim_only_role']=='INTERFACE' else teacher_row
        row = reducer(case,result)
        performance = result.get('performance')
        row.update(world=case['d2_world_id'],cpus=case['cpus'],
                   result_sha256=pin['sha256'],performance=performance,
                   protocol=case.get('d2_probe_motion'), probe_summary=result.get('detail')
                   if case['d2_stage']=='E' else None)
        if case['d2_stage']!='E' and row['task_and_mobile_success'] and row['completed']:
            row['G7']=g7_receipt(case,path,pin['sha256'])
            row['G7_pass']=row['G7']['passed']
        return row

    try:
        with ExitStack() as stack:
            # Preserve compatibility with legacy project actors. The parent
            # owns the old shared lock and the child inherits only its own
            # registered world lock, never the sibling lock.
            stack.enter_context(closing(OwnerLock(binding['original_global_lock'])))
            owners = {w['id']:stack.enter_context(closing(OwnerLock(w['mobile_world_lock'])))
                      for w in binding['slots']}
            write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),epoch=time.time(),
                                                      source_sha=spec['source_sha']))
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=len(owners)))
            while head is not None or active:
                stopped = (root/'STOP.json').exists()
                deadline = time.time() >= spec['deadline_epoch']-spec['case_lead_s']
                futile = spec['d2_stage']=='Q' and not matrix_still_possible([c for _,c in cards],rows)
                if not (stopped or deadline or futile or fatal):
                    busy = {case['d2_world_id'] for _,case in active.values()}
                    while head is not None:
                        case_index,(path,case) = head
                        slot = case['d2_world_id']
                        if slot in busy or not paired_dispatch_allowed(
                                case, (item for _,item in active.values())):
                            break
                        future = executor.submit(finish_case,case_index,path,case,owners[slot].handle.fileno())
                        active[future] = (case_index,case)
                        busy.add(slot)
                        head = next(pending,None)
                progress('IN_FLIGHT' if active else 'NO_MORE_DISPATCH')
                if not active:
                    break
                sealed, _ = wait(active,timeout=10,return_when=FIRST_COMPLETED)
                for future in sealed:
                    _,case = active.pop(future)
                    key=case['panel_entry_id']
                    try:
                        row = future.result()
                        rows[key] = row
                        if fatal_control_violation(row):
                            fatal = 'D2_ORIGINAL_STOP_OR_TICK_VIOLATION'
                            if not (root/'STOP.json').exists():
                                write(root/'STOP.json',dict(reason=fatal))
                        elif row['technical_failure']:
                            fatal = 'D2_PRESERVED_TECHNICAL_FAILURE_NO_AUTO_REPLAY'
                        elif spec['d2_stage']=='E' and (not row['completed'] or row['hard_STOP']
                                                       or not row['interface_protocol_pass']):
                            fatal = 'D2_ENGINEERING_CASE_FAILED_NO_PROMOTION'
                    except BaseException:
                        fatal = traceback.format_exc()
                write(root/'RECORDS.private.json',[rows[c['panel_entry_id']] for _,c in cards
                      if c['panel_entry_id'] in rows])
    except BaseException:
        fatal = traceback.format_exc()
    for _,case in cards:
        path = Path(case['registered_root'])
        if (case['panel_entry_id'] not in launched and not (path/'SUPERVISOR.private.json').exists()
                and read(path/'control/LEASE.json')['status']=='ACTIVE'):
            close(case,0,not_started=True,reason='D2_UNSTARTED_FINITE_SUFFIX')
    ordered = [rows[c['panel_entry_id']] for _,c in cards if c['panel_entry_id'] in rows]
    code = int(bool(fatal) or len(rows)!=len(cards))
    write(root/'RESULT.private.json',dict(schema='SLA_D2_MOBILE_BATCH_ACTUAL',source_sha=spec['source_sha'],
          failure=fatal,exit_code=code,records=ordered,launched_count=len(launched),
          new_verified_native_requests=sum(r['actual_native_requests'] for r in ordered),
          native_ticks=sum(r['native_ticks'] for r in ordered),
          unresolved_case_launches=len(launched)-len(rows),unstarted=len(cards)-len(launched),
          hard_STOP=sum(r['hard_STOP'] for r in ordered),technical=sum(r['technical_failure'] for r in ordered),
          stage=spec['d2_stage'],profile=spec['d2_profile'],wall_s=time.monotonic()-wall,
          optimizer_updates=0,training_visits=0,generations=0,hardware_requests=0))
    close(spec,code)
    return code
