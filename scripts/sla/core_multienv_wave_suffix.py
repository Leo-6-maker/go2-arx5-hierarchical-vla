"""Reconcile the original Wave A ledger before a cold storage handoff."""
import argparse
import json
from pathlib import Path
import subprocess

from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest
from scripts.pi0294.e1.journal import Journal
from scripts.sla.core_multienv_scope import PLAN, order
from scripts.sla.core_multienv_train import PRODUCER_SOURCE, PRODUCER_SPEC
from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases


def read(path):
    return json.loads(path.read_bytes())


def suffix(ordered, records, *, storage_timeout=None, reconciled=()):
    """Replay the original family/instruction pause rule in roster order."""
    technical = set(reconciled) | ({storage_timeout} if storage_timeout else set())
    cursor = 0
    streak, paid, paused, pending, skipped = {}, [], set(), [], []
    for slot, card in enumerate(ordered, 1):
        cid = card['id']
        cell = (card['family'], card['instruction'])
        settled = cursor < len(records) and records[cursor]['condition'] == cid and cid in technical
        if card['split'] == 'TRAIN' and cell in paused and not settled:
            skipped.append(dict(slot=slot, condition=cid, reason='PAUSED_REPEAT_CONTACT_MECHANISM'))
            continue
        if cursor == len(records):
            pending.append(dict(slot=slot, condition=cid, split=card['split'], max_ticks=card['native_max_ticks']))
            continue
        stored = records[cursor]
        if (stored['condition'] != cid or stored['split'] != card['split'] or stored['requests'] != 1
                or not (0 if settled else 1) <= stored['ticks'] <= card['native_max_ticks']
                or bool(stored['technical_failure']) != (cid in technical)
                or (stored['hard_STOP'] and stored['post_STOP_ticks'] != 0)):
            raise ValueError('Original Wave ledger is not an ordered, reconciled prefix: '+cid)
        paid.append(dict(slot=slot, condition=cid, split=card['split'], ticks=stored['ticks'], admitted=stored['admitted']))
        cursor += 1
        if card['split'] == 'TRAIN' and cid not in technical:
            if stored['admitted']:
                streak.pop(cell, None)
            else:
                signature = stored.get('first_failure_signature')
                previous, count = streak.get(cell, (None, 0))
                count = count + 1 if signature is not None and signature == previous else 1
                streak[cell] = signature, count
                if count >= 2:
                    paused.add(cell)
    if cursor != len(records):
        raise ValueError('Original ledger contains an unknown or already paused request')
    checkpoints = [dict(family=cell[0], instruction=cell[1], signature=signature, count=n)
                   for cell, (signature,n) in sorted(streak.items())]
    return dict(paid=paid, pending=pending, paused=skipped, retained_paused_cells=sorted(paused),
                retained_contact_streak=checkpoints,
                paid_ticks=sum(r['ticks'] for r in paid),
                paid_train=sum(r['split']=='TRAIN' for r in paid),
                admitted_train=sum(r['split']=='TRAIN' and r['admitted'] for r in paid),
                pending_tick_cap=sum(x['max_ticks'] for x in pending))


def readback(root, timeout=None, *, reconciled=()):
    spec_path = root/'EXECUTION_SPEC.json'
    spec = read(spec_path)
    if (digest(spec_path) != PRODUCER_SPEC or spec['registered_root'] != str(root)
            or spec['goal_plan_sha'] != PLAN or spec['source_sha'] != PRODUCER_SOURCE):
        raise PermissionError('Expected the original PR425 Wave A collector')
    progress_path = root/'PROGRESS.private.json'
    records_path = root/'RECORDS.private.json'
    progress = read(progress_path)
    raw = records_path.read_bytes()
    records = json.loads(raw)
    if progress['current'] is not None or progress['completed'] != len(records):
        raise RuntimeError('Original Wave A has an unreconciled condition')
    population = read_pin(spec['compiled_population'])
    ordered = order(population)
    settled = [*([timeout] if timeout is not None else []),*reconciled]
    manifest = suffix(ordered, records+settled, reconciled=[row['condition'] for row in settled])
    allowed = {r['condition'] for r in manifest['pending'] + manifest['paused']}
    require_unstarted_cases(spec['registry'], allowed)
    for r in records:
        c = root/'cases'/r['condition']
        lease = read(c/'control'/'LEASE.json')
        native = read_pin(r['result'])
        if (lease['status'] != 'CLOSED' or lease['spec_sha256'] != digest(c/'EXECUTION_SPEC.json')
                or Path(r['result']['path']) != c/'RESULT.private.json' or native['source_sha'] != PRODUCER_SOURCE
                or native['physical_requests'] != 1 or native['actual_native_integrations'] != r['ticks']):
            raise ValueError('Original case terminal identity or denominator changed: '+r['condition'])
    if (records_path.read_bytes() != raw or read(progress_path)['current'] is not None
            or any((root/'cases'/cid).exists() for cid in allowed)):
        raise RuntimeError('Original producer advanced during the readback')
    return dict(status='COLD_SUFFIX_LEDGER_RECONCILED', original_spec=PRODUCER_SPEC,
                original_records_sha256=digest(records_path), population=spec['compiled_population']['sha256'],
                original_running_progress=progress['stage'], original_paid_requests=len(records),
                cold_owner_registered=False, new_native_requests=0, new_native_ticks=0, **manifest)


def closed_storage_timeout(origin, refs):
    """Account for the first NFS case after its wall-time failure and clean teardown."""
    from .core_multienv_payload import LIVE_CONTROL

    previous = read_pin(refs['spec'])
    terminal = read_pin(refs['terminal'])
    root = Path(previous['registered_root'])
    first, next_case = previous['multienv_cold_handoff']['conditions']
    records, = terminal['records']
    case = root/'cases'/first
    outcome = Journal(root/'control/journal',root.name).completed(first,dict(
        condition=first, population=previous['compiled_population']['sha256'],source=previous['source_sha']))
    member, native = read_pin(outcome['spec']), read_pin(outcome['result'])
    group_lease, child_lease = (read(p/'control/LEASE.json') for p in (root,case))
    state = subprocess.check_output(['systemctl','--user','show','wm02-'+root.name+'.service',
                                    '-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],text=True)
    raw = case/'native-01'
    trace = native['trace_audit']
    catalog = read_pin(refs['catalog_terminal'])
    if (refs['spec']['sha256'] != '9dae4fe8021718b2b1d45115487ba2befe340442293e9b62f0aa6f9bd78718c0'
            or refs['terminal']['sha256'] != 'a38b13f6cf9c815fadeebd8f4563cbe6fd9e5de2e61696d6000ede80efa7f532'
            or previous['multienv_cold_handoff']['producer'] != origin
            or Path(refs['spec']['path']) != root/'EXECUTION_SPEC.json'
            or Path(refs['terminal']['path']) != root/'RESULT.private.json'
            or dict(x.split('=',1) for x in state.splitlines()) != {'MainPID':'0','ActiveState':'failed','ExecMainStatus':'1'}
            or group_lease['status'] != 'CLOSED' or group_lease['spec_sha256'] != refs['spec']['sha256']
            or child_lease['status'] != 'CLOSED' or child_lease['spec_sha256'] != outcome['spec']['sha256']
            or terminal['unreconciled'] != first or terminal['records'] != read(root/'RECORDS.private.json')
            or terminal['source_sha'] != native['source_sha'] or native['source_sha'] != previous['source_sha']
            or Path(outcome['spec']['path']) != case/'EXECUTION_SPEC.json'
            or Path(outcome['result']['path']) != case/'RESULT.private.json' or outcome['result'] != records['result']
            or terminal['exit_code'] != outcome['exit'] or outcome['exit'] != child_lease['worker_exit_code'] or outcome['exit'] != 1
            or member['campaign_root'] != str(root) or member['panel_entry_id'] != first or records['condition'] != first
            or member['multienv_payload_storage'].get('live_control') == LIVE_CONTROL
            or any((p/'STOP.json').exists() for p in (root,case,raw)) or (root/'cases'/next_case).exists()
            or not native['failure'].strip().endswith('TimeoutError: AUTO1_FINITE_ENDPOINT_OR_OPERATOR_STOP')
            or 'after=adapter.advance_to(step+25)' not in native['failure']
            or native['wall_s'] < member['wall_limit_s'] or child_lease['closed_epoch'] >= member['deadline_epoch']
            or not native['shutdown_clean'] or not native['technical_failure'] or native['unsafe']
            or records['hard_STOP'] or records['admitted'] or terminal['admitted_train']
            or native['physical_requests'] != records['requests'] or records['requests'] != terminal['paid_requests'] or terminal['paid_requests'] != 1
            or native['actual_native_integrations'] != records['ticks'] or records['ticks'] != terminal['verified_ticks']
            or native['actual_native_integrations'] != trace['native_step_records'] or trace['status'] != 'RECORDED_TRACE_CHECKS_PASS'
            or trace['STOP_observed'] or trace['errors'] or len(trace['input_files_sha256']) != 3
            or any(digest(raw/name) != sha for name,sha in trace['input_files_sha256'].items())
            or not catalog['producer_closed'] or catalog['source'] != previous['source_sha'] or catalog['paid_TRAIN'] != 24
            or catalog['new_admitted'] != 10 or catalog['sealed_new_members'] != 10 or catalog['latest']['count'] != 69):
        raise PermissionError('The storage timeout must be fully reconciled without reusing its paid card')
    return records, next_case, previous['deadline_epoch']


def closed_storage_chain(origin, refs):
    """Settle the paid 24–26 prefix while retaining both timeouts and the launch failure."""
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
    from .core_multienv_red_hold_pilot import SLOT, SLOT25, SLOT25_SPEC, after_slot25

    wave, pilot = read_pin(refs['wave25']), read_pin(refs['pilot'])
    handoff = wave['multienv_cold_handoff']
    if (refs['wave25']['sha256'] != SLOT25_SPEC or handoff['producer'] != origin
            or handoff['conditions'] != [SLOT25] or pilot['reopen']['slot25'] != refs['wave25']):
        raise PermissionError('Expected the exact Wave A slot 25 then slot 26 registration')
    timeout, expected, _ = closed_storage_timeout(origin,handoff['closed_storage_predecessor'])
    closed = after_slot25(wave,refs['wave25'])
    last = read_pin(refs['wave25_terminal'])
    catalog = read_pin(refs['wave25_catalog'])
    if (expected != SLOT25 or not closed or closed['terminal'] != refs['wave25_terminal']
            or closed['admitted'] or not last['records'][0]['technical_failure']
            or not catalog['producer_closed'] or catalog['source'] != wave['source_sha']
            or catalog['paid_TRAIN'] != 25 or catalog['new_admitted'] != 10
            or catalog['sealed_new_members'] != 10 or catalog['latest']['count'] != 69):
        raise PermissionError('The second timeout and its finished catalog must agree on the paid denominator')
    root = Path(pilot['registered_root'])
    case = root/'cases'/SLOT
    terminal = read_pin(refs['pilot_terminal'])
    result = terminal['result']
    native = read_pin(result['native'])
    journal = read(root/'control'/'journal'/(SLOT+'.json'))
    lease, child_lease = (read(p/'control/LEASE.json') for p in (root,case))
    state = unit_state('wm02-'+root.name)
    if (refs['pilot']['sha256'] != '5559d892531efc06e43f98848f70356303744613efc076310416dd2764af6894'
            or refs['pilot_terminal']['sha256'] != '3ff311fafd9030a329309f4f90e3e01caf9a3ef0ca9d079f2bade1d6e26efe43'
            or Path(refs['pilot']['path']) != root/'EXECUTION_SPEC.json' or Path(refs['pilot_terminal']['path']) != root/'RESULT.private.json'
            or state['MainPID'] != '0' or state['ActiveState'] != 'failed' or lease['status'] != 'CLOSED'
            or lease['spec_sha256'] != refs['pilot']['sha256'] or child_lease['status'] != 'CLOSED'
            or child_lease['spec_sha256'] != digest(case/'EXECUTION_SPEC.json') or journal['status'] != 'COMPLETE'
            or journal['result']['native'] != result['native'] or journal['result']['exit'] != child_lease['worker_exit_code']
            or Path(result['native']['path']) != case/'RESULT.private.json' or terminal['exit_code'] != 1
            or terminal['condition'] != SLOT or result['admitted'] or terminal['native_requests'] != 1
            or not native['technical_failure'] or native['unsafe'] or native['physical_requests'] != 1
            or native['actual_native_integrations'] or native['owned_process_exits']
            or (case/'native-01'/'physics.jsonl').exists() or (case/'native-01'/'LIVE_CONTROL.private.json').exists()
            or not native['failure'].strip().endswith('PermissionError: Expected the registered private Wave A NFS raw output')):
        raise PermissionError('Retain the paid slot 26 launch failure with independently verified zero physics')
    failed_pilot = dict(condition=SLOT,split='TRAIN',requests=1,ticks=0,technical_failure=True,
                        hard_STOP=False,post_STOP_ticks=None,admitted=False)
    return timeout,last['records'][0],failed_pilot


def closed_handoff(spec):
    """Verify a cold predecessor terminal before handing an original prefix to a new owner."""
    refs = spec['multienv_cold_handoff']
    origin = refs['producer']
    previous = read_pin(origin)
    root = Path(previous['registered_root'])
    terminal = refs['terminal']
    local = Path(spec['registered_root'])
    frozen = ('compiled_population', 'registered_panel', 'registry', 'mobile_world_lock',
              'goal_plan_sha', 'interface_predecessor', 'campaign_deadline_epoch', 'C20_statistics_path',
              'owner', 'host', 'uid', 'cpus', 'ROS_DOMAIN_ID')
    if (origin['sha256'] != PRODUCER_SPEC or Path(origin['path']) != root/'EXECUTION_SPEC.json'
            or Path(terminal['path']) != root/'RESULT.private.json' or local == root
            or spec['deadline_epoch'] > previous['deadline_epoch'] or not spec.get('multienv_payload_storage')
            or any(spec[key] != previous[key] for key in frozen) or spec['multienv'] != previous['multienv']):
        raise PermissionError('Cold continuation changed an original site or condition binding')
    old = read_pin(terminal)
    lease = read(root/'control'/'LEASE.json')
    status = subprocess.check_output(['systemctl', '--user', 'show', 'wm02-'+root.name+'.service',
                                      '-p', 'MainPID', '-p', 'ActiveState'], text=True)
    state = dict(row.split('=',1) for row in status.splitlines())
    if (state != {'MainPID':'0','ActiveState':'inactive'} or lease['status'] != 'CLOSED'
            or lease['spec_sha256'] != PRODUCER_SPEC or old['exit_code'] or old['unreconciled']
            or old['source_sha'] != PRODUCER_SOURCE or not (root/'STOP.json').is_file()
            or old['records'] != read(root/'RECORDS.private.json')):
        raise PermissionError('Expected a clean predecessor handoff and released original owner')
    cards = refs['conditions']
    timeout = None
    additional = ()
    if 'closed_storage_chain' in refs:
        if spec.get('multienv_local_archive') is not True or 'closed_storage_predecessor' in refs:
            raise PermissionError('The settled suffix requires the verified local raw path')
        timeout,*additional = closed_storage_chain(origin,refs['closed_storage_chain'])
    if 'closed_storage_predecessor' in refs:
        from .core_multienv_payload import LIVE_CONTROL

        timeout, next_case, deadline = closed_storage_timeout(origin,refs['closed_storage_predecessor'])
        if cards != [next_case] or spec['deadline_epoch'] > deadline or spec['multienv_payload_storage']['live_control'] != LIVE_CONTROL:
            raise PermissionError('Only the unopened second card and original time window can follow the storage timeout')
    ledger = readback(root,timeout,reconciled=additional)
    settled = [*([timeout] if timeout else []),*additional]
    if (not cards or cards != [row['condition'] for row in ledger['pending'][:len(cards)]]
            or old['paid_requests']+len(settled) != len(ledger['paid'])
            or old['verified_ticks']+sum(row['ticks'] for row in settled) != ledger['paid_ticks']):
        raise PermissionError('Cold continuation must use the next qualified original conditions')
    return ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('producer', type=Path)
    options = parser.parse_args()
    print(json.dumps(readback(options.producer.resolve()), sort_keys=True, allow_nan=False))
