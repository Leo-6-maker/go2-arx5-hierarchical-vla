"""Reconcile the original Wave A ledger before a cold storage handoff."""
import argparse
import json
from pathlib import Path
import subprocess

from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest
from scripts.sla.core_multienv_scope import PLAN, order
from scripts.sla.core_multienv_train import PRODUCER_SOURCE, PRODUCER_SPEC
from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases


def read(path):
    return json.loads(path.read_bytes())


def suffix(ordered, records):
    """Replay the original family/instruction pause rule in roster order."""
    cursor = 0
    streak, paid, paused, pending, skipped = {}, [], set(), [], []
    for slot, card in enumerate(ordered, 1):
        cid = card['id']
        cell = (card['family'], card['instruction'])
        if card['split'] == 'TRAIN' and cell in paused:
            skipped.append(dict(slot=slot, condition=cid, reason='PAUSED_REPEAT_CONTACT_MECHANISM'))
            continue
        if cursor == len(records):
            pending.append(dict(slot=slot, condition=cid, split=card['split'], max_ticks=card['native_max_ticks']))
            continue
        stored = records[cursor]
        if (stored['condition'] != cid or stored['split'] != card['split'] or stored['requests'] != 1
                or not 0 < stored['ticks'] <= card['native_max_ticks'] or stored['technical_failure']
                or (stored['hard_STOP'] and stored['post_STOP_ticks'] != 0)):
            raise ValueError('Original Wave ledger is not an ordered, reconciled prefix: '+cid)
        paid.append(dict(slot=slot, condition=cid, split=card['split'], ticks=stored['ticks'], admitted=stored['admitted']))
        cursor += 1
        if card['split'] == 'TRAIN':
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


def readback(root):
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
    manifest = suffix(ordered, records)
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
    ledger = readback(root)
    cards = refs['conditions']
    if (not cards or cards != [row['condition'] for row in ledger['pending'][:len(cards)]]
            or old['paid_requests'] != len(ledger['paid']) or old['verified_ticks'] != ledger['paid_ticks']):
        raise PermissionError('Cold continuation must use the next qualified original conditions')
    return ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('producer', type=Path)
    options = parser.parse_args()
    print(json.dumps(readback(options.producer.resolve()), sort_keys=True, allow_nan=False))
