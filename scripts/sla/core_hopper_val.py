"""Freeze held-out expert windows at VAL24 completion or 48 h using only native events and labels."""
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from hopper_parent_custody import PLAN, digest, write_once

T0 = 1795236112
POP_SHA = '9e483c85eb90cc2e9d1da66b9f3f8dff6606c6173b66755bcffcc4c7e5f91279'


def checked(pin):
    path = Path(pin['path'])
    if digest(path) != pin['sha256']:
        raise ValueError('Held-out evidence changed: '+path.name)
    return json.loads(path.read_bytes())


def select(record):
    ep = checked(record['episode'])
    terminal = checked(record['result'])
    score = terminal['detail']['score']
    if (not record['admitted'] or record['hard_STOP'] or not record['task_and_mobile_success']
            or ep['split'] != 'VAL' or ep['id'] != record['condition'] or not score['task_success']
            or ep['native_result_sha256'] != record['result']['sha256']):
        raise ValueError('Not an independently admitted VAL episode')
    member = Path(record['episode']['path']).parent.parent / ep['labels']['file']
    if digest(member) != ep['labels']['sha256']:
        raise ValueError('Captured VAL labels changed')
    event = int(terminal['detail']['terminal_native_step']) if ep['task'].startswith('HOLD') else (
        score['events'].get('released', {}).get('native_step'))
    with np.load(member, allow_pickle=False) as lab:
        steps = lab['native_step']; n = len(steps)
        if n != ep['windows'] or np.any(np.diff(steps) <= 0):
            raise ValueError('Nonmonotonic VAL query clock')
        out = []
        positions = {}
        for i, (tick, phase) in enumerate(zip(steps, lab['phase'], strict=True)):
            mask = lab['consumed_mask'][i]
            if (not lab['supervision_mask'][i].all() or not np.array_equal(np.flatnonzero(mask), np.arange(10))
                    or abs(lab['previous_actual_command'][i, 2] - tick*.002) > 1e-8):
                continue
            previous, source_s, query_s = lab['previous_actual_command'][i]
            if not np.isfinite(previous) or abs(query_s - source_s - .050) > 1e-8:
                raise ValueError('Missing previous-command clock')
            widths = np.r_[previous, .002+.040*lab['nominal_policy_H40'][i, :, 9]]
            due = bool(np.any((widths[:-1] >= .022) & (widths[1:] < .022) & mask))
            for name, match in [('approach', phase in ('PREGRASP', 'DESCEND')),
                                ('p10_due_close', due), ('lift_or_carry', phase in ('LIFT', 'CARRY')),
                                ('hold_complete_or_release', event is not None and tick < event <= tick + 25*10)]:
                if match and name not in positions:
                    positions[name] = (i, int(tick))
        used = set()
        for name in ('approach', 'p10_due_close', 'lift_or_carry', 'hold_complete_or_release'):
            if name not in positions:
                continue
            idx, tick = positions[name]
            if idx in used:
                continue
            used.add(idx)
            out.append(dict(criterion=name, index=idx, native_tick=tick, frame=int(lab['frame_index'][idx]),
                            phase=str(lab['phase'][idx]), event_tick=event if name=='hold_complete_or_release' else None))
    return dict(condition=ep['id'], episode_sha256=record['episode']['sha256'],
                labels_sha256=ep['labels']['sha256'], native_sha256=record['result']['sha256'], selected=out)


def run(root):
    cfg = json.loads((root/'MOUNT.private.json').read_bytes())
    producer = Path(cfg['producer'])
    if (cfg['plan'] != PLAN or cfg['T0_epoch'] != T0 or os.getuid() != cfg['uid']
            or digest(Path(__file__)) != cfg['worker_sha256'] or digest(producer/'EXECUTION_SPEC.json') != cfg['producer_spec']
            or digest(root/'POPULATIONS.json') != POP_SHA):
        raise ValueError('Selector source, producer, population or phase origin changed')
    ids = [v['condition_id'] for v in json.loads((root/'POPULATIONS.json').read_bytes())['populations']['VAL']]
    if len(set(ids)) != 24 or [r['condition_id'] for r in cfg['fixed_val']] != ids:
        raise ValueError('Expected the original 24 VAL conditions')
    deadline = T0 + 48*3600
    while True:
        rfile = producer/'RECORDS.private.json'
        rows = json.loads(rfile.read_bytes()) if rfile.exists() else []
        rows = [r for r in rows if r['split']=='VAL' and r['requests']==1]
        now = time.time()
        if (root/'STOP.json').exists():
            write_once(root/'STOPPED.private.json', dict(epoch=now, status='STOPPED_BEFORE_FREEZE'))
            return
        if any(r['condition'] not in ids for r in rows) or len({r['condition'] for r in rows}) != len(rows):
            raise ValueError('Unexpected or repeated held-out native request')
        processed = [r for r in rows if r['admitted'] or not r['task_and_mobile_success'] or r['first_failure'].startswith('ADMISSION:')]
        if len(processed) == 24 or now >= deadline:
            terminal_times = [Path(r['result']['path']).stat().st_mtime for r in processed]
            all_finished = len(terminal_times) == 24 and max(terminal_times) <= deadline
            cutoff = max(terminal_times) if all_finished else deadline
            by_id = {r['condition']:r for r in rows if Path(r['result']['path']).stat().st_mtime <= cutoff}
            episodes = []
            conditions = []
            for eid in ids:
                row = by_id.get(eid)
                status = 'NO_TERMINAL_BY_FREEZE'
                if row:
                    status = ('SUCCESS_WITHOUT_ADMISSION_AT_FREEZE' if row['task_and_mobile_success']
                              else row['first_failure'])
                    if row['first_failure'].startswith('ADMISSION:'):
                        status = row['first_failure']
                    if row.get('admitted') and Path(row['episode']['path']).stat().st_mtime <= deadline:
                        episodes.append(select(row))
                        status = 'ADMITTED'
                conditions.append(dict(condition=eid, status=status))
            case_dir = producer/'cases'
            started = set(by_id)
            for eid in ids:
                receipt = case_dir/eid/'native-01/BEFORE_FIRST_INTEGRATION.json'
                if receipt.is_file() and receipt.stat().st_mtime <= cutoff:
                    started.add(eid)
            started = sorted(started)
            write_once(root/'PANEL.private.json', dict(plan=PLAN, split='VAL', selector=cfg['source'],
                       selector_sha256=cfg['worker_sha256'], population_sha256=POP_SHA, registered_attempts=24,
                       launched_observed=len(started), launched_ids_by_freeze=started, actual_terminal_by_freeze=len(by_id),
                       admitted=len(episodes), selected_windows=sum(len(ep['selected']) for ep in episodes),
                       conditions=conditions, episodes=episodes, freeze_epoch=cutoff, read_epoch=now,
                       freeze_trigger='VAL24_TERMINAL' if all_finished else 'T0_PLUS_48H',
                       old_TEST_read=False, optimization_visits=0, model_generations=0, native_dispatched=0))
            return
        pulse = dict(epoch=now, pid=os.getpid(), registered=24, processed=len(processed), deadline_epoch=deadline)
        tmp = root/'heartbeat.pending'; tmp.write_text(json.dumps(pulse)); os.replace(tmp,root/'HEARTBEAT.private.json')
        time.sleep(min(45, max(.1, deadline-time.time())))


if __name__ == '__main__':
    os.umask(0o077)
    root=Path.cwd()
    with (root/'OWNER.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(root)
        except Exception as exc:
            write_once(root/'FAILURE.private.json', dict(error=str(exc), epoch=time.time()))
            raise
