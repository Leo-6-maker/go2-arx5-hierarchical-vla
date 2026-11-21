"""Write a bounded Core-native accounting receipt from the actual expert and student owners."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import os
from pathlib import Path
import time

from scripts.pi0292.durable import OwnerLock, read
from scripts.sla.hopper_parent_custody import PLAN, digest, write_once
from scripts.sla.hopper_site import DEADLINE, T0, pinned


def counts(cases):
    result = Counter(dict.fromkeys(('confirmed_requests','native_terminals','terminal_ticks',
        'terminal_model_generations','unreconciled_cases','reconciled','scored','success','admitted',
        'hard_STOP','STOP_with_tick_evidence','post_STOP_ticks'),0))
    reasons = Counter()
    for row in cases:
        result['confirmed_requests'] += row['requests']
        if not row['terminal']:
            result['unreconciled_cases'] += 1
            result['open_confirmed_requests'] += row['requests']
            result['launched_pending_request_evidence'] += int(row['requests']==0)
            continue
        result['native_terminals'] += 1
        result['terminal_ticks'] += row['ticks']
        result['terminal_model_generations'] += row['generations']
        if row.get('expert_queries') is not None:
            result['expert_queries'] += row['expert_queries']
            result['expert_queries_measured_terminals'] += 1
        if not row['reconciled']:
            result['unreconciled_cases'] += 1
        if row['scored']:
            result['scored'] += 1
            result['score_through_terminal'] += int(row['score_tick']==row['ticks'])
            result['score_ends_before_terminal'] += int(row['score_tick']<row['ticks'])
            for key in ('grasp', 'lift20', 'base80', 'hold1s', 'correct_region', 'release', 'retreat50'):
                result[key] += row[key]
        if row['reconciled']:
            result['reconciled'] += 1
            result['success'] += row['success']
            result['hard_STOP'] += row['hard_STOP']
            if row['hard_STOP'] and row['post_STOP_ticks'] is not None:
                result['STOP_with_tick_evidence'] += 1
                result['post_STOP_ticks'] += row['post_STOP_ticks']
            result['admitted'] += row['admitted']
            reasons[row['outcome']] += 1
    return dict(**result,first_outcomes=dict(reasons))


def observe(folder, source, split, condition, record, student=False):
    card = folder/'EXECUTION_SPEC.json'
    native = folder/'RESULT.private.json'
    first = folder/'native-01/BEFORE_FIRST_INTEGRATION.json'
    if not native.is_file() and not first.is_file() and not (folder/'SUPERVISOR.private.json').is_file():
        return None
    spec = read(card)
    if (spec['source_sha'] != source or spec['registered_root'] != str(folder)
            or spec['panel_entry_id'] != condition or spec['hardware_connections_allowed']
            or spec['new_physical_requests_budget'] != 1):
        raise ValueError('Case source, identity or scope differs from its registered owner')
    row = dict(id=condition,split=split,family=spec['mobile_condition']['family'],
               instruction=spec['task_instruction_id'],source=source,
               requests=int(first.is_file()),terminal=native.is_file())
    if student:
        row['endpoint'] = spec['arm']
    if not row['terminal']:
        progress = folder/'native-01/MULTITASK_SCORE_PROGRESS.private.json'
        row['last_score_tick'] = read(progress)['native_step'] if progress.is_file() else None
        return row
    actual = read(native)
    sha = digest(native)
    paid = actual['physical_requests']
    if (paid not in (0,1) or paid < row['requests'] or actual['source_sha'] != source
            or not 0 <= actual['actual_native_integrations'] <= spec['max_ticks']):
        raise ValueError('Inconsistent native terminal or budget')
    row.update(terminal_sha256=sha,requests=paid,ticks=actual['actual_native_integrations'],
               generations=actual['model_generations'],reconciled=record is not None)
    detail = actual.get('detail') or {}
    row['expert_queries'] = detail.get('expert_queries')
    capture = folder/'native-01/LIVE_DATA/MANIFEST.private.json'
    if not student and capture.is_file():
        cap = read(capture)
        if (cap['schema']!='MFREE_LIVE_FINAL_PLAN_CAPTURE_V1' or cap['final_native_step']!=row['ticks']
                or cap['frozen_scene_sha256']!=spec['frozen_files'][spec['model_path']]
                or cap['queries']<0 or cap['planned_suffix_is_executed']):
            raise ValueError('Expert capture does not match the native terminal')
        row['capture_sha256'] = digest(capture)
        row['captured_queries'] = cap['queries']
        if not cap['writer_error']:
            if row['expert_queries'] is not None and row['expert_queries']!=cap['queries']:
                raise ValueError('Terminal and actual capture query counts differ')
            row['expert_queries'] = cap['queries']
    score = detail.get('score')
    if record is not None:
        if (record['result_sha256'] if student else record['result']['sha256']) != sha:
            raise ValueError('Owner and native terminal digests differ')
        if not student and (record['ticks'] != row['ticks'] or record['requests'] != paid):
            raise ValueError('Owner and native denominator differ')
        score = record.get('score') or score
        failure = record.get('first_failure')
        success = bool(record['task_and_mobile_success'])
        row.update(success=success,hard_STOP=record['hard_STOP'],
                   post_STOP_ticks=record.get('integrations_after_STOP' if student else 'post_STOP_ticks'),
                   admitted=False if student else record['admitted'],
                   outcome='TASK_SUCCESS' if success and (student or record['admitted']) else (failure or 'UNCLASSIFIED'))
    score_file = folder/'native-01/MULTITASK_SCORE_PROGRESS.private.json'
    if score is None and score_file.is_file():
        score = read(score_file)
        row['progress_sha256'] = digest(score_file)
    row['scored'] = isinstance(score,dict) and 'ever_target_grasp' in score
    if row['scored']:
        if not 0<=score['native_step']<=row['ticks'] or score['instruction']!=row['instruction']:
            raise ValueError('Score does not belong to this condition or native prefix')
        row['score_tick'] = score['native_step']
        events = score['events']
        row.update(grasp=score['ever_target_grasp'], lift20=score['ever_target_lift20mm'],
                   base80=bool(events.get('base80mm')), hold1s=score['longest_grasp_lift_hold_s']>=1.,
                   correct_region=bool(events.get('correct_region')),release=bool(events.get('released')),
                   retreat50=bool(events.get('retreated50mm')))
    return row


def report(cfg):
    catalog = pinned(cfg['catalog'])
    population = pinned(catalog['population'])
    experts = {r['id']:s for s in ('TRAIN','VAL') for r in population['conditions'][s]}
    if len(experts) != 264 or catalog['plan'] != PLAN:
        raise ValueError('Unexpected expert population or scope')
    cases = {}
    def add(row):
        if row is not None:
            if row['id'] in cases:
                raise ValueError('Same registered condition used by two owners')
            cases[row['id']] = row
    producers = {}
    for spec_sha, dirname in catalog['producers'].items():
        root = Path(dirname)
        spec = pinned(dict(path=str(root/'EXECUTION_SPEC.json'),sha256=spec_sha))
        records = root/'RECORDS.private.json'
        ledger = read(records) if records.is_file() else []
        rows = {r['condition']:r for r in ledger}
        if len(ledger)!=len(rows):
            raise ValueError('Duplicate terminal in producer ledger')
        producers[root.name] = dict(spec_sha256=spec_sha,source=spec['source_sha'],
                                   owner_terminal=(root/'RESULT.private.json').is_file())
        for folder in sorted((root/'cases').glob('CHV1-*')):
            add(observe(folder,spec['source_sha'],experts[folder.name],folder.name,rows.get(folder.name)))
    students = pinned(cfg['students'])
    gates_root = Path(students['gate']['path']).parent
    pinned(students['gate'])
    gates = {f.stem:read(f) for f in sorted((gates_root/'endpoints').glob('*.json'))}
    blocks = Path(cfg['students']['path']).parent/'blocks'
    for folder in sorted(blocks.glob('*')):
        card = folder/'EXECUTION_SPEC.json'
        if not card.is_file():
            continue
        block = read(card)
        terminal = folder/'RESULT.private.json'
        rows = {r['id']:r for r in read(terminal)['records'] if r['role']=='STUDENT'} if terminal.is_file() else {}
        for case in block['cases']:
            child = Path(case).parent
            if not Path(case).is_file():
                continue
            spec = read(Path(case))
            proof = pinned(spec['student_proof'])
            arm = next((key for key,g in gates.items() if g==proof),None)
            split = spec['evaluation_split']
            if (not arm or proof['gate']['status']!='ENDPOINT_RGB16_VERIFIED'
                    or child.name not in {c['id'] for c in proof['panels'][split]}):
                raise ValueError('Student case lacks its registered RGB16 release')
            add(observe(child,students['source'],split,spec['panel_entry_id'],rows.get(child.name),student=True))
    populations = defaultdict(list)
    cells = defaultdict(list)
    for row in cases.values():
        populations[row['split']].append(row)
        cells[(row['split'],row['family'],row['instruction'])].append(row)
    base = Path(cfg['catalog']['path']).parent
    entries = sorted((base/'catalog').glob('TRAIN-???.json'))
    latest = read(entries[-1]) if entries else None
    if latest:
        seal = pinned(dict(path=latest['path'],sha256=latest['sha256']))
        if seal['episode_sha256']!=latest['membership'] or seal['count']!=latest['count']:
            raise ValueError('Latest TRAIN catalog changed')
    return dict(plan=PLAN,scope='CORE_NATIVE',producers=producers,
                populations={s:counts(populations[s]) for s in ('TRAIN','VAL','DEV','CONF')},
                cells={'/'.join(k):counts(v) for k,v in sorted(cells.items())},
                current_train_seal={k:latest[k] for k in ('count','windows','sha256')} if latest else None,
                endpoints={k:dict(status=g['gate']['status'],exact=g['gate']['exact'],
                    checkpoint=g['gate']['checkpoint'],results={s:counts([r for r in populations[s]
                    if r['endpoint']==k]) for s in ('DEV','CONF')}) for k,g in gates.items()},cases=list(cases.values()))


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('root',type=Path)
    cli.add_argument('config_sha256')
    args = cli.parse_args()
    os.umask(0o077)
    root = args.root
    cfg = pinned(dict(path=str(root/'CONFIG.json'),sha256=args.config_sha256))
    now = time.time()
    if (cfg['plan']!=PLAN or cfg['T0']!=T0 or cfg['deadline']!=DEADLINE or cfg['uid']!=os.getuid()
            or cfg['host']!=os.uname().nodename or not os.environ.get('INVOCATION_ID')
            or any(digest(root/'source'/name)!=value for name,value in cfg['source_members'].items())):
        raise PermissionError('Daily receipt requires the registered Core owner and source')
    if T0 <= now < DEADLINE and not (root/'STOP').exists():
        slot = 'pre-deadline' if now>=DEADLINE-300 else f'day-{int((now-T0)//86400):02d}'
        dest = root/'receipts'/(slot+'.json')
        with closing(OwnerLock(root/'owner.lock')):
            if not dest.exists():
                actual = report(cfg)
                write_once(dest,dict(actual,read_start=now,read_end=time.time(),slot=slot,
                    source=cfg['source'],invocation=os.environ['INVOCATION_ID'],pid=os.getpid()))
