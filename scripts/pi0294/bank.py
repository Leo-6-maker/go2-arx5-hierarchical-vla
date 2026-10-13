"""Freeze paired multitask banks without rewriting any legacy episode or stats."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

from scripts.iteration.queue import digest, value_digest
from scripts.pi0292.durable import OwnerLock, read, write

PHASES = ('approach', 'contact', 'transport', 'withdraw', 'hold')
STAGE_PHASE = dict(approach='approach', align='approach', side_approach='approach',
    push_align='approach', close='contact', verify_grasp='contact', establish_contact='contact',
    contact='contact', push_contact='contact', lift='transport', carry='transport',
    lower='transport', push='transport', advance='transport', release='withdraw',
    retreat='withdraw', withdraw='withdraw', separate='withdraw', detach='withdraw', hold='hold', done='hold')


def validate_cpu_ticket(root, reservation_path, manifest_path=None):
    from .run_admission import validate
    root = Path(root).resolve(strict=True)
    ticket = read(reservation_path)
    if ticket.get('schema') != 'PI0294_CPU_RESERVATION_V1' or ticket.get('fixture') is not False:
        raise ValueError('SOLE_COORDINATOR_CPU_RESERVATION_REQUIRED')
    if Path(ticket['root']).resolve() != root:
        raise ValueError('CPU_RESERVATION_WRONG_ROOT')
    queue = read(root / 'LIVE_QUEUE.json')
    if queue['auxiliary'].get(ticket['id']) != ticket:
        raise ValueError('CPU_RESERVATION_NOT_IN_AUTHORITATIVE_QUEUE')
    path = Path(manifest_path) if manifest_path else root / ticket.get('manifest_relative','RUN_MANIFEST.json')
    if digest(path) != ticket['run_manifest_sha256']:
        raise ValueError('CPU_MANIFEST_IDENTITY_CHANGED')
    manifest = read(path)
    validate(root, profile=manifest['execution']['profile'], manifest_path=path)
    if manifest['code']['vla_commit'] != ticket['source_sha']:
        raise ValueError('CPU_SOURCE_IDENTITY_CHANGED')
    if time.time() >= min(ticket['deadline_epoch'], 1791939600):
        raise TimeoutError('CPU_STAGE_RESERVATION_EXPIRED')
    cpus = ticket['cpus']
    if not cpus or len(cpus) > 16 or len(cpus) != len(set(cpus)) or not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError('CPU_STAGE_EXACT_ALLOWED_AFFINITY_REQUIRED')
    if any(ticket['reservation'][k] != 0 for k in ('physics_starts', 'physics_steps', 'rgb_pairs')):
        raise ValueError('CPU_STAGE_CANNOT_EXECUTE_PHYSICS')
    os.sched_setaffinity(0, set(cpus))
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def cutoff(signum, frame):
        raise TimeoutError('CPU_STAGE_CUTOFF')
    signal.signal(signal.SIGALRM, cutoff)
    signal.alarm(max(1, int(ticket['deadline_epoch'] - time.time())))
    return ticket


def stages(folder):
    samples = read(Path(folder) / 'samples.json')
    result = {phase: [] for phase in PHASES}
    unknown = {}
    for anchor, sample in enumerate(samples[:601]):
        stage = sample.get('privileged', {}).get('teacher_stage')
        if stage in STAGE_PHASE:
            result[STAGE_PHASE[stage]].append(anchor)
        else:
            unknown.setdefault(str(stage), []).append(anchor)
    return result, unknown


def new_admission(root, scene_plan, queue, *, allow_partial_extension=False):
    root = Path(root)
    cards = scene_plan['cards']
    expected = {c['id']: c for c in cards}
    if len(expected) != len(cards):
        raise ValueError('DUPLICATE_FIRST_ATTEMPT_CARD')
    groups = {}
    for card in cards:
        if card['split'] not in ('TRAIN', 'DEV') or card['task'] not in ('T1', 'T2'):
            raise ValueError('ONLY_FORMAL_PAIRED_DATA')
        groups.setdefault(card['group_id'], []).append(card)
    for group, pair in groups.items():
        if len(pair) != 2 or {c['task'] for c in pair} != {'T1', 'T2'}:
            raise ValueError('TWO_TASKS_PER_SCENE_REQUIRED')
        if len({c['split'] for c in pair}) != 1 or len({c['initial_state_sha256'] for c in pair}) != 1:
            raise ValueError('PAIRED_INITIAL_STATE_AND_SPLIT_REQUIRED')
        ignored = {'id', 'task', 'task_id', 'instruction'}
        if {k: v for k, v in pair[0].items() if k not in ignored} != {k: v for k, v in pair[1].items() if k not in ignored}:
            raise ValueError('PAIRED_CARDS_DIFFER_BEYOND_LANGUAGE_AND_AUDIT_ID')
    records = {r['id']: r for r in queue['records'] if r['id'] in expected}
    partial = bool(allow_partial_extension and scene_plan['version'] == 'v3.1')
    if set(queue['reservations']) & set(expected) or len(records) != len(expected) and not partial:
        raise ValueError('ALL_SCHEDULED_FIRST_ATTEMPTS_MUST_BE_TERMINAL')
    output = []
    for card in cards:
        identifier = card['id']
        folder = root / 'episodes' / identifier
        if identifier not in records:
            if folder.exists():
                raise ValueError('UNRECONCILED_PARTIAL_OUTPUT_CANNOT_BE_OMITTED')
            output.append(dict(id=identifier, seed=card['seed'], group=card['group_id'],
                split=card['split'], task=card['task'], family=card['family'], origin='new',
                origin_authority='PI0294', admitted=False, raw_root=str(folder),
                samples_sha256=None, first_attempt=False, terminal='NOT_ATTEMPTED_PRESERVED',
                initial_state_sha256=card['initial_state_sha256'],
                worker_status='NOT_ATTEMPTED', audit_status='NOT_APPLICABLE'))
            continue
        worker = read(folder / 'WORKER_RECEIPT.json')
        if worker != records[identifier] or worker['fixture'] is not False:
            raise ValueError('EXACT_DURABLE_ACTUAL_FIRST_ATTEMPT_REQUIRED')
        if worker['card_sha256'] != value_digest(card):
            raise ValueError('PREREGISTERED_CARD_IDENTITY_CHANGED')
        audit = read(folder / 'AUDIT.json')
        ready = read(folder / 'READY.json') if (folder / 'READY.json').is_file() else None
        admitted = bool(ready and ready['status'] == 'SEALED_AUDITED_EPISODE'
            and ready['fixture'] is False and ready['admitted'] and audit['admitted']
            and audit['status'] == 'PASS_INDEPENDENT_RAW_AUDIT')
        if ready and (ready['audit_sha256'] != digest(folder / 'AUDIT.json')
                or ready['worker_receipt_sha256'] != digest(folder / 'WORKER_RECEIPT.json')
                or ready['raw_manifest_sha256'] != digest(folder / 'FILE_MANIFEST.json')):
            raise ValueError('SEALED_EPISODE_CHANGED_AFTER_AUDIT')
        if admitted and ready['samples_sha256'] != digest(folder / 'samples.json'):
            raise ValueError('ADMITTED_SAMPLE_CHANGED')
        output.append(dict(id=identifier, seed=card['seed'], group=card['group_id'],
            split=card['split'], task=card['task'], family=card['family'], origin='new',
            origin_authority='PI0294', admitted=admitted, raw_root=str(folder.resolve()),
            samples_sha256=digest(folder / 'samples.json'), first_attempt=True,
            terminal='ADMITTED' if admitted else 'REJECTED_PRESERVED',
            initial_state_sha256=card['initial_state_sha256'],
            worker_status=worker['status'], audit_status=audit['status']))
    byid = {row['id']: row for row in output}
    physical_keys = ('qpos', 'qvel', 'object_pose', 'tcp', 'model_xml_sha256', 'paired_scene_sha256')
    for pair in groups.values():
        paths = [root / 'episodes' / c['id'] / 'INITIAL_STATE.json' for c in pair]
        if all(path.is_file() for path in paths):
            snapshots = [read(path) for path in paths]
            if any(snapshots[0].get(k) != snapshots[1].get(k) for k in physical_keys):
                raise ValueError('ACTUAL_INITIAL_STATE_DIFFERS_BETWEEN_TASKS')
        elif all(byid[c['id']]['admitted'] for c in pair):
            raise ValueError('ADMITTED_PAIR_MUST_HAVE_ACTUAL_INITIAL_STATES')
    return output


def admission(root, scene_plan, old_native, old_admission, *, allow_partial_extension=False):
    root, old_native = Path(root), Path(old_native)
    old_manifest = read(old_native / 'DATASET_MANIFEST.json')
    prior = read(old_admission)
    if prior['status'] != 'ARX5_BANK_V2_QUALIFIED':
        raise ValueError('QUALIFIED_ORIGINAL_BANK_REQUIRED')
    if digest(old_native / 'statistics.json') != old_manifest['statistics_sha256']:
        raise ValueError('LEGACY_STATISTICS_CHANGED')
    originals = {r['id']: r for r in prior['episodes'] if r['admitted']}
    oldrows = []
    for row in old_manifest['episodes']:
        if row['id'] not in originals or row['split'] != originals[row['id']]['split']:
            raise ValueError('ORIGINAL_EPISODE_ID_OR_SPLIT_CHANGED')
        numeric = old_native / row['file']
        if digest(numeric) != row['sha256']:
            raise ValueError('LEGACY_NUMERIC_EXPORT_CHANGED')
        folder = Path(row['raw_root'])
        if digest(folder / 'samples.json') != row['samples_sha256']:
            raise ValueError('LEGACY_ORIGINAL_SAMPLE_CHANGED')
        oldrows.append(dict(id=row['id'], seed=row['seed'], group=row['group'], split=row['split'],
            task='T1', family='F0', origin='legacy', origin_authority='PI0292', admitted=True,
            raw_root=str(folder.resolve()), samples_sha256=row['samples_sha256'],
            original_native_sha256=row['sha256'], recollected=False, first_attempt=True,
            terminal='REUSED_QUALIFIED_UNCHANGED'))
    if {s: sum(r['split'] == s for r in oldrows) for s in ('TRAIN', 'DEV')} != {'TRAIN': 64, 'DEV': 16}:
        raise ValueError('EXACT_ORIGINAL64_TRAIN16_DEV_REQUIRED')
    newrows = new_admission(root, scene_plan, read(root / 'LIVE_QUEUE.json'),
                            allow_partial_extension=allow_partial_extension)
    counts = {task: {split: dict(planned=sum(r['task'] == task and r['split'] == split for r in newrows),
        admitted=sum(r['task'] == task and r['split'] == split and r['admitted'] for r in newrows))
        for split in ('TRAIN', 'DEV')} for task in ('T1', 'T2')}
    version = scene_plan['version']
    if version == 'v3.0':
        if any(counts[t]['TRAIN']['planned'] != 32 or counts[t]['DEV']['planned'] != 8 for t in counts):
            raise ValueError('EXACT_V3_0_FIRST_ATTEMPT_PLAN_REQUIRED')
        qualified = all(counts[t]['TRAIN']['admitted'] >= 28 and counts[t]['DEV']['admitted'] >= 7 for t in counts)
    elif version == 'v3.1':
        if any(counts[t]['TRAIN']['planned'] != 96 or counts[t]['DEV']['planned'] != 16 for t in counts):
            raise ValueError('V3_1_TOTAL224_INCLUDES_V3_0')
        qualified = all(counts[t]['TRAIN']['admitted'] >= 28 and counts[t]['DEV']['admitted'] >= 7 for t in counts)
    else:
        raise ValueError('UNREGISTERED_VERSION')
    rows = oldrows + newrows
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('OLD_AND_NEW_ID_CONFLICT')
    if {r['group'] for r in rows if r['split'] == 'TRAIN'} & {r['group'] for r in rows if r['split'] == 'DEV'}:
        raise ValueError('SCENE_GROUP_SPLIT_LEAKAGE')
    terminal = all(row['first_attempt'] for row in newrows)
    return dict(schema='PI0294_BANK_ADMISSION_V1', version=version,
        status='PI0294_BANK_QUALIFIED' if qualified else 'PI0294_BANK_QUALITY_GATE_FAIL',
        counts=counts, episodes=rows, all_planned_first_attempts_terminal=terminal,
        partial_extension_archive_only=not terminal, scheduled_new_episodes=len(newrows),
        actual_new_attempts=sum(row['first_attempt'] for row in newrows),
        original_statistics_sha256=old_manifest['statistics_sha256'], original_admission_sha256=digest(old_admission),
        original_native_manifest_sha256=digest(old_native / 'DATASET_MANIFEST.json'),
        source_scene_plan_sha256=value_digest(scene_plan), fit_DEV=False,
        old_originals_modified=False, hot_updates_active_training=False)


def build(root, scene_plan_path, old_native, old_admission, output, *, allow_partial_extension=False):
    root, output = Path(root), Path(output)
    version = read(scene_plan_path)['version']
    owner = OwnerLock(root / ('BANK_' + version.replace('.', '_') + '.lock'))
    began = time.monotonic()
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    try:
        evidence = root / 'evidence'
        evidence.mkdir(exist_ok=True)
        admission_path = evidence / ('BANK_' + version.replace('.', '_') + '_ADMISSION.json')
        if admission_path.exists() or output.exists():
            raise FileExistsError('PRESERVE_FIRST_BANK_FREEZE_INCLUDING_FAILURES')
        value = admission(root, read(scene_plan_path), old_native, old_admission,
                          allow_partial_extension=allow_partial_extension)
        write(admission_path, value)
        if value['status'] != 'PI0294_BANK_QUALIFIED':
            return value
        view = root / ('bank_' + version.replace('.', '_') + '_input')
        (view / 'episodes').mkdir(parents=True, exist_ok=False)
        for row in value['episodes']:
            if row['admitted']:
                (view / 'episodes' / row['id']).symlink_to(Path(row['raw_root']))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pi0291_arx5_data'))
        exporter = importlib.import_module('scripts.pi0291_arx5_data.export_dataset')
        manifest = exporter.export(view, admission_path, output)
        byid = {r['id']: r for r in value['episodes']}
        for row in manifest['episodes']:
            original = byid[row['id']]
            stage_anchors, unknown = stages(original['raw_root'])
            row.update(task=original['task'], family=original['family'], origin=original['origin'],
                origin_authority=original['origin_authority'], stage_anchors=stage_anchors,
                unmapped_stages=unknown, first_attempt=True)
        manifest.update(schema='PI0294_FROZEN_MULTITASK_NATIVE_V1', version=version,
            frozen=True, fixture=False, task_metadata_policy_input=False, phase_metadata_policy_input=False,
            original_statistics_sha256=value['original_statistics_sha256'], statistics_fit_count=1,
            original_native_manifest_sha256=value['original_native_manifest_sha256'],
            all_planned_first_attempts_terminal=value['all_planned_first_attempts_terminal'],
            partial_extension_archive_only=value['partial_extension_archive_only'],
            hot_update_active_training=False,
            new_admission_counts=value['counts'], processor_relative_conversion_count=1,
            task_ids=['T1', 'T2'], origin_authorities=['PI0292', 'PI0294'])
        write(output / 'DATASET_MANIFEST.json', manifest)
        write(output / 'task_map.json', dict(T1='PICK_PLACE', T2='PUSH',
            language_source='Exact original policy observation, never substituted',
            task_ids_model_input=False, stages_model_input=False))
        provenance = read(output / 'STATISTICS_PROVENANCE.json')
        provenance.update(fit_count=1, version=version, shared_tasks=['T1', 'T2'],
            original_statistics_sha256=value['original_statistics_sha256'], originals_modified=False)
        write(output / 'STATISTICS_PROVENANCE.json', provenance)
        # This export is usable raw data; model/processor qualification is separate.
        write(output / 'BANK_SEAL.json', dict(schema='PI0294_BANK_SEAL_V1', version=version,
            status='FROZEN_NATIVE_EXPORT_PENDING_PROCESSOR_QUALIFICATION', fixture=False,
            manifest_sha256=digest(output / 'DATASET_MANIFEST.json'), statistics_sha256=digest(output / 'statistics.json'),
            admission_sha256=digest(admission_path), statistics_fit_count=1,
            file_sha256={p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()},
            model_loads=0, integrations=0))
        if digest(Path(old_native) / 'statistics.json') != value['original_statistics_sha256']:
            raise ValueError('OLD_STATISTICS_MUTATED_DURING_EXPORT')
        return manifest
    finally:
        own = resource.getrusage(resource.RUSAGE_SELF)
        receipt = dict(status='BANK_CPU_STAGE_FINISHED', version=version,
            CPU_s=own.ru_utime + own.ru_stime - cpu0.ru_utime - cpu0.ru_stime,
            wall_s=time.monotonic() - began, model_loads=0, integrations=0,
            output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) if output.exists() else 0)
        path = root / 'evidence' / ('BANK_' + version.replace('.', '_') + '_CPU.json')
        if path.parent.exists() and not path.exists():
            write(path, receipt)
        owner.close()


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'scene-plan', 'old-native', 'old-admission', 'output', 'reservation'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--manifest')
    p.add_argument('--allow-partial-extension', action='store_true')
    args = p.parse_args()
    validate_cpu_ticket(args.root, args.reservation, args.manifest)
    value = build(args.root, args.scene_plan, args.old_native, args.old_admission, args.output,
                  allow_partial_extension=args.allow_partial_extension)
    print(json.dumps(dict(status=value.get('status', 'FROZEN_NATIVE_EXPORTED'),
                         version=value.get('version'), fixture=False)))
    return 2 if value.get('status') == 'PI0294_BANK_QUALITY_GATE_FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
