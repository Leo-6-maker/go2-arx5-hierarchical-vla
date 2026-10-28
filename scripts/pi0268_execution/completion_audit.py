"""Reconcile final saved evidence and cumulative budgets; no model imports or calls."""
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

BASE = Path('/home/liuyu-nfs/Projects/go2-arx5')
ROOT = BASE / 'PI0268/R1'
RAW = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0268/R1')
POINTS = ('OLD0', 'ROT0', 'OLD2048', 'ROT2048')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def lines(path, cap):
    rows = [json.loads(s) for s in path.read_text().splitlines() if s]
    assert len(rows) <= cap and [r['ordinal'] for r in rows] == list(range(1, len(rows) + 1)), path
    return rows


def main():
    # ponytail: reconcile existing receipts in one CPU pass, without a new workflow engine.
    evidence = []
    def record(path, status=None):
        value = read(path)
        if status is not None:
            assert value['status'] == status, path
        evidence.append(dict(path=str(path), sha256=sha(path), status=value.get('status')))
        return value

    card = record(ROOT / 'design/APPROVED_RUN_CARD.json')
    assert card['release'] == 'e5ec6d52100f6bb2e32be5b9661c756b0adb61f4' and card['units_per_arm'] == 2048
    assert card['seed'] == 26741 and card['microbatches_per_unit'] == 2
    difference = record(ROOT / 'design/EXACT_STATISTICS_DIFF.json')
    assert difference['allowed_only'] and difference['source_unmodified'] and len(difference['changes']) == 860
    for row in difference['changes']:
        assert row['block'] in ('state', 'relative_action') and 3 <= row['feature'] < 9
        assert row['field'] in ('min', 'q01', 'max', 'q99')
        assert row['new'] == (-1 if row['field'] in ('min', 'q01') else 1)
    factor = record(ROOT / 'evidence/FACTOR_CODEC_ADMISSION.json', 'PASS_304_FIXED_FACTOR_INPUTS_AND_TWO_CODECS')
    assert factor['admitted'] == factor['proposed'] == 304 and len(factor['roundtrips']) == 608
    assert factor['frame_pairs_new'] == 192 and factor['frame_pairs_reused'] == 112
    evaluation = record(ROOT / 'evidence/EVALUATION_CPU_CHECK.json', 'PASS_ACTUAL_TRAIN_EVAL_CONDITIONS_FACTOR_INPUTS_AND_RELIABILITY_BOUNDARIES')
    assert len(evaluation['checks']) == 832 and evaluation['model_queries'] == evaluation['gradient_queries'] == 0
    cache = record(ROOT / 'evidence/OLD0_EXACT_CACHE_BINDING.json', 'PASS_EXACT_OLD0_MODEL_PROCESSOR_INPUT_NOISE_CONVENTIONS')
    assert len(cache['legacy']) == 440 and len(cache['virtual']) == 64 and cache['ROT0_reuse_forbidden']
    training = record(ROOT / 'evidence/MAIN_TRAINING_AUDIT.json', 'PASS_BOTH_2048_MATCHED_TRAINING_INTEGRITY')
    assert training['optimizer_updates'] == {'OLD': 2048, 'ROT': 2048}
    assert all(training[k] for k in ('actual_inputs_match_Hopper_CPU_plan', 'actual_random_packets_equal', 'actual_S4_gradient_loss_mode_checks'))
    assert len(training['FIT112_consumption']) == 224 and all(r['actual_seen'] for r in training['FIT112_consumption'])
    harvested = ROOT / 'evidence/training_hopper'
    hopper = record(harvested / 'evidence/HOPPER_CPU_ADMISSION.json', 'PASS_PI0268_HOPPER_ACTUAL_PROCESSOR_BOUNDARY')
    assert hopper['OLD_bit_identity'] and hopper['unchanged_pixels_tokens_raw_labels_XYZ_gripper_masks']
    transfer = record(ROOT / 'evidence/TRAINING_EVIDENCE_TRANSFER.json', 'PASS_COMPLETE_TRAINING_JOURNALS_TRANSFER_AND_MEMBER_HASHES')
    manifest = read(harvested / 'TRAINING_EVIDENCE_MANIFEST.json')
    assert sha(harvested / 'TRAINING_EVIDENCE_MANIFEST.json') == transfer['source_manifest_sha256']
    assert len(manifest['files']) == transfer['files'] == 169
    for row in manifest['files']:
        path = harvested / row['path']
        if not path.exists() and path.suffix in ('.ER', '.OU'):
            path = path.with_name(path.name + '.log')
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], path
    updates = lines(harvested / 'raw_completed_updates.jsonl', 4384)
    trainheads = lines(harvested / 'raw_training_heads.jsonl', 90000)
    assert len(updates) == training['raw_completed_updates'] == 4136
    assert len(trainheads) == training['raw_training_heads'] == 74504
    assert sha(harvested / 'raw_completed_updates.jsonl') == training['completed_updates_sha256']
    assert sha(harvested / 'raw_training_heads.jsonl') == training['training_heads_sha256']
    for arm in ('OLD', 'ROT'):
        canary = record(harvested / arm / 'canary_compare.json', 'PASS_PI0268_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY')
        assert canary['units'] == 8 and canary['model_max_abs'] == canary['optimizer_max_abs'] == canary['scheduler_lr_abs'] == 0
        assert canary['packet_equal'] and canary['mode_restored'] and all(canary['saved_RNG_states_equal'].values())
    checkpoints = []
    for source in training['complete_checkpoints']:
        arm, step = source['arm'], source['step']
        receipt = record(ROOT / 'evidence' / arm / f'checkpoint-{step}.full_readback.json', 'PASS_FULL_FILE_READBACK')
        assert receipt['complete_trainer'] and receipt['files'] == source['entries'] and len(receipt['files']) == 18
        checkpoints.append(dict(arm=arm, step=step, revision=receipt['revision'], prefix=receipt['prefix'], tree_sha256=receipt['tree_sha256']))
    assert {(r['arm'], r['step']) for r in checkpoints} == set(itertools.product(('OLD', 'ROT'), (1024, 2048)))

    verdict = record(RAW / 'delivery/PRIMARY_VERDICT.json', 'PRIMARY_COMPLETE')
    cpu = record(RAW / 'delivery/CPU_REPLAY.json', 'PASS_SAVED_ARRAY_REPLAY')
    assert verdict['offline_rows'] == len(cpu['metrics']) == 3872 and verdict['virtual_episodes'] == len(cpu['episodes']) == 256
    assert cpu['model_queries'] == cpu['gradient_queries'] == cpu['physics_steps'] == 0
    attempts = lines(ROOT / 'evidence/model_attempts.jsonl', 24576)
    heads = lines(ROOT / 'evidence/model_heads.jsonl', 98304)
    virtual = lines(ROOT / 'evidence/virtual_model_attempts.jsonl', 19200)
    starts = lines(ROOT / 'evidence/virtual_starts.jsonl', 208)
    frames = lines(ROOT / 'evidence/virtual_frame_starts.jsonl', 19696)
    factors = lines(ROOT / 'evidence/factor_frame_starts.jsonl', 304)
    assert len(frames) + len(factors) <= 20000
    assert Counter(r['attempt'] for r in heads) == Counter({r['ordinal']: 4 for r in attempts})
    predictions = {r['file']: r['sha256'] for r in cpu['predictions']}
    assert len(predictions) == len(cpu['predictions']) == len(attempts) == 16326
    assert {r['request']['output'] for r in attempts} == set(predictions)
    for attempt in attempts:
        receipt = read(Path(attempt['request']['output']).with_suffix('.receipt.json'))
        assert receipt['attempt'] == attempt['ordinal'] and receipt['heads'] == 4 and receipt['native_S4']
        assert receipt['point'] == attempt['point'] and receipt['output_sha256'] == predictions[attempt['request']['output']]
        assert receipt['input_sha256'] == attempt['input_sha256'] and receipt['noise_sha256'] == attempt['noise_sha256']
        assert receipt['no_teacher_or_privileged_condition']
    assert all(attempts[r['global_attempt'] - 1]['request'] == r['request'] for r in virtual)
    assert len(attempts) - len(virtual) == 3432 and len(virtual) == len(frames) == 12894
    families = [f['family_id'] for f in read(BASE / 'PI0267/design/SCENE_CARD.json')['families'] if f['split'] == 'NEW_SIM_EVAL']
    expected = {f'{f}-{t}-c{c}-{b}' for f, t, c, b in itertools.product(families, ('M1', 'M3'), (0, 1), ('E0', 'E1'))}
    assert len(expected) == 64 and len(starts) == 192
    episode_queries = 0
    for point in POINTS:
        folder = RAW.parents[1] / 'PI0267/R1/virtual/donor' if point == 'OLD0' else RAW / 'virtual' / point
        paths = list(folder.glob('*/result.json'))
        assert {p.parent.name for p in paths} == expected
        for path in paths:
            episode = read(path)
            assert episode['validity'] == 'VALID' and episode['physics_steps'] == 0
            assert not episode['teacher_used_for_control'] and not episode['reference_retimed']
            assert sha(path.parent / 'trajectory.npz') == episode['arrays_sha256']
            if point != 'OLD0':
                episode_queries += len(episode['queries'])
    assert episode_queries == len(virtual)
    videos = []
    visual = record(ROOT / 'evidence/VIDEO_VISUAL_QA.json', 'PASS_VISUAL_INSPECTION_FIXED_CHAPTER_START_MIDDLE_END')
    assert visual['total_inspected_frames'] == 96 and visual['videos'] == 4
    for point in POINTS:
        receipt = record(RAW / 'delivery' / (point + '-VIDEO_RECEIPT.json'), 'PASS_RECORDED_RGB_VIDEO_FULL_DECODE')
        assert len(receipt['chapters']) == 8 and receipt['model_queries'] == receipt['new_renders'] == receipt['physics_steps'] == 0
        assert sha(Path(receipt['file'])) == receipt['sha256']
        videos.append(dict(point=point, sha256=receipt['sha256'], frames=receipt['frames']))
    archives = [record(ROOT / 'evidence' / ('RESULT_ARCHIVE_' + p + '.json'), 'PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS') for p in POINTS]
    old = record(ROOT / 'evidence/PI0267_CLOSEOUT_WITH_SECONDARY_PARTIAL.json', 'PI0267_PRIMARY_COMPLETE_SECONDARY_PARTIAL_ARCHIVES_COMPLETE')
    assert old['secondary_status'] == 'SECONDARY_PARTIAL' and old['secondary_WBC_model_queries'] == 3
    for name, n in (('OFFLINE3872.csv', 3872), ('VIRTUAL256.csv', 256), ('PAIRED_OFFLINE_CHANGES.csv', 3872)):
        with (RAW / 'delivery' / name).open() as stream:
            assert len(list(csv.DictReader(stream))) == n
        evidence.append(dict(path=str(RAW / 'delivery' / name), sha256=sha(RAW / 'delivery' / name)))
    result = dict(status='PASS_COMPLETE_PRIMARY_ARTIFACTS_AND_CUMULATIVE_BUDGET_RECONCILIATION',
        final_summary_archive='PENDING_SEPARATE_FIXED_REVISION_READBACK', optimizer_updates=training['optimizer_updates'],
        raw_training_updates=len(updates), raw_training_heads=len(trainheads), full_Trainer_checkpoints=checkpoints,
        offline_logical_rows=3872, exact_OLD0_offline_reuse=440, new_offline_queries=3432,
        virtual_logical_episodes=256, exact_OLD0_virtual_reuse=64, new_virtual_starts=len(starts),
        new_virtual_queries=len(virtual), inference_attempts=len(attempts), inference_heads=len(heads),
        new_factor_frames=len(factors), new_virtual_frames=len(frames), new_frame_pairs=len(factors)+len(frames),
        quality=verdict['interpretation'], virtual_research_candidates=verdict['reliable_virtual_research_candidates'],
        model_queries_by_audit=0, gradient_queries_by_audit=0, new_WBC_physics_hardware=0,
        old_secondary_status=old['secondary_status'], videos=videos, archives=archives, evidence=evidence,
        selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    path = ROOT / 'evidence/PRIMARY_COMPLETION_AUDIT.json'
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('evidence', 'archives')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
