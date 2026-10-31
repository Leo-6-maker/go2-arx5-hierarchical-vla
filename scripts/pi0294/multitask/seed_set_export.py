"""Freeze five qualified book episodes; fit new statistics only on TRAIN."""
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from scripts.pi0292.train_adapter import DOMAIN, TAG, OFFICIAL_SHA, sha
from .source_export import statistics

TRAIN_OFFSETS = [[0., 0.], [.004, 0.], [-.004, 0.], [0., .004], [0., -.004]]
DEV_OFFSETS = [[.006, 0.], [-.006, 0.], [0., .006], [0., -.006], [.004, .004],
               [.004, -.004], [-.004, .004], [-.004, -.004], [.008, .002], [-.008, -.002]]


def raw(v):
    return (json.dumps(v, indent=2, allow_nan=False) + '\n').encode()


def validate_seal(seal):
    coordinates = [r['xy_offset_m'] for r in seal['TRAIN'] + seal['DEV']]
    if any(type(v) is not list or len(v) != 2 or any(type(x) not in (int, float) or not np.isfinite(x) for x in v) for v in coordinates):
        raise ValueError('TYPED_FINITE_TRAIN_DEV_COORDINATES')
    if (seal['schema'] != 'BOOK5_TRAIN_DEV10_FROZEN_SEED_SET_V1'
            or [r['id'] for r in seal['TRAIN']] != [f'BOOK_SOURCE_SEED{i:03}' for i in range(5)]
            or [r['xy_offset_m'] for r in seal['TRAIN']] != TRAIN_OFFSETS
            or [r['id'] for r in seal['DEV']] != [f'BOOK_DEV{i:03}' for i in range(10)]
            or [r['xy_offset_m'] for r in seal['DEV']] != DEV_OFFSETS
            or seal['DEV_for_training_or_statistics'] is not False):
        raise ValueError('FIVE_DECLARED_TRAIN_TEN_DISJOINT_DEV_CONDITIONS')
    if set(map(tuple, TRAIN_OFFSETS)) & set(map(tuple, DEV_OFFSETS)):
        raise ValueError('TRAIN_DEV_INITIAL_CONDITION_LEAKAGE')
    return seal


def fit_statistics(packets):
    """Equal full-H40/window-timestep weighting, anchored at actual TCP."""
    states, actions, relative = [], [], []
    future = np.arange(601)[:, None] + np.arange(40)
    for p in packets:
        s, c = p['actual_state10'][:601], p['committed_absolute10']
        if s.shape != (601, 10) or c.shape != (640, 10) or not np.all(p['valid_window'][:601]):
            raise ValueError('ONLY_FULL_ACTUAL_COMMAND_WINDOWS_IN_TRAIN_STATISTICS')
        states.append(s); actions.append(c[future])
        relative.append(pose10(np.linalg.inv(matrix_from10(s))[:, None] @ matrix_from10(c)[future], c[future, 9])[..., :9])
    s, a, r = np.concatenate(states), np.concatenate(actions), np.concatenate(relative)
    return {TAG: dict(state=dict(eef_9d=statistics(s[:, :9], 0), gripper_position=statistics(s[:, 9:], 0)),
        action=dict(eef_9d=statistics(a[..., :9], (0, 1)), gripper_position=statistics(a[..., 9:], (0, 1))),
        relative_action=dict(eef_9d=statistics(r, (0, 1))))}


def prepare(candidates, output, seal):
    """Candidate root only; optimization remains disabled until native codec."""
    validate_seal(seal)
    if len(candidates) != 5:
        raise ValueError('EXACT_FIVE_SEED_CANDIDATES')
    output = Path(output)
    if output.exists():
        raise FileExistsError('PRESERVE_PRIOR_SEED_SET')
    rows, packets = [], []
    for expected, case in zip(seal['TRAIN'], candidates):
        root = Path(case['root']); manifest = root / 'DATASET_MANIFEST.json'
        if sha(manifest) != expected['candidate_manifest_sha256']:
            raise ValueError('CANDIDATE_MANIFEST_CHANGED')
        m = json.loads(manifest.read_bytes()); r, = m['episodes']
        q = Path(case['qualification'])
        if sha(q) != expected['codec_result_sha256']:
            raise ValueError('NATIVE_CODEC_RECEIPT_CHANGED')
        qualification = json.loads(q.read_bytes())
        if (m['training_admitted'] is not False or r['id'] != expected['id'] or r['split'] != 'TRAIN'
                or r['raw_sha256'] != expected['raw_sha256'] or r['audit_sha256'] != expected['audit_sha256']
                or r['sha256'] != expected['numeric_sha256'] or r['windows'] != 601
                or qualification['manifest_sha256'] != expected['candidate_manifest_sha256']
                or qualification['codec']['status'] != 'PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP'
                or qualification['new_models_queries_optimizer_physics_PBS'] != 0):
            raise PermissionError('ACTUAL_QUALIFIED_COMPLETE_SOURCE_SEED_AND_OFFICIAL_CODEC_ONLY')
        rawroot = Path(r['raw_root'])
        if sha(rawroot / 'ACTUAL_RECORDS.jsonl') != expected['raw_sha256']:
            raise ValueError('RAW_CHANGED_AT_SEED_SET_VERSION_BOUNDARY')
        if sha(rawroot / 'INDEPENDENT_RAW_AUDIT.json') != expected['audit_sha256']:
            raise ValueError('INDEPENDENT_AUDIT_CHANGED')
        numeric = root / r['file']
        if not numeric.resolve().is_relative_to(root.resolve()) or sha(numeric) != expected['numeric_sha256']:
            raise ValueError('ACTUAL_NUMERIC_LABEL_FILE_CHANGED')
        with np.load(numeric, allow_pickle=False) as z:
            p = {k: z[k].copy() for k in z.files}
        packets.append(p); rows.append(dict(r, source_candidate_manifest_sha256=expected['candidate_manifest_sha256'],
                                           source_codec_result_sha256=expected['codec_result_sha256']))
    output.mkdir(parents=True, exist_ok=False); (output / 'episodes').mkdir()
    for r, case in zip(rows, candidates):
        destination = output / 'episodes' / (r['id'] + '.npz')
        shutil.copyfile(Path(case['root']) / r['file'], destination)
        if sha(destination) != r['sha256']:
            raise ValueError('NUMERIC_COPY_READBACK')
        r['file'] = destination.relative_to(output).as_posix()
    def put(name, v):
        with (output / name).open('xb') as f:
            f.write(raw(v))
    put('statistics.json', fit_statistics(packets))
    put('TRAIN_DEV_SEAL.json', seal)
    put('STATISTICS_PROVENANCE.json', dict(rule='FIVE_TRAIN_EPISODES_ALL_FULL_H40_EQUAL_WINDOW_TIMESTEP_MINMAX_V1',
        TRAIN_ids=[r['id'] for r in rows], state_anchor_rows=3005, action_window_timestep_rows=120200,
        DEV_PILOT_FAILED_or_old_E1_E2_in_fit=False, old_E1_E2_R0_overwritten=False))
    m = dict(schema='BOOK5_TRAIN_DEV10_NATIVE_DATASET_V1', domain=DOMAIN, purpose='NEW_BOOK_SKILL_SEED_SET',
        training_admitted=False, official_source_sha=OFFICIAL_SHA, episodes=rows,
        statistics_sha256=sha(output / 'statistics.json'), TRAIN_DEV_seal_sha256=sha(output / 'TRAIN_DEV_SEAL.json'),
        TRAIN_windows=3005, DEV_windows=0, DEV_conditions=10, new_model_queries_optimizer_physics=0)
    put('DATASET_MANIFEST.json', m)
    return m, packets


def admit(root):
    """Called only after the original official CPU codec/loader actual pass."""
    root = Path(root); qpath = root / 'OFFICIAL_QUALIFICATION.json'; q = json.loads(qpath.read_bytes())
    mpath = root / 'DATASET_MANIFEST.json'; m = json.loads(mpath.read_bytes())
    if (q['candidate_manifest_sha256'] != sha(mpath) or q['all_3005_loader_anchors'] is not True
            or q['all_641_RGB_pairs_each_reverified'] is not True or q['new_models_queries_optimizer_physics_PBS'] != 0
            or q['CUDA_initialized'] is not False or len(q['codecs']) != 5
            or any(v['status'] != 'PASS_FIXED_ORIGINAL_V3_0_NO_SILENT_CLIP' or v['H40_windows_actually_checked'] != 601
                   or v['native_inverse_clip_components_float32'] != 0 for v in q['codecs'])):
        raise PermissionError('ACTUAL_FIVE_EPISODE_JOINT_STATISTICS_OFFICIAL_QUALIFICATION_REQUIRED')
    validate_seal(json.loads((root / 'TRAIN_DEV_SEAL.json').read_bytes()))
    admitted = root / 'admitted'; admitted.mkdir(exist_ok=False); (admitted / 'episodes').mkdir()
    for r in m['episodes']:
        shutil.copyfile(root / r['file'], admitted / r['file'])
        if sha(admitted / r['file']) != r['sha256']:
            raise ValueError('ADMITTED_NUMERIC_COPY_CHANGED')
    for name in ('statistics.json', 'TRAIN_DEV_SEAL.json', 'STATISTICS_PROVENANCE.json'):
        shutil.copyfile(root / name, admitted / name)
    m.update(training_admitted=True, native_official_qualification_sha256=sha(qpath),
             original_candidate_manifest_sha256=sha(mpath), raw_individual_candidate_flags_unchanged=True)
    with (admitted / 'DATASET_MANIFEST.json').open('xb') as f:
        f.write(raw(m))
    return admitted, m
