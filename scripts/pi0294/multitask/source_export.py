"""Qualified source records -> the existing committed-command NPZ adapter.

This is a new-skill seed candidate, separate from the frozen E1/E2 dataset.
The immutable raw audit remains unchanged. Processor qualification and a
versioned seed-set admission must precede optimization.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from scripts.pi0292.train_adapter import DOMAIN, OFFICIAL_SHA, TAG, sha

KINDS = ('observation', 'planned_H40', 'submitted_interval', 'completed_interval')
OBS_KEYS = {'instruction', 'actual_tcp', 'actual_gripper', 'body_rgb', 'wrist_rgb'}


def packet(rows, card):
    """Select actual pre-command inputs and only completed submitted labels."""
    kept = [r for r in rows if r['kind'] in KINDS]
    expected = []
    for i in range(640):
        expected += ['observation'] + (['planned_H40'] if i % 8 == 0 else [])
        expected += ['submitted_interval', 'completed_interval']
    expected += ['observation']
    if [r['kind'] for r in kept] != expected:
        raise ValueError('COMPLETE_ORDERED_641_OBSERVATIONS_640_EXECUTED_COMMANDS')
    obs = [r['value'] for r in kept if r['kind'] == 'observation']
    sent = [r['value'] for r in kept if r['kind'] == 'submitted_interval']
    done = [r['value'] for r in kept if r['kind'] == 'completed_interval']
    plans = [r['value'] for r in kept if r['kind'] == 'planned_H40']
    for i, o in enumerate(obs):
        p = o['policy_obs']
        if (set(p) != OBS_KEYS or p['instruction'] != card['instruction']
                or o['index'] != i or o['native_step'] != 2500 + i * 25
                or abs(o['time_s'] - 5 - i / 20) > 1e-8
                or not np.array_equal(p['actual_tcp'], o['actual_state']['tcp'])
                or p['actual_gripper'] != o['actual_state']['gripper_fraction']):
            raise ValueError('ACTUAL_PRECOMMAND_POLICY_OBSERVATION_IDENTITY')
        for name in ('body_rgb', 'wrist_rgb'):
            pin = p[name]
            path = PurePosixPath(pin['file'])
            if path.is_absolute() or len(path.parts) != 1 or '..' in path.parts:
                raise ValueError('OWN_RAW_IMAGE_BASENAME_ONLY')
            if abs(o['calibration'][name]['timestamp_s'] - o['time_s']) > 1e-8:
                raise ValueError('IMAGE_STATE_CLOCK_MISMATCH')
    for k, p in enumerate(plans):
        a = np.asarray(p['rows10'])
        identity = json.dumps(dict(index=p['observation_index'], source_time_s=p['source_time_s'],
                                   rows10=a.tolist()), sort_keys=True, separators=(',', ':'), allow_nan=False)
        if (p['ref_id'] != hashlib.sha256(identity.encode()).hexdigest()
                or p['observation_index'] != k * 8 or a.shape != (40, 10)
                or abs(p['source_time_s'] - 5 - k * .4) > 1e-8
                or p['committed_prefix_count'] != 0
                or p['role'] != 'UNEXECUTED_REFERENCE_EVIDENCE_NOT_TRAINING_LABEL'):
            raise ValueError('UNCHANGED_H40_REFERENCE_IDENTITY')
    for i, (s, c) in enumerate(zip(sent, done)):
        p = plans[i // 8]
        if (s['completed'] is not False or c['completed'] is not True
                or s['training_admitted'] is not False or c['training_admitted'] is not False
                or s['observation_index'] != i or c['observation_index'] != i
                or s['ref_id'] != c['ref_id'] or c['ref_id'] != p['ref_id']
                or s['prefix_index'] != i % 8 or c['prefix_index'] != i % 8
                or s['start_native_step'] != 2500 + i * 25
                or c['start_native_step'] != s['start_native_step']
                or s['expected_end_native_step'] != 2525 + i * 25
                or c['end_native_step'] != s['expected_end_native_step']
                or abs(s['source_time_s'] - 5 - i / 20) > 1e-8
                or s['source_time_s'] != c['source_time_s']
                or not np.array_equal(s['action_EEF10'], c['action_EEF10'])
                or not np.array_equal(c['action_EEF10'], p['rows10'][i % 8])):
            raise ValueError('ONLY_ACTUALLY_SUBMITTED_AND_COMPLETED_PREFIX_LABELS')
    states = np.asarray([pose10(o['policy_obs']['actual_tcp'], o['policy_obs']['actual_gripper']) for o in obs])
    commands = np.asarray([c['action_EEF10'] for c in done], dtype=np.float64)
    if (states.shape != (641, 10) or commands.shape != (640, 10)
            or not np.isfinite(states).all() or not np.isfinite(commands).all()
            or np.any(commands[:, 9] < 0) or np.any(commands[:, 9] > 1)
            or np.any(states[:, 9] < -.001 / .044) or np.any(states[:, 9] > 1 + .001 / .044)
            or not np.allclose(pose10(matrix_from10(commands), commands[:, 9]), commands, atol=1e-8, rtol=0)
            or not np.allclose(pose10(matrix_from10(states), states[:, 9]), states, atol=1e-8, rtol=0)):
        raise ValueError('FINITE_PHYSICAL_ABSOLUTE_EEF10_WITHOUT_CLIPPING')
    return dict(actual_state10=states, committed_absolute10=commands,
                valid_window=np.arange(640) < 601,
                rgb_paths=np.asarray([[o['policy_obs'][n]['file'] for n in ('body_rgb', 'wrist_rgb')] for o in obs]),
                rgb_sha256=np.asarray([[o['policy_obs'][n]['sha256'] for n in ('body_rgb', 'wrist_rgb')] for o in obs]),
                anchor_time_s=np.arange(601) / 20, offset_s=np.arange(40) / 20,
                instruction=np.asarray(card['instruction']),
                label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'))


def statistics(values, axis):
    return {k: np.asarray(v).tolist() for k, v in dict(
        min=values.min(axis=axis), max=values.max(axis=axis), mean=values.mean(axis=axis),
        std=values.std(axis=axis), q01=np.quantile(values, .01, axis=axis),
        q99=np.quantile(values, .99, axis=axis)).items()}


def source_identity(card, audit):
    """Keep native book and custom cabinet success contracts distinct."""
    if card.get('task') == 'BOOK':
        if audit.get('native_LIBERO_goal_final') is not True:
            raise PermissionError('NATIVE_LIBERO_GOAL_REQUIRED_FOR_BOOK')
        if 'book_geometry_index'in card:
            from .libero_book_teacher import checked_route_card
            checked_route_card(card)
            return dict(episode_id='BOOK_GEOMETRY_TRAIN'+str(card['book_geometry_index']).zfill(3),
                task='BOOK',family='LIBERO_ARX5_LEFT_NEW_TRAIN_GEOMETRY_PORT',
                purpose='NEW_BOOK_YAW_OR_GRASP_TRAIN_CANDIDATE_NOT_OLD_CONFIRMATION')
        return dict(episode_id='BOOK_SOURCE_SEED' + str(card.get('book_seed_index', 0)).zfill(3),
                    task='BOOK', family='LIBERO_ARX5_LEFT_PORT',
                    purpose='NEW_BOOK_SKILL_SEED_CANDIDATE_SEPARATE_FROM_E1_E2')
    if (card.get('task') == 'DOOR'
            and card.get('source_port') == 'ROBOCASA_SINGLE_CABINET_ARX5_SOURCE_PORT_V1'
            and card.get('port_revision') == 'ROBOCASA_ARX5_PORT08_EDGE_HINGE_BRAKED_TURN'
            and card.get('cabinet', {}).get('mechanical_variant') == 'OUTER_PANEL_EDGE_AXIS_EXACT_QUARTER_TURN_V1'
            and card.get('original_benchmark_robot_or_seed') is False
            and audit.get('schema') == 'NIGHT_SOURCE_CABINET_INDEPENDENT_RAW_AUDIT_V1'
            and audit.get('native_ROBOCASA_goal_final') is True):
        seed = card.get('cabinet_seed_index', 0)
        if 'cabinet_seed_index' in card:
            from .cabinet_repair02 import checked_card
            checked_card(card, 'right')
        return dict(episode_id='CUSTOM_CABINET_PORT08_SOURCE_SEED' + str(seed).zfill(3), task='DOOR',
                    family='ROBOCASA_ARX5_CUSTOM_EDGE_HINGE_BRAKED_PORT',
                    purpose='NEW_CUSTOM_CABINET_SKILL_SEED_CANDIDATE_SEPARATE_FROM_E1_E2')
    raise PermissionError('EXACT_QUALIFIED_SOURCE_TASK_AND_NATIVE_GOAL_REQUIRED')


def export(folder, output, *, audit_sha256, terminal_sha256, raw_sha256):
    """Export a pinned qualified book or explicit custom cabinet, never resimulate."""
    folder, output = Path(folder).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('PRESERVE_PRIOR_DERIVATIVE_AND_RAW_RECORDS')
    audit_path, terminal_path = folder / 'INDEPENDENT_RAW_AUDIT.json', folder / 'TERMINAL.json'
    if sha(audit_path) != audit_sha256 or sha(terminal_path) != terminal_sha256:
        raise ValueError('PINNED_INDEPENDENT_AUDIT_AND_TERMINAL_CHANGED')
    audit = json.loads(audit_path.read_bytes())
    if (audit['status'] != 'COMPLETE_SOURCE_PILOT_QUALIFIED'
            or any(audit[k] is not True for k in ('task_success', 'qualified_source_pilot', 'complete_episode',
                   'original_native_safety_pass', 'additional_support_release_withdraw_hold_pass'))
            or audit['integrity_problems'] or audit['raw_sha256'] != raw_sha256
            or audit['terminal_sha256'] != terminal_sha256):
        raise PermissionError('COMPLETE_SAFE_INDEPENDENTLY_SCORED_SOURCE_SEED_REQUIRED')
    card = json.loads((folder / 'CARD.json').read_bytes())
    identity = source_identity(card, audit)
    root = folder.parent.parent
    lease = json.loads((root / 'control/NATIVE_SOURCE_LEASE.json').read_bytes())
    chain = json.loads((root / 'outputs/SOURCE_CHAIN_TERMINAL.json').read_bytes())
    if (lease['status'] != 'CLOSED' or lease['all_children_terminal'] is not True
            or chain['closed_lease_sha256'] != sha(root / 'control/NATIVE_SOURCE_LEASE.json')):
        raise PermissionError('CLOSED_SOURCE_OWNER_REQUIRED')
    h, rows = hashlib.sha256(), []
    with (folder / 'ACTUAL_RECORDS.jsonl').open('rb') as f:
        for line in f:
            h.update(line)
            row = json.loads(line)
            if row['kind'] in KINDS:
                rows.append(row)
    if h.hexdigest() != raw_sha256:
        raise ValueError('RAW_RECORDS_CHANGED_SINCE_INDEPENDENT_AUDIT')
    arrays = packet(rows, card)
    from PIL import Image
    for o in (r['value'] for r in rows if r['kind'] == 'observation'):
        for name in ('body_rgb', 'wrist_rgb'):
            pin = o['policy_obs'][name]
            image = folder / pin['file']
            if image.is_symlink() or image.stat().st_size != pin['bytes'] or sha(image) != pin['sha256']:
                raise ValueError('ACTUAL_RGB_FILE_CHANGED_SINCE_AUDIT')
            with Image.open(image) as opened:
                if opened.mode != 'RGB' or opened.size != (256, 256):
                    raise ValueError('ACTUAL_256_RGB_REQUIRED')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'episodes').mkdir()
    episode_id = identity['episode_id']
    numeric = output / 'episodes' / (episode_id + '.npz')
    np.savez_compressed(numeric, **arrays)
    future = np.arange(601)[:, None] + np.arange(40)
    state, commands = arrays['actual_state10'][:601], arrays['committed_absolute10']
    relative = pose10(np.linalg.inv(matrix_from10(state))[:, None] @ matrix_from10(commands)[future], commands[future, 9])[..., :9]
    absolute = commands[future]
    fitted = {TAG: dict(state=dict(eef_9d=statistics(state[:, :9], 0), gripper_position=statistics(state[:, 9:], 0)),
        action=dict(eef_9d=statistics(absolute[..., :9], (0, 1)), gripper_position=statistics(absolute[..., 9:], (0, 1))),
        relative_action=dict(eef_9d=statistics(relative, (0, 1))))}
    def put(name, value):
        with (output / name).open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
    put('statistics.json', fitted)
    manifest = dict(schema='SOURCE_SKILL_SEED_CANDIDATE_V1', domain=DOMAIN,
        purpose=identity['purpose'], training_admitted=False,
        official_source_sha=OFFICIAL_SHA, statistics_sha256=sha(output / 'statistics.json'),
        statistics_rule='THIS_SOURCE_TRAIN_CANDIDATE_ONLY_601_FULL_H40_MINMAX_NOT_E1_E2_R0',
        TRAIN_windows=601, DEV_windows=0, model_calls=0, optimizer_updates=0,
        raw_audit_training_admitted_flag_unchanged=audit['training_admitted'],
        required_before_optimization=['official_processor_codec_qualification', 'frozen_seed_set_and_disjoint_DEV_admission'],
        episodes=[dict(id=episode_id, task=identity['task'], family=identity['family'], origin='SOURCE_TEACHER',
            split='TRAIN', windows=601, raw_root=str(folder), file=numeric.relative_to(output).as_posix(),
            sha256=sha(numeric), source_sha=json.loads((root / 'RUN_MANIFEST.json').read_bytes())['source_sha'],
            raw_sha256=raw_sha256, audit_sha256=audit_sha256, terminal_sha256=terminal_sha256,
            instruction=card['instruction'], complete_commands=640, actual_observations=641)])
    if identity['task'] == 'DOOR':
        manifest['custom_source_variant'] = dict(port_revision=card['port_revision'],
            mechanical_variant=card['cabinet']['mechanical_variant'],
            original_RoboCasa_benchmark=False, target_Go2_WBC_qualification=False,
            native_goal_field='native_ROBOCASA_goal_final')
    put('DATASET_MANIFEST.json', manifest)
    return manifest
