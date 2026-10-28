"""T0 fixtures plus saved public COV metadata; no renderer/native qualification."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.pi0294.generalization import spec as S
from scripts.pi0294.generalization import target_eef10_contract as C
from go2arx5_vla.target_sim.dataset import pose10

ROOT = Path(__file__).resolve().parents[2]


def seal(value):
    result = deepcopy(value)
    result.pop('sha256', None)
    result['sha256'] = S.digest(result)
    return result


def pose(x=.45):
    return dict(tcp_xyz=[x, .02, .55], tcp_quaternion_xyzw=[.1, .2, .3, float(np.sqrt(.86))], gripper=.5)


def scene(x=.45, *, split='G_TRAIN', lineage='CPU_TARGET_PARENT', visual='V0'):
    layout = dict(generator_version=S.LAYOUT_VERSION, source_xy_m=[x, -.085], goal_xy_m=[x, .085],
                  cube_yaw_rad=0., direction='+Y', distance_bin='NEAR')
    return S.make_spec(layout, split=split, lineage_id=lineage, source_population_sha256='1' * 64,
                       physics_profile_sha256='2' * 64, seeds=dict(geometry=1, visual=2, reset=3),
                       visual=visual, reset='GEN_COMMON_HOME')


def home():
    return C.common_home_design('CPU_COMMON_HOME_V1', pose(), calibration_sha256='3' * 64)


def context():
    identity = {key: S.digest(key) for key in C.IDENTITIES}
    identity['calibration'] = home()['calibration_sha256']
    return C.episode_context(scene(), home(), task='T1', episode_id='CPU_TARGET_EPISODE', identity=identity,
                             evidence_kind='CPU_SYNTHETIC_NOT_PHYSICS')


def receipt_chain(ctx, proposal, index, *, point=None):
    point = index - proposal['observation_index'] if point is None else point
    binding = dict(context_sha256=ctx['sha256'], segment_id=proposal['segment_id'], sequence=index,
                   proposal_sha256=proposal['sha256'], point_index=point, command_time_s=index * C.DT,
                   ref_id=f'CPU_REF_{proposal["observation_index"]}', clock_id=ctx['identity']['clock'])
    chain, parent = {}, proposal['sha256']
    for stage in C.STAGES:
        row = dict(binding, schema=C.RECEIPT_SCHEMA, stage=stage, outcome=C.OUTCOMES[stage], parent_sha256=parent,
                   event_time_s=(index + (stage == 'executed')) * C.DT,
                   command=deepcopy(proposal['requested'][point]),
                   source_receipt_sha256=S.digest(['CPU_ONLY', index, stage]))
        chain[stage] = seal(row)
        parent = chain[stage]['sha256']
    return chain


def recording(count=640):
    ctx = context()
    observations, proposals, commands = [], [], []
    for i in range(count + 1):
        obs = dict(pose(.4), instruction=ctx['instruction'])
        for camera in ('body_rgb', 'wrist_rgb'):
            obs[camera] = dict(path=f'{camera}/{i:06d}.png', sha256=S.digest(['CPU_RGB', camera, i]),
                               timestamp_s=i * C.DT, valid=True, source='mujoco_render',
                               snapshot_id=f'CPU_SNAPSHOT_{i}', frame_index=i)
        observations.append(dict(context_sha256=ctx['sha256'], index=i, segment_id=0, time_s=i * C.DT,
                                 snapshot_id=f'CPU_SNAPSHOT_{i}', policy_observation=obs,
                                 clock_id=ctx['identity']['clock']))
    for start in range(0, count, C.PREFIX):
        # Deliberately impossible unexecuted suffix; it must never become labels.
        requested = [pose(.45 + (start + p) * .00001 if p < C.PREFIX else 999.) for p in range(C.HORIZON)]
        p = C.proposal(ctx, segment_id=0, observation_index=start, requested=requested)
        proposals.append(p)
        commands.extend(receipt_chain(ctx, p, i) for i in range(start, min(start + C.PREFIX, count)))
    return ctx, observations, proposals, commands


def test_641_observations_640_real_commands_601_windows_not_602_or_unexecuted_suffix():
    fixture = recording()
    result = C.inspect_commands(*fixture)
    assert result['observation_count'] == 641 and result['command_count'] == 640
    assert result['valid_window_count'] == 601 and result['windows'][-1]['anchor'] == 600
    w = result['windows'][0]
    assert w['action'].shape == (40, 10) and np.max(w['action'][:, 0]) < 1
    assert np.array_equal(w['action'][0], pose10(fixture[3][0]['submitted']['command']))
    assert not np.array_equal(w['action'][0], w['state'])
    assert w['rot6d_layout'] == 'GR00T_FIRST_TWO_ROWS'
    expected_xyz = np.linalg.inv(C._native_pose(fixture[1][0]['policy_observation'])[0]) @ C._native_pose(pose())[0]
    assert np.allclose(w['relative_action'][0, :3], expected_xyz[:3, 3])
    assert not result['native_execution_authorized'] and not result['training_export_authorized']
    assert result['status'] == 'CPU_CONTRACT_ONLY_NOT_NATIVE_OR_DATA_ADMISSION'


def test_short_records_have_only_real_windows_and_final_observation_cannot_be_padded():
    assert C.inspect_commands(*recording(40))['valid_window_count'] == 1
    assert C.inspect_commands(*recording(39))['valid_window_count'] == 0
    ctx, observations, proposals, commands = recording(40)
    with pytest.raises(ValueError, match='FINAL_OBSERVATION'):
        C.inspect_commands(ctx, observations[:-1], proposals, commands)


@pytest.mark.parametrize('stage,field,value,reason', [
    ('accepted', 'sequence', 99, 'SAME_RECEIPT_TRANSACTION'),
    ('installed', 'parent_sha256', '0' * 64, 'PRIOR_AND_RAW'),
    ('installed', 'event_time_s', 4., 'TIME_ORDER'),
    ('executed', 'command', pose(.4), 'ACHIEVED_STATE'),
    ('submitted', 'source_receipt_sha256', None, 'PRIOR_AND_RAW'),
    ('accepted', 'outcome', 'REJECTED', 'REJECTED_OR_UNKNOWN'),
    ('accepted', 'sequence', False, 'NO_BOOLEANS'),
])
def test_actual_receipt_identity_time_and_achieved_state_substitution_rejected(stage, field, value, reason):
    ctx, observations, proposals, commands = recording(40)
    commands[0][stage][field] = value
    commands[0][stage] = seal(commands[0][stage])
    # Update only child hash links so the selected semantic check is exercised.
    for i in range(C.STAGES.index(stage) + 1, len(C.STAGES)):
        commands[0][C.STAGES[i]]['parent_sha256'] = commands[0][C.STAGES[i - 1]]['sha256']
        commands[0][C.STAGES[i]] = seal(commands[0][C.STAGES[i]])
    with pytest.raises(ValueError, match=reason):
        C.inspect_commands(ctx, observations, proposals, commands)


def test_unexecuted_prefix_suffix_and_missing_ack_are_not_success_supervision():
    ctx, observations, proposals, commands = recording(40)
    commands[8] = receipt_chain(ctx, proposals[0], 8, point=8)
    with pytest.raises(ValueError, match='UNEXECUTED_H40_SUFFIX'):
        C.inspect_commands(ctx, observations, proposals, commands)
    ctx, observations, proposals, commands = recording(40)
    del commands[0]['installed']
    with pytest.raises(ValueError, match='FOUR_ORIGINAL'):
        C.inspect_commands(ctx, observations, proposals, commands)


@pytest.mark.parametrize('kind', C.BOUNDARIES)
def test_windows_never_cross_reset_abort_takeover_or_clock_break(kind):
    result = C.inspect_commands(*recording(40), boundaries=[dict(kind=kind, observation_index=20)])
    assert result['valid_window_count'] == 0
    assert result['rejected_windows'] == [dict(anchor=0, reason='BOUNDARY_OR_SEGMENT_CROSSING')]


def test_silent_segment_change_privileged_input_stale_rgb_and_byte_tampering_rejected():
    for mutation, reason in (
        (lambda f: f[1][10].update(segment_id=1), 'SEGMENT_CHANGE_NEEDS_BOUNDARY'),
        (lambda f: f[1][10]['policy_observation'].update(object_pose=[0, 0, 0]), 'POLICY_ONLY'),
        (lambda f: f[1][10]['policy_observation']['body_rgb'].update(timestamp_s=.4), 'STALE_RGB'),
        (lambda f: f[3][10]['submitted'].update(command=pose(88.)), 'SHA256_MISMATCH'),
    ):
        f = recording(40)
        mutation(f)
        with pytest.raises(ValueError, match=reason):
            C.inspect_commands(*f)


def test_fixed_common_home_and_transitive_geometry_lineage_reuse_original_audit():
    report = C.bind_scene_population([scene(), scene(visual='V1')], home())
    assert len({row['home_sha256'] for row in report['scenes']}) == 1
    assert report['split_audit']['status'] == 'EXCLUSION_INDEX_INCOMPLETE'
    assert not report['training_export_authorized']
    chain = [scene(.45, lineage='a'), scene(.4515, lineage='b'), scene(.453, lineage='c', split='G_TEST')]
    with pytest.raises(ValueError, match='CROSS_SPLIT_LINEAGE'):
        C.bind_scene_population(chain, home())
    bad = home()
    bad['scene_dependent'] = True
    with pytest.raises(ValueError, match='COMMON_HOME'):
        C.bind_scene_population([scene()], seal(bad))


def test_saved_public_cov_geometry_retains_train_origin_without_target_qualification():
    path = ROOT / 'reports/PI0294/DATA_COV1/DATA_COV1_FROZEN_POPULATION.json'
    raw = path.read_bytes()
    groups = [g for g in json.loads(raw)['P24_groups'] if g['replicate'] == 1][:6]
    scenes = []
    import hashlib
    for group in groups:
        old = S.import_cov1_train(group, source_population_sha256=hashlib.sha256(raw).hexdigest(),
                                 physics_profile_sha256='2' * 64, visual_master_seed=10, reset_master_seed=11)
        new = S.make_spec(old['layout'], split=old['study_split'], lineage_id=old['lineage_id'],
                          source_population_sha256=old['source_population_sha256'],
                          physics_profile_sha256=old['physics_profile_sha256'], seeds=old['seeds'],
                          visual=old['visual_profile'], reset='GEN_COMMON_HOME', origin=old['origin'])
        scenes.append(new)
    bound = C.bind_scene_population(scenes, home())
    assert len(bound['scenes']) == 6
    assert all(s['split'] == 'G_TRAIN' for s in bound['scenes'])
    assert {s['lineage_id'] for s in bound['scenes']} == {g['group_id'] for g in groups}
    assert bound['status'] == 'T0_METADATA_ONLY'
    assert all(s['qualification'] == S.QUALIFICATION for s in scenes)


def test_import_is_cpu_only_and_old_recording_versions_remain_rejected():
    code = ('import sys; from scripts.pi0294.generalization import target_eef10_contract; '
            'assert not ({"mujoco", "torch", "rclpy", "OpenGL"} & set(sys.modules))')
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'))
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, check=True, timeout=20)
    f = recording(40)
    f[0]['schema'] = 'PI0288_RECORDING_V1'
    with pytest.raises(ValueError):
        C.inspect_commands(seal(f[0]), *f[1:])
