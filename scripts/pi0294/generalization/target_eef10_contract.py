"""T0 standing Go2+ARX5 EEF10 recording contract; pure CPU, no data admission.

This is a new sidecar contract for the existing target recorder/exporter. It
does not rename PI0288 records, remove EEF13 BASE criteria, run a Teacher, or
claim that caller-supplied receipts prove native execution. A future original
native recorder must bind these receipts to independently read raw originals.
"""
from copy import deepcopy
import math

import numpy as np

from go2arx5_vla.target_sim.dataset import POLICY_FIELDS, pose10, ROT6D_LAYOUT
from go2arx5_vla.target_sim.recorder import _native_pose
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows
from scripts.pi0294.scene_cards import INSTRUCTIONS
from . import population, spec

SCHEMA = 'GO2_ARX5_TARGET_EEF10_RECORDING_CONTRACT_V1'
HOME_SCHEMA = 'GO2_ARX5_TARGET_COMMON_HOME_DESIGN_V1'
RECEIPT_SCHEMA = 'GO2_ARX5_TARGET_EEF10_COMMAND_RECEIPT_V1'
IDENTITIES = ('core_wbc', 'task_executor', 'teacher', 'clock', 'scene', 'codec', 'calibration')
STAGES = ('submitted', 'accepted', 'installed', 'executed')
OUTCOMES = dict(zip(STAGES, ('SUBMITTED', 'ACCEPTED', 'INSTALLED', 'CONSUMED')))
BOUNDARIES = ('reset', 'abort', 'native_abort', 'takeover', 'invalid_reference', 'clock_break')
POSE_FIELDS = {'tcp_xyz', 'tcp_quaternion_xyzw', 'gripper'}
HORIZON, PREFIX, DT = 40, 8, .05
EPS = 1e-8


def require(ok, reason):
    if not ok:
        raise ValueError('TARGET_EEF10_' + reason)


def _integer(value):
    return type(value) is int and value >= 0


def _time(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'FINITE_SOURCE_TIME')
    return value


def _pose(value):
    require(type(value) is dict and set(value) == POSE_FIELDS, 'POSE_ONLY_NO_BASE_OR_PRIVILEGED_LABELS')
    require(type(value['tcp_xyz']) is list and len(value['tcp_xyz']) == 3
            and type(value['tcp_quaternion_xyzw']) is list and len(value['tcp_quaternion_xyzw']) == 4
            and all(spec.number(n) for n in value['tcp_xyz'] + value['tcp_quaternion_xyzw'] + [value['gripper']]),
            'FINITE_NUMERIC_POSE_NO_BOOLEANS')
    _native_pose(value)
    return pose10(value)


def _sealed(value):
    require(type(value) is dict and spec.sha256(value.get('sha256')), 'CONTENT_SHA256_REQUIRED')
    require(value['sha256'] == spec.digest({k: v for k, v in value.items() if k != 'sha256'}),
            'CONTENT_SHA256_MISMATCH')


def _seal(value):
    result = deepcopy(value)
    result['sha256'] = spec.digest(result)
    return result


def common_home_design(profile_id, tcp, *, calibration_sha256):
    """Seal one caller-chosen home, independent of scene/task; never qualify it."""
    require(isinstance(profile_id, str) and bool(profile_id), 'EXPLICIT_HOME_VERSION')
    _pose(tcp)
    require(spec.sha256(calibration_sha256), 'EXPLICIT_CALIBRATION_BINDING')
    return _seal(dict(schema=HOME_SCHEMA, profile_id=profile_id, tcp=deepcopy(tcp),
                      calibration_sha256=calibration_sha256, selection_scope='TRAIN_ONLY',
                      scene_dependent=False, native_qualification='NOT_RUN'))


def validate_home(home):
    _sealed(home)
    require(set(home) == {'schema', 'profile_id', 'tcp', 'calibration_sha256', 'selection_scope',
                         'scene_dependent', 'native_qualification', 'sha256'}, 'EXACT_HOME_FIELDS')
    require(home == common_home_design(home['profile_id'], home['tcp'],
                                      calibration_sha256=home['calibration_sha256']),
            'COMMON_HOME_IS_INDEPENDENT_DESIGN_NOT_NATIVE_RESET_PROOF')


def bind_scene_population(scenes, home, *, historical_groups=(), historical_source_sha256=None):
    """Reuse GEN1's transitive geometry/lineage split audit, with one sealed home.

    Historical source hashes remain coverage claims, not proof of completeness.
    Source-domain geometry may be reused as metadata; it gains no target IK,
    reset, camera, Teacher, or physical qualification here.
    """
    validate_home(home)
    report = population.audit_splits(scenes, historical_groups=historical_groups,
                                    historical_source_sha256=historical_source_sha256)
    population.require_separated_metadata(report)
    require(all(s['reset_profile'] == 'GEN_COMMON_HOME' for s in scenes), 'COMMON_HOME_EXPLICIT_SCENE_VERSION')
    rows = [dict(design_sha256=s['design_hash'], geometry_id=s['geometry_id'], lineage_id=s['lineage_id'],
                 split=s['study_split'], home_sha256=home['sha256'], tasks=list(s['tasks']))
            for s in scenes]
    return _seal(dict(schema=SCHEMA + '/SCENE_BINDING', scenes=sorted(rows, key=lambda s: s['design_sha256']),
                      home=deepcopy(home), split_audit=report, native_execution_authorized=False,
                      training_export_authorized=False, status='T0_METADATA_ONLY'))


def episode_context(scene, home, *, task, episode_id, identity, evidence_kind):
    spec.validate(scene)
    validate_home(home)
    require(scene['reset_profile'] == 'GEN_COMMON_HOME', 'SOURCE_RELATIVE_RESET_NOT_COMMON_HOME')
    require(task in INSTRUCTIONS and task in scene['tasks'], 'ORIGINAL_T1_T2_ONLY')
    require(isinstance(episode_id, str) and bool(episode_id), 'EPISODE_ID')
    require(type(identity) is dict and set(identity) == set(IDENTITIES)
            and all(spec.sha256(v) for v in identity.values()), 'FROZEN_IDENTITY_HASHES')
    require(identity['calibration'] == home['calibration_sha256'], 'HOME_CALIBRATION_IDENTITY')
    require(evidence_kind in ('CPU_SYNTHETIC_NOT_PHYSICS', 'SAVED_NATIVE_RECORDS_UNAUDITED'), 'HONEST_EVIDENCE_KIND')
    return _seal(dict(schema=SCHEMA, episode_id=episode_id, task=task, instruction=INSTRUCTIONS[task],
                      scene_design_sha256=scene['design_hash'], geometry_id=scene['geometry_id'],
                      lineage_id=scene['lineage_id'], split=scene['study_split'], home_sha256=home['sha256'],
                      identity=deepcopy(identity), evidence_kind=evidence_kind,
                      mode='STANDING_FREEBASE_FROZEN_WBC', action_dim=10, sample_hz=20,
                      horizon=HORIZON, prefix=PREFIX, label_layer='submitted',
                      label_source='ACTUALLY_SUBMITTED_ACCEPTED_INSTALLED_EEF10',
                      native_execution_authorized=False, training_export_authorized=False))


def _context(context):
    _sealed(context)
    require(set(context) == {'schema', 'episode_id', 'task', 'instruction', 'scene_design_sha256',
                             'geometry_id', 'lineage_id', 'split', 'home_sha256', 'identity', 'evidence_kind',
                             'mode', 'action_dim', 'sample_hz', 'horizon', 'prefix', 'label_layer',
                             'label_source', 'native_execution_authorized', 'training_export_authorized', 'sha256'},
            'EXACT_CONTEXT_FIELDS')
    require(context.get('schema') == SCHEMA and context.get('task') in INSTRUCTIONS
            and context.get('instruction') == INSTRUCTIONS[context['task']], 'NEW_VERSION_ORIGINAL_LANGUAGE')
    require(context.get('mode') == 'STANDING_FREEBASE_FROZEN_WBC'
            and (context.get('action_dim'), context.get('sample_hz'), context.get('horizon'), context.get('prefix'))
            == (10, 20, HORIZON, PREFIX), 'EEF10_TIMING_AND_FROZEN_MODE')
    require(context.get('label_layer') == 'submitted'
            and context.get('label_source') == 'ACTUALLY_SUBMITTED_ACCEPTED_INSTALLED_EEF10', 'EXPLICIT_INTERFACE_LABEL_LAYER')
    require(context.get('native_execution_authorized') is False
            and context.get('training_export_authorized') is False, 'COLD_CONTRACT_CANNOT_GRANT_AUTHORITY')
    require(context.get('evidence_kind') in ('CPU_SYNTHETIC_NOT_PHYSICS', 'SAVED_NATIVE_RECORDS_UNAUDITED'),
            'NO_NATIVE_PASS_FROM_METADATA')
    require(set(context['identity']) == set(IDENTITIES)
            and all(spec.sha256(v) for v in context['identity'].values()), 'FROZEN_IDENTITY_HASHES')
    require(context['split'] in spec.SPLITS and all(isinstance(context[k], str) and bool(context[k])
            for k in ('episode_id', 'geometry_id', 'lineage_id'))
            and spec.sha256(context['scene_design_sha256']) and spec.sha256(context['home_sha256']),
            'EXPLICIT_EPISODE_SCENE_LINEAGE_HOME')


def proposal(context, *, segment_id, observation_index, requested):
    """A preserved H40 proposal is not 40 executed commands."""
    _context(context)
    require(_integer(segment_id) and _integer(observation_index) and observation_index % PREFIX == 0,
            'NONNEGATIVE_SEGMENT_AND_FIXED_PREFIX_REPLAN_GRID')
    require(type(requested) is list and len(requested) == HORIZON, 'EXACT_H40_PROPOSAL')
    for value in requested:
        _pose(value)
    return _seal(dict(schema=SCHEMA + '/PROPOSAL', context_sha256=context['sha256'],
                      segment_id=segment_id, observation_index=observation_index,
                      issue_time_s=observation_index * DT, requested=deepcopy(requested)))


def _observation(row, context, index):
    require(set(row) == {'context_sha256', 'index', 'segment_id', 'time_s', 'snapshot_id',
                         'policy_observation', 'clock_id'}, 'EXACT_OBSERVATION_FIELDS')
    require(row['context_sha256'] == context['sha256'] and type(row['index']) is int and row['index'] == index,
            'OBSERVATION_CONTEXT_AND_SEQUENCE')
    require(_integer(row['segment_id']) and abs(_time(row['time_s']) - index * DT) <= EPS,
            'CONTIGUOUS_OBSERVED_20HZ')
    require(row['clock_id'] == context['identity']['clock'] and bool(row['snapshot_id']), 'ACTUAL_SOURCE_CLOCK_AND_SNAPSHOT')
    obs = row['policy_observation']
    require(type(obs) is dict and set(obs) == POLICY_FIELDS and obs['instruction'] == context['instruction'],
            'POLICY_ONLY_MEASURED_POSE_DUAL_RGB_LANGUAGE')
    _pose({key: obs[key] for key in POSE_FIELDS})
    paths = []
    for name in ('body_rgb', 'wrist_rgb'):
        camera = obs[name]
        require(camera.get('valid') is True and camera.get('source') == 'mujoco_render'
                and camera.get('snapshot_id') == row['snapshot_id']
                and type(camera.get('frame_index')) is int and camera['frame_index'] == index,
                'ACTUAL_CAMERA_SNAPSHOT_FRAME_REQUIRED')
        require(spec.sha256(camera.get('sha256')) and isinstance(camera.get('path'), str) and bool(camera['path']),
                'ORIGINAL_CAMERA_REFERENCE_REQUIRED')
        age = row['time_s'] - _time(camera['timestamp_s'])
        require(-EPS <= age <= .025 + EPS, 'NO_FUTURE_OR_STALE_RGB')
        paths.append(camera['path'])
    require(paths[0] != paths[1], 'TWO_DISTINCT_CAMERA_PATHS')


def inspect_commands(context, observations, proposals, commands, *, boundaries=()):
    """Validate the command/ACK chain and index H40 windows from actual prefixes.

    Each command contains four content-hashed receipts. Every receipt binds the
    same source command, context, reference and point; accepted/installed may
    retain documented transforms, but this version labels the submitted layer.
    An executed receipt confirms consumption, never supplies an achieved-TCP
    target. Missing/rejected receipts cannot be filled from the H40 proposal.
    Returned packet arrays are CPU preparation, not export/admission permission.
    """
    _context(context)
    require(type(commands) is list and type(observations) is list
            and 0 < len(commands) <= 640 and len(observations) == len(commands) + 1,
            'ONE_FINAL_OBSERVATION_NOT_AN_EXTRA_COMMAND')
    for i, row in enumerate(observations):
        _observation(row, context, i)
    require(len({row['snapshot_id'] for row in observations}) == len(observations), 'SNAPSHOT_REUSE')
    for camera in ('body_rgb', 'wrist_rgb'):
        require(len({r['policy_observation'][camera]['path'] for r in observations}) == len(observations),
                'CAMERA_FILE_REUSE')
    require(type(proposals) is list and 0 < len(proposals) <= 640, 'BOUNDED_PROPOSALS')
    proposals_by_sha = {}
    for p in proposals:
        _sealed(p)
        require(p == proposal(context, segment_id=p['segment_id'], observation_index=p['observation_index'],
                              requested=p['requested']), 'EXACT_PRESERVED_PROPOSAL')
        require(p['sha256'] not in proposals_by_sha, 'DUPLICATE_PROPOSAL')
        require(p['observation_index'] < len(commands)
                and p['segment_id'] == observations[p['observation_index']]['segment_id'], 'PROPOSAL_OBSERVATION_SEGMENT')
        proposals_by_sha[p['sha256']] = p
    boundary_at = set()
    require(isinstance(boundaries, (tuple, list)), 'BOUNDARY_LIST')
    for event in boundaries:
        require(set(event) == {'kind', 'observation_index'} and event['kind'] in BOUNDARIES
                and _integer(event['observation_index']) and event['observation_index'] <= len(commands),
                'EXPLICIT_BOUNDARY_EVENT')
        boundary_at.add(event['observation_index'])
    for i in range(1, len(observations)):
        if observations[i]['segment_id'] != observations[i - 1]['segment_id']:
            require(i in boundary_at and observations[i]['segment_id'] == observations[i - 1]['segment_id'] + 1,
                    'SEGMENT_CHANGE_NEEDS_BOUNDARY')
    labels, refs, used, differences = [], [], set(), []
    for index, chain in enumerate(commands):
        require(type(chain) is dict and set(chain) == set(STAGES), 'ALL_FOUR_ORIGINAL_COMMAND_RECEIPTS')
        submitted = chain['submitted']
        p = proposals_by_sha.get(submitted.get('proposal_sha256'))
        require(p is not None, 'ACTUAL_COMMAND_MUST_BIND_PRESERVED_PROPOSAL')
        point = submitted.get('point_index')
        require(_integer(point) and point < PREFIX, 'UNEXECUTED_H40_SUFFIX_FORBIDDEN')
        require(index == p['observation_index'] + point, 'ACTUAL_PREFIX_SOURCE_SLOT')
        require((p['sha256'], point) not in used, 'DUPLICATE_PREFIX_CONSUMPTION')
        used.add((p['sha256'], point))
        binding = dict(context_sha256=context['sha256'], segment_id=p['segment_id'],
                       sequence=index, proposal_sha256=p['sha256'], point_index=point,
                       command_time_s=index * DT, ref_id=submitted.get('ref_id'),
                       clock_id=context['identity']['clock'])
        require(isinstance(binding['ref_id'], str) and bool(binding['ref_id']), 'EXPLICIT_REFERENCE_ID')
        require(binding['segment_id'] == observations[index]['segment_id'], 'COMMAND_OBSERVATION_SEGMENT')
        parent, last_time = p['sha256'], p['issue_time_s']
        for stage in STAGES:
            receipt = chain[stage]
            _sealed(receipt)
            require(set(receipt) == set(binding) | {'schema', 'stage', 'parent_sha256', 'event_time_s',
                                                   'outcome', 'command', 'source_receipt_sha256', 'sha256'}, 'EXACT_RECEIPT_FIELDS')
            require(receipt['schema'] == RECEIPT_SCHEMA and receipt['stage'] == stage
                    and all(receipt.get(k) == v for k, v in binding.items()), 'SAME_RECEIPT_TRANSACTION')
            require(all(_integer(receipt[k]) for k in ('sequence', 'segment_id', 'point_index')),
                    'INTEGER_TRANSACTION_IDENTITY_NO_BOOLEANS')
            _time(receipt['command_time_s'])
            require(receipt['outcome'] == OUTCOMES[stage], 'REJECTED_OR_UNKNOWN_RECEIPT_IS_NOT_SUPERVISION')
            require(receipt['parent_sha256'] == parent and spec.sha256(receipt['source_receipt_sha256']),
                    'EXACT_PRIOR_AND_RAW_RECEIPT_HASH')
            stamp = _time(receipt['event_time_s'])
            require(last_time - EPS <= stamp <= observations[index + 1]['time_s'] + EPS,
                    'ISSUE_ACCEPT_INSTALL_EXECUTE_TIME_ORDER')
            _pose(receipt['command'])
            parent, last_time = receipt['sha256'], stamp
        require(submitted['command'] == p['requested'][point], 'LABEL_DIFFERS_FROM_ACTUALLY_SUBMITTED_PROPOSAL_POINT')
        require(chain['executed']['event_time_s'] >= observations[index]['time_s'] - EPS,
                'CONSUMPTION_MUST_BE_IN_ACTUAL_COMMAND_INTERVAL')
        require(chain['executed']['command'] == chain['installed']['command'], 'EXECUTED_RECEIPT_IS_NOT_ACHIEVED_STATE_LABEL')
        labels.append(_pose(submitted['command']))
        refs.append((binding['ref_id'], index))
        differences.append(dict(submitted_accepted_equal=submitted['command'] == chain['accepted']['command'],
                                accepted_installed_equal=chain['accepted']['command'] == chain['installed']['command']))
    windows, rejected = [], []
    for start in range(max(0, len(commands) - HORIZON + 1)):
        end = start + HORIZON
        if (any(start <= b <= end for b in boundary_at)
                or len({row['segment_id'] for row in observations[start:end + 1]}) != 1):
            rejected.append(dict(anchor=start, reason='BOUNDARY_OR_SEGMENT_CROSSING'))
            continue
        actual, _ = _native_pose(observations[start]['policy_observation'])
        relative = []
        for chain in commands[start:end]:
            target, grip = _native_pose(chain['submitted']['command'])
            rel = np.linalg.inv(actual) @ target
            relative.append(np.r_[rel[:3, 3], rotation_matrix_to_rot6d_rows(rel[:3, :3]), grip])
        windows.append(dict(anchor=start, command_ids=refs[start:end], action=np.asarray(labels[start:end]),
                            relative_action=np.asarray(relative), state=pose10(observations[start]['policy_observation']),
                            point_offsets_s=np.arange(HORIZON) * DT, rot6d_layout=ROT6D_LAYOUT))
    return dict(schema=SCHEMA + '/CPU_INSPECTION', status='CPU_CONTRACT_ONLY_NOT_NATIVE_OR_DATA_ADMISSION',
                context_sha256=context['sha256'], observation_count=len(observations), command_count=len(commands),
                supplied_input_sha256=dict(observations=spec.digest(observations), proposals=spec.digest(proposals),
                                           commands=spec.digest(commands), boundaries=spec.digest(boundaries)),
                valid_window_count=len(windows), windows=windows, rejected_windows=rejected,
                receipt_transform_differences=differences, label_layer='submitted',
                native_execution_authorized=False, training_export_authorized=False,
                remaining_native_gates=['ORIGINAL_RAW_RECEIPT_READBACK', 'ACTUAL_REFERENCE_INSTALL_AND_PREFIX',
                                        'FULL_NATIVE_SAFETY_AND_OBJECT_TASK_SCORE', 'RGB_BYTES_AND_RESET_STATE',
                                        'HISTORY_EXCLUSION_AND_ORIGINAL_PROCESSOR_R0_ROUNDTRIP'])
