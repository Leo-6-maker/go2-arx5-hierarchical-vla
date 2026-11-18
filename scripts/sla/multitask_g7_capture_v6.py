"""Verify nominal H40, exact native P10 and paused RGB for one closed expert."""

import json
from pathlib import Path
import shutil

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.codec_v1 import eef, rigid
from go2arx5_vla.sla_adapter.dense_g7_v6 import CLOCK_FIELDS, label_margin_report
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin


def read_capture(condition, case, native):
    detail = native['detail']
    score = detail['score']
    end = native['actual_native_integrations']
    clock = native['trace_audit']
    if (not native['completed'] or native['unsafe'] or native['technical_failure']
            or not native['metrics']['task_and_required_motion_success'] or not score['task_success']
            or score['instruction'] != condition['instruction'] or score['native_step'] != end
            or score['states'] != end-5703+1 or score['missing_native_intervals'] or score['unsafe_states']
            or clock['status'] != 'RECORDED_TRACE_CHECKS_PASS' or clock['native_step_records'] != end):
        raise ValueError('ACTUAL_COMPLETE_2MS_TEACHER_SUCCESS_REQUIRED')
    path = Path(case['registered_root'])/'native-01'
    capture = path/'LIVE_DATA'
    if Path(detail['capture']['path']) != capture/'MANIFEST.private.json':
        raise ValueError('EXACT_NATIVE_RESULT_CAPTURE_REQUIRED')
    manifest = read_pin(detail['capture'])
    count = manifest['queries']
    if (manifest['schema'] != 'MFREE_LIVE_FINAL_PLAN_CAPTURE_V1' or manifest['writer_error'] is not None
            or manifest['final_native_step'] != end or count != detail['expert_queries']
            or not 1 <= count <= case['policy_queries'] or end != 5703+250*count
            or detail['consumed_task_atoms'] != 10*count or manifest['planned_suffix_is_executed'] is not False
            or manifest['frozen_scene_sha256'] != case['frozen_files'][case['model_path']]
            or manifest['language'] != case['language'] or manifest['data_contract'] != case['collection_contract']
            or Path(manifest['labels']['path']) != capture/'QUERIES.private.json'):
        raise ValueError('COMPLETE_LIVE_H40_RESULT_CLOCK_AND_CAPTURE_IDENTITY_REQUIRED')
    rows = read_pin(manifest['labels'])
    if len(rows) != count:
        raise ValueError('EXACT_TEACHER_QUERY_DENOMINATOR_REQUIRED')
    commands = [json.loads(line) for line in (path/'MOBILE_SAMPLES.private.jsonl').read_text().splitlines()]
    joined = {row['native_start_step']: row for row in commands}
    if len(joined) != len(commands):
        raise ValueError('UNIQUE_REAL_COMMAND_CLOCK_REQUIRED')
    frame = rigid(rows[0]['episode_fixed_T_policy_from_mount'])
    transform = rigid(case['policy_from_world'])
    codec = GripLogitCodec(frame)
    state10, nominal, state7, latent7, contexts = [], [], [], [], []
    for index, row in enumerate(rows):
        step = 5703+250*index
        context = row['previous_confirmed_command']
        prior = joined[step-25]
        if (row['query'] != index or row['native_step'] != step or abs(row['native_time_s']-step*.002) > 1e-8
                or row['instruction'] != condition['instruction'] or row['actual_elapsed_prefix_ticks'] != 250
                or row['partial_interval_ticks'] or row['full_EEF_H40_planner_check'] is not True
                or row['rgb_provenance'] != 'CURRENT_OWNED_PAUSED_NATIVE_PACKET'
                or row['object_phase_memory_in_student_inputs'] is not False
                or row['supervision_mask'] != [True]*40 or row['consumed_mask'] != [True]*10+[False]*30
                or not np.array_equal(row['episode_fixed_T_policy_from_mount'], frame)
                or prior['native_end_step'] != step or prior['ticks'] != 25
                or context['width_m'] != .044*prior['installed_EEF10'][9]
                or context['source_native_s'] != prior['gripper_command']['source_native_s']
                or context['query_native_s'] != row['native_time_s'] or len(row['intervals']) != 10):
            raise ValueError(f'FULL_H40_REAL_PREVIOUS_COMMAND_AND_P10_REQUIRED:{index}')
        for atom, sample in enumerate(row['intervals']):
            start = step + 25*atom
            observed = joined[start]
            if (sample['atom'] != atom or sample['start_native_step'] != start
                    or sample['end_native_step'] != start+25 or sample['confirmed_ticks'] != 25
                    or observed['ticks'] != 25 or observed['native_end_step'] != start+25
                    or observed['gripper_command']['sequence'] != 128+index*10+atom
                    or sample['installed_reference_sha256'] != observed['installed_reference_sha256']
                    or sample['boundary']['native_step'] != start+25 or not sample['boundary']['exact_receiver_knot']
                    or not observed['safety_valid']):
                raise ValueError(f'P10_INTERVAL_MUST_MATCH_NATIVE_CONSUMPTION:{index}:{atom}')
        world = eef(row['nominal_world_H40'], (40, 10))
        margin = label_margin_report(world, context)
        if margin != row['original_label_G7_margin'] or margin['original_execution_speed_exceeded']:
            raise ValueError(f'EXACT_RAW_G7_LABEL_MARGIN_AND_NATIVE_SPEED_REQUIRED:{index}')
        state = eef(row['state_eef10'], (10,))
        measured = row['actual_state']
        expected_state = pose10(transform @ np.asarray(measured['tcp']), measured['gripper_fraction'])
        # Native capture and offline export can use different floating-point
        # matrix kernels. Preserve the captured bytes and allow only roundoff
        # when independently recomputing the pose.
        if state[9] != expected_state[9] or not np.allclose(state[:9], expected_state[:9], rtol=0, atol=1e-12):
            raise ValueError('SAME_PAUSED_MEASURED_STATE_AND_POLICY_FRAME_REQUIRED')
        target = pose10(transform @ matrix_from10(world), world[:, 9])
        s, a = codec.encode(state, target)
        state10.append(state); nominal.append(target); state7.append(s); latent7.append(a)
        contexts.append([context[key] for key in CLOCK_FIELDS])
    arrays = dict(state7=np.asarray(state7), latent7=np.asarray(latent7), state_eef10=np.asarray(state10),
        nominal_policy_H40=np.asarray(nominal), native_step=np.asarray([r['native_step'] for r in rows]),
        frame_index=np.arange(count), phase=np.asarray([r['phase'] for r in rows]),
        previous_actual_command=np.asarray(contexts), supervision_mask=np.ones((count,40), dtype=bool),
        consumed_mask=np.asarray([r['consumed_mask'] for r in rows], dtype=bool))
    return capture, manifest, frame, codec, arrays


def copy_shards(source, manifest, data, output=None):
    shards = []
    start = 0
    for raw in manifest['frame_shards']:
        path = Path(raw['path'])
        count = raw['count']
        if (path.parent != source or path.name != f'frames-{len(shards):03d}.npz'
                or raw['first_query'] != start or not 1 <= count <= 16):
            raise ValueError('IMMUTABLE_CONTIGUOUS_SOURCE_RGB_SHARDS_REQUIRED')
        observed = path
        if output is not None:
            observed = output/path.name
            shutil.copyfile(path, observed)
        if sha(observed) != raw['sha256']:
            raise ValueError('BYTE_EXACT_ORIGINAL_PAUSED_RGB_TRANSFER_REQUIRED')
        with np.load(observed, allow_pickle=False) as rgb:
            span = slice(start, start+count)
            if (rgb['rgb'].shape != (count, 2, 256, 256, 3) or rgb['rgb'].dtype != np.uint8
                    or not np.array_equal(rgb['native_step'], data['native_step'][span])
                    or not np.array_equal(rgb['state_eef10'], data['state_eef10'][span])):
                raise ValueError('EXACT_CURRENT_RGB_STATE_AND_TEACHER_QUERY_JOIN_REQUIRED')
        file = str(observed.relative_to(output.parent)) if output is not None else observed.name
        shards.append(dict(file=file, sha256=raw['sha256'],
                           first_frame=start, frames=count))
        start += count
    if start != len(data['native_step']):
        raise ValueError('ALL_TEACHER_QUERIES_HAVE_MATCHING_RGB_SHARDS')
    return shards
