from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from scripts.pi0291_arx5_data.native_dataset import pose10
from scripts.pi0292.durable import write
from scripts.sla.export_multitask_g7_v6 import copy_shards, read_capture
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
from go2arx5_vla.sla_adapter.multitask_data_v6 import visit_bank
from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS


def pin(path):
    return dict(path=str(path), sha256=sha(path))


def test_complete_h40_retains_only_ten_independently_confirmed_atoms(tmp_path):
    start, end = 5703, 5953
    path = tmp_path/'native-01'
    capture = path/'LIVE_DATA'; capture.mkdir(parents=True)
    condition = dict(id='example', instruction='HOLD-red')
    scene = str(tmp_path/'example.mjb')
    frame = np.eye(4).tolist()
    case = dict(registered_root=str(tmp_path), policy_queries=120, policy_from_world=frame,
                model_path=scene, frozen_files={scene: 'a'*64}, language='Grasp and hold the red cube',
                collection_contract='EXAMPLE_TEST_ONLY')
    state = pose10(frame, .5)
    state[3] = np.nextafter(state[3], np.inf)
    h40 = np.tile(state, (40,1)); h40[10:,0] = .03
    context = dict(width_m=.022, source_native_s=(start-25)*.002, query_native_s=start*.002)
    wire = 'b'*64
    intervals = [dict(atom=i, start_native_step=start+25*i, end_native_step=start+25*(i+1),
        installed_reference_sha256=wire, confirmed_ticks=25,
        boundary=dict(native_step=start+25*(i+1), exact_receiver_knot=True)) for i in range(10)]
    commands = [dict(native_start_step=start+25*i, native_end_step=start+25*(i+1), ticks=25,
        installed_EEF10=state.tolist(), installed_reference_sha256=wire, safety_valid=True,
        gripper_command=dict(sequence=128+i, source_native_s=(start+25*i)*.002)) for i in range(-1,10)]
    log = path/'MOBILE_SAMPLES.private.jsonl'
    log.write_text(''.join(json.dumps(row)+'\n' for row in commands))
    labels = capture/'QUERIES.private.json'
    write(labels, [dict(query=0, native_step=start, native_time_s=start*.002, instruction=condition['instruction'],
        actual_elapsed_prefix_ticks=250, partial_interval_ticks=0, full_EEF_H40_planner_check=True,
        rgb_provenance='CURRENT_OWNED_PAUSED_NATIVE_PACKET', object_phase_memory_in_student_inputs=False,
        supervision_mask=[True]*40, consumed_mask=[True]*10+[False]*30,
        nominal_world_H40=h40.tolist(), episode_fixed_T_policy_from_mount=frame, previous_confirmed_command=context,
        intervals=intervals, original_label_G7_margin=label_margin_report(h40, context),
        state_eef10=state.tolist(), actual_state=dict(tcp=frame, gripper_fraction=.5), phase='HOLD')])
    rgb = capture/'frames-000.npz'
    np.savez(rgb, rgb=np.zeros((1,2,256,256,3), np.uint8), state_eef10=state[None], native_step=[start])
    manifest = capture/'MANIFEST.private.json'
    write(manifest, dict(schema='MFREE_LIVE_FINAL_PLAN_CAPTURE_V1', writer_error=None, final_native_step=end,
        queries=1, planned_suffix_is_executed=False, frozen_scene_sha256='a'*64,
        language=case['language'], data_contract=case['collection_contract'], labels=pin(labels),
        frame_shards=[dict(**pin(rgb), count=1, first_query=0)]))
    native = dict(completed=True, unsafe=False, technical_failure=False, actual_native_integrations=end,
        metrics=dict(task_and_required_motion_success=True),
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_PASS', native_step_records=end),
        detail=dict(expert_queries=1, consumed_task_atoms=10, capture=pin(manifest), score=dict(task_success=True,
            instruction=condition['instruction'], native_step=end, states=251, missing_native_intervals=0, unsafe_states=0)))
    source, data_manifest, _, _, arrays = read_capture(condition, case, native)
    assert arrays['supervision_mask'].sum() == 40 and arrays['consumed_mask'].sum() == 10
    np.testing.assert_array_equal(arrays['nominal_policy_H40'][0,10:,0], h40[10:,0])
    out = tmp_path/'export'/'teacher'; out.mkdir(parents=True)
    copied, = copy_shards(source, data_manifest, arrays, out)
    assert sha(Path(out.parent/copied['file'])) == sha(rgb)
    wrong_frame = np.asarray(frame).copy()
    wrong_frame[0,3] = 1e-6
    with pytest.raises(ValueError, match='SAME_PAUSED_MEASURED_STATE_AND_POLICY_FRAME_REQUIRED'):
        read_capture(condition, dict(case, policy_from_world=wrong_frame.tolist()), native)
    commands[-1]['installed_reference_sha256'] = 'c'*64
    log.write_text(''.join(json.dumps(row)+'\n' for row in commands))
    with pytest.raises(ValueError, match='P10_INTERVAL_MUST_MATCH_NATIVE_CONSUMPTION:0:9'):
        read_capture(condition, case, native)


def test_both_arms_share_exact_superblock_and_instruction_quotas():
    def dataset(tasks, mark):
        rows = [dict(split='TRAIN', task=task, episode=f'{task}-0', phase='observed', anchor=0)
                for task in tasks]
        return SimpleNamespace(catalog_rows=lambda: rows, manifest_sha=mark*64)

    training = SimpleNamespace(legacy=dataset(['PICK_HOLD'], '1'), new=dataset(INSTRUCTIONS, '2'))
    visits, receipt = visit_bank(training)
    again, second = visit_bank(training)
    np.testing.assert_array_equal(again, visits)
    assert receipt == second
    assert np.all((visits.reshape(200,160) == 0).sum(axis=1) == 32)
    counts = list(receipt['teacher_task_visits'].values())
    assert sum(counts) == 25600 and max(counts)-min(counts) == 1
