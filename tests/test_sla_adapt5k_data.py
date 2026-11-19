import json

import numpy as np
import pytest

from go2arx5_vla.sla_adapter.adapt5k_data import CAPTURE_SCHEMA, CompletePlanBatch, fit_target_state
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
from go2arx5_vla.sla_adapter.dense_g7_v6 import VERSION
from go2arx5_vla.sla_adapter.multitask_data_v6 import C20_STATISTICS


def test_new_complete_plan_reader_preserves_plan_and_p10_masks(tmp_path):
    def record(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=sha(path))

    state = np.zeros((1, 10))
    step = np.array([5703])
    labels = tmp_path / 'labels.npz'
    np.savez(labels, state7=np.ones((1, 7)), latent7=np.full((1, 40, 7), 3.), state_eef10=state,
             nominal_policy_H40=np.zeros((1, 40, 10)), native_step=step, frame_index=[0],
             supervision_mask=np.ones((1, 40), dtype=bool), consumed_mask=np.arange(40)[None, :] < 10,
             phase=['PLACE'], previous_actual_command=[[.04, step[0]*.002-.05, step[0]*.002]])
    camera = tmp_path / 'frames.npz'
    np.savez(camera, rgb=np.zeros((1, 2, 256, 256, 3), dtype=np.uint8), native_step=step, state_eef10=state)
    keys = dict(id='first', scene_sha256='source-scene', reset=11, native_result_sha256='native-result')
    episode = dict(keys, split='TRAIN', task='PLACE-red-left', family='F1', language='Place the red cube left.',
                   windows=1, mount={'T_policy_from_mount': np.eye(4).tolist()},
                   labels=dict(file=labels.name, sha256=sha(labels)), individual_pass=True,
                   frame_shards=[dict(file=camera.name, sha256=sha(camera), first_frame=0, frames=1)],
                   label_kind='expert_query_plan_H40', expert_stage_audit_only=['PLACE'])
    manifest = record('manifest.json', dict(schema=CAPTURE_SCHEMA, representation=VERSION,
                      statistics_sha256=C20_STATISTICS, live_SCALE_training_data_changed=False,
                      episodes=[episode], windows=1))
    actual = dict(keys, instruction=episode['task'], admitted=True, windows=1, confirmed_P10_atoms=10)
    outcome = dict(cases=[actual], dataset_sha256=manifest['sha256'], admitted_episodes=1, admitted_windows=1)
    proof = record('actual.json', outcome)
    stats = Quantiles(np.zeros(7), np.full(7, 2.))
    kwargs = dict(statistics={'state': stats, 'actions': stats}, resize_with_pad=None, cache_bytes=1024**2)
    reader = CompletePlanBatch(manifest, proof, **kwargs)
    assert len(reader) == 1 and reader.identity(0)['native_step'] == 5703
    assert len(reader.raw(0)[2]) == 2
    assert reader.labels['first']['supervision_mask'].sum() == 40
    assert reader.labels['first']['consumed_mask'].sum() == 10
    report = fit_target_state([reader], stats)
    assert report['state']['q01'] == [1.]*7
    assert report['action_outside_frozen_quantiles_fraction'] == [1.]*7
    assert report['action_points'] == 40
    invalid = record('short.json', dict(outcome, cases=[dict(actual, confirmed_P10_atoms=9)]))
    with pytest.raises(ValueError, match='ORIGINAL_P10'):
        CompletePlanBatch(manifest, invalid, **kwargs)
