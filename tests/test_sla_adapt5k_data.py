import json

import numpy as np
import pytest

from go2arx5_vla.sla_adapter.adapt5k_data import CAPTURE_SCHEMA, CompletePlanBatch, fit_target_state
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
from go2arx5_vla.sla_adapter.dense_g7_v6 import VERSION
from go2arx5_vla.sla_adapter.multitask_data_v6 import C20_STATISTICS
from go2arx5_vla.sla_adapter import adapt5k_contract as contract
from go2arx5_vla.sla_adapter import adapt5k_checkpoint as phase_checkpoint
from go2arx5_vla.sla_adapter.scale20k_contract import bank_sha256


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


def test_adapt5k_phase_clocks_and_auxiliary_weights_keep_independent_meanings():
    jax = pytest.importorskip('jax')
    rates = contract.phase_rates()
    assert rates[0] == 5e-6 and rates[-1] == 1e-6 and np.all(np.diff(rates) < 0)
    assert [slot for update in range(16) for slot in contract.auxiliary_slots(update)] == list(range(64))
    before = contract.progress(500, 2000)
    switched = contract.progress(500, 2000, 2000)
    assert switched['stage_local_cursors'] == dict.fromkeys(contract.DOMAINS, 0)
    assert switched['lineage_cursors'] == before['lineage_cursors']
    assert before['lineage_total_updates'] == 2500

    fm = {'target': {'shared': 1., 'private': 2.}, 'robodojo': {'shared': 3., 'private': 4.},
          'libero': {'shared': 5., 'private': 6.}}
    out = contract.joint_gradient(fm, {'shared': 7., 'private': 8.}, tree_map=jax.tree.map)
    assert out == {'shared': 17., 'target': 12., 'robodojo': 4., 'libero': 6.}


def test_due_close_requires_a_consumed_crossing_at_or_after_the_seam():
    label = np.ones((40, 10))
    label[9:, 9] = .4
    p10 = np.arange(40) < 10
    assert contract.due_close(label, p10, .04)
    label[9, 9] = 1.
    assert not contract.due_close(label, p10, .04)
    label[:, 9] = .4
    assert contract.due_close(label, p10, .04)
    assert not contract.due_close(label, p10, .019)


def test_phase_checkpoint_restores_full_state_and_pins_d2_invariants(tmp_path):
    jax = pytest.importorskip('jax')
    groups = ('shared', *contract.DOMAINS)
    marker = 'a'*64
    binding = dict.fromkeys(('parent_checkpoint_sha256','parent_training_spec_sha256',
        'target_state_sha256','target_D1_sha256','optimizer_sha256','tokenizer_sha256',
        'partition_sha256','parent_manifest_sha256','source_sha256','R200_membership_sha256',
        'LIBERO_membership_sha256','stage_target_data_sha256'), marker)
    binding.update(plan=contract.PLAN, recipe=phase_checkpoint.RECIPE, parent_step=500,
        learning_rates_sha256=bank_sha256(contract.phase_rates()), stage_start=0, stage_stop=2000,
        stage_bank_sha256=dict.fromkeys(contract.DOMAINS,marker),
        frozen_C20_action_statistics_sha256=C20_STATISTICS)
    parameters = {d:{str(i):np.array([i, i+.5],dtype=np.float32) for i in range(12-(d=='libero'))}
                  for d in groups}
    optimizer = {d:{str(i):np.array([i+1],dtype=np.float32) for i in range(23)} for d in groups}
    state = dict(parameters=parameters, optimizer=optimizer, rng=jax.random.PRNGKey(9))
    original = phase_checkpoint.save(tmp_path/'phase00025',phase_step=25,binding=binding,**state)
    restored,clocks = phase_checkpoint.restore(original,binding=binding,**state)
    assert clocks['parent_global_step'] == 500 and clocks['lineage_total_updates'] == 525
    assert clocks['phase_cursors'] == dict(target=1600,robodojo=800,libero=800)
    assert all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(state),strict=True))
    following = dict(binding,stage_start=2000,stage_stop=5000,stage_target_data_sha256='b'*64)
    assert phase_checkpoint.next_stage_binding(binding,following,2000) == following
    with pytest.raises(ValueError,match='SINGLE_COLD_2K'):
        phase_checkpoint.next_stage_binding(binding,dict(following,target_state_sha256='c'*64),2000)
