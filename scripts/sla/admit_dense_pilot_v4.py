"""Derive a separately named, limited TRAIN pilot from two sealed native runs.

Original engineering receipts keep their original eligibility flags. This
new admission is restricted to one task/layout/object and cannot satisfy D100.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from scripts.pi0292.durable import write
from scripts.pi0294.e1.artifacts import atomic_npz
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles, roundtrip_error, pad32
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec, VERSION, EPSILON
from go2arx5_vla.sla_adapter.dense_dataset_v4 import SCHEMA, sha, manifest_contract


def admit(panel_root, output):
    panel_root, output = Path(panel_root), Path(output)
    prefix = panel_root / 'V4_RECONCILED_PREFIX4.private.json'
    proof = json.loads(prefix.read_bytes())
    if proof['schema'] != 'SLA_V4_FIXED16_RECONCILED_PREFIX4_ACTUAL' or proof['expert_task_successes'] != 4:
        raise ValueError('RECONCILED_CLOSED_NATIVE_PREFIX_REQUIRED')
    selected = [r for r in proof['rows'] if r['run'].endswith('-p10')]
    if len(selected) != 2:
        raise ValueError('TWO_FIXED_P10_RESTARTS_REQUIRED')
    output.mkdir()
    write(output / 'STATE.private.json', dict(state='preparing', D100_new=False))
    episodes, all_state, all_actions = [], [], []
    for source in selected:
        root = panel_root / 'runs' / source['run']
        original_path = root / 'data/sla-pt-shards/MANIFEST.private.json'
        if str(original_path) != source['manifest']['path'] or sha(original_path) != source['manifest']['sha256']:
            raise ValueError('EXACT_CLOSED_NATIVE_SHARD_MANIFEST_REQUIRED')
        original = json.loads(original_path.read_bytes())
        worker = json.loads((root / 'WORKER_EXIT.private.json').read_bytes())
        lease = json.loads((root / 'control/LEASE.json').read_bytes())
        result_path = root / 'native-01/RESULT.json'
        if (worker['exit_code'] != lease['worker_exit_code'] or worker['exit_code'] != 0
                or lease['status'] != 'CLOSED' or worker['failure'] is not None
                or original['training_dataset_admitted'] is not False
                or sha(result_path) != original['source_proof']['result']['sha256']):
            raise ValueError('ACTUAL_CLOSED0_SOURCE_AND_UNMODIFIED_ENGINEERING_IDENTITY')
        result = json.loads(result_path.read_bytes())
        q = result['reference_qualification']
        if (not result['shutdown_clean'] or result['physics_stops'] or not q['task_condition_success']
                or result['actual_target_integrations'] != 21703):
            raise ValueError('FULL_NATIVE_TASK_SUCCESS_WITHOUT_HARD_STOP_REQUIRED')
        spec_path = root / 'EXECUTION_SPEC.json'
        if sha(spec_path) != original['source_proof']['spec']['sha256']:
            raise ValueError('ORIGINAL_SOURCE_WORLD_SPEC_CHANGED')
        spec = json.loads(spec_path.read_bytes())
        restart = spec['expert_cadence_restart']
        if (spec['task_learning_grasp']['prefix_atoms'] != 10 or restart not in (1, 2)
                or original['scene'] != 'TASK_LEARNING_UNDER_HAND_TRAIN01'):
            raise ValueError('EXACT_ONE_LAYOUT_P10_PILOT_SCOPE_REQUIRED')
        eid = f'TRAIN01-P10-r{restart}'
        folder = output / eid;folder.mkdir()
        shutil.copyfile(original_path, folder / 'ORIGINAL_ENGINEERING_MANIFEST.private.json')
        frames, frame_states = [], []
        for row in original['frame_shards']:
            p = original_path.parent / row['file']
            if sha(p) != row['sha256']:
                raise ValueError('ORIGINAL_FRAME_SHARD_CHANGED')
            os.link(p, folder / row['file'])
            with np.load(p, allow_pickle=False) as f:
                frame_states.extend(f['state_eef10'].copy())
            frames.append(dict(row, file=eid+'/'+row['file']))
        p = original_path.parent / original['label_shard']['file']
        if sha(p) != original['label_shard']['sha256']:
            raise ValueError('ORIGINAL_CONSUMED_LABEL_SHARD_CHANGED')
        with np.load(p, allow_pickle=False) as f:
            old = {k: f[k].copy() for k in f.files}
        if not np.array_equal(old['command_native_step'], 5703+np.arange(640)*25):
            raise ValueError('CONTIGUOUS_CONSUMED_COMMAND_CLOCK_REQUIRED')
        phases = {r['sha256']: r['stage'] for r in q['teacher_decisions']}
        tail = q['recorded_expert_success_tail']
        if not tail['passes'] or tail['consumed_commands'] != 40 or tail['end_step']-tail['start_step'] != 1000:
            raise ValueError('ACTUAL_TWO_SECOND_COMMAND_TAIL_REQUIRED')
        phases[tail['command_sha256']] = 'HOLD'
        policy_from_world = np.asarray(spec['task_phase_source_from_target'], dtype=float)
        codec = GripLogitCodec(original['source_proof']['mount']['T_policy_from_mount'])
        raw_states, raw_actions, frame_indices, annotations = [], [], [], []
        for i, tick in enumerate(old['native_step']):
            command = int((tick-5703)//25);frame = int((tick-5703)//50)
            state = np.asarray(frame_states[frame])
            world = old['command_nominal_eef10'][command:command+40]
            nominal = pose10(policy_from_world @ matrix_from10(world), world[:, 9])
            state7, latent7 = codec.encode(state, nominal)
            if (not np.allclose(state7, old['state7'][i], rtol=0, atol=1e-10)
                    or not np.allclose(latent7, old['nominal_latent7'][i], rtol=0, atol=1e-10)):
                raise ValueError('DERIVED_LABELS_DIFFER_FROM_CONSUMPTION_AUDITED_CODEC')
            error = roundtrip_error(nominal, codec.decode(state, pad32(latent7)))
            if error['xyz_m'] > 1e-6 or error['so3_rad'] > 1e-5 or error['gripper'] > EPSILON+1e-12:
                raise ValueError('PILOT_NOMINAL_COMMAND_ROUNDTRIP_FAILED')
            raw_states.append(state);raw_actions.append(nominal);frame_indices.append(frame)
            annotations.append(phases[str(old['command_source_sha256'][command])])
        atomic_npz(folder/'TRAIN_WINDOWS.npz', state7=old['state7'], latent7=old['nominal_latent7'],
                   state_eef10=np.asarray(raw_states), nominal_policy_H40=np.asarray(raw_actions),
                   native_step=old['native_step'], frame_index=np.asarray(frame_indices),
                   phase=np.asarray(annotations))
        all_state.extend(old['state7']);all_actions.extend(old['nominal_latent7'])
        episodes.append(dict(id=eid, task='PICK_HOLD', split='TRAIN', layout='TRAIN01', restart=restart,
            prefix_atoms=10, native_integrations=21703, windows=61, frames=321, commands=640,
            language=original['language'], mount=original['source_proof']['mount'],
            labels=dict(file=eid+'/TRAIN_WINDOWS.npz', sha256=sha(folder/'TRAIN_WINDOWS.npz')),
            frame_shards=frames, supervised_anchors=list(range(61)), expert_stage_audit_only=annotations,
            original_engineering_manifest_sha256=source['manifest']['sha256']))
        del result, q
    states, actions = np.asarray(all_state), np.asarray(all_actions).reshape(-1, 7)
    stats = dict(state=Quantiles.fit(states).summary(states), actions=Quantiles.fit(actions).summary(actions),
                 fit_unique_windows=122, fitted_before_oversampling=True, split='TRAIN', codec_version=VERSION,
                 epsilon=EPSILON, state_gripper_transformed=False)
    write(output/'STATISTICS.private.json', stats)
    manifest = dict(schema=SCHEMA, state='committed',
        scope='NAMED_PICK_HOLD_TRAIN01_P10_FITTING_PILOT_ONLY',
        dataset_id='T_PICK_HOLD_TRAIN01_P10_2EP_V4_1', tasks=['PICK_HOLD'], split='TRAIN',
        D100_new=False, PICK_PLACE_admitted=False, legacy_engineering_receipts_modified=False,
        phase_input_to_student=False, consumption_proven=True, actual_success_tail_s=2.,
        codec=VERSION, physics_dt_s=.002, frame_period_ticks=50, anchor_period_ticks=250,
        command_period_ticks=25, episodes=episodes, supervised_windows=122,
        source_prefix_sha256=sha(prefix), source_panel_sha256=proof['panel_sha256'],
        statistics=dict(file='STATISTICS.private.json', sha256=sha(output/'STATISTICS.private.json')))
    manifest_contract(manifest)
    write(output/'MANIFEST.private.json', manifest)
    write(output/'STATE.private.json', dict(state='committed', D100_new=False, supervised_windows=122))
    return dict(dataset_id=manifest['dataset_id'], manifest_sha256=sha(output/'MANIFEST.private.json'),
                episodes=2, supervised_windows=122, frames=642, consumed_commands=1280,
                D100_new=False, PICK_PLACE_admitted=False, new_physical_requests=0, optimizer_updates=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(admit(args.panel_root, args.output)))
