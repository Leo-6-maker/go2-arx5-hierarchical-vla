"""Seal the first twenty declared fresh production assignments, without substitution.

Source recording receipts remain unchanged. A live production suffix is allowed;
each selected episode must already have its own actual CLOSED0 worker and shard.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from scripts.pi0292.durable import write
from scripts.pi0294.e1.artifacts import atomic_npz
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles, pad32, roundtrip_error
from go2arx5_vla.sla_adapter.dense_dataset_v4 import member, sha
from go2arx5_vla.sla_adapter.gripper_v2 import EPSILON, VERSION, GripLogitCodec
from go2arx5_vla.sla_adapter.named_dataset_v5 import CONDITIONS, DATASET_ID, SCHEMA, SCOPE, manifest_contract


def selected_assignments(panel):
    from go2arx5_vla.world_model.sla_named_expert_v5 import validate, PRODUCTION_SCHEMA
    if panel.get('schema') != PRODUCTION_SCHEMA:
        raise ValueError('FRESH_PRODUCTION_PANEL_REQUIRED_NOT_OLD_ENGINEERING')
    rows = validate(panel)[:20]
    if [(r['layout'], r['restart']) for r in rows] != list(CONDITIONS):
        raise ValueError('FIXED_FIRST20_ASSIGNMENTS_NO_SUCCESS_BASED_SUBSTITUTION')
    return rows


def source_episode(root, row, panel_pin):
    from go2arx5_vla.world_model.corelab_expert_semantics_v3 import ROLE
    root = Path(root)
    spec_path = root/'EXECUTION_SPEC.json'
    spec = json.loads(spec_path.read_bytes())
    lease = json.loads((root/'control/LEASE.json').read_bytes())
    worker = json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    original_path = root/'data/sla-pt-shards/MANIFEST.private.json'
    original = json.loads(original_path.read_bytes())
    proof = original['source_proof']
    result_path = root/'native-01/RESULT.json'
    if (lease['status'] != 'CLOSED' or lease['worker_exit_code'] != worker['exit_code']
            or worker['exit_code'] != 0 or worker['failure'] is not None
            or lease['spec_sha256'] != sha(spec_path)
            or spec['registered_root'] != str(root) or root.name != row['native_run']
            or spec['task_learning_role'] != ROLE
            or spec['scene_id'] != original['scene'] or original['scene'] != row['parent']
            or original['scene_sha256'] != row['model_sha256']
            or spec['expert_cadence_restart'] != row['restart']
            or spec['task_learning_grasp']['prefix_atoms'] != 10 or row['prefix_atoms'] != 10
            or spec['stable_task_panel'] != panel_pin['path']
            or spec['stable_task_panel_sha256'] != panel_pin['sha256']
            or spec['source_sha'] != original['source_sha']
            or original['schema'] != 'SLA_DENSE_EXPERT_EPISODE_SHARDS_V3'
            or original['state'] != 'committed' or original['split'] != 'TRAIN'
            or original['training_dataset_admitted'] is not False
            or original['source_frame_count'] != 321 or original['source_command_count'] != 640
            or original['label_count'] != 61 or proof['encoded_windows'] != 61
            or proof['actual_two_second_success_tail'] is not True
            or proof['spec'] != dict(path=str(spec_path), sha256=sha(spec_path))
            or proof['result'] != dict(path=str(result_path), sha256=sha(result_path))):
        raise ValueError('EXACT_FRESH_CLOSED0_NATIVE_CONSUMED_SHARD_REQUIRED')
    result = json.loads(result_path.read_bytes())
    q = result['reference_qualification']
    if (result['status'] != 'NATIVE_STAGE_PASSED' or not result['shutdown_clean']
            or result['physics_stops'] or not q['task_condition_success']
            or result['actual_target_integrations'] != 21703):
        raise ValueError('COMPLETE_NATIVE_PICK_HOLD_SUCCESS_REQUIRED')
    return original_path, original, spec, q


def derive_episode(root, output, row, panel_pin):
    original_path, original, spec, q = source_episode(root, row, panel_pin)
    eid = f"{row['layout']}-P10-r{row['restart']}"
    folder = output/eid
    folder.mkdir()
    shutil.copyfile(original_path, folder/'ORIGINAL_RECORDING_MANIFEST.private.json')
    frames, frame_states = [], []
    for descriptor in original['frame_shards']:
        path = member(original_path.parent, descriptor['file'])
        if sha(path) != descriptor['sha256']:
            raise ValueError('SEALED_FRESH_RGB_SHARD_CHANGED')
        os.link(path, folder/descriptor['file'])
        with np.load(path, allow_pickle=False) as f:
            frame_states.extend(f['state_eef10'].copy())
        frames.append(dict(descriptor, file=eid+'/'+descriptor['file']))
    label = original['label_shard']
    path = member(original_path.parent, label['file'])
    if sha(path) != label['sha256']:
        raise ValueError('SEALED_FRESH_CONSUMED_COMMAND_SHARD_CHANGED')
    with np.load(path, allow_pickle=False) as f:
        old = {k: f[k].copy() for k in f.files}
    if (not np.array_equal(old['command_native_step'], 5703+np.arange(640)*25)
            or not np.array_equal(old['native_step'], 5703+np.arange(61)*250)):
        raise ValueError('COMPLETE_FRESH_COMMAND_AND_ANCHOR_CLOCK_REQUIRED')
    stages = {r['sha256']: r['stage'] for r in q['teacher_decisions']}
    tail = q['recorded_expert_success_tail']
    if not tail['passes'] or tail['consumed_commands'] != 40 or tail['end_step']-tail['start_step'] != 1000:
        raise ValueError('ACTUAL_TWO_SECOND_EXPERT_HOLD_TAIL_REQUIRED')
    stages[tail['command_sha256']] = 'HOLD'
    policy_from_world = np.asarray(spec['task_phase_source_from_target'], dtype=float)
    codec = GripLogitCodec(original['source_proof']['mount']['T_policy_from_mount'])
    states, actions, annotations = [], [], []
    for i, tick in enumerate(old['native_step']):
        command, frame = int((tick-5703)//25), int((tick-5703)//50)
        state = np.asarray(frame_states[frame])
        world = old['command_nominal_eef10'][command:command+40]
        nominal = pose10(policy_from_world @ matrix_from10(world), world[:, 9])
        state7, latent7 = codec.encode(state, nominal)
        if (not np.allclose(state7, old['state7'][i], rtol=0, atol=1e-10)
                or not np.allclose(latent7, old['nominal_latent7'][i], rtol=0, atol=1e-10)):
            raise ValueError('FRESH_DERIVATION_MUST_MATCH_ORIGINAL_CONSUMPTION_AUDIT')
        error = roundtrip_error(nominal, codec.decode(state, pad32(latent7)))
        if error['xyz_m'] > 1e-6 or error['so3_rad'] > 1e-5 or error['gripper'] > EPSILON+1e-12:
            raise ValueError('FRESH_COMMAND_CODEC_ROUNDTRIP_FAILED')
        states.append(state); actions.append(nominal)
        annotations.append(stages[str(old['command_source_sha256'][command])])
    atomic_npz(folder/'TRAIN_WINDOWS.npz', state7=old['state7'], latent7=old['nominal_latent7'],
        state_eef10=np.asarray(states), nominal_policy_H40=np.asarray(actions), native_step=old['native_step'],
        frame_index=np.arange(61)*5, phase=np.asarray(annotations))
    episode = dict(id=eid, task='PICK_HOLD', split='TRAIN', layout=row['layout'], restart=row['restart'],
        prefix_atoms=10, native_integrations=21703, windows=61, frames=321, commands=640,
        language=original['language'], mount=original['source_proof']['mount'],
        labels=dict(file=eid+'/TRAIN_WINDOWS.npz', sha256=sha(folder/'TRAIN_WINDOWS.npz')),
        frame_shards=frames, supervised_anchors=list(range(61)), expert_stage_audit_only=annotations,
        original_recording_manifest_sha256=sha(original_path), source_run=Path(root).name,
        source_spec_sha256=sha(Path(root)/'EXECUTION_SPEC.json'),
        source_native_result_sha256=sha(Path(root)/'native-01/RESULT.json'))
    return episode, old['state7'], old['nominal_latent7']


def admit(panel_root, output, *, expected_panel_sha256):
    panel_root, output = Path(panel_root), Path(output)
    panel_path = panel_root/'PANEL.private.json'
    if sha(panel_path) != expected_panel_sha256:
        raise ValueError('FIXED_FRESH_PRODUCTION_PANEL_BYTES_REQUIRED')
    panel = json.loads(panel_path.read_bytes())
    rows = selected_assignments(panel)
    pin = dict(path=str(panel_path), sha256=expected_panel_sha256)
    # Do not leave a partially published dataset merely because the next
    # declared condition is still running. No replacement with an easier one.
    for row in rows:
        source_episode(panel_root/'runs'/row['native_run'], row, pin)
    started = time.time()
    output.mkdir()
    write(output/'STATE.private.json', dict(state='preparing', D100_new=False))
    episodes, states, actions = [], [], []
    try:
        for row in rows:
            ep, s, a = derive_episode(panel_root/'runs'/row['native_run'], output, row, pin)
            episodes.append(ep); states.extend(s); actions.extend(a)
        states, actions = np.asarray(states), np.asarray(actions).reshape(-1, 7)
        write(output/'STATISTICS.private.json', dict(
            state=Quantiles.fit(states).summary(states), actions=Quantiles.fit(actions).summary(actions),
            fit_unique_windows=1220, fitted_before_oversampling=True, split='TRAIN', codec_version=VERSION,
            epsilon=EPSILON, state_gripper_transformed=False))
        manifest = dict(schema=SCHEMA, state='committed', dataset_id=DATASET_ID, scope=SCOPE,
            tasks=['PICK_HOLD'], objects=1, split='TRAIN', D100_new=False, D300_new=False,
            PICK_PLACE_admitted=False, legacy_engineering_receipts_modified=False,
            old_2EP_reused=0, old_engineering16_reused=0, phase_input_to_student=False,
            consumption_proven=True, actual_success_tail_s=2., codec=VERSION, physics_dt_s=.002,
            frame_period_ticks=50, anchor_period_ticks=250, command_period_ticks=25,
            selection_rule='FIRST20_DECLARED_ASSIGNMENTS_ALL_COMPLETE_NO_SUBSTITUTION',
            source_panel=pin, full_source_population=24, selected_assignments=20,
            episodes=episodes, supervised_windows=1220,
            statistics=dict(file='STATISTICS.private.json', sha256=sha(output/'STATISTICS.private.json')))
        manifest_contract(manifest)
        write(output/'MANIFEST.private.json', manifest)
        write(output/'STATE.private.json', dict(state='committed', supervised_windows=1220, D100_new=False))
    except BaseException as error:
        write(output/'STATE.private.json', dict(state='incomplete', error=repr(error), D100_new=False))
        raise
    return dict(dataset_id=DATASET_ID, manifest_sha256=sha(output/'MANIFEST.private.json'),
        episodes=20, supervised_windows=1220, dual_camera_frames=6420, consumed_commands=12800,
        D100_new=False, PICK_PLACE_admitted=False, new_physical_requests=0, optimizer_updates=0,
        wall_s=time.time()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-root', type=Path, required=True)
    parser.add_argument('--panel-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(admit(args.panel_root, args.output, expected_panel_sha256=args.panel_sha256)))
