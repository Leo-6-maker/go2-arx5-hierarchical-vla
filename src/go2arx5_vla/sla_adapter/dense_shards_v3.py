"""Seal verified dense expert episodes as immutable local lossless shards."""
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.pi0292.durable import write
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0294.e1.artifacts import atomic_npz
from ..world_model.corelab_expert_semantics_v3 import native_endpoint
from ..world_model.fusion_observation import shadow_inference_payload
from ..world_model.stable_task_dataset_v1 import load
from ..world_model.storage import digest
from ..world_model.task_learning_grasp_v1 import B, need
from .dense_windows_v3 import encode_episode

SCHEMA = 'SLA_DENSE_EXPERT_EPISODE_SHARDS_V3'
FRAMES_PER_SHARD = 128
MAX_SHARD_BYTES = 1024**3
CAMERAS = ('body_rgb', 'wrist_rgb')


def _read_frame(root, record, spec_pin, scene_sha256):
    uuid = record['query_uuid']
    directory = root / 'native-01' / 'PAUSED_OBSERVATIONS' / uuid
    path = directory / 'PAUSED_OBSERVATION.private.json'
    need(Path(uuid).name == uuid and not uuid.startswith('.') and str(path) == record['path']
         and record['committed'] is True, 'CANONICAL_COMMITTED_ORIGINAL_FRAME')
    frame, _ = load(path, record['sha256'])
    body = unseal(frame)
    shadow = unseal(body['original_shadow_ABI'])
    payload = shadow_inference_payload(body['original_shadow_ABI'])
    target = shadow['target_observation']
    state10 = np.asarray(payload['state_eef10'], dtype=np.float64)
    need(body['query_uuid'] == uuid and shadow['native_step'] == body['native_step'] == record['native_step']
         and body['native_time_s'] == record['native_time_s']
         and body['world']['execution_spec_sha256'] == target['target_runtime_sha256'] == spec_pin['sha256']
         and body['world']['scene_sha256'] == scene_sha256 and body['world']['native_output'] == str(root / 'native-01')
         and body['transport']['immutable_snapshot_sha256'] == record['memory_snapshot_sha256']
         and target['native_state_sha256'] == record['state_sha256'] == body['files']['STATE.private.bin'],
         'EXACT_FRAME_WORLD_CLOCK_STATE_AND_CAMERA_BINDING')
    paths = ('STATE.private.bin', *(f'images/000000-{name}.png' for name in CAMERAS))
    for relative in paths:
        member = directory / relative
        need(member.resolve() == member and digest(member) == body['files'][relative],
             'ORIGINAL_RGB_AND_STATE_BYTES_VERIFIED')
    pixels = []
    for relative in paths[1:]:
        with Image.open(directory / relative) as img:
            need(img.mode == 'RGB' and img.size == (256, 256), 'EXACT_NATIVE_DUAL_RGB256')
            pixels.append(np.asarray(img).copy())
    return np.stack(pixels), state10, payload['language'], body['files']['images/000000-calibration.json']


def _write_shard(output, filename, arrays):
    path = output / filename
    atomic_npz(path, **arrays)
    size = path.stat().st_size
    need(0 < size <= MAX_SHARD_BYTES, 'FINITE_IMMUTABLE_SHARD_SIZE')
    with np.load(path, allow_pickle=False) as checked:
        need(set(checked.files) == arrays.keys() and
             all(np.array_equal(checked[key], value) for key, value in arrays.items()),
             'LOSSLESS_SEALED_SHARD_READBACK')
    return dict(file=filename, sha256=digest(path), bytes=size)


def seal_episode(root, output, *, result_sha256, archive_sha256):
    """Package one already closed TRAIN episode. The cohort decides BC admission."""
    root, output = Path(root), Path(output)
    numeric, proof = encode_episode(root, expected_result_sha256=result_sha256,
                                    expected_archive_sha256=archive_sha256)
    need(proof['actual_two_second_success_tail'], 'NATIVE_SUCCESS_TAIL_BEFORE_SHARD_PUBLICATION')
    spec, spec_pin = load(Path(proof['spec']['path']), proof['spec']['sha256'])
    archive, _ = load(Path(proof['archive']['path']), archive_sha256)
    confirmed, _ = load(Path(proof['commands']['path']), proof['commands']['sha256'])
    records = archive['records']
    steps = [row['native_step'] for row in records]
    endpoint = native_endpoint(spec)
    need(steps == list(range(B, endpoint + 1, 50)), 'CONTIGUOUS_ACTUAL_10HZ_FRAMES')
    need(all(row['dense_anchor'] == ((row['native_step'] - B) % 250 == 0) for row in records),
         'EXACT_2HZ_ANCHOR_IDENTITY')
    need(confirmed['period_s'] == .05 and
         [row['native_start_step'] for row in confirmed['records']] == list(range(B, endpoint, 25)),
         'CONTIGUOUS_REAL_20HZ_EXPERT_AND_HOLD_COMMANDS')

    output.mkdir(parents=False, exist_ok=False)
    status = output / 'STATE.private.json'
    write(status, dict(state='writing', schema=SCHEMA, episode=root.name))
    try:
        scene_sha256 = proof['mount']['scene_sha256']
        descriptors = []
        language = None
        for start in range(0, len(records), FRAMES_PER_SHARD):
            batch = records[start:start + FRAMES_PER_SHARD]
            frames = [_read_frame(root, row, spec_pin, scene_sha256) for row in batch]
            pixels, states, prompts, calibrations = zip(*frames, strict=True)
            if language is None:
                language = prompts[0]
            need(all(prompt == language for prompt in prompts), 'ONE_ORIGINAL_EPISODE_LANGUAGE')
            arrays = dict(rgb=np.stack(pixels), state_eef10=np.stack(states),
                          native_step=np.asarray(steps[start:start + len(batch)], dtype=np.int64),
                          native_time_s=np.asarray([r['native_time_s'] for r in batch], dtype=np.float64),
                          source_query_uuid=np.asarray([r['query_uuid'] for r in batch]),
                          source_frame_sha256=np.asarray([r['sha256'] for r in batch]),
                          calibration_sha256=np.asarray(calibrations))
            shard = _write_shard(output, f'frames-{start // FRAMES_PER_SHARD:03d}.npz', arrays)
            descriptors.append(dict(shard, first_frame=start, frames=len(batch)))

        commands = confirmed['records']
        numbers = _write_shard(output, 'labels.npz', dict(
            **numeric, command_native_step=np.asarray([r['native_start_step'] for r in commands], dtype=np.int64),
            command_nominal_eef10=np.asarray([r['policy_EEF10'] for r in commands], dtype=np.float64),
            command_installed_eef10=np.asarray([r['installed_consumed_EEF10'] for r in commands], dtype=np.float64),
            command_source_sha256=np.asarray([r['Teacher_decision_sha256'] for r in commands])))
        manifest = dict(schema=SCHEMA, state='committed', episode=root.name, split='TRAIN',
                        scene=spec['scene_id'], scene_sha256=scene_sha256, source_sha=spec['source_sha'],
                        camera_profile=spec['observation_camera_profile'], cameras=list(CAMERAS),
                        language=language, frame_period_ticks=50, anchor_period_ticks=250,
                        command_period_ticks=25, physics_dt_s=.002, source_frame_count=len(records),
                        label_count=proof['encoded_windows'], source_command_count=len(commands),
                        frame_shards=descriptors, label_shard=numbers, source_proof=proof,
                        training_dataset_admitted=False)
        write(output / 'MANIFEST.private.json', manifest)
    except Exception as error:
        write(status, dict(state='incomplete', schema=SCHEMA, episode=root.name, failure=repr(error)))
        raise
    # MANIFEST is the sole commit marker. A process killed before this status
    # update leaves a discoverable commit with all referenced bytes already durable.
    write(status, dict(state='committed', schema=SCHEMA, episode=root.name))
    return manifest
