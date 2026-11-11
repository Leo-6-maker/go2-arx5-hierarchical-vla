"""Audit actual RGB Processor/collator and separate geometric mask copies.

No masks or privileged geometry enter a policy example. No model is loaded.
Final patch unpacking is checked against actual RGB before counting mask pixels.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

from go2arx5_vla.world_model.native_camera_transfer_v1 import coverage, CAMERA_ID
from scripts.wm02.export_native_camera_transfer_v1 import read, write, sha, STATS


def unpack(patches, grid, *, patch_size, temporal_size, merge_size):
    t, h, w = map(int, grid)
    m, p = merge_size, patch_size
    a = np.asarray(patches).reshape(t, h//m, w//m, m, m, 3, temporal_size, p, p)
    return a.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(t*temporal_size, 3, h*p, w*p)


def audit(dataset, output, official_source, cosmos, *, deadline):
    from scripts.pi0292.train_adapter import build_processor
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError('FRESH_PROCESSOR_AUDIT_REQUIRED')
    manifest_raw = read(dataset/'DATASET_MANIFEST.json')
    manifest = json.loads(manifest_raw)
    raw_coverage = json.loads(read(dataset/'RAW_COVERAGE.private.json'))
    if manifest['profile'] != CAMERA_ID or raw_coverage['coverage']['captures'] != 1500:
        raise ValueError('COMPLETE_DECLARED_CAMERA_DATA_REQUIRED')
    stats = json.loads(read(dataset/'statistics.json', STATS))
    processor = build_processor(Path(official_source), Path(cosmos), stats)
    ip = processor.collator.processor.image_processor
    if type(ip).__name__ != 'Qwen2VLImageProcessorFast':
        raise ValueError('REGISTER_NEW_FINAL_IMAGE_PROCESSOR_BEFORE_AUDIT')
    from gr00t.data.types import VLAStepData, MessageType
    from gr00t.data.embodiment_tags import EmbodimentTag
    params = dict(patch_size=ip.patch_size, temporal_size=ip.temporal_patch_size, merge_size=ip.merge_size)
    output.mkdir()
    write(output/'PROCESSOR_IDENTITY.private.json', dict(
        official_source=str(official_source), cosmos=str(cosmos), statistics_sha256=STATS,
        image_processor_config=ip.to_dict(), eval_transform=str(processor.eval_image_transform),
        final_processor_source_sha256=sha(Path(inspect.getfile(type(ip))).read_bytes()),
        project_processor_source_sha256=sha(Path(inspect.getfile(type(processor))).read_bytes()),
        masks_are_student_inputs=False, model_loads=0),)
    rows = [json.loads(line) for line in (dataset/'audit/windows.private.jsonl').read_text().splitlines()]
    index = {(r['episode'], r['anchor']): r for r in rows}
    start = time.monotonic()
    calls = 0
    with (output/'windows.private.jsonl').open('x') as stream:
        for ep in manifest['episodes']:
            with np.load(dataset/ep['file'], allow_pickle=False) as archive:
                packet = {k: archive[k] for k in archive.files}
            if sha((dataset/ep['file']).read_bytes()) != ep['sha256']:
                raise ValueError('NEW_EPISODE_CHANGED')
            for anchor in range(15):
                if time.time() >= deadline:
                    raise TimeoutError('FINITE_PROCESSOR_AUDIT_DEADLINE')
                row = dict(index[ep['id'], anchor])
                images = []
                for camera in range(2):
                    path = dataset/str(packet['rgb_paths'][anchor, camera])
                    raw = read(path, str(packet['rgb_sha256'][anchor, camera]))
                    with Image.open(path) as frame:
                        images.append(np.asarray(frame).copy())
                state = packet['actual_state10'][anchor]
                atom = int(packet['native_command_anchor'][anchor])
                commands = packet['committed_absolute10'][atom:atom+40]
                if commands.shape != (40, 10):
                    raise ValueError('ORIGINAL_H40_ANCHOR_MUST_BE_PRESERVED')
                value = VLAStepData(images={'external_rgb':[images[0]], 'wrist_rgb':[images[1]]},
                    states={'eef_9d':state[None,:9], 'gripper_position':state[None,9:]},
                    actions={'eef_9d':commands[:,:9], 'gripper_position':commands[:,9:]},
                    masks=None, text=packet['instruction'].item(),
                    embodiment=EmbodimentTag.NEW_EMBODIMENT, metadata={})
                processed = processor([{'type':MessageType.EPISODE_STEP.value, 'content':value}])
                actual = processor.collator([processed])['inputs']
                calls += 1
                grids = actual['image_grid_thw'].cpu().numpy()
                pixels = actual['pixel_values'].cpu().numpy()
                if grids.shape != (2, 3):
                    raise ValueError('EXACT_TWO_FINAL_IMAGE_GRIDS_REQUIRED')
                vlm_images = processed['vlm_content']['images']
                if len(vlm_images) != 2:
                    raise ValueError('TWO_ORDERED_CAMERA_INPUTS_REQUIRED')
                with np.load(dataset/'audit'/row['masks_file'], allow_pickle=False) as saved:
                    masks = [saved[k].copy() for k in ('body_rgb', 'wrist_rgb')]
                final_mask_inputs = []
                offset = 0
                for image, vlm_image, mask, grid in zip(images, vlm_images, masks, grids):
                    transformed = processor.eval_image_transform(image)
                    if not np.array_equal(vlm_image.numpy(), transformed.numpy()):
                        raise ValueError('COLLATOR_CAMERA_ORDER_OR_EVAL_TRANSFORM_MISMATCH')
                    # This fixed profile has 256x256 resize/crop and no spatial
                    # augmentation. Prove identity for each actual image/mask.
                    if not np.array_equal(transformed.numpy().transpose(1, 2, 0), image):
                        raise ValueError('REGISTER_MASK_GEOMETRY_FOR_NONIDENTITY_EVAL_TRANSFORM')
                    n = int(np.prod(grid))
                    reconstructed = unpack(pixels[offset:offset+n], grid, **params)
                    offset += n
                    normalized = image.transpose(2, 0, 1).astype(np.float32)/255.
                    normalized = (normalized - np.asarray(ip.image_mean, np.float32)[:, None, None]) / np.asarray(ip.image_std, np.float32)[:, None, None]
                    if reconstructed.shape[1:] != normalized.shape or not np.allclose(
                            reconstructed, normalized[None], rtol=0, atol=2e-6):
                        raise ValueError('ACTUAL_FINAL_VLM_PIXELS_NOT_FULL_RGB256_GEOMETRY')
                    mask_image = np.repeat((mask.astype(np.uint8)*255)[:, :, None], 3, axis=2)
                    mask_input = processor.eval_image_transform(mask_image)
                    if not np.array_equal(mask_input.numpy().transpose(1, 2, 0), mask_image):
                        raise ValueError('AUDIT_MASK_EVAL_GEOMETRY_DIFFERS')
                    final_mask_inputs.append(mask_input)
                # A separate invocation with normalization disabled lets us
                # unpack visible masks; resize/crop/patch geometry is unchanged.
                final_masks = ip(images=final_mask_inputs, return_tensors='pt',
                                 do_rescale=False, do_normalize=False)
                if not np.array_equal(final_masks['image_grid_thw'].cpu().numpy(), grids):
                    raise ValueError('MASK_AND_ACTUAL_IMAGE_GRID_GEOMETRY_DIFFERS')
                counts, offset = {}, 0
                for camera, mask, grid in zip(('body_rgb', 'wrist_rgb'), masks, grids):
                    n = int(np.prod(grid))
                    decoded = unpack(final_masks['pixel_values'][offset:offset+n].cpu().numpy(), grid, **params)
                    offset += n
                    if not np.array_equal(decoded, np.broadcast_to(mask[None, None]*255, decoded.shape)):
                        raise ValueError('FINAL_MASK_SPATIAL_GEOMETRY_DIFFERS')
                    counts[camera] = int(np.count_nonzero(decoded[0, 0] >= 128))
                row.update(processed_visible_pixels=counts, final_image_grid_thw=grids.tolist(),
                    final_RGB_pixel_values_sha256=sha(pixels.tobytes()), masks_are_student_inputs=False)
                index[ep['id'], anchor] = row
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
            write(output/'PROGRESS.private.json', dict(actual_Processor_calls=calls, wall_s=time.monotonic()-start))
    final = coverage(list(index.values()), processed=True)
    result = dict(schema='NATIVE_CAMERA_ACTUAL_PROCESSOR_COVERAGE_V1',
        camera_id=CAMERA_ID, dataset_manifest_sha256=sha(manifest_raw),
        raw_coverage=raw_coverage['coverage'], final_coverage=final,
        camera_coverage_passes=raw_coverage['coverage']['passes'] and final['passes'],
        actual_Processor_calls=calls, actual_model_calls=0, optimizer_updates=0,
        physics_integrations=0, training_admitted=False, wall_s=time.monotonic()-start)
    write(output/'RESULT.private.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'output', 'official_source', 'cosmos'):
        parser.add_argument('--'+key.replace('_', '-'), required=True, type=Path)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(**vars(args))))


if __name__ == '__main__':
    main()
