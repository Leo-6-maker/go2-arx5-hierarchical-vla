"""Prepare only the named pilot's current RGB; cache the unchanged frozen teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.dense_dataset_v4 import manifest_contract

VIEW = 'SLA_NAMED_DENSE_PICK_HOLD_RGB224_TEACHER_VIEW_V4_1'
CACHE = 'SLA_FROZEN_OMNIVGGT_NAMED_DENSE_PICK_HOLD_CACHE_V4_1'


def contract(spec):
    root = Path(spec['dataset'])
    if sha(root/'MANIFEST.private.json') != spec['dense_manifest_sha256']:
        raise ValueError('EXACT_NAMED_TRAIN_DATASET_REQUIRED_FOR_DENSE_SPATIAL_PREPARATION')
    manifest = json.loads((root/'MANIFEST.private.json').read_bytes())
    manifest_contract(manifest)
    audit = spec['target_audit']
    if sha(audit['path']) != audit['sha256']:
        raise ValueError('NATIVE_CONSUMPTION_AND_ALL_WINDOW_READER_PROOF_CHANGED')
    q = json.loads(Path(audit['path']).read_bytes())
    if (q.get('schema') != 'SLA_NAMED_DENSE_PILOT_ALL_WINDOW_CPU_READBACK_V4_1'
            or not q['passes'] or q['read_windows'] != 122 or q['manifest_sha256'] != spec['dense_manifest_sha256']):
        raise ValueError('NAMED_122_WINDOW_TARGET_ONLY_SPATIAL_SCOPE_REQUIRED')
    return manifest


def pixels(spec):
    contract(spec)
    from PIL import __version__ as pillow_version
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.dense_dataset_v4 import DensePilotDataset
    if pillow_version != '11.3.0':
        raise ValueError('EXACT_STUDENT_PILLOW_REQUIRED')
    dataset = DensePilotDataset(spec['dataset'], spec['dense_manifest_sha256'], resize_with_pad=resize_with_pad)
    root = Path(spec['registered_root'])/'rgb224';root.mkdir()
    rows = []
    for i in range(len(dataset)):
        if time.time() >= spec['deadline_epoch']:
            raise TimeoutError('FINITE_NAMED_DENSE_RGB_VIEW')
        item = dataset[i]
        names = ('base_0_rgb', 'left_wrist_0_rgb')
        pixels = np.stack([item['image'][name] for name in names])
        masks = np.stack([item['image_padding_mask'][name] for name in names])
        if pixels.shape != (2,224,224,3) or pixels.dtype != np.uint8 or masks.shape != (2,224,224) or masks.dtype != bool:
            raise ValueError('EXACT_CURRENT_STUDENT_RGB_AND_PIXEL_MASKS_REQUIRED')
        if not all(item['image_mask'][name] for name in names) or item['image_mask']['right_wrist_0_rgb']:
            raise ValueError('ONLY_THE_TWO_VALID_BODY_AND_WRIST_VIEWS')
        p = root/f'{i:04d}.npz';np.savez(p, pixels=pixels, padding_mask=masks)
        identity = dataset.identity(i)
        rows.append(dict(index=i, episode_id=identity['episode_id'], anchor=identity['anchor'], file=p.name,
            sha256=sha(p), pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
            padding_mask_sha256=hashlib.sha256(masks.tobytes()).hexdigest()))
    manifest = dict(schema=VIEW, passes=True, rows=rows, source_dataset_sha256=spec['dense_manifest_sha256'],
        target_audit=spec['target_audit'], actual_pixel_window_reads=122, privileged_inputs_saved=False,
        state_actions_or_labels_saved=False, source_cameras=list(names), missing_third_camera_fed_to_teacher=False,
        model_loads=0, optimizer_updates=0, optimization_visits=0, physical_requests=0)
    write(root/'MANIFEST.private.json', manifest)
    return dict(passes=True, pixel_view=dict(path=str(root),sha256=sha(root/'MANIFEST.private.json')),
                actual_pixel_window_reads=122, model_loads=0, optimizer_updates=0, physical_requests=0)


def cache(spec, counts):
    contract(spec)
    root = Path(spec['pixel_view']['path'])
    if sha(root/'MANIFEST.private.json') != spec['pixel_view']['sha256']:
        raise ValueError('EXACT_NEW_PILOT_PIXEL_VIEW_REQUIRED')
    view = json.loads((root/'MANIFEST.private.json').read_bytes())
    if (view['schema'] != VIEW or view['source_dataset_sha256'] != spec['dense_manifest_sha256']
            or view['target_audit'] != spec['target_audit'] or len(view['rows']) != 122
            or view['state_actions_or_labels_saved']):
        raise ValueError('MATCHED_CURRENT_TARGET_PIXELS_ONLY_NO_TEACHER_PRIVILEGED_INPUTS')
    from scripts.sla.cache_teacher_v1 import run
    result = run(spec, counts, view_schema=VIEW, window_count=122, cache_schema=CACHE)
    result.update(schema='SLA_NAMED_DENSE_FROZEN_TEACHER_CACHE_ACTUAL_V4_1',
                  dataset_id='T_PICK_HOLD_TRAIN01_P10_2EP_V4_1', D100_new=False, PICK_PLACE_admitted=False)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('spec', type=Path);p.add_argument('--mode', choices=['pixels','cache'], required=True)
    a = p.parse_args();s = json.loads(a.spec.read_bytes())
    counts = dict(teacher_model_load_attempts=0, teacher_model_constructions=0, teacher_model_loads=0,
        constructor_preinit_load_attempts=0, constructor_preinit_loads=0, constructor_hub_load_attempts=0,
        teacher_forward_attempts=0, teacher_forwards_completed=0, complete_cached_TRAIN_windows=0)
    try:
        result = pixels(s) if a.mode == 'pixels' else cache(s, counts)
    except BaseException as e:
        write(Path(s['registered_root'])/'FAILED.private.json',dict(error=repr(e),actual=counts,
              optimizer_updates=0,physical_requests=0))
        raise
    write(Path(s['registered_root'])/'RESULT.private.json', result)
