"""Save exact pinned student RGB224 pixels for the frozen teacher's own process.

This pixel bundle is neither a learned-feature cache nor a training checkpoint.
Only already sealed TRAIN input images/masks are saved; no state, action, stage,
object pose, camera matrix, depth or Teacher label enters the teacher bundle.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.runtime_v1 import data
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha


def run(spec):
    from PIL import __version__ as pillow_version
    if pillow_version != '11.3.0':
        raise ValueError('EXACT_STUDENT_PILLOW_REQUIRED_FOR_RGB224')
    dataset = data(spec)
    if len(dataset) != 1450:
        raise ValueError('COMPLETE_SEALED_TRAIN1450_REQUIRED')
    root = Path(spec['registered_root'])/'rgb224'
    root.mkdir(exist_ok=False)
    rows = []
    for index in range(len(dataset)):
        if time.time() >= spec['deadline_epoch']:
            raise TimeoutError('FINITE_SPATIAL_PIXEL_VIEW')
        item = dataset[index]
        names = ('base_0_rgb', 'left_wrist_0_rgb')
        pixels = np.stack([item['image'][name] for name in names])
        mask = np.stack([item['image_padding_mask'][name] for name in names])
        if pixels.dtype != np.uint8 or pixels.shape != (2,224,224,3) or not all(item['image_mask'][n] for n in names) or item['image_mask']['right_wrist_0_rgb']:
            raise ValueError('ONLY_TWO_VALID_STUDENT_RGB_VIEWS_REQUIRED')
        file = root/f'{index:04d}.npz'
        np.savez(file, pixels=pixels, padding_mask=mask)
        identity = dataset.original.identity(index)
        rows.append(dict(index=index, episode_id=identity['episode_id'], anchor=identity['anchor'],
                         file=file.name, sha256=sha(file),
                         pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                         padding_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
        if (index+1)%100 == 0:
            write(Path(spec['registered_root'])/'PROGRESS.private.json',
                  dict(pixel_windows_completed=index+1, requested=1450, model_calls=0, optimizer_updates=0, physical_requests=0))
    value = dict(schema='SLA_EXACT_TRAIN_RGB224_TEACHER_VIEW_V1', passes=True,
                 source_dataset_sha256=dataset.original.manifest_sha,
                 target_audit=spec['target_audit'], camera_audit=spec['camera_audit'],
                 actual_pixel_window_reads=len(rows), source_cameras=list(names),
                 missing_third_camera_fed_to_teacher=False, original_student_pillow=pillow_version,
                 image_shape=[2,224,224,3], image_dtype='uint8', rows=rows,
                 privileged_inputs_saved=False, state_actions_or_labels_saved=False,
                 pretrained_model_loads=0, teacher_loads=0, optimizer_updates=0, physical_requests=0)
    write(root/'MANIFEST.private.json', value)
    return dict(schema='SLA_SPATIAL_RGB224_VIEW_ACTUAL_V1', passes=True,
                pixel_view=dict(path=str(root),sha256=sha(root/'MANIFEST.private.json')),
                actual_TRAIN_pixel_window_reads=1450, model_loads=0, teacher_calls=0,
                optimizer_updates=0, optimization_visits=0, physical_requests=0)


if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_bytes())
    write(Path(spec['registered_root'])/'RESULT.private.json', run(spec))
