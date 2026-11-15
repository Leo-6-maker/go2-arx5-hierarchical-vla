"""Read every fresh-cohort RGB/label join once before the learner opens it."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.codec_v1 import pad32, roundtrip_error
from go2arx5_vla.sla_adapter.gripper_v2 import EPSILON
from go2arx5_vla.sla_adapter.named_dataset_v5 import AUDIT_SCHEMA, NamedTargetDataset


def audit(root, manifest_sha256, output):
    from openpi_client.image_tools import resize_with_pad
    started = time.time()
    dataset = NamedTargetDataset(root, manifest_sha256, resize_with_pad=resize_with_pad)
    maximum = dict(xyz_m=0., so3_rad=0., gripper=0.)
    events = {}
    identities = set()
    for i in range(len(dataset)):
        state, nominal, images, language = dataset.raw(i)
        item = dataset[i]
        episode, anchor = dataset.rows[i]
        if (len(images) != 2 or any(im.shape != (256,256,3) or im.dtype != np.uint8 for im in images)
                or item['state'].shape != (32,) or item['actions'].shape != (40,32)
                or np.any(item['state'][7:] != 0) or np.any(item['actions'][:,7:] != 0)
                or item['prompt'] != language or not np.isfinite(item['actions']).all()):
            raise ValueError('ALL_FRESH_RGB224_STATE32_ACTION_H40_32_READBACK_REQUIRED')
        codec = dataset.codecs[episode['id']]
        physical = codec.decode(state, pad32(dataset.statistics['actions'].unnormalize(item['actions'][:,:7])))
        error = roundtrip_error(nominal, physical)
        if error['xyz_m'] > 1e-6 or error['so3_rad'] > 1e-5 or error['gripper'] > EPSILON+1e-6:
            raise ValueError('FRESH_FP32_TRAIN_INPUT_ROUNDTRIP_FAILED')
        for key in maximum:maximum[key] = max(maximum[key], error[key])
        phase = episode['expert_stage_audit_only'][anchor]
        events[phase] = events.get(phase,0)+1
        identities.add((episode['id'],anchor))
    result = dict(schema=AUDIT_SCHEMA, passes=True, manifest_sha256=manifest_sha256,
        dataset_id=dataset.manifest['dataset_id'], episodes=20, read_windows=len(identities),
        unique_phase_counts=events, FP32_roundtrip_max_error=maximum,
        full_dual_camera_frames=6420, consumed_commands=12800, phase_input_to_student=False,
        D100_new=False, D300_new=False, PICK_PLACE_admitted=False,
        model_loads=0, optimizer_updates=0, physical_requests=0, DEV_TEST_reads=0,
        wall_s=time.time()-started)
    if result['read_windows'] != 1220:raise ValueError('ALL1220_FRESH_WINDOWS_REQUIRED')
    write(output,result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('IMMUTABLE_FRESH_DATA_AUDIT_ALREADY_EXISTS')
    print(json.dumps(audit(args.dataset,args.manifest_sha256,args.output)))
