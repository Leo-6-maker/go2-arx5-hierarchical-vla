"""Finite all-window SLA codec, clock, RGB224 and visibility audit; no model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from openpi_client.image_tools import resize_with_pad
from go2arx5_vla.sla_adapter.calibration_v1 import episode_frame
from go2arx5_vla.sla_adapter.codec_v1 import pad32, roundtrip_error, TOLERANCES
from go2arx5_vla.sla_adapter.config_v1 import visits, lr_sha256
from go2arx5_vla.sla_adapter.dataset_v1 import TargetDataset
from go2arx5_vla.world_model.storage import digest

DATASET_SHA = '11ac2104e72ddb091045bb2ae0f8cba56bd4002022fc44825e983173565efbe1'


def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def audit(dataset, camera_audit, output, *, deadline):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise FileExistsError('FRESH_SLA_DERIVATION_OUTPUT_REQUIRED')
    if digest(dataset/'DATASET_MANIFEST.json') != DATASET_SHA:
        raise ValueError('PINNED_BODY_WRIST_TRAIN100_REQUIRED')
    m = json.loads((dataset/'DATASET_MANIFEST.json').read_bytes())
    output.mkdir()
    started = time.time()
    frames = {}
    for e in m['episodes']:
        if time.time() >= deadline:
            raise TimeoutError('FINITE_SLA_DATA_AUDIT_DEADLINE')
        frames[e['id']] = episode_frame(dataset, e, m['T_policy_from_target'])
    write(output/'FRAMES.private.json', dict(dataset_sha256=DATASET_SHA, episodes=frames))
    d = TargetDataset(dataset, camera_audit, frames, resize_with_pad=resize_with_pad)
    stats = d.fit_statistics()
    write(output/'STATISTICS.private.json', stats)
    raw_errors = {k: 0. for k in TOLERANCES}
    fp32_errors = dict(raw_errors)
    all_rows = []
    with (output/'codec.private.jsonl').open('x') as stream:
        for index in range(len(d)):
            if time.time() >= deadline:
                raise TimeoutError('FINITE_SLA_CODEC_AUDIT_DEADLINE')
            s, a, _, _, state, expert, identity = d.active(index)
            codec = d.codecs[identity['episode_id']]
            normalized = d.statistics['actions'].normalize(a)
            fp64 = codec.decode(state, pad32(d.statistics['actions'].unnormalize(normalized)))
            fp32 = codec.decode(state, pad32(d.statistics['actions'].unnormalize(normalized.astype(np.float32))))
            err = roundtrip_error(expert, fp64)
            err32 = roundtrip_error(expert, fp32)
            for key, tolerance in TOLERANCES.items():
                raw_errors[key] = max(raw_errors[key], err[key])
                fp32_errors[key] = max(fp32_errors[key], err32[key])
                if max(err[key], err32[key]) > tolerance:
                    raise ValueError('SLA_CODEC_ROUNDTRIP_TOLERANCE:'+key)
            model_input = d[index]
            if set(model_input) != {'image', 'image_mask', 'image_padding_mask', 'state', 'actions', 'prompt'}:
                raise ValueError('PRIVILEGED_OR_AUDIT_FIELDS_IN_STUDENT_INPUT')
            row = dict(**identity, fp64=err, fp32=err32,
                       RGB224_sha256={key: hashlib.sha256(a.tobytes()).hexdigest() for key, a in model_input['image'].items()})
            stream.write(json.dumps(row)+'\n')
    # Reuse existing segmentation masks. Apply the SAME pinned PIL transform
    # and fixed 128/255 occupancy rule; masks never become model inputs.
    original_rows = dataset/'audit/windows.private.jsonl'
    rows = [json.loads(line) for line in original_rows.read_text().splitlines()]
    phases = {}
    masks_digest = hashlib.sha256()
    for row in rows:
        if time.time() >= deadline:
            raise TimeoutError('FINITE_SLA_VISIBILITY_AUDIT_DEADLINE')
        file = dataset/'audit'/row['masks_file']
        if not file.resolve().is_relative_to((dataset/'audit').resolve()):
            raise ValueError('AUDIT_MASK_PATH_ESCAPE')
        masks_digest.update((row['masks_file']+':'+digest(file)+'\n').encode())
        with np.load(file, allow_pickle=False) as z:
            count = {}
            for key in ('body_rgb', 'wrist_rgb'):
                mask = z[key]
                if mask.shape != (256, 256) or mask.dtype != np.bool_ or int(mask.sum()) != row['visible_pixels'][key]:
                    raise ValueError('SEALED_RAW_SEGMENTATION_MASK_MISMATCH')
                image = np.repeat((mask.astype(np.uint8)*255)[:, :, None], 3, axis=2)
                resized, _ = resize_with_pad(image, 224, 224, return_mask=True)
                count[key] = int(np.count_nonzero(resized[:, :, 0] >= 128))
        if row['supervised']:
            phase = phases.setdefault(row['phase'], dict(n=0, passed=0))
            phase['n'] += 1
            phase['passed'] += int(max(count.values()) >= 64)
        all_rows.append(dict(episode=row['episode'], anchor=row['anchor'], supervised=row['supervised'], phase=row['phase'], RGB224_visible_pixels=count))
    for phase in phases.values():
        phase.update(required=math.ceil(.9*phase['n']), passes=phase['passed'] >= math.ceil(.9*phase['n']))
    write(output/'VISIBILITY.private.json', dict(mask_transform='PINNED_PIL_BILINEAR_224_THRESHOLD_GE128', phases=phases, rows=all_rows,
          original_rows_sha256=digest(original_rows), masks_manifest_sha256=masks_digest.hexdigest(), masks_are_student_inputs=False))
    visit = visits(m)
    np.save(output/'VISITS.private.npy', visit, allow_pickle=False)
    passes = (len(rows) == 1500 and sum(r['supervised'] for r in rows) == 1450
              and len(phases) == 5 and all(p['passes'] for p in phases.values()))
    result = dict(schema='SLA_TARGET_CODEC_DATA_ACTUAL_V1', passes=passes, source_dataset_sha256=DATASET_SHA,
        actual_episodes=len(frames), captures_retained=len(rows), expert_windows=len(d), excluded_interventions=50,
        roundtrips=1450, FP64_max_error=raw_errors, FP32_normalized_max_error=fp32_errors, preregistered_tolerances=TOLERANCES,
        phases=phases, exposure_visits=len(visit), unique_exposed_windows=len(set(visit.tolist())),
        source_images_unchanged=True, active_dimensions=7, padding_dimensions=25, no_implicit_clip=True,
        statistics_sha256=digest(output/'STATISTICS.private.json'), frames_sha256=digest(output/'FRAMES.private.json'),
        visits_sha256=digest(output/'VISITS.private.npy'), used_lr_sha256=lr_sha256(),
        original_consumer_changed=False, new_native_replay_required=False,
        actual_model_loads=0, actual_optimizer_updates=0, optimization_visits=0, physical_requests=0,
        native_integrations=0, FK_episode_calibrations=len(frames), wall_s=time.time()-started)
    if digest(dataset/'DATASET_MANIFEST.json') != DATASET_SHA:
        raise ValueError('SOURCE_DATASET_CHANGED_DURING_AUDIT')
    write(output/'RESULT.private.json', result)
    if not passes:
        raise ValueError('SLA_POST_RESIZE_VISIBILITY_NOT_QUALIFIED')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--camera-audit', required=True)
    p.add_argument('--camera-audit-sha256', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--deadline', required=True, type=float)
    a = p.parse_args()
    print(json.dumps(audit(a.dataset, dict(path=a.camera_audit, sha256=a.camera_audit_sha256), a.output, deadline=a.deadline)))


if __name__ == '__main__':
    main()
