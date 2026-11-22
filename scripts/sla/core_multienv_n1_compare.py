"""N1 engineering readbacks over real native receipts and the original G7 gate."""

import json
from pathlib import Path
import sys
import time

import numpy as np

from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest


def samples(path):
    with path.open() as stream:
        rows = [row for line in stream if (row := json.loads(line))['native_start_step'] >= 5703]
    if [row['native_start_step'] for row in rows] != list(range(5703, 10703, 25)):
        raise ValueError('Original 200 consecutive task intervals required')
    if any(row['ticks'] != 25 or row['native_end_step'] != row['native_start_step'] + 25
           or not row['safety_valid'] for row in rows):
        raise ValueError('Exact safe fixed-clock task intervals required')
    return rows


def pose_delta(before, after, name):
    a, b = (np.asarray([row[name] for row in rows]) for rows in (before, after))
    trans = np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1)
    cos = (np.sum(a[:, :3, :3] * b[:, :3, :3], axis=(1, 2)) - 1) / 2
    return float(trans.max()), float(np.arccos(np.clip(cos, -1, 1)).max())


def compare(spec, row, root, detail):
    if not detail.get('interface_protocol_pass'):
        return dict(status='INTERFACE_FAILED', first_failure=detail.get('terminal_reason'))
    from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState
    import mujoco

    historical = Path(row['recording']['path'])
    if digest(historical) != row['recording']['sha256']:
        raise PermissionError('Historical recording changed')
    template = read_pin(row['reference'])
    authority = read_pin(spec['sim_only_project_authorization'])
    group = template['mobile_condition']['family'] + '-' + template['d2_probe_motion']
    tolerance = read_pin(authority['tolerance'])['engineering_comparison_tolerance'][group]
    before, after = (samples(p) for p in (historical, root/'native-01/MOBILE_SAMPLES.private.jsonl'))
    model = mujoco.MjModel.from_binary_path(template['model_path'])
    fingers = TwoObjectState(model, 'red').finger_q
    tcp, tcp_angle = pose_delta(before, after, 'actual_TCP')
    mount, mount_angle = pose_delta(before, after, 'actual_mount')
    jaw = [(np.asarray([r['qpos'] for r in sequence])[:, fingers]).mean(axis=1)/.044
           for sequence in (before, after)]
    observed = dict(EEF_translation_m=tcp, EEF_angle_rad=tcp_angle,
                    mount_translation_m=mount, mount_angle_rad=mount_angle,
                    jaw_fraction=float(np.abs(jaw[0]-jaw[1]).max()))
    windows = detail['RGB_pairs']
    if [r['native_step'] for r in windows] != list(range(5703, 10703, 250)):
        raise ValueError('Exact paused 20 RGB pairs required')
    return dict(status='WITHIN_FROZEN_STATE_TOLERANCE' if all(
        observed[k] <= tolerance[k] for k in tolerance) else 'OUTSIDE_FROZEN_STATE_TOLERANCE',
        matched_atoms=len(after), paired_RGB=len(windows), limits=tolerance, maximum=observed,
        historical_recording_sha256=row['recording']['sha256'],
        controller_selected_reference='NOT_EXPORTED_BY_FROZEN_CONTROLLER')


def g7(request):
    from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
    from go2arx5_vla.sla_adapter.multitask_data_v6 import C20_STATISTICS
    from scripts.sla.export_multitask_g7_v6 import fp32_audit
    from scripts.sla.multitask_g7_capture_v6 import copy_shards, read_capture

    started = time.monotonic()
    case, native = read_pin(request['case']), read_pin(request['native'])
    if case['expert_training_eligible'] is not False or case['collection_split'] != 'ENGINEERING':
        raise PermissionError('Independent engineering G7 readback only')
    raw = read_pin(dict(path=request['statistics'], sha256=C20_STATISTICS))
    stats = {key: Quantiles(raw[key]['q01'], raw[key]['q99']) for key in ('state', 'actions')}
    try:
        source, manifest, _, codec, arrays = read_capture(dict(instruction=case['task_instruction_id']), case, native)
        copy_shards(source, manifest, arrays)
        audit = fp32_audit([(case['panel_entry_id'], codec, arrays, arrays['previous_actual_command'])], stats)
        failed = any(g['over_original_gate'] or (g['codec_path'] == 'fp32_codec' and g['over_margin'])
                     for g in audit['label_margin_by_domain_codec_phase'])
        result = dict(status='G7_FAILED' if failed else 'G7_PASS',
                      first_failure='Original G7 FP32 margin' if failed else None,
                      windows=manifest['queries'], actual_P10_atoms=int(arrays['consumed_mask'].sum()))
    except (KeyError, ValueError) as error:
        result = dict(status='G7_FAILED', first_failure=str(error))
    return dict(result, wall_s=time.monotonic()-started, training_admissions=0)


if __name__ == '__main__':
    path, sha = sys.argv[1:]
    print(json.dumps(g7(read_pin(dict(path=path, sha256=sha)))))
