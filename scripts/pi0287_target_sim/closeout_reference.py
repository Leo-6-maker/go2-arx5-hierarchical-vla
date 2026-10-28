"""Recompute saved hard-limit readouts and encode only actual saved RGB; no simulator."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    repo = Path(__file__).resolve().parents[2]
    output = repo / 'reports/PI0287/execution'
    evidence = output / 'core-v2/evidence/reference-fixtures-hard-limits-v2'
    names = ('hold', 'horizontal_5cm', 'vertical_5cm', 'pure_rotation_0p10rad',
             'vertical_return_5cm', 'hotswap_0p4s')
    dofs = json.loads((output / 'core/evidence/scene-smoke-n1/RESULT.json').read_text())['dof_names']
    results = []
    for name in names:
        path = evidence / (name + '.json')
        value = json.loads(path.read_text())
        violations = []
        for row in value['trace']:
            state = row['actual_DOF_hard_limit_state']
            q, lower, upper = (np.asarray(state[k]) for k in ('positions', 'lower', 'upper'))
            assert q.shape == lower.shape == upper.shape == (len(dofs),)
            assert np.isfinite(np.r_[q, lower, upper]).all()
            penetration = np.maximum(np.maximum(lower - q, q - upper), 0.)
            assert np.array_equal(penetration, np.asarray(state['penetration']))
            assert bool(np.any(penetration > 0)) == row['hard_limit'] == state['hard_limit']
            assert [dofs[i] for i in np.flatnonzero(penetration)] == state['offending']
            for i in np.flatnonzero(penetration):
                violations.append(dict(time_s=row['time_s'], joint=dofs[i],
                    position_rad=float(q[i]), lower_rad=float(lower[i]), upper_rad=float(upper[i]),
                    penetration_rad=float(penetration[i])))
        duration = value['trace'][-1]['time_s']
        prefix_xyz = [float(np.linalg.norm(np.asarray(row['actual_tcp'][:3]) - row['installed_tcp'][:3]))
                      for row in value['trace']]
        assert duration < 10. and value['metrics'] is None
        assert value['failure'].startswith('NATIVE_ABORT ') or violations
        assert value['status'] == 'VALID_REFERENCE_QUALITY_FAIL'
        frames = np.load(evidence / (name + '-RGB.npz'))['frames']
        assert len(frames) == len(value['camera_trace']) and frames.shape[1:] == (96, 192, 3)
        results.append(dict(name=name, verdict='FAIL', completed_ticks=len(value['trace']),
            last_saved_time_s=duration, failure=value['failure'], violations=violations,
            terminal_abort_state='UNOBSERVED' if value['failure'].startswith('NATIVE_ABORT ') else 'SAVED',
            late2s_tracking_metrics='UNOBSERVED_EARLY_TERMINATION',
            observed_prefix_last_XYZ_m=prefix_xyz[-1], observed_prefix_max_XYZ_m=max(prefix_xyz),
            observed_prefix_last_base_height_m=value['trace'][-1]['base'][2],
            hotswap_completed=False if name == 'hotswap_0p4s' else None,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), RGB_pairs=len(frames)))
    report = dict(status='CPU_RECOMPUTATION_PASS_REFERENCE_QUALITY_FAIL', pass_count=0, denominator=6,
        fixtures=results, no_threshold_relaxation=True, original_soft_limit_attempts='ENGINEERING_INVALID_RETAINED',
        controller_class='HISTORICAL_EXECUTOR_BASELINE', root_cause='UNRESOLVED_NOT_A_GENERAL_WBC_VERDICT',
        G1='PARTIAL_SUBCHECKS_ONLY', G2='REFERENCE_QUALITY_FAIL', G3='NOT_RUN', G4='NOT_RUN', G5='NOT_RUN',
        overall='PARTIAL', selected='NONE', deployment='NONE', new_neural_queries=0, new_physics=0)
    (output / 'REFERENCE_CPU_RECOMPUTATION.json').write_text(json.dumps(report, indent=2) + '\n')

    # Saved sparse RGB only. Playback is explicitly slowed; no invented missing frames.
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio
    video = output / 'PI0287_ACTUAL_FAILED_REFERENCES.mp4'
    arrays = [np.load(evidence / (n + '-RGB.npz'))['frames'] for n in names]
    common = min(len(a) for a in arrays)
    with imageio.get_writer(video, fps=2, codec='libx264', quality=8, macro_block_size=1) as writer:
        for index in range(common):
            canvas = Image.new('RGB', (768, 460), '#181818')
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), 'PI0287 ACTUAL RGB | 6 FAILED reference fixtures | historical executor', fill='white')
            for j, array in enumerate(arrays):
                x, y = (j % 2) * 384, 36 + (j // 2) * 132
                draw.text((x + 4, y), names[j] + ' | body / wrist', fill='white')
                frame = Image.fromarray(array[index]).resize((384, 108), Image.Resampling.NEAREST)
                canvas.paste(frame, (x, y + 18))
            draw.text((12, 436), f'Saved capture {index+1}/{common}; slowed playback; NOT grasp/task success', fill='white')
            writer.append_data(np.asarray(canvas))
    (output / 'VIDEO_RECEIPT.json').write_text(json.dumps(dict(path=video.name,
        sha256=hashlib.sha256(video.read_bytes()).hexdigest(), source='SAVED_REAL_DUAL_RGB_NPZ',
        common_frames_per_condition=common, presentation_fps=2, inference='NONE',
        omitted_later_frames='RETAINED_IN_RAW_NPZ', task_success=False), indent=2) + '\n')
    (output / 'GATE_SUMMARY.json').write_text(json.dumps(dict(terminal='REFERENCE_QUALITY_FAIL',
        overall='PARTIAL', G0='CPU_ASSET_IDENTITY_COMPLETE_HISTORICAL_BASELINE',
        G1='PARTIAL_SUBCHECKS_ONLY', G2='FAIL_0_OF_6', G3='NOT_RUN', G4='NOT_RUN', G5='NOT_RUN',
        old_soft_limit_attempt='ENGINEERING_INVALID_RETAINED',
        current_reference_attempt='RAW_HARD_LIMIT_RECOMPUTED_NATIVE_ABORT_RETAINED',
        no_grasp_or_deployment_claim=True, selected='NONE', deployment='NONE'), indent=2) + '\n')
    (output / 'DATA_MANIFEST.json').write_text(json.dumps(dict(status='NOT_ADMITTED_REFERENCE_GATE_FAIL',
        task='GO2_PLACE_V1', teacher_episodes=0, TRAIN=0, DEV=0, PILOT=0, samples=0,
        statistics_fit=False, official_processor_read=False, raw_reference_diagnostics=str(evidence),
        failed_reference_denominator=6, selected='NONE', deployment='NONE'), indent=2) + '\n')
    (output / 'TRAINING_READINESS.json').write_text(json.dumps(dict(status='NOT_READY_REFERENCE_GATE_FAIL',
        dataset_admitted=False, launch_allowed=False, official_CPU_processor='NOT_RUN_NO_TASK_DATA',
        GR00T_forwards=0, GR00T_updates=0, WBC_updates=0, selected='NONE', deployment='NONE'), indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], video=str(video), fixtures=6)))


if __name__ == '__main__':
    main()
