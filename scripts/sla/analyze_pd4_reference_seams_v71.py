"""Read the original15 queries without generating or certifying new actions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(group, joint_analysis):
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0292.relay_protocol import unseal
    from scripts.pi0291_arx5_data.native_dataset import pose10

    group, joint_analysis = Path(group), Path(joint_analysis)
    assert sha(group/'RESULT.private.json') == 'd9185b7dff07e364368efccd04a1e2d26c382900c6b02000c7a8904d6e6a2401'
    assert sha(joint_analysis) == '6d2f6d5fba81176055c7e083489d3cc5f30d77871694225da22b5b9d715b3980'
    joints = {r['native_run']: r for r in read(joint_analysis)['rows']}
    assert read(group/'control/LEASE.json')['status'] == 'CLOSED'
    started, cpu = time.time(), time.process_time()
    rows = []
    for outcome in read(group/'RESULT.private.json')['outcomes']:
        root = group/'runs'/outcome['native_run']
        native, gpu = root/'native-01', root.with_name(root.name+'-gpu')/'fit-01'
        assert sha(native/'RESULT.json') == outcome['result_sha256']
        assert sha(gpu/'RESULT.private.json') == outcome['GPU_result_sha256']
        queries = read(gpu/'RESULT.private.json')['records']
        captures = read(native/'TASK_LEARNING_CAPTURES.private.json')['records']
        path = native/'TASK_LEARNING_CONFIRMED_COMMANDS.private.json'
        commands = read(path)['records'] if path.exists() else []
        history = list(read(native/'STABLE_TASK_CYCLES/-1/RESULT.private.json')['windows'])
        history.extend(dict(r, actual_installed_consumed_EEF10_BASE7=r['installed_consumed_EEF10'])
                       for r in commands)
        events = [json.loads(line) for line in (native/'COMMANDS.private.jsonl').read_text().splitlines()]
        for index, (query, capture) in enumerate(zip(queries, captures, strict=True)):
            assert sha(capture['path']) == query['observation_sha256']
            observation = unseal(read(capture['path']))
            shadow = unseal(observation['original_shadow_ABI'])
            actual = matrix10(shadow['target_observation']['actual_state_eef10'])
            frame = np.asarray(shadow['source_policy_view']['T_source_from_target'])
            packet = gpu/'queries'/f'{index:02d}'/'ACTION.npz'
            assert sha(packet) == query['action_sha256']
            with np.load(packet, allow_pickle=False) as arrays:
                raw = arrays['physical_H40']
                target = pose10(np.linalg.inv(frame) @ matrix10(raw), raw[:, 9])
            previous = [r for r in history if r['native_end_step'] <= capture['native_step']][-2:]
            assert [r['native_end_step'] for r in previous] == [capture['native_step']-25, capture['native_step']]
            previous = [matrix10(r['actual_installed_consumed_EEF10_BASE7'][:10]) for r in previous]
            reference = previous[-1]
            target_poses = matrix10(target)
            connected = np.empty_like(target_poses)
            connected[:] = np.eye(4)
            connected[:, :3, 3] = reference[:3, 3] + target[:, :3] - actual[:3, 3]
            connected[:, :3, :3] = reference[:3, :3] @ actual[:3, :3].T @ target_poses[:, :3, :3]
            translation_delta = np.max(np.abs(np.diff(connected[:, :3, 3], axis=0)-np.diff(target[:, :3], axis=0)))
            raw_relative = np.swapaxes(target_poses[:-1, :3, :3], -1, -2) @ target_poses[1:, :3, :3]
            installed_relative = np.swapaxes(connected[:-1, :3, :3], -1, -2) @ connected[1:, :3, :3]
            rotation_delta = np.max(np.abs(raw_relative-installed_relative))
            assert translation_delta < 1e-12 and rotation_delta < 1e-12
            xyz = np.concatenate([np.stack(previous)[:, :3, 3], connected[:, :3, 3]])
            acceleration = np.diff(xyz, n=2, axis=0)/.05**2
            reference_rotation = reference[:3, :3].T @ actual[:3, :3]
            rotation_error = float(np.arccos(np.clip((np.trace(reference_rotation)-1)/2, -1, 1)))
            terminal = index == len(queries)-1
            classification = ('ACCEPTED_P10' if not terminal else 'JOINT_REJECT'
                              if outcome['native_run'] in joints else 'EEF_REJECT')
            if classification != 'EEF_REJECT':
                candidates = [e for e in events if e.get('native_step') == capture['native_step']
                    and e.get('phase') in ('proposed', 'reference-certificate-rejected') and 'action_EEF10' in e]
                assert len(candidates) == 1
                assert np.allclose(matrix10(candidates[0]['action_EEF10']), connected, rtol=0, atol=1e-12)
            row = dict(layout=outcome['layout'],view=outcome['method'],query=index,
                classification=classification,native_step=capture['native_step'],
                reference_minus_actual_xyz_m=(reference[:3, 3]-actual[:3, 3]).tolist(),
                reference_to_actual_translation_m=float(np.linalg.norm(reference[:3, 3]-actual[:3, 3])),
                reference_to_actual_rotation_rad=rotation_error,
                raw_start_displacement_m=float(np.linalg.norm(target[0, :3]-actual[:3, 3])),
                previous_reference_velocity_m_s=((previous[-1][:3, 3]-previous[-2][:3, 3])/.05).tolist(),
                join_translation_acceleration_m_s2=acceleration[:2].tolist(),
                maximum_join_translation_acceleration_m_s2=float(np.linalg.norm(acceleration[:2], axis=1).max()),
                maximum_interior_translation_acceleration_m_s2=float(np.linalg.norm(acceleration[2:], axis=1).max()),
                connected_internal_translation_difference_error_m=float(translation_delta),
                connected_internal_relative_rotation_error=float(rotation_delta),
                connected_H40_matches_recorded_native_candidate=classification != 'EEF_REJECT',
                raw_action_sha256=query['action_sha256'],observation_sha256=query['observation_sha256'],
                consumed_points=10 if not terminal else 0)
            if classification == 'JOINT_REJECT':
                diagnostic = joints[outcome['native_run']]['diagnostic']['joint_rates']
                row.update(first_joint_exceedance=diagnostic['first_acceleration_exceedance'],
                    maximum_joint_acceleration_rad_s2=diagnostic['maximum_acceleration_rad_s2'])
            rows.append(row)
    assert len(rows) == 15 and Counter(r['classification'] for r in rows) == dict(ACCEPTED_P10=7, JOINT_REJECT=7, EEF_REJECT=1)
    summaries = {}
    for classification in ('ACCEPTED_P10', 'JOINT_REJECT', 'EEF_REJECT'):
        subset = [r for r in rows if r['classification'] == classification]
        summaries[classification] = dict(queries=len(subset), episodes=len({r['layout'] for r in subset}))
        for field in ('reference_to_actual_translation_m', 'reference_to_actual_rotation_rad',
                      'raw_start_displacement_m', 'maximum_join_translation_acceleration_m_s2'):
            values = [r[field] for r in subset]
            summaries[classification][field] = dict(minimum=min(values), median=statistics.median(values), maximum=max(values))
    return dict(schema='SLA_PD4_ORIGINAL15_REFERENCE_SEAM_OBSERVATIONAL_V71',rows=rows,summaries=summaries,
        original8_result_sha256=sha(group/'RESULT.private.json'),joint_analysis_sha256=sha(joint_analysis),
        descriptive_only=True,independent_scenes=4,causal_reference_error_claim=False,
        full_WBC_state_restored=False,same_student_state_expert_final_plan_pairs=0,
        model_loads=0,new_generations=0,certifier_calls=0,optimizer_updates=0,training_visits=0,
        physical_requests=0,physics_integrations=0,CPU_s=time.process_time()-cpu,wall_s=time.time()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--joint-analysis', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.group, args.joint_analysis)), flush=True)
