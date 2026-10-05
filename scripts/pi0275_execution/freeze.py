"""Freeze a private PI0275 source tree and cards without physics or neural calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from contract import AUTH, RELEASE, REPO, bound_source, self_check


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True); a = parser.parse_args()
    sources = dict(object_transfer_teacher='05935f80750aef853e6c7a6f26d9c449c25bf37918d7588f6f4133d1ea175c66',
                   object_transfer_proposals='dd9ea7746b374d47dc81228e1d8b09eb37edc09b49f4414fdc032fa5903d8b34')
    # Generator hash is bound independently below; the saved proposal is also compared structurally.
    proposal = REPO / 'scripts/object_transfer_proposals.py'
    sys.modules['object_transfer_teacher'] = bound_source('object_transfer_teacher', sources['object_transfer_teacher'])
    scene_source = bound_source('object_transfer_proposals', sources['object_transfer_proposals'])
    scene = scene_source.build()
    saved = json.loads((REPO / 'reports/object_transfer/preparation/SCENE_AND_SPLIT_PROPOSAL_DRAFT.json').read_text())
    expected = dict(saved); expected.pop('source_sha256')
    if scene != expected or saved['source_sha256'] != sha(proposal):
        raise ValueError('frozen proposal does not match reviewed generator and saved card')
    teacher = REPO / 'scripts/object_transfer_teacher.py'
    if sha(teacher) != sources['object_transfer_teacher']:
        raise ValueError('reviewed teacher identity changed')
    old = teacher.read_text(); old_check = 'if not 0 <= width <= .08 or min(speed, penetration) < 0:'
    if old.count(old_check) != 1:
        raise ValueError('private native measurement adapter source mismatch')
    adapted = old.replace(old_check, 'if not -1e-5 <= width <= .08 + 1e-5 or min(speed, penetration) < 0:')
    target = REPO / 'scripts/pi0275_execution/teacher.py'
    # The reviewed teacher inserts parents[1]/src; this private location needs the real root.
    adapted = adapted.replace("Path(__file__).resolve().parents[1] / 'src'", "Path(__file__).resolve().parents[2] / 'src'")
    target.write_bytes(adapted.encode('utf-8'))
    legacy_path = REPO / 'scripts/object_transfer_task_score.py'
    if sha(legacy_path) != 'b682429848f119ff722dec7e86060f36e96a4eafe7fa6ca8f3a17e2c2de5cd49':
        raise ValueError('reviewed strict scorer changed')
    legacy_source = legacy_path.read_text()
    width_check = "np.any(values['measured_gripper_width_m'] < 0)"
    if legacy_source.count(width_check) != 1:
        raise ValueError('native strict scorer measurement adapter source mismatch')
    legacy_native = REPO / 'scripts/pi0275_execution/score_legacy_native.py'
    legacy_native.write_bytes(legacy_source.replace(width_check, "np.any(values['measured_gripper_width_m'] < -1e-5)").encode('utf-8'))
    asset = REPO / 'assets/object_transfer/draft_tool.urdf'
    if sha(asset) != '44e62ac4fe50e54529fbfc0a59a9fcddb589adec4a872d67d736e7c5364669ca':
        raise ValueError('reviewed tool asset changed')
    # Seeds are deterministic identities, not scenario randomization.
    for slot in scene['teacher_slots']:
        slot['engine_seed'] = int.from_bytes(hashlib.sha256(('PI0275:OBJECT_TRANSFER_V1:' + slot['episode_id']).encode()).digest()[:8], 'big') % 2**31
    files = sorted(set(list((REPO / 'scripts/pi0275_execution').glob('*.py')) + list((REPO / 'src').rglob('*.py'))
                       + [teacher, proposal, REPO / 'scripts/object_transfer_task_score.py', asset,
                          REPO / 'reports/object_transfer/preparation/SCENE_AND_SPLIT_PROPOSAL_DRAFT.json']))
    card = dict(schema='PI0275_FROZEN_EXECUTION_MANIFEST_V1', authority=AUTH, release=RELEASE, issue11=5999317212,
                scene_card=scene, files=[dict(path=p.relative_to(REPO).as_posix(), sha256=sha(p)) for p in files],
                teacher_adaptation=dict(base_sha256=sha(teacher), private_sha256=sha(target),
                    changes=['native measured width numerical tolerance 1e-5m; preserve raw value', 'private source root parents[2]'], semantic_stages_unchanged=True),
                strict_scorer_native_parser=dict(base_sha256=sha(legacy_path), private_sha256=sha(legacy_native),
                    change='permit protocol native width numerical tolerance; preserve all raw data and original strict geometric scoring'),
                physics=dict(dt=1/120, substeps=2, use_gpu_pipeline=False, physx_use_gpu=False, solver_type=1,
                             position_iterations=8, velocity_iterations=2, gravity=[0, 0, -9.81],
                             contact_collection='CC_LAST_SUBSTEP', contact_lambda='installed SDK contact force magnitude; N, no impulse/dt conversion',
                             contact_offset_m=.001, rest_offset_m=0., friction=.8, restitution=0.),
                controller=dict(translation_Kp=600., translation_Kd=40., max_force_norm_N=20., angular_Kp=4., angular_Kd=.3,
                                max_torque_norm_Nm=2., finger_Kp=2000., finger_Kd=5., finger_force_N=10.,
                                command_translation_m_s=.10, command_angular_rad_s=.5, command_opening_m_s=.04),
                frames=dict(source='PI0275_OBJECT_TASK_WORLD', tcp='PI0275_TOOL_PALM_FACE', position_anchor_m=[0, 0, 0]),
                budget=dict(primary_starts=68, total_starts=74, native_conditions=2, controls=4, physics_ticks=278880,
                            teacher_updates=261120, RGB_pairs=48000, learned_calls=0, WBC_runs=0, hardware_runs=0),
                actual_native_engine_inventory='REQUIRED_AT_FIRST_DECLARED_NATIVE_START',
                native_or_task_PASS=False, CPU_contract=self_check(), selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')
    a.root.mkdir(parents=True, exist_ok=True)
    out = a.root / 'EXECUTION_MANIFEST.json'
    if out.exists() and json.loads(out.read_text()) != card:
        raise ValueError('existing execution manifest differs; preserve it and explicitly version repair')
    out.write_bytes((json.dumps(card, sort_keys=True, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(dict(status='FROZEN_PRE_NATIVE_NOT_PASS', manifest_sha256=sha(out), files=len(files), physics_calls=0)))


if __name__ == '__main__':
    main()
