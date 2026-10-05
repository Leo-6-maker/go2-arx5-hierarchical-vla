"""Draft object-result scorer, CPU only; fixtures are not simulated episodes."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np


def score(trace, target_index, bin_bounds):
    """Evidence-side only. No reference, teacher phase or task state is installed."""
    t = np.asarray(trace['time'], float)
    n = len(t)
    if t.ndim != 1 or n < 2 or not np.isfinite(t).all() or abs(t[0]) > 1e-9:
        raise ValueError('finite episode clock starting at zero required')
    if np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > 1/120 + 1e-8:
        raise ValueError('full physics-rate evidence required, not RGB subsampling')
    if target_index not in (0, 1):
        raise ValueError('two-object draft requires target index 0 or 1')
    shapes = dict(object_position=(n, 2, 3), object_rotation=(n, 2, 3, 3),
                  object_velocity=(n, 2, 3), tcp_position=(n, 3), half_extents=(2, 3),
                  support_z=(2,), measured_gripper_width_m=(n,), target_tool_contact=(n,), bilateral_grasp_contact=(n,),
                  dangerous_contact=(n,), penetration_m=(n,), gate_reject=(n,))
    values = {}
    for key, shape in shapes.items():
        x = np.asarray(trace[key])
        if x.shape != shape or not np.isfinite(x).all():
            raise ValueError(f'{key}: expected finite {shape}')
        values[key] = x
    r = values['object_rotation']
    if not np.allclose(r @ np.swapaxes(r, -1, -2), np.eye(3), atol=1e-6) or not np.allclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError('actual object rotations must be SO3')
    bounds = np.asarray(bin_bounds, float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError('declared container interior bounds required')
    if np.any(values['half_extents'] <= 0) or np.any(values['measured_gripper_width_m'] < 0) or np.any(values['penetration_m'] < 0):
        raise ValueError('invalid geometry, opening or penetration')
    for key in ('target_tool_contact', 'bilateral_grasp_contact', 'dangerous_contact', 'gate_reject'):
        if not np.all(np.isin(values[key], [False, True])):
            raise ValueError('contact and reject fields must be boolean flags')
    if np.any(values['bilateral_grasp_contact'].astype(bool) & ~values['target_tool_contact'].astype(bool)):
        raise ValueError('bilateral target contact contradicts aggregate tool contact')
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    local = values['half_extents'][:, None, :] * signs
    corners = values['object_position'][:, :, None, :] + np.einsum('noij,okj->noki', r, local)
    bottom = corners[..., 2].min(axis=-1)
    task = t <= 30 + 1e-9
    final = (t >= 28 - 1e-9) & task
    reasons = []
    if t[-1] < 30 - 1e-9:
        reasons.append('EARLY_STOP_KEEP_AS_TASK_FAIL')
    lift = bottom[:, target_index] - values['support_z'][target_index]
    if not np.any(lift[task] >= .08 - 1e-9):
        reasons.append('TARGET_NOT_LIFTED_80MM')
    grasp = task & (lift >= .08 - 1e-9) & values['bilateral_grasp_contact'].astype(bool)
    starts = np.flatnonzero(grasp & ~np.r_[False, grasp[:-1]])
    ends = np.flatnonzero(grasp & ~np.r_[grasp[1:], False])
    if not any(t[end] - t[start] >= .3 - 1e-9 for start, end in zip(starts, ends)):
        reasons.append('NO_SUSTAINED_PHYSICAL_GRASP_DURING_LIFT')
    other = 1 - target_index
    moved = np.linalg.norm(values['object_position'][:, other] - values['object_position'][0, other], axis=-1)
    if np.any(moved[task] >= .010 - 1e-9) or np.any((bottom[:, other] - values['support_z'][other])[task] >= .08 - 1e-9):
        reasons.append('NON_TARGET_MOVED_OR_LIFTED')
    target_corners = corners[:, target_index]
    inside = ((target_corners >= bounds[0] - 1e-9) & (target_corners <= bounds[1] + 1e-9)).all(axis=(1, 2))
    if not final.any() or not inside[final].all():
        reasons.append('FINAL_FULL_OBJECT_NOT_IN_COMMAND_BIN')
    if not final.any() or np.any(values['measured_gripper_width_m'][final] < .070 - 1e-9) or np.any(values['target_tool_contact'][final]):
        reasons.append('NOT_RELEASED')
    distance = np.linalg.norm(values['tcp_position'] - values['object_position'][:, target_index], axis=-1)
    if not final.any() or np.any(distance[final] < .100 - 1e-9):
        reasons.append('NOT_WITHDRAWN')
    if not final.any() or np.any(np.linalg.norm(values['object_velocity'][final, target_index], axis=-1) >= .020):
        reasons.append('FINAL_OBJECT_NOT_STABLE')
    if np.any(values['dangerous_contact']) or np.any(values['penetration_m'] > .002):
        reasons.append('DANGEROUS_CONTACT_OR_PENETRATION')
    if np.any(values['gate_reject']):
        reasons.append('GATE_REJECT')
    return dict(success=not reasons, reasons=reasons, target_index=target_index,
                task_window_s=[0, 30], final_window_s=[28, 30],
                recorded_end_s=float(t[-1]), final_ticks=int(final.sum()),
                all_recorded_safety_retained=True, max_target_lift_m=float(lift[task].max()),
                max_non_target_displacement_m=float(moved[task].max()),
                evidence_scope='DRAFT_OBJECT_TASK_SCORE_NOT_WBC_OR_DEPLOYMENT', qualification_authority=False)


def demo():
    t = np.arange(3601) / 120
    n = len(t)
    p = np.tile(np.array([[-.2, 0, .665], [0, -.2, .665]]), (n, 1, 1))
    p[(t >= 5) & (t < 15), 0, 2] = .765
    p[t >= 15, 0] = [.4, .2, .665]
    trace = dict(time=t, object_position=p, object_rotation=np.tile(np.eye(3), (n, 2, 1, 1)),
                 object_velocity=np.zeros((n, 2, 3)), tcp_position=np.tile([.4, .2, .9], (n, 1)),
                 half_extents=np.array([[.02, .015, .015], [.02, .015, .015]]), support_z=np.array([.65, .65]),
                 measured_gripper_width_m=np.full(n, .08), target_tool_contact=np.zeros(n, bool),
                 bilateral_grasp_contact=(t >= 5) & (t < 15),
                 dangerous_contact=np.zeros(n, bool), penetration_m=np.zeros(n), gate_reject=np.zeros(n, bool))
    trace['target_tool_contact'][(t >= 5) & (t < 15)] = True
    trace['measured_gripper_width_m'][(t >= 5) & (t < 15)] = .03
    bounds = np.array([[.32, .12, .65], [.48, .28, .85]])
    assert score(trace, 0, bounds)['success']
    cases = {}
    for name in ('NEVER_LIFTED', 'NO_GRASP', 'WRONG_OBJECT', 'WRONG_BIN', 'STILL_GRASPING', 'FINGERS_NOT_OPEN', 'NO_RETREAT', 'LATE_DROP', 'MOVING_OBJECT', 'DANGEROUS_CONTACT', 'PENETRATION', 'GATE_REJECT', 'EARLY_STOP', 'PARTIAL_OBJECT_OUTSIDE'):
        x = {key: value.copy() for key, value in trace.items()}
        target = 0
        if name == 'NEVER_LIFTED': x['object_position'][:, 0, 2] = .665
        elif name == 'NO_GRASP': x['bilateral_grasp_contact'][:] = False
        elif name == 'WRONG_OBJECT': target = 1
        elif name == 'WRONG_BIN': x['object_position'][t >= 15, 0, 1] = -.2
        elif name == 'STILL_GRASPING': x['target_tool_contact'][t >= 28] = True
        elif name == 'FINGERS_NOT_OPEN': x['measured_gripper_width_m'][t >= 28] = .03
        elif name == 'NO_RETREAT': x['tcp_position'][t >= 28] = x['object_position'][t >= 28, 0]
        elif name == 'LATE_DROP': x['object_position'][-1, 0, 2] = .60
        elif name == 'MOVING_OBJECT': x['object_velocity'][-1, 0, 0] = .03
        elif name == 'DANGEROUS_CONTACT': x['dangerous_contact'][100] = True
        elif name == 'PENETRATION': x['penetration_m'][100] = .003
        elif name == 'GATE_REJECT': x['gate_reject'][100] = True
        elif name == 'PARTIAL_OBJECT_OUTSIDE': x['object_position'][t >= 28, 0, 0] = .475
        elif name == 'EARLY_STOP':
            x = {key: value[:3000].copy() if key not in ('half_extents', 'support_z') else value for key, value in x.items()}
        result = score(x, target, bounds)
        assert not result['success'], name
        cases[name] = result['reasons']
    # Data corruption is incomplete evidence, separate from valid task failures.
    broken = dict(trace, object_position=np.full_like(p, np.nan))
    try: score(broken, 0, bounds)
    except ValueError: pass
    else: raise AssertionError('nonfinite achieved state accepted')
    return dict(status='PASS_CPU_DRAFT_OBJECT_RESULT_AND14_NEGATIVE_TRACES',
                negative_cases=cases, valid_failure_count=14, corrupt_trace_rejected=True,
                synthetic_fixture_only=True, actual_physical_episodes=0, teacher_calls=0,
                renders=0, physics_steps=0, model_queries=0, optimizer_updates=0,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                launch_allowed=False, selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = demo()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(result, sort_keys=True))
