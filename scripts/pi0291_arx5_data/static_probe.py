"""Explicitly zero-integration native-chain and sensor diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
try:
    from .config import PARAMETERS as P, layout
    from .engine import Engine
    from .scene import DOWNWARD_ROTATION, PAD_INNER_GAP_AT_ZERO, TCP_OFFSET
except ImportError:
    from config import PARAMETERS as P, layout
    from engine import Engine
    from scene import DOWNWARD_ROTATION, PAD_INNER_GAP_AT_ZERO, TCP_OFFSET

OFFSETS = [(0., 0., .0605), (.02, 0., .04), (-.264, 0., 0.),
           (.245, 0., .056), (.06775, 0., .0845), (.02895, 0., -.0845)]
AXES = [(0., 0., 1.), (0., 1., 0.), (0., -1., 0.),
        (0., -1., 0.), (0., 0., -1.), (1., 0., 0.)]


def independent_fk(q):
    result = np.eye(4)
    result[:3, 3] = [.05, 0., .50]
    for position, axis, angle in zip(OFFSETS, AXES, q):
        result[:3, 3] += result[:3, :3] @ position
        a = np.asarray(axis)
        cross = np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])
        rotation = np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*cross @ cross
        result[:3, :3] = result[:3, :3] @ rotation
    result[:3, 3] += result[:3, :3] @ TCP_OFFSET
    return result


def probe(asset_root, identifiers, output, render=False):
    rows = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for identifier in identifiers:
        card = layout(identifier)
        engine = Engine(card, asset_root=asset_root)
        state = engine.snapshot()
        forward = independent_fk(state['arm_q'])
        fk_error = float(np.max(np.abs(forward-state['tcp'])))
        targets = []
        for xy in (card['source_xy'], card['goal_xy']):
            for height in (.024, .104, .154):
                target = np.eye(4)
                target[:3, :3] = DOWNWARD_ROTATION
                target[:3, 3] = [*xy, P['table_top']+height]
                q, error = engine.solve_ik(target, initial=engine.arm_targets, iterations=150)
                targets.append(dict(xy=xy, height=height, q=q.tolist(), error=error))
        gaps = []
        geometric_gaps = []
        for stroke in (0., .022, .044):
            engine.data.qpos[engine.finger_q] = stroke
            mujoco.mj_forward(engine.model, engine.data)
            gaps.append(engine.snapshot()['gripper_gap_m'])
            y_axis = engine.pose('tool')[:3, 1]
            center_difference = engine.data.geom_xpos[engine.geoms['left_pad']]-engine.data.geom_xpos[engine.geoms['right_pad']]
            geometric_gaps.append(float(np.dot(center_difference, y_axis)-.006))
        assert np.allclose(gaps, PAD_INNER_GAP_AT_ZERO+2*np.array([0., .022, .044]))
        assert np.allclose(gaps, geometric_gaps, atol=1e-12)
        assert fk_error < 1e-9
        engine.data.qpos[engine.finger_q] = .044
        mujoco.mj_forward(engine.model, engine.data)
        state = engine.snapshot()
        camera = None
        if render:
            from PIL import Image
            images, camera = engine.render(state)
            for name, pixels in images.items():
                Image.fromarray(pixels).save(output/f'{identifier}_{name}.png')
        rows.append(dict(id=identifier, steps=engine.steps, time_s=float(engine.data.time),
                         nq=engine.model.nq, nv=engine.model.nv, nu=engine.model.nu,
                         initial_tcp=state['tcp'].tolist(), independent_fk_max_error=fk_error,
                         initial_contacts=state['contacts'], initial_safety_valid=state['safety_valid'],
                         gap_m_at_stroke_0_022_044=gaps, independently_measured_geom_gap_m=geometric_gaps,
                         actual_arm_damping=engine.model.dof_damping[engine.arm_v].tolist(),
                         actual_finger_damping=engine.model.dof_damping[engine.finger_v].tolist(),
                         targets=targets, camera=camera))
        assert engine.steps == 0 and engine.data.time == 0.
        engine.close()
    result = dict(status='ZERO_INTEGRATION_ASSET_CHECK', integration_steps=0,
                  rendered_pairs=len(rows) if render else 0, rows=rows,
                  task_success_claim=False, WBC_runs=0, asset_original_modified=False)
    (output/'STATIC_PROBE.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root', required=True)
    parser.add_argument('--ids', nargs='+', default=['C29100'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    probe(args.asset_root, args.ids, args.output, args.render)
