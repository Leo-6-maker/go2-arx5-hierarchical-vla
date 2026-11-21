"""Compare the original HOLD scorer's simultaneous base and grasp eligibility on saved TRAIN states."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from go2arx5_vla.world_model import multitask_score_v6 as score_module
from go2arx5_vla.world_model.multitask_score_v6 import BASE_MOTION, HOLD_SECONDS, LIFT, Score, Task
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, packets
from go2arx5_vla.world_model.multitask_terminal_audit_v1 import START
from go2arx5_vla.world_model.native_replay import forbid_integration
from scripts.sla.hopper_parent_custody import PLAN, digest, write_once


def read(path):
    return json.loads(path.read_bytes())


def compare(case, expected):
    import mujoco
    import numpy as np

    spec = read(case/'EXECUTION_SPEC.json')
    terminal = read(case/'RESULT.private.json')
    native = case/'native-01'
    score_file = native/'MULTITASK_SCORE_PROGRESS.private.json'
    cached = read(score_file)
    scene = Path(spec['model_path'])
    if (digest(case/'RESULT.private.json') != expected or spec['goal_plan_sha'] != PLAN
            or '-TRAIN-' not in case.name or spec['panel_entry_id'] != case.name
            or spec['scope'] != 'SLA_CORE_HOPPER_V1_EXPERT' or digest(scene) != spec['frozen_files'][str(scene)]):
        raise ValueError('Unbound terminal, scene or TRAIN case')
    task = Task.parse(spec['task_instruction_id'])
    if task.operation != 'HOLD' or terminal['physical_requests'] != 1:
        raise ValueError('Expected one original HOLD episode')
    counts = Counter()
    scorer = None
    best = None
    first_hold = last_base = None
    started = time.process_time()
    with forbid_integration(mujoco):
        reader = TwoObjectState(mujoco.MjModel.from_binary_path(str(scene)), task.color)
        for packet in packets(native/'state.bin.journal'):
            tick = packet['step']
            if tick < START:
                continue
            sample = reader.update(packet)
            if scorer is None:
                if tick != START:
                    raise ValueError('Missing original task start')
                scorer = Score(task.instruction,sample,spec['multitask_geometry'])
            state = scorer.update(sample)
            motion = state['base_directional_motion_m']
            hold = bool(state['target_grasp'] and state['lift_m'] >= LIFT and scorer.hold_since is not None
                        and sample['time_s'] - scorer.hold_since >= HOLD_SECONDS - 1e-10)
            base = motion >= BASE_MOTION
            counts['base'] += base
            counts['qualified_hold'] += hold
            counts['both'] += base and hold
            if base:
                last_base = tick
            if hold:
                if first_hold is None:
                    first_hold = tick
                if best is None or motion > best['motion_m']:
                    best = dict(tick=tick,motion_m=float(motion))
    score = scorer.result()
    fields = ('native_step','states','target_grasp','ever_target_lift20mm','task_success',
              'wrong_object_grasp_ever','missing_native_intervals','unsafe_states','events')
    if (score['native_step'] != terminal['actual_native_integrations'] or
            any(score[key] != cached[key] for key in fields) or
            any(not np.isclose(score[k],cached[k],rtol=0,atol=1e-10) for k in
                ('maximum_lift_m','longest_grasp_lift_hold_s','base_directional_motion_m'))):
        raise ValueError('Replayed states did not reproduce the native terminal score')
    return dict(condition=case.name,scene_sha256=digest(scene),native_result=expected,
        scorer_source=digest(Path(score_module.__file__)),journal_sha256=digest(native/'state.bin.journal'),
        cached_sha256=digest(score_file),task_ticks=[START,score['native_step']],scored_states=score['states'],
        base_gate_m=BASE_MOTION,hold_gate_s=HOLD_SECONDS,eligible_ticks=dict(counts),
        last_base_tick=last_base,first_qualified_hold_tick=first_hold,best_qualified_hold=best,
        success=score['task_success'],terminal_base_m=score['base_directional_motion_m'],
        longest_hold_s=score['longest_grasp_lift_hold_s'],cpu_s=time.process_time()-started,
        new_integrations=0,model_generations=0,optimizer_updates=0)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('case',type=Path)
    p.add_argument('terminal_sha256')
    p.add_argument('output',type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_once(args.output,compare(args.case,args.terminal_sha256))
