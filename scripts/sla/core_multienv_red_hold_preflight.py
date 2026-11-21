"""Readback and no-dynamics geometry checks for one unopened F2 red HOLD slot."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from go2arx5_vla.trajectory.so3 import quat_to_matrix
from go2arx5_vla.world_model.mobile_continuous_teacher_v2 import ANGULAR_SPEED, TRANSLATION_SPEED, rotation_vector
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import read
from scripts.sla.core_hopper_red_place_preflight import base_hold, recorded_knot
from scripts.sla.core_multienv_place_sweep import ArmShadow, scene_delta
from scripts.sla.core_multienv_red_descent import compare_path, pose_goals
from scripts.sla.core_multienv_red_hold_teacher import AWAY_M, PITCH_DEG, RISE_M, VERSION, RedHoldDetour
from scripts.sla.core_multienv_red_lift import lift_from_projection
from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases

PRODUCER_SPEC = '45d949a883c7c10ec31c0caef9a40bdd0a4c51225d5417dac68c7b05432e870b'
SOURCE = 'eccd8e2e2089bb2505c50043f959959d07094d56'
SLOT = 'MEV1-TRAIN-F2-NEAR_POS-00-HOLD-red'
SCENE_SHA = '4d5e050e945f5a533bf43044a704b8d98504fa9d05f938442b8952cd0e9e00ff'
FIXTURES = (
    ('MEV1-TRAIN-F2-NEAR_NEG-00-HOLD-red',
     '1852d281138cdac76bf9bad070742dc5713e23646e78c8b04eb75e9df5c02b27',
     '67a4f8d4dec52f86049648683092bf1f49465f8e014cec59a705efabbeeec4ee'),
    ('MEV1-TRAIN-F2-NEAR_NEG-01-HOLD-red',
     'f64d6e7f819f10569ed18a79e4f5d4e5b4b714803a42d7703046354c7fb162c9',
     '74473a45fbcb1bbc6119126c4ce7a54c359d735abf7d354aeace6bf491fe76da'),
)
CONFIG = (PITCH_DEG, AWAY_M * 1000, RISE_M * 1000)
VECTORS = ('velocity', 'inner_velocity', 'angular_velocity', 'inner_angular_velocity')


def plans(folder, spec, initial, query, base):
    sample = query['actual_state']
    accepted = recorded_knot(folder, query['recent_reference'])[:7]
    reference = np.eye(4)
    reference[:3, :3] = quat_to_matrix(np.asarray(accepted[3:]))
    reference[:3, 3] = accepted[:3]
    args = (initial, spec['mobile_condition'], spec['multitask_geometry'], 'HOLD-red')
    old, new = MultitaskTeacher(*args), RedHoldDetour(*args, None)
    memory = query['teacher_memory_audit_only']
    for teacher in (old, new):
        for name in VECTORS:
            setattr(teacher, name, np.asarray(memory[name]))
        teacher.last_committed_step = memory['last_committed_step']
    prior, _ = old.decide(sample, reference, base)
    if (not np.array_equal(prior.decoded_H40, query['nominal_world_H40'])
            or not np.array_equal(prior.reference_path, query['installed_reference_H40'])):
        raise ValueError('Saved original candidate boundary did not replay exactly')
    candidate, audit = new.decide(sample, reference, base)
    if (candidate.decoded_H40.shape != (40, 10) or candidate.reference_path.shape != (40, 10)
            or audit['teacher_version'] != VERSION or new.stage != 'PREGRASP'):
        raise ValueError('Expected the original H40 checks and deferred descent')
    return dict(native_step=query['native_step'], original_exact=True, candidate_H40_gate_pass=True,
                candidate_turn_started=new.turn_anchor is not None, closing_started=new.close_start is not None,
                candidate_changed=not np.array_equal(prior.reference_path, candidate.reference_path),
                candidate_installed_sha256=hashlib.sha256(candidate.reference_path.tobytes()).hexdigest())


def geometry(shadow, anchor, intent):
    sample = anchor['actual_state']
    start = np.asarray(sample['tcp'])
    shadow.data.qpos[:] = sample['qpos']
    mujoco.mj_fwdPosition(shadow.model, shadow.data)
    turn, detour, grasp = pose_goals(shadow, start, np.asarray(intent['orientation']),
                                    np.asarray(intent['goal']), CONFIG)
    angle = np.linalg.norm(rotation_vector(turn[:3, :3] @ start[:3, :3].T))
    length = np.linalg.norm(detour[:3, 3] - turn[:3, 3]) + np.linalg.norm(grasp[:3, 3] - detour[:3, 3])
    estimate = (anchor['native_step'] - 5703) * .002 + angle / ANGULAR_SPEED
    estimate += (length + .026) / TRANSLATION_SPEED + 2. + 1.
    approach = compare_path(shadow, anchor, intent, CONFIG)
    lift = lift_from_projection(shadow, approach) if approach['passed'] else None
    return dict(approach=approach, lift=lift, nominal_rate_bound_s=float(estimate))


def run(root):
    spec_path = root / 'EXECUTION_SPEC.json'
    base = read(spec_path)
    if (digest(spec_path) != PRODUCER_SPEC or mujoco.__version__ != '3.2.7'
            or mujoco.get_mjcb_control() is not None):
        raise PermissionError('Expected the original producer and geometry runtime')
    require_unstarted_cases(base['registry'], {SLOT})
    pop = read_pin(base['compiled_population'])
    slot = next(r for r in pop['conditions']['TRAIN'] if r['id'] == SLOT)
    pin = next(c['scene'] for c in pop['clusters'] if c['id'] == slot['cluster'])
    scene = Path(pin['path'])
    if (pin['sha256'] != SCENE_SHA or digest(scene) != SCENE_SHA or (root / 'cases' / SLOT).exists()
            or slot['native_max_ticks'] != 35703):
        raise PermissionError('The reserved next TRAIN slot has changed')
    records = {r['condition']: r for r in read(root / 'RECORDS.private.json')}
    target = mujoco.MjModel.from_binary_path(str(scene))
    outcomes = []
    with forbid_integration(mujoco) as calls:
        for cid, terminal_sha, queries_sha in FIXTURES:
            case = root / 'cases' / cid
            native = case / 'native-01'
            spec = read(case / 'EXECUTION_SPEC.json')
            queries = native / 'LIVE_DATA/QUERIES.private.json'
            record = records[cid]
            old_scene = Path(spec['model_path'])
            if (record['result']['sha256'] != terminal_sha or digest(case / 'RESULT.private.json') != terminal_sha
                    or record['first_failure_signature'] != ['ILLEGAL_CONTACT', 'DESCEND', [['task_blue_cube', 'x5_link6']]]
                    or record['post_STOP_ticks'] != 0 or spec['source_sha'] != SOURCE
                    or spec['collection_split'] != 'TRAIN' or digest(queries) != queries_sha
                    or digest(old_scene) != spec['frozen_files'][str(old_scene)]):
                raise PermissionError('Original F2 HOLD evidence changed')
            q = read(queries)
            i = next(i for i, row in enumerate(q) if row['phase'] == 'DESCEND')
            before = q[i-1:i+1]
            if before[0]['phase'] != 'PREGRASP' or q[0]['native_step'] != 5703:
                raise ValueError('Original PREGRASP boundary missing')
            base_hold_pose = base_hold(native, q[0]['recent_reference'])
            shadow = ArmShadow(target)
            old = mujoco.MjModel.from_binary_path(str(old_scene))
            shifts = scene_delta(old, target)
            results = []
            for row in before:
                check = plans(native, spec, q[0]['actual_state'], row, base_hold_pose)
                state = row['actual_state']
                positions = np.array(state['qpos'])
                for s, d in shifts:
                    positions[s] += d
                anchor = dict(row, actual_state=dict(state, qpos=positions))
                raw = q[i]['raw_intent']
                intent = dict(raw, goal=np.asarray(raw['goal']) + shifts[0][1])
                results.append(dict(H40=check, next_scene=geometry(shadow, anchor, intent)))
            outcomes.append(dict(condition=cid, terminal_sha256=terminal_sha,
                                 queries_sha256=queries_sha, samples=results))
    if calls:
        raise RuntimeError('Cold qualification attempted physics integration')
    require_unstarted_cases(base['registry'], {SLOT})
    return dict(status='F2_RED_HOLD_DETOUR_COLD_CHECK', teacher=VERSION, source=SOURCE,
                unopened=slot, scene=pin, outcomes=outcomes, native_requests=0, native_ticks=0,
                model_generations=0, optimizer_updates=0, VAL_inputs=0, native_pilot='NOT_RUN')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('producer', type=Path)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    result = run(args.producer)
    with args.output.open('x') as stream:
        json.dump(result, stream, allow_nan=False, sort_keys=True, indent=2)
