"""Measure target and distractor approach on sealed GRASP24 state packets."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from go2arx5_vla.world_model import sla_followup_native_v1 as plan
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.multitask_score_v6 import Task
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, packets
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import read


def measure(card, result, spec, startup):
    scene = spec['model_path']
    endpoint = result['actual_native_integrations']
    target = Task.parse(spec['task_instruction_id']).color
    other = 'blue' if target == 'red' else 'red'
    closest = {}
    contacts = {target:0,other:0}
    first_contacts = {}
    last = startup-1
    ending = None
    if digest(scene) != spec['frozen_files'][scene]:
        raise ValueError('GRASP24_GEOMETRY_ORIGINAL_SCENE_CHANGED')
    with forbid_integration(mujoco):
        reader = TwoObjectState(mujoco.MjModel.from_binary_path(scene),target)
        for packet in packets(card.parent/'native-01/state.bin.journal'):
            tick = packet['step']
            if tick < startup:
                continue
            if tick != last+1 or tick > endpoint:
                raise ValueError('GRASP24_GEOMETRY_EXACT_TASK_CLOCK_REQUIRED')
            sample = reader.update(packet)
            p,q = reader.data.geom_xpos[reader.pad]
            midpoint = .5*(p+q)
            tcp = sample['tcp'][:3,3]
            distances = {}
            for color in (target,other):
                obj = sample['objects'][color]
                xyz = obj['pose'][:3,3]
                distances[color] = dict(pad_midpoint_m=float(np.linalg.norm(xyz-midpoint)),
                                        tcp_m=float(np.linalg.norm(xyz-tcp)))
                contacts[color] += int(obj['hand_contact'])
            record = dict(tick=tick,objects_m=distances,gripper_open_fraction=sample['gripper_open_fraction'])
            for color, metrics in distances.items():
                for name,value in metrics.items():
                    key = color+'_'+name
                    if key not in closest or value < closest[key]['objects_m'][color][name]:
                        closest[key] = record
                if sample['objects'][color]['hand_contact'] and color not in first_contacts:
                    first_contacts[color] = record
            ending = record
            last = tick
    if ending is None or last != endpoint:
        raise ValueError('GRASP24_GEOMETRY_PACKET_PREFIX_INCOMPLETE')
    return dict(case=card.parent.name,arm=spec['arm'],instruction=spec['task_instruction_id'],target=target,
        distractor=other,task_packets=last-startup+1,terminal_tick=last,closest=closest,
        robot_contact_packets=contacts,first_robot_contacts=first_contacts,terminal_geometry=ending,
        terminal_contact_STOP=any(e['reason']=='ILLEGAL_CONTACT' for e in (result.get('physics_stops') or ())),
        case_spec_sha256=digest(card),scene_sha256=spec['frozen_files'][scene])


def audit(root, spec_sha, count):
    root = root.resolve(strict=True)
    scope = read_pin(dict(path=str(root/'EXECUTION_SPEC.json'),sha256=spec_sha))
    if (scope['scope'] != plan.SCOPE or scope['evaluation_stage'] != plan.GRASP
            or scope['compiled_population']['sha256'] != plan.POPULATION_SHA
            or not 0 < count <= plan.CAPS[plan.GRASP][0]):
        raise ValueError('GRASP24_GEOMETRY_FIXED_SCOPE_REQUIRED')
    records = {row['id']:row for row in read(root/'RECORDS.private.json') if row['role']=='STUDENT'}
    began = time.monotonic()
    cases = []
    for name in scope['cases'][:count]:
        card = Path(name).resolve(strict=True)
        summary = records[card.parent.name]
        lease = read(card.parent/'control/LEASE.json')
        spec = read_pin(dict(path=str(card),sha256=lease['spec_sha256']))
        pin = dict(path=str(card.parent/'RESULT.private.json'),sha256=summary['result_sha256'])
        raw = read_pin(pin)
        if (not card.is_relative_to(root/'cases') or lease['status']!='CLOSED'
                or raw['actual_native_integrations']!=summary['native_ticks'] or raw['source_sha']!=scope['source_sha']):
            raise ValueError('GRASP24_GEOMETRY_REQUIRES_AN_ORIGINAL_SEALED_CASE')
        row = measure(card,raw,spec,5703)
        cases.append(dict(row,result_sha256=pin['sha256']))
    return dict(spec_sha256=spec_sha,source_sha=scope['source_sha'],cases=cases,read_only=True,
                new_integrations=0,new_model_generations=0,new_optimizer_updates=0,
                observed_wall_s=time.monotonic()-began)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('sha256')
    parser.add_argument('count',type=int)
    args = parser.parse_args()
    print(json.dumps(audit(args.root,args.sha256,args.count),allow_nan=False))
