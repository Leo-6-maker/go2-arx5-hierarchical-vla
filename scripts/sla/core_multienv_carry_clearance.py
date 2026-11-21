"""Replay only saved positions to measure the F1 blue-right link2/table failure."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from scripts.sla.core_hopper_carry_audit import last_stop

CELL = 'CHV1-TRAIN-F1-PLACE-blue-right'


def contact_geometry(model, qpos, pairs):
    state = mujoco.MjData(model)
    state.qpos[:] = qpos
    mujoco.mj_fwdPosition(model,state)
    return dict(table_gaps_mm=[1000*float(mujoco.mj_geomDistance(
        model,state,link,table,.20,None)) for link,table in pairs],
        link_positions_world_mm=[[float(v*1000) for v in state.geom_xpos[link]] for link,_ in pairs])


def audit(parent):
    if mujoco.__version__!='3.2.7' or mujoco.get_mjcb_control() is not None:
        raise PermissionError('Expected the original physics library without a controller')
    forbid_integration()
    records = json.loads((parent/'RECORDS.private.json').read_bytes())
    failed = [r for r in records if r['split']=='TRAIN' and r['condition'].startswith(CELL+'-')
              and r['first_failure']=='ILLEGAL_CONTACT']
    passed = [r for r in records if r['split']=='TRAIN' and r['condition'].startswith(CELL+'-')
              and r['admitted']]
    outputs = []
    for row in [*failed,*passed]:
        folder = parent/'cases'/row['condition']
        spec = json.loads((folder/'EXECUTION_SPEC.json').read_bytes())
        native = read_pin(row['result'])
        if native['physical_requests']!=1 or native['actual_native_integrations']!=row['ticks']:
            raise ValueError('Original request or tick identity changed')
        model_file = spec['model_path']
        if digest(model_file)!=spec['frozen_files'][model_file]:
            raise ValueError('Historical scene changed')
        model = mujoco.MjModel.from_binary_path(model_file)
        link = model.body('x5_link2').id
        table = model.geom('task_table').id
        links = np.where((model.geom_bodyid==link) & (model.geom_contype!=0))[0]
        pairs = [(int(g),table) for g in links]
        if not pairs:
            raise ValueError('Missing the frozen collidable link2 and table')
        qp = folder/'native-01/LIVE_DATA/QUERIES.private.json'
        captures = json.loads(qp.read_bytes())
        carry = [q for q in captures if q['phase']=='CARRY']
        if not carry:
            raise ValueError('No observed carry phase')
        samples = [contact_geometry(model,q['actual_state']['qpos'],pairs) for q in carry]
        result = dict(id=row['condition'],native_result=row['result']['sha256'],
                      scene=spec['frozen_files'][model_file],queries=digest(qp),native_ticks=row['ticks'],
                      carry_queries=len(carry),admitted=row['admitted'],
                      sampled_carry_minimum_mm=min(min(s['table_gaps_mm']) for s in samples),
                      last_carry_tick=carry[-1]['native_step'],last_carry=samples[-1],
                      link_geom_names=[model.geom(g).name for g,_ in pairs],
                      max_gap_query_resolution_ms=500)
        if row in failed:
            terminal = last_stop(folder/'native-01/physics.jsonl')
            contacts = {(min(c['geom1'],c['geom2']),max(c['geom1'],c['geom2'])):1000*c['dist']
                        for c in terminal['contacts']}
            pair_depths = [contacts[(min(g,table),max(g,table))] for g,_ in pairs
                           if (min(g,table),max(g,table)) in contacts]
            if terminal['step']!=row['ticks'] or not pair_depths or row['post_STOP_ticks']!=0:
                raise ValueError('Missing exact link2/table STOP and recorded cessation')
            result.update(contacts_at_stop_mm=pair_depths,stop=contact_geometry(model,terminal['qpos'],pairs),
                          last_carry_to_stop_ms=2*(row['ticks']-carry[-1]['native_step']))
        outputs.append(result)
    return dict(scope='ORIGINAL_F1_TRAIN_BLUE_RIGHT_CARRY',failed=len(failed),passed=len(passed),
                results=outputs,VAL_cases=0,native_requests=0,native_integrations=0,
                optimizer_updates=0,model_generations=0)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('parent', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.parent)))
