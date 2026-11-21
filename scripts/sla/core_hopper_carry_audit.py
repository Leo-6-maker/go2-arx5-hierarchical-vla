"""Read the original frozen contact predicate and statically check one unopened TRAIN carry pilot."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.sla.core_hopper_blue_carry import CELL, PILOT, PRIOR, remaining_clearance, planar, TRANSFER_CLEARANCE_M
from scripts.sla.core_hopper_expert import POP
from go2arx5_vla.world_model.multitask_teacher_v6 import swept_footprints_overlap
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest

OBSERVER='7e305c296ba2706ae0d546048d6f4c46a6a65ac374d43afd537eebf029daf272'
BASE='7090e2ad238e76cd6278cf65405f7785604abfdb'


def read(path):
    return json.loads(path.read_bytes())


def last_stop(path):
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-256*1024))
        rows=[json.loads(s) for s in f.read().splitlines()[1:]]
    if rows[-1].get('reason')!='ILLEGAL_CONTACT':
        raise ValueError('Expected the original terminal contact stop')
    return next(x for x in reversed(rows) if x.get('contact_stop') and x['step']==rows[-1]['steps'])


def stopped_pairs(model,row,target):
    invalid=[]
    for c in row['contacts']:
        ids=[c[k] for k in ('geom1','geom2')]
        names=[model.geom(i).name for i in ids]
        floor='floor' in names and any(n in names for n in ('FL','FR','RL','RR'))
        task=('task_table' in names and any(n in names for n in ('task_cube_geom','task_blue_cube_geom'))
              or target in names and any(n in names for n in ('task_left_pad','task_right_pad')))
        if not (floor or task) or (task and c['dist']<-.004):
            invalid.append(dict(geoms=names,bodies=[model.body(model.geom_bodyid[i]).name for i in ids],
                                signed_distance_mm=1000*c['dist']))
    return invalid


def history(root,entry):
    name=entry['condition']
    folder=root/'cases'/name
    s=read(folder/'EXECUTION_SPEC.json')
    actual=read_pin(entry['result'])
    scene=Path(s['model_path'])
    observer=Path(s['observer_library']).with_suffix('.cpp')
    if (s['collection_split']!='TRAIN' or s['source_sha']!=BASE or actual['source_sha']!=BASE
            or digest(scene)!=s['frozen_files'][str(scene)] or digest(observer)!=OBSERVER
            or digest(Path(s['observer_library']))!=read_pin(s['compiled_observer_result'])['observer_candidate']['library_sha256']
            or entry['requests']!=1 or not entry['hard_STOP'] or entry['post_STOP_ticks']!=0):
        raise ValueError('Missing original producer, scene or terminal safety proof')
    m=mujoco.MjModel.from_binary_path(str(scene))
    native=folder/'native-01'
    stop=last_stop(native/'physics.jsonl')
    if stop['step']!=entry['ticks']:
        raise ValueError('Native terminal tick changed')
    target='task_cube_geom' if '-red' in name else 'task_blue_cube_geom'
    pairs=stopped_pairs(m,stop,target)
    qp=native/'LIVE_DATA/QUERIES.private.json'
    captures=read(qp)
    final=captures[-1]
    out=dict(condition=name,result_sha256=entry['result']['sha256'],query_sha256=digest(qp),scene_sha256=digest(scene),
             terminal_tick=stop['step'],last_query_tick=final['native_step'],phase=final['phase'],pairs=pairs,
             last_query_native_grasp=final['actual_state']['native_grasp'],post_STOP_ticks=0)
    if name not in PRIOR:
        return out
    if final['phase']!='CARRY' or not pairs or any(set(c['geoms'])!={'task_cube_geom','task_blue_cube_geom'} for c in pairs):
        raise ValueError('This pilot requires both original blue-red carry stops')
    g=s['multitask_geometry']
    q=next(x for x in captures if x['phase']=='CARRY')
    obj,obstacle,halves=planar(q['actual_state'],g)
    zone=np.asarray(g['zones']['right']['center_world_XY'])
    out['two_leg_bound_mm']=[1000*v for v in remaining_clearance(obj,obj[0],zone,obstacle,halves)]
    out['original_swept_footprints_overlap']=bool(swept_footprints_overlap(obj,zone,obstacle,halves))
    data=mujoco.MjData(m)
    blue,red=(m.geom(n).id for n in ('task_blue_cube_geom','task_cube_geom'))
    zs=[]
    for pos in (final['actual_state']['qpos'],stop['qpos']):
        data.qpos[:]=pos
        mujoco.mj_fwdPosition(m,data)
        zs.append(float(data.geom_xpos[blue,2]))
    out['query_to_stop_blue_center_drop_mm']=1000*(zs[0]-zs[1])
    out['query_to_stop_ms']=2*(stop['step']-final['native_step'])
    if min(out['two_leg_bound_mm'])<=1000*TRANSFER_CLEARANCE_M or not out['original_swept_footprints_overlap']:
        raise ValueError('Proposed planar bypass lacks static clearance')
    return out


def audit(root):
    spec=read(root/'EXECUTION_SPEC.json')
    if (mujoco.__version__!='3.2.7' or mujoco.get_mjcb_control() is not None or
            spec['source_sha']!=BASE or spec['compiled_population']['sha256']!=POP):
        raise ValueError('Audit requires frozen producer, scene bank and geometry-only MuJoCo')
    cells={}
    for row in read(root/'RECORDS.private.json'):
        if row['split']=='TRAIN':
            cells.setdefault(row['condition'].rsplit('-',1)[0],[]).append(row)
    with forbid_integration(mujoco):
        pairs=[history(root,r) for rows in cells.values() for r in rows[:2]
               if len(rows)>=2 and all(x['first_failure']=='ILLEGAL_CONTACT' for x in rows[:2])]
        pop=read_pin(spec['compiled_population'])
        row=next(v for v in pop['conditions']['TRAIN'] if v['id']==PILOT)
        cluster=next(v for v in pop['clusters'] if v['id']==row['scene_family'])
        scene=Path(cluster['scene']['path'])
        if digest(scene)!=cluster['scene']['sha256']:
            raise ValueError('Registered pilot scene changed')
        model=mujoco.MjModel.from_binary_path(str(scene))
        data=mujoco.MjData(model)
        mujoco.mj_fwdPosition(model,data)
        ids=[model.geom(n).id for n in ('task_blue_cube_geom','task_cube_geom')]
        obj,obstacle=data.geom_xpos[ids,:2]
        extent=sum((abs(data.geom_xmat[i].reshape(3,3))@model.geom_size[i])[:2] for i in ids)
        zone=np.asarray(cluster['scoring_geometry']['zones']['right']['center_world_XY'])
        gaps=remaining_clearance(obj,obj[0],zone,obstacle,extent)
    if {r['condition'] for r in pairs if r['condition'] in PRIOR}!=set(PRIOR) or min(gaps)<=TRANSFER_CLEARANCE_M:
        raise ValueError('Prior evidence or unopened scene qualification failed')
    return dict(status='TRAIN_STATIC_CARRY_BYPASS_VERIFIED',producer=BASE,observer_source_sha256=OBSERVER,
                target_cell=CELL,prior=PRIOR,pilot=PILOT,pilot_scene_sha256=digest(scene),prior_rows=pairs,
                pilot_two_leg_centered_clearance_mm=[1000*v for v in gaps],new_native_requests=0,
                new_integrations=0,new_generations=0,validation_outcomes_used=0)


if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('producer',type=Path)
    args.add_argument('output',type=Path)
    opts=args.parse_args()
    result=audit(opts.producer)
    with opts.output.open('x') as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
