"""Continue two successful TRAIN pilots on their unopened fixed variants, with the original expert code."""
import argparse
import json
from pathlib import Path
import time

from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest

BLUE = 'CHV1-TRAIN-F2-PLACE-blue-right'
RED = 'CHV1-TRAIN-F2-HOLD-red'
# Terminal, native result and admitted RGB+action episode of each successful pilot.
PILOTS = {
    'b29a56c7c083208408fddbcf34023d8af9374ef1b4cda20e0f90dc57df3c0aa3':
        (BLUE, 'e939f7d315dd2873143609cb6ae987c9ff801ff7673af4035baad850e9c69395',
         'cc98103d647c1b5273b9a1f05fc0a143cd685d2fe084b880b4bbca52aa474d85',
         '5c1cc64b27dd7301d93ffe22ddd1fd1adbf394616e509366b197bf54577cac4e'),
    'deaa9a88dc1426a8a74a46a026a7c9f7ee190bca3256ba301ff98c0b1f9920dc':
        (RED, 'ea1cda419c9cbb691d997a79feebfac6255f6adbd63bf3b498536f87830b5097',
         '975ce5c6a3e78a8666312ee407eb301f49bb0d96d9978f6b357600c400c65061',
         '5db29ebc84fe687aebb8b5bddd4bb0bd216b1e139e94c404a1fb58190030529e'),
}


def identity(pilot_pin):
    spec = read_pin(pilot_pin)
    cell, terminal_sha, native_sha, episode_sha = PILOTS[pilot_pin['sha256']]
    root = Path(spec['registered_root'])
    closed = read_pin(dict(path=str(root/'RESULT.private.json'),sha256=terminal_sha))
    if (closed['source_sha'] != spec['source_sha'] or closed['exit_code'] or closed['unreconciled']
            or closed['paid_requests'] != 1 or closed['admitted_train'] != 1 or len(closed['records']) != 1):
        raise ValueError('Successful closed single-pilot owner required')
    r = closed['records'][0]
    if (r['condition'] != cell+'-02' or r['split'] != 'TRAIN' or r['first_failure'] != 'TASK_SUCCESS'
            or r['requests'] != 1 or r['hard_STOP'] or r['technical_failure'] or not r['admitted']
            or r['result']['sha256'] != native_sha or r['episode']['sha256'] != episode_sha):
        raise ValueError('Pilot native result or admission differs')
    native, episode = read_pin(r['result']), read_pin(r['episode'])
    if (native['source_sha'] != spec['source_sha'] or native['physical_requests'] != 1
            or episode['id'] != r['condition'] or episode['split'] != 'TRAIN' or episode['windows'] <= 0):
        raise ValueError('Wrong source, native denominator or training episode')
    return cell, spec


def variants(pop, cell):
    rows = sorted((r for r in pop['conditions']['TRAIN'] if r['id'].rsplit('-',1)[0]==cell
                   and r['geometry_variant']>=3),key=lambda r:r['geometry_variant'])
    if ([r['id'] for r in rows] != [cell+f'-{n:02d}' for n in range(3,20)]
            or any(r['id']!=r['condition_id'] or r['split']!='TRAIN' or r['native_request_limit']['max_actual_requests']!=1
                   for r in rows)):
        raise ValueError('The fixed remaining TRAIN conditions changed')
    return rows


def blue_clearance(model, geometry):
    import mujoco
    import numpy as np
    from scripts.sla.core_hopper_blue_carry import remaining_clearance, TRANSFER_CLEARANCE_M
    data=mujoco.MjData(model)
    mujoco.mj_fwdPosition(model,data)
    ids=[model.geom(n).id for n in ('task_blue_cube_geom','task_cube_geom')]
    obj,obstacle=data.geom_xpos[ids,:2]
    extent=sum((abs(data.geom_xmat[i].reshape(3,3))@model.geom_size[i])[:2] for i in ids)
    zone=np.asarray(geometry['zones']['right']['center_world_XY'])
    gaps=remaining_clearance(obj,obj[0],zone,obstacle,extent)
    return dict(qualified=min(gaps)>TRANSFER_CLEARANCE_M,leg_clearance_mm=[1000*float(v) for v in gaps])


def red_clearance(model, original, records):
    from scripts.sla.core_hopper_red_approach_audit import analyze_case, world_pitch, CANDIDATE
    gaps=[]
    bilateral=[]
    for record in records:
        witness=analyze_case(original,record,(CANDIDATE,),world_pitch,model)
        stages=witness['by_angle_deg'][str(CANDIDATE)]
        close=witness['closing_projection'][str(CANDIDATE)]
        gaps.extend(v[k]['gap_mm'] for v in stages.values() for k in ('other_cube','table','selected_nonpad'))
        gaps.extend(close['forbidden_min_gap_mm'].values())
        bilateral.append(close['both_pads_intersect_red'])
    return dict(qualified=bool(min(gaps)>0 and all(bilateral)),minimum_forbidden_gap_mm=float(min(gaps)),
                bilateral_on_both_witnesses=all(bilateral))


def geometry(pilot_pin):
    import mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration
    if mujoco.__version__!='3.2.7' or mujoco.get_mjcb_control() is not None:
        raise ValueError('Expected the isolated frozen geometry runtime')
    cell,pilot=identity(pilot_pin)
    pop=read_pin(pilot['compiled_population'])
    clusters={c['id']:c for c in pop['clusters']}
    primary=Path(read_pin(pilot['carry_repair']['original_producer'])['registered_root'])
    records={r['condition']:r for r in json.loads((primary/'RECORDS.private.json').read_bytes()) if r['split']=='TRAIN'}
    rows=[]
    started=time.process_time()
    with forbid_integration(mujoco) as calls:
        for entry in variants(pop,cell):
            cluster=clusters[entry['scene_family']]
            ref=cluster['scene']
            if digest(Path(ref['path']))!=ref['sha256'] or (primary/'cases'/entry['id']).exists():
                raise ValueError('Scene identity changed or condition is already open')
            model=mujoco.MjModel.from_binary_path(ref['path'])
            if cell==BLUE:
                measured=blue_clearance(model,cluster['scoring_geometry'])
            else:
                measured=red_clearance(model,primary,[records[cell+'-00'],records[cell+'-01']])
            rows.append(dict(condition=entry['id'],scene_sha256=ref['sha256'],**measured))
        if calls:
            raise RuntimeError('Geometry qualification attempted integration')
    return dict(status='TRAIN_FIXED_SERIES_GEOMETRY',pilot_spec=pilot_pin['sha256'],cell=cell,
                population=pilot['compiled_population']['sha256'],rows=rows,cpu_s=time.process_time()-started,
                VAL_inputs=0,model_generations=0,native_requests=0,native_ticks=0)


def qualified_prefix(pop,repair):
    cell,pilot=identity(repair['completed_pilot'])
    prior={k:v for k,v in repair.items() if k not in ('completed_pilot','series_geometry')}
    audit=read_pin(repair['series_geometry'])
    if (prior!=pilot['carry_repair'] or audit['status']!='TRAIN_FIXED_SERIES_GEOMETRY'
            or audit['pilot_spec']!=repair['completed_pilot']['sha256'] or audit['cell']!=cell
            or audit['population']!=pilot['compiled_population']['sha256']
            or any(audit[k] for k in ('VAL_inputs','model_generations','native_requests','native_ticks'))):
        raise ValueError('Wrong pilot, scope or static evidence')
    rows=variants(pop,cell)
    clusters={c['id']:c['scene']['sha256'] for c in pop['clusters']}
    eligible=[]
    stopped=False
    for row,test in zip(rows,audit['rows'],strict=True):
        if row['id']!=test['condition'] or test['scene_sha256']!=clusters[row['scene_family']]:
            raise ValueError('Static rows differ from the registered population')
        if not test['qualified']:
            stopped=True
        if not stopped:
            eligible.append(row)
    if not eligible:
        raise ValueError('First remaining condition did not qualify; keep this cell closed')
    return eligible


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('pilot',type=Path)
    parser.add_argument('sha256')
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=geometry(dict(path=str(args.pilot),sha256=args.sha256))
    with args.output.open('x') as target:
        json.dump(result,target,sort_keys=True,indent=2,allow_nan=False)
