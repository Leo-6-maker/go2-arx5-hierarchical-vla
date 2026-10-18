"""Zero-integration geometry validation of saved FD2 states, not new episodes."""
import itertools
import json
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.fd1 import put,sha


def box_gap_and_footprint(engine):
    table=engine.geoms['table'];center=engine.data.geom_xpos[table];half=engine.model.geom_size[table]
    if not np.allclose(engine.data.geom_xmat[table].reshape(3,3),np.eye(3),atol=1e-12,rtol=0):
        raise ValueError('FD2_UNCHANGED_AXIS_ALIGNED_TABLE_REQUIRED')
    table_top=float(center[2]+half[2]);gaps=[];inside=[]
    for side in ['left','right']:
        g=engine.geoms[side+'_pad'];axes=engine.data.geom_xmat[g].reshape(3,3)
        corners=np.asarray(list(itertools.product(*[[-s,s] for s in engine.model.geom_size[g]])))
        corners=corners@axes.T+engine.data.geom_xpos[g]
        gaps.append(float(corners[:,2].min()-table_top))
        inside.append(bool(np.all(corners[:,:2]>=center[:2]-half[:2]-1e-12)
            and np.all(corners[:,:2]<=center[:2]+half[:2]+1e-12)))
    return np.asarray(gaps),inside,table_top


def run(root):
    import mujoco
    from .fd2 import validate
    from .scene import make_engine
    root=Path(root);lock=OwnerLock(root/'FD2_OWNER.lock');m,inputs=validate(root)
    binding=read(root/'plans/GEOMETRY_AUDIT_BINDING.private.json');l=read(root/'RESOURCE_LEDGER.json')
    if (Path(__file__).resolve()!=Path(binding['file']).resolve() or sha(__file__)!=binding['sha256']
            or l['status']!='GEOMETRY_AUDIT_RESERVED' or l['CPU_reserved_s']!=900
            or l['unknown_effects'] or l['engine_starts']+4>700):
        raise PermissionError('FD2_EXACT_SAVED_GEOMETRY_AUDIT_BINDING_AND_FOUR_START_RESERVE')
    for p in (root/'worker_processes').glob('*.json'):
        w=read(p)
        if w['operation']=='geometry-audit':continue
        stat=Path('/proc',str(w['pid']),'stat');f=stat.read_text().split() if stat.exists() else None
        if f and f[2]!='Z' and f[21]==str(w['proc_start']):
            raise PermissionError('FD2_ALL_PRODUCERS_AND_ARCHIVERS_TERMINAL_FIRST')
    started=time.process_time();rows=[];starts=0;failure=None;result=None
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU,(600,600));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def forbidden(*a,**k):raise PermissionError('FD2_GEOMETRY_AUDIT_FORBIDS_ALL_INTEGRATION')
    def expire(*a):raise TimeoutError('FD2_FINITE_GEOMETRY_AUDIT')
    original=[mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2]
    mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=forbidden
    signal.signal(signal.SIGALRM,expire)
    signal.alarm(max(1,int(min(900,m['limits']['hard_close_epoch']-time.time()))))
    try:
        plan=read(root/'plans/PILOT_PLAN.json')
        for group in plan['groups']:
            cards=[c for c in plan['cards'] if c['group_id']==group['group_id']]
            l['engine_starts']+=1;starts+=1;write(root/'RESOURCE_LEDGER.json',l)
            engine=make_engine(cards[0],inputs['asset_root'])
            try:
                if any(engine.model.geom_type[engine.geoms[n]]!=mujoco.mjtGeom.mjGEOM_BOX
                        for n in ['table','left_pad','right_pad']):
                    raise ValueError('FD2_FROZEN_THREE_BOX_GEOMS_REQUIRED')
                for card in cards:
                    f=root/'episodes'/card['id'];xml=read(f/'scene.xml.json')
                    if xml['xml']!=engine.xml:raise ValueError('FD2_PAIR_MODEL_XML_MUST_MATCH_ORIGINAL')
                    with np.load(f/'highrate.npz',allow_pickle=False) as z:
                        high=z['rows'].copy();nq=int(z['nq']);nv=int(z['nv'])
                    with np.load(f/'DYNAMIC_CLEARANCE.npz',allow_pickle=False) as z:live=z['live'].copy()
                    if len(high)!=len(live):raise ValueError('FD2_SAVED_TICK_JOIN')
                    max_gap_error=max_query_error=0.;all_inside=True;minimum=[float('inf'),float('inf')]
                    for index,row in enumerate(high):
                        engine.data.qpos[:]=row[1:1+nq];engine.data.qvel[:]=row[1+nq:1+nq+nv]
                        engine.data.ctrl[:]=row[1+nq+nv:1+nq+nv+8];engine.data.time=row[0]
                        mujoco.mj_forward(engine.model,engine.data)
                        gap,inside,table_top=box_gap_and_footprint(engine)
                        if abs(table_top-.44)>1e-12:raise ValueError('FD2_FROZEN_TABLE_TOP')
                        all_inside &= all(inside);minimum=np.minimum(minimum,gap).tolist()
                        query=np.asarray([mujoco.mj_geomDistance(engine.model,engine.data,engine.geoms[s+'_pad'],
                            engine.geoms['table'],2.,np.zeros(6)) for s in ['left','right']])
                        max_gap_error=max(max_gap_error,float(np.max(np.abs(gap-live[index,1:3]))))
                        max_query_error=max(max_query_error,float(np.max(np.abs(query-live[index,3:5]))))
                    if max_gap_error>1e-10 or max_query_error>1e-10:
                        raise ValueError('FD2_SAVED_GEOMETRY_CAPTURE_RECONSTRUCTION_DIFFERS')
                    rows.append(dict(id=card['id'],saved_states=len(high),
                        maximum_vertical_gap_reconstruction_error_m=max_gap_error,
                        maximum_original_native_query_reconstruction_error_m=max_query_error,
                        all_pad_box_vertices_inside_finite_table_XY=all_inside,
                        minimum_vertex_minus_actual_table_top_m=minimum,
                        vertical_gap_is_exact_separation_for_inside_above_boxes=all_inside and min(minimum)>=0,
                        raw_native_query_not_asserted_equal_to_exact_Euclidean_gap=True,
                        native_query_distmax_m=2.,native_enableflags=int(engine.model.opt.enableflags),
                        raw_highrate_sha256=sha(f/'highrate.npz'),raw_measurement_sha256=sha(f/'DYNAMIC_CLEARANCE.npz')))
            finally:engine.close()
        result=dict(status='PASS_SAVED_GEOMETRY_RECONSTRUCTION_NO_NEW_PHYSICS',rows=rows,
            engine_starts=starts,integrations=0,Teacher_episodes=0,model_queries=0,
            mujoco_version=mujoco.__version__,runtime_source=m['code']['vla_commit'],audit_source=binding['source_sha'],
            surface_gap_primary='EXACT_PAD_BOX_VERTEX_TO_ACTUAL_FINITE_TABLE_TOP_WHEN_FOOTPRINT_INSIDE',
            query_metric_caveat='Original 3.2.7 mj_geomDistance with distmax=2 uses a collision-function query. Raw outputs are retained and reproduced, not silently rescaled or asserted equal to analytic top-plane gaps.',
            official_implementation='https://github.com/google-deepmind/mujoco/blob/3.2.7/src/engine/engine_support.c#L1295')
        put(root/'evidence/GEOMETRY_AUDIT_RESULT.private.json',result)
        return result
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        put(root/'evidence/GEOMETRY_AUDIT_FAILURE.private.json',dict(error=failure,spent_starts=starts,integrations=0))
        raise
    finally:
        signal.alarm(0);mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2=original
        l.update(status='GEOMETRY_AUDIT_FAILED_PRESERVED' if failure else 'GEOMETRY_AUDIT_TERMINAL',
            Core_CPU_s=l['Core_CPU_s']+time.process_time()-started,CPU_reserved_s=0,
            saved_geometry_audit_engine_starts=starts,saved_geometry_audit_new_ticks=0)
        write(root/'RESOURCE_LEDGER.json',l);lock.close()
