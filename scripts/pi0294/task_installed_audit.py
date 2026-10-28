"""Independent numerical audit of the saved native14D published references."""
import json
from pathlib import Path
import numpy as np
from .task_capture_runtime import cpu_stage,sha
from go2arx5_vla.trajectory.so3 import quat_slerp


def run(root):
    from scripts.pi0292.durable import write
    root=Path(root)
    with cpu_stage(root,'TASK_NATIVE_INSTALLED_REFERENCE_CPU',cores=1,cpu_s_cap=600):
        output=[]
        for name in ('TASK-PILOT-W0-T2-engineering04','TASK-PILOT-W1-T1','TASK-PILOT-W1-T2'):
            p=root/'episodes'/name;rows=[json.loads(s) for s in (p/'planner.jsonl').read_text().splitlines()]
            accepts={r['ref_id']:r for r in rows if r['kind']=='accept'}
            n=0;max_position=0.;max_quaternion=0.;covered=set()
            for row in rows:
                if row['kind']!='installed' or row.get('ref_id') not in accepts:continue
                accept=accepts[row['ref_id']];doc=json.loads(accept['raw']);source=accept['source_mpc_s']
                times=np.asarray(row['times']);poses=np.asarray(row['poses']);inputs=np.asarray(row['inputs'])
                if poses.shape!=(len(times),14) or inputs.shape!=(len(times),12) or inputs.any():
                    raise ValueError('ORIGINAL_NATIVE14D_POSE_AND_ZERO12D_INPUT_REQUIRED')
                if not np.isfinite(poses).all() or np.any(np.diff(times)<=0):raise ValueError('INSTALLED_FINITE_ORDERED_REFERENCE')
                for part,offset in (('ee',0),('base',7)):
                    points=doc[part]['waypoints'];grid=np.asarray([w['time'] for w in points])
                    xyz=np.asarray([w['position'] for w in points]);q=np.asarray([w['quaternion_xyzw'] for w in points])
                    for t,actual in zip(times,poses[:,offset:offset+7]):
                        dt=t-source
                        position=np.asarray([np.interp(dt,grid,xyz[:,k]) for k in range(3)])
                        index=int(np.clip(np.searchsorted(grid,dt,side='right')-1,0,len(grid)-2))
                        fraction=float(np.clip((dt-grid[index])/(grid[index+1]-grid[index]),0,1))
                        expected=quat_slerp(q[index],q[index+1],fraction)
                        got=actual[3:]/np.linalg.norm(actual[3:]);expected=expected/np.linalg.norm(expected)
                        max_position=max(max_position,float(np.max(np.abs(actual[:3]-position))))
                        max_quaternion=max(max_quaternion,float(min(np.max(np.abs(got-expected)),np.max(np.abs(got+expected)))))
                covered.add(row['ref_id']);n+=1
            if len(covered)!=640 or max_position>1e-6 or max_quaternion>1e-6:
                raise ValueError('PUBLISHED_NATIVE_REFERENCE_NUMERICALLY_DIFFERS_FROM_SUBMITTED_WORLD_PATH')
            output.append(dict(episode=name,status='PASS_SAVED_NATIVE14D_INSTALLATION_PARITY',transactions=len(covered),
                installed_messages=n,max_position_abs_m=max_position,max_sign_invariant_unit_quaternion_component=max_quaternion,
                tolerance=1e-6,original_actual_sample_times_used=True,no_time_reanchor=True,planner_sha256=sha(p/'planner.jsonl')))
        target=root/'evidence/TASK_NATIVE_INSTALLED_REFERENCE_CPU.json'
        if target.exists():raise FileExistsError('FIRST_INSTALLATION_AUDIT_PRESERVED')
        value=dict(status='PASS',episodes=output,new_physics=0,new_model_queries=0,controller_queries=0)
        write(target,value);print(json.dumps(value))
