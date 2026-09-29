"""Independent CPU reconstruction of the bounded preparation packet."""
from pathlib import Path
import json
import sys
if '--native-canonical-replay' in sys.argv:
    import isaacgym
import numpy as np
import pi0256_preparation as prep
from pi0255_motion3 import sha,write
from go2arx5_vla.trajectory.so3 import quat_to_matrix,geodesic_angle

ROOT=prep.ROOT


def replay():
    correction='--tcp-sync-correction' in sys.argv
    registry=json.loads((ROOT/('design/PREPARATION_V2_TCP_SYNC_CONTRACT.json' if correction else 'design/PREPARATION_V2_CONTRACT.json')).read_text())
    summaries=[]
    for case in registry['cases']:
        out=ROOT/'episodes'/case['case_id'];file=out/'result.json'
        if not file.exists():continue
        result=json.loads(file.read_text())
        for name,identity in result['files'].items():
            p=out/name;assert p.stat().st_size==identity['bytes'] and sha(p)==identity['sha256'],p
        if result['validity']!='VALID':
            summaries.append(dict(case=case['case_id'],group=case['group'],validity=result['validity'],engineering=result.get('engineering_error')));continue
        ticks=[json.loads(x) for x in (out/'raw_ticks.jsonl').read_text().splitlines()]
        first=ticks[0]['pre_step'] if ticks else json.loads((out/'post_reset.json').read_text())
        previous=np.asarray(first['TCP_world']);q=np.asarray(first['quaternion_xyzw']);previous_time=first['time_s']
        start=json.loads((out/'physics_started.json').read_text())['time_s'];state='PREPARE';stable=None;best_interval=0.;qualified_ticks=0
        for row in ticks:
            post=row['post_step'];p=np.asarray(post['TCP_world']);rotation=np.asarray(post['quaternion_xyzw']);dt=post['time_s']-previous_time
            computed=dict(speed_mps=float(np.linalg.norm(p-previous)/dt),
                angular_speed_radps=float(geodesic_angle(quat_to_matrix(rotation),quat_to_matrix(q))/dt),
                position_error_m=float(np.linalg.norm(p-post['desired_world'])),
                rotation_error_rad=float(geodesic_angle(quat_to_matrix(rotation),quat_to_matrix(post['desired_quaternion']))))
            for k,v in computed.items():assert np.isclose(v,row[k],atol=1e-10,rtol=1e-10),(case['case_id'],k)
            assert row['failure_pose_world']==prep.failure_pose(post,row['pre_autoreset_event'])
            assert row['post_step_is_failure_pose']==(row['pre_autoreset_event'] is None)
            assert row['joint_limit_violation']==post['telemetry']['joint_limit_violation']
            state,stable=prep.advance(state,dict(row,**computed),start,stable)
            assert state==row['state'] and stable==row['stable_since']
            qualified_ticks+=int(prep.qualified(row))
            if stable is not None:best_interval=max(best_interval,row['time_s']-stable)
            previous,q,previous_time=p,rotation,post['time_s']
        if result['success']:assert state=='SETTLED' and best_interval>=.5-1e-8
        steps=[json.loads(x) for x in (out/'physics_steps.jsonl').read_text().splitlines()]
        assert len(steps)==len(ticks)+1,'implicit reset step absent or unrecorded control step'
        calls=[json.loads(x) for x in (out/'actor_calls.jsonl').read_text().splitlines()]
        assert len(calls)==2*len(ticks)
        for call in calls:
            with np.load(out/call['input_file'],allow_pickle=False) as arrays:
                import hashlib
                assert hashlib.sha256(np.ascontiguousarray(arrays['history']).tobytes()).hexdigest()==call['array_sha256']
        clean=[r for r in ticks if r['clean']]
        fk_errors=[float(np.linalg.norm(np.asarray(r['post_step']['TCP_world'])-r['post_step']['FK_TCP_world'])) for r in clean]
        fk_rotations=[float(geodesic_angle(quat_to_matrix(r['post_step']['quaternion_xyzw']),quat_to_matrix(r['post_step']['FK_quaternion_xyzw']))) for r in clean]
        summaries.append(dict(case=case['case_id'],group=case['group'],validity='VALID',success=result['success'],failure=result['failure'],
            ticks=len(ticks),control_steps=len(steps),end_time_s=result['end_time_s'],qualified_ticks=qualified_ticks,stable_interval_max_s=best_interval,
            clean_XYZ_RMS_m=float(np.sqrt(np.mean([r['position_error_m']**2 for r in clean]))) if clean else None,
            clean_SO3_RMS_rad=float(np.sqrt(np.mean([r['rotation_error_rad']**2 for r in clean]))) if clean else None,
            clean_speed_max_mps=max([r['speed_mps'] for r in clean],default=None),
            clean_angular_speed_max_radps=max([r['angular_speed_radps'] for r in clean],default=None),
            soft_joint_violation_ticks=sum(r['joint_limit_violation'] for r in clean),
            hard_joint_violation_ticks=sum(any(v<0 for v in r['post_step']['hard_URDF_joint_margins'].values()) for r in clean),
            FK_vs_native_TCP_max_m=max(fk_errors,default=None),
            FK_vs_native_SO3_max_rad=max(fk_rotations,default=None),
            terminal_event=ticks[-1]['pre_autoreset_event'] if ticks else None,
            result_sha256=sha(file)))
    pairs=[]
    for i in range(6):
        a=ROOT/'episodes'/[c for c in registry['cases'] if c['group']=='A'][i]['case_id']
        b=ROOT/'episodes'/[c for c in registry['cases'] if c['group']=='B'][i]['case_id']
        if not (a/'pre_first_step_identity.npz').exists() or not (b/'pre_first_step_identity.npz').exists():continue
        with np.load(a/'pre_first_step_identity.npz',allow_pickle=False) as x,np.load(b/'pre_first_step_identity.npz',allow_pickle=False) as y:
            match={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
        a0=json.loads((a/'pre_first_step.json').read_text());b0=json.loads((b/'pre_first_step.json').read_text())
        match.update(python_rng=a0['python_rng']==b0['python_rng'],numpy_cursor=a0['numpy_cursor']==b0['numpy_cursor'])
        assert all(match.values()),('PAIR_INITIAL_IDENTITY_MISMATCH',i,match)
        pairs.append(dict(family=i,pre_first_physics_matches=match,post_intervention_reference_history='PACKAGE_DEPENDENT'))
    groups={g:dict(valid=sum(x['group']==g and x['validity']=='VALID' for x in summaries),
        passed=sum(x['group']==g and x.get('success',False) for x in summaries)) for g in 'ABC'}
    d=dict(schema='PI0256_INDEPENDENT_PREPARATION_REPLAY_V1',status='PASS_RECORDED_CPU_RECONSTRUCTION',
           groups=groups,cases=summaries,pairs=pairs,new_physics=0,new_GR00T=0,selected_checkpoint='NONE')
    write(ROOT/('evidence/preparation_replay_tcp_sync.json' if correction else 'evidence/preparation_replay.json'),d)
    print(json.dumps(dict(status=d['status'],groups=groups,pairs=len(pairs))),flush=True)


def tcp_alias_probe():
    """Source-backed cache-copy correction, using original recorded camera refresh."""
    import ast
    import types
    import torch
    source=json.loads((ROOT/'evidence/runtime_readback.json').read_text())
    entry=next(x for x in source['files'] if x['name']=='wbc_env')
    assert sha(entry['path'])==entry['sha256']
    tree=ast.parse(Path(entry['path']).read_text())
    node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='_arm_post_physics_hook')
    node.body=node.body[:4]  # actual copy + TCP shift only; no controller/physics invocation
    a=ROOT/'episodes/pi0256-prep-A00'
    before=json.loads((a/'post_reset.json').read_text())
    after=json.loads((a/'raw_ticks.jsonl').read_text().splitlines()[0])['pre_step']
    offset=np.array([.1424,0.,.0001057]);rotation=quat_to_matrix(before['quaternion_xyzw'])
    actual_delta=np.asarray(before['TCP_world'])-after['TCP_world']
    assert before['time_s']==after['time_s'] and np.allclose(actual_delta,rotation@offset,atol=1e-7,rtol=0)
    scope={'quat_apply':lambda q,v:v@torch.as_tensor(rotation.T,dtype=v.dtype)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),entry['path'],'exec'),scope)
    checks=[]
    for detach in (False,True):
        raw=torch.zeros(1,1,13);raw[0,0,:3]=torch.tensor([.2,-.1,.6]);raw[0,0,3:7]=torch.as_tensor(before['quaternion_xyzw'])
        physical=raw.clone();view=raw[:,0]
        obj=types.SimpleNamespace(arm_time_buf=torch.zeros(1),force_time_buf=torch.zeros(1),num_envs=1,num_bodies=1,
            rigid_body_state=raw,ee_idx=0,end_effector_state=view.clone() if detach else view,
            ee_local_offset=torch.as_tensor(offset,dtype=torch.float32).view(1,3))
        scope['_arm_post_physics_hook'](obj)
        desired=obj.end_effector_state.clone()
        raw_unchanged=bool(torch.equal(raw,physical))
        assert raw_unchanged==detach
        raw.copy_(physical)  # fetch_results updates native raw rigid-body cache
        correct=torch.equal(obj.end_effector_state,desired)
        assert correct==detach
        checks.append(dict(detached=detach,render_refresh_preserves_TCP=correct,raw_physical_pose_unmodified_before_refresh=raw_unchanged))
    receipt=dict(schema='PI0256_SOURCE_PROVEN_TCP_CACHE_ALIAS_CORRECTION_V1',status='PASS_TCP_ALIAS_CAUSAL_CPU_PROOF',
        source_sha256=entry['sha256'],actual_same_tick_delta_m=actual_delta.tolist(),predicted_rotated_TCP_offset_m=(rotation@offset).tolist(),
        A00_result_sha256=sha(a/'result.json'),tests=checks,physics=0,GR00T=0,
        correction='detach derived EEF cache once in project-owned B/C constructor; native hook continues unchanged',
        trajectory_pose_threshold_weight_gain_history_changes=False)
    write(ROOT/'evidence/tcp_alias_cpu_proof.json',receipt);print(receipt['status'],flush=True)


def canonical_replay():
    import torch
    torch.set_num_threads(1)
    from modules.trajectory import to_canonical,quat_to_mat,TrajectoryBatch
    rows=[]
    for file in sorted((ROOT/'episodes').glob('*/result.json')):
        result=json.loads(file.read_text())
        if result['validity']!='VALID':continue
        out=file.parent
        for line in (out/'reference_loads.jsonl').read_text().splitlines():
            item=json.loads(line);event=item['event']
            with np.load(out/item['file'],allow_pickle=False) as data:
                dt=float(np.diff(data['timestamps'])[0])
                gamma,law=to_canonical(torch.as_tensor(data['raw_position'],dtype=torch.float32),
                                       quat_to_mat(torch.as_tensor(data['raw_quaternion'],dtype=torch.float32)),dt)
                batch=TrajectoryBatch(1,data['gamma_s'].shape[1],data['tl_t'].shape[1],device='cpu')
                batch.load([0],[gamma],[law])
                batch.gamma_p+=torch.as_tensor(event['anchor_world_m'],dtype=torch.float32).view(1,1,3)
                errors={k:float(np.max(np.abs(getattr(batch,k).numpy()-data[k]))) for k in
                    ('gamma_s','gamma_p','gamma_quat','gamma_tangent','L','tl_t','tl_s','tl_sdot','T')}
                assert all(np.isfinite(v) for v in errors.values())
                rows.append(dict(case=out.name,file=item['file'],native_reference_sha256=item['sha256'],cpu_max_abs_errors=errors,
                    gamma_capacity=data['gamma_s'].shape[1],time_capacity=data['tl_t'].shape[1],raw_points=len(data['timestamps'])))
    d=dict(status='ACTUAL_NATIVE_CANONICAL_CPU_RECOMPUTED',references=rows,
        max_abs_by_array={k:max(x['cpu_max_abs_errors'][k] for x in rows) for k in rows[0]['cpu_max_abs_errors']},
        physics=0,GR00T=0,numerical_errors_reported_without_posthoc_threshold_change=True)
    write(ROOT/'evidence/native_canonical_cpu_replay.json',d)
    print(json.dumps(dict(status=d['status'],references=len(rows),max_abs_by_array=d['max_abs_by_array'])),flush=True)


if __name__=='__main__':
    if '--tcp-alias-probe' in sys.argv:tcp_alias_probe()
    elif '--native-canonical-replay' in sys.argv:canonical_replay()
    else:replay()
