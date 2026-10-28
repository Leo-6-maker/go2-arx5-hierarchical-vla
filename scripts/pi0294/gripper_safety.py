"""Adopted R2: saved-state mechanics and finite D0 causal replays, no NN."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

RUN = '20261014T072142Z-incoming-pi0294-r2'
AUTHORITY = '1df3e8cbae0ebd24143868483d74e6b03a6c48e2'
GO_COMMENT = 6163255025
T0, CUTOFF, DEADLINE = 1791962502, 1792002102, 1792005702
SAFETY_DEADLINE = T0 + 4*3600
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'src')]
if not __package__:
    __package__ = 'scripts.pi0294'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admitted(root):
    root = Path(root)
    m = read(root/'RUN_MANIFEST.json')
    if (root.name != RUN or m['run_uuid'] != RUN or m['owner'] != 'EXISTING_LOCAL_PC'
            or m['authority'] != dict(protocol_ref=AUTHORITY, go_comment=GO_COMMENT,
                                     approval='DIRECT_HUMAN_MERGED_B_TO_E_ADOPTION')
            or m['limits']['T0_epoch'] != T0 or m['limits']['deadline_epoch'] != DEADLINE
            or not T0 <= time.time() < SAFETY_DEADLINE):
        raise PermissionError('EXACT_ADOPTED_R2_SAFETY_WINDOW_REQUIRED')
    e = m['execution']
    if e['profile'] != '4080-simulation' or socket.gethostname() not in e['hostnames'] or os.getuid() != e['uid']:
        raise PermissionError('EXACT_EXISTING_HOST_PROFILE_OWNER_REQUIRED')
    binding = read(root/'OUTPUT_BINDING.private.json')
    if (sha(root/'OUTPUT_BINDING.private.json') != e['output_binding_sha256']
            or Path(binding['run_root']).resolve() != root.resolve() or root.is_symlink()
            or not root.resolve().is_relative_to(Path(binding['registered_parent']).resolve())):
        raise PermissionError('REGISTERED_R2_OUTPUT_REQUIRED')
    registry, lease = read(binding['registry']), read(binding['lease'])
    if (not any(r['role'] == e['private_path_role'] and Path(r['path']).resolve() == root.resolve()
                and r['executor_node'] == socket.gethostname() for r in registry['entries'])
            or lease['run_uuid'] != RUN or lease['owner'] != 'EXISTING_LOCAL_PC'
            or lease['status'] != 'ACTIVE' or lease['deadline_epoch'] != DEADLINE):
        raise PermissionError('LIVE_R2_REGISTRY_AND_LEASE_REQUIRED')
    seal_path = root/m['code']['source_seal_relative']
    if (sha(seal_path) != m['code']['source_seal_sha256']
            or REPO.resolve() != (root/m['code']['capsule_relative']).resolve()
            or not REPO.resolve().is_relative_to(root.resolve()) or REPO.is_symlink()):
        raise PermissionError('SEALED_WORKER_SOURCE_REQUIRED')
    for name, digest in read(seal_path)['files'].items():
        path = REPO/name
        if path.is_symlink() or not path.resolve().is_relative_to(REPO.resolve()) or sha(path) != digest:
            raise PermissionError('SEALED_MEMBER_CHANGED:'+name)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PermissionError('CPU_ONLY_NO_CUDA_REQUIRED')
    return m, read(root/'INPUT_BINDINGS.private.json')


def saved_inputs(binding):
    import numpy as np
    from scripts.pi0292.relay_protocol import unseal
    folder = Path(binding['original_failure'])
    expected = {'highrate.npz':'92b266d9489d4a0c86d8cb1bf1769ca6a8512b5f1f59148d09ee8fb84bafac68',
                'INITIAL_STATE.json':'2e1cb6f75ec8702e6365d674c9416ed5b972c1e77009cce262db4ebc7203b641'}
    if any(sha(folder/name) != digest for name,digest in expected.items()):
        raise ValueError('ORIGINAL_SAVED_INPUT_IDENTITY_CHANGED')
    with np.load(folder/'highrate.npz', allow_pickle=False) as data:
        rows, nq, nv = data['rows'].copy(), int(data['nq']), int(data['nv'])
    commands=[]
    for i in range(6):
        bundle=read(folder/'relay/commands'/f'{i:03d}.json')
        body=unseal(bundle['command'])
        if body['kind']!='install_prefix8' or body['sequence']!=i or body['tick']!=i*200:
            raise ValueError('ORIGINAL_PACKET_SEQUENCE_REQUIRED')
        commands.extend(body['prefix_eef10'])
    values=np.asarray(commands,dtype=float)
    if len(rows)!=1081 or values.shape!=(48,10) or not np.isfinite(values).all():
        raise ValueError('EXACT_ORIGINAL_ROWS_AND_SIX_PREFIXES_REQUIRED')
    return read(folder/'CARD.json'), read(folder/'INITIAL_STATE.json'), rows, nq, nv, values


def create_original(card, initial, binding):
    import numpy as np
    from .scene import make_engine
    engine=make_engine(card,binding['asset_root'])
    if (sha_bytes(engine.xml.encode()) != initial['model_xml_sha256']
            or not np.array_equal(engine.data.qpos,np.asarray(initial['qpos']))
            or not np.array_equal(engine.data.qvel,np.asarray(initial['qvel']))):
        engine.close()
        raise ValueError('EXACT_ORIGINAL_XML_AND_INITIAL_STATE_REQUIRED')
    return engine


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def force_balance(root, binding, output, guard):
    import numpy as np
    import mujoco
    from .saved_gripper_trace import analyze
    card, initial, rows, nq, nv, commands=saved_inputs(binding)
    engine=create_original(card,initial,binding)
    m=engine.model
    mapping={name:dict(qpos=int(engine.finger_q[i]),qvel=int(engine.finger_v[i]),ctrl=int(engine.finger_ctrl[i]))
             for i,name in enumerate(('left_slide','right_slide'))}
    arithmetic=analyze(rows,[commands[i:i+8] for i in range(0,48,8)],mapping,initial['qpos'],nq=nq,nv=nv)
    d=mujoco.MjData(m);matrix=np.zeros((nv,nv));force_rows=[];maximum_residual=0.
    q=np.asarray(initial['qpos']);v=np.asarray(initial['qvel'])
    try:
        for i,row in enumerate(rows):
            guard()
            mujoco.mj_resetData(m,d)
            d.qpos[:],d.qvel[:],d.ctrl[:]=q,v,row[1+nq+nv:1+nq+nv+m.nu]
            d.time=i*.002
            mujoco.mj_forward(m,d)
            mujoco.mj_fullM(m,matrix,d.qM)
            residual=matrix@d.qacc+d.qfrc_bias-d.qfrc_actuator-d.qfrc_passive-d.qfrc_constraint-d.qfrc_applied
            maximum_residual=max(maximum_residual,float(np.max(np.abs(residual))))
            acceleration_proxy=(row[1+nq:1+nq+nv]-v)/.002
            cross=matrix[np.ix_(engine.finger_v,engine.arm_v)]@acceleration_proxy[engine.arm_v]
            columns=np.r_[d.time,d.qpos[engine.finger_q],d.qvel[engine.finger_v],
                d.ctrl[engine.finger_ctrl],d.qfrc_actuator[engine.finger_v],d.qfrc_bias[engine.finger_v],
                d.qfrc_passive[engine.finger_v],d.qfrc_constraint[engine.finger_v],
                d.qacc[engine.finger_v],cross,np.diag(matrix)[engine.finger_v]]
            force_rows.append(columns)
            q,v=row[1:1+nq].copy(),row[1+nq:1+nq+nv].copy()
        matrix_rows=np.asarray(force_rows)
        np.savez_compressed(output/'reconstructed_mechanics.npz',rows=matrix_rows,
            fields=np.asarray(['pre_time','pre_q_L','pre_q_R','pre_v_L','pre_v_R','saved_ctrl_L','saved_ctrl_R',
                'recomputed_actuator_L','recomputed_actuator_R','recomputed_bias_L','recomputed_bias_R',
                'recomputed_passive_L','recomputed_passive_R','recomputed_constraint_L','recomputed_constraint_R',
                'recomputed_qacc_L','recomputed_qacc_R','discrete_arm_coupling_proxy_L','discrete_arm_coupling_proxy_R',
                'mass_diagonal_L','mass_diagonal_R']))
        result=dict(status='SAVED_PRE_POST_TIMING_AND_RECOMPUTED_MECHANICS_COMPLETE',saved_rows=1081,
            prefix_points_sent=48,prefix_points_started=44,new_integrations=0,model_constructions=1,
            dynamics_forward_evaluations=1081,arithmetic=arithmetic,mapping=mapping,
            joints=[dict(name=name,axis=m.jnt_axis[engine.joints[name]].tolist(),
                gear=m.actuator_gear[engine.finger_ctrl[i]].tolist(),damping=float(m.dof_damping[engine.finger_v[i]]),
                armature=float(m.dof_armature[engine.finger_v[i]])) for i,name in enumerate(('left_slide','right_slide'))],
            final_pre_step=matrix_rows[-1].tolist(),maximum_recomputed_equation_residual=maximum_residual,
            measured_fields=['saved_post_qpos','saved_post_qvel','saved_pre_step_ctrl','initial_qpos_qvel','prefix_packets'],
            reconstructed_fields=['pre_qpos_qvel_from_previous_saved_row','bias','passive','constraint','M','qacc'],
            UNOBSERVED=['original_warmstart','original_solver_internal_state','original_qfrc_bias_passive_constraint','original_qacc'],
            exact_original_full_state_restore_claim=False,coupling_proxy_is_not_native_continuous_acceleration=True,
            original_FAILURE_preserved=True,physics_starts=0,RGB_pairs=0,neural_queries=0)
        return result
    finally:
        engine.close()


def causal_replay(root,binding,output,guard,case):
    import numpy as np
    from scripts.pi0292.eval_contract import matrix10
    from .scene import qualify_engine
    if case not in ('A','B','C','D'):
        raise ValueError('FROZEN_TWO_BY_TWO_CASE_REQUIRED')
    card,initial,old,nq,nv,commands=saved_inputs(binding)
    engine=create_original(card,initial,binding)
    rows=[];error=None;started=time.monotonic()
    try:
        initial_state=engine.snapshot()
        qualification=qualify_engine(engine,render=False,guard=guard)
        if not qualification['geometry_passed']:
            raise ValueError('ORIGINAL_STATIC_GEOMETRY_NOT_REPRODUCED')
        try:
            for i in range(1200):
                guard()
                if i%25==0:engine.snapshot()
                point=commands[i//25]
                target=matrix10(point) if case in ('A','B') else initial_state['tcp']
                opening=float(point[9]) if case in ('A','C') else 1.
                try:
                    engine.step(target,opening)
                except Exception as exc:
                    error=f'{type(exc).__name__}: {exc}'
                engine.snapshot()  # same post-step observation call as original record_tick
                rows.append(np.r_[engine.data.time,engine.data.qpos,engine.data.qvel,engine.data.ctrl,
                    engine.data.xfrc_applied[engine.ids['tool']],float(bool(engine.safety))])
                if error is not None:break
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
        trace=np.asarray(rows)
        np.savez_compressed(output/'actual_replay.npz',rows=trace,nq=nq,nv=nv)
        prefix=min(len(trace),len(old))
        return dict(case=case,status='VALID_SAFETY_FAIL' if engine.safety else
            'ENGINEERING_FAILURE_PRESERVED' if error else 'BOUNDED_DIAGNOSTIC_COMPLETE',
            error=error,safety=list(engine.safety),physical_starts=1,physics_steps=engine.steps,RGB_pairs=0,
            new_neural_queries=0,wall_s=time.monotonic()-started,
            recorded_arm='ORIGINAL_REFERENCE' if case in ('A','B') else 'ORIGINAL_CONTROLLER_INITIAL_TCP_HOLD',
            recorded_finger='ORIGINAL_REFERENCE' if case in ('A','C') else 'ALL_OPEN_1',
            maximum_original_row_abs_difference=float(np.max(np.abs(trace[:prefix]-old[:prefix]))) if prefix else None,
            original_reproduced=bool(case=='A' and len(trace)==1081 and error=='RuntimeError: JOINT_LIMIT'
                and np.allclose(trace,old,atol=1e-10,rtol=0)),
            final_finger_q=engine.data.qpos[engine.finger_q].tolist(),
            final_finger_qvel=engine.data.qvel[engine.finger_v].tolist(),
            final_motor_ctrl=engine.data.ctrl[engine.finger_ctrl].tolist(),
            not_online_policy_or_training_data=True,original_safety_unchanged=True)
    finally:
        engine.close()


def stress(root,binding,output,guard,case,driver_path):
    import numpy as np
    import mujoco
    from scripts.pi0292.eval_contract import matrix10
    from .gripper_driver import bind
    cases=('open_hold','closed_hold','open_close','close_open','high_end','failure_prefix_hold')
    if case not in cases:raise ValueError('EXACT_SIX_FROZEN_STRESS_CASES_REQUIRED')
    spec_path=Path(driver_path)
    if not spec_path.resolve().is_relative_to((root/'plans').resolve()):raise ValueError('REGISTERED_FROZEN_DRIVER_SPEC_REQUIRED')
    spec=read(spec_path)
    card,initial,old,nq,nv,commands=saved_inputs(binding)
    engine=bind(create_original(card,initial,binding),spec)
    rows=[];error=None;final=None
    try:
        if case in ('closed_hold','close_open'):
            # Pre-registered initial finger condition, before the first integration.
            engine.data.qpos[engine.finger_q]=0.;engine.data.qvel[engine.finger_v]=0.;engine.finger_targets[:]=0.
            mujoco.mj_forward(engine.model,engine.data)
        first=engine.snapshot();target=first['tcp'].copy()
        initial_state=dict(qpos=engine.data.qpos.tolist(),qvel=engine.data.qvel.tolist(),finger_targets=engine.finger_targets.tolist())
        try:
            for i in range(2000):
                guard()
                t=i*.002
                if i%25==0:engine.snapshot()
                pose=target
                if case=='open_hold':opening=1.
                elif case=='closed_hold':opening=0.
                elif case=='open_close':opening=1. if t<1 else 0.
                elif case=='close_open':opening=0. if t<1 else 1.
                elif case=='high_end':opening=float(commands[min(i//25,39),9])
                else:
                    point=commands[min(i//25,47)];pose=matrix10(point);opening=float(point[9])
                try:engine.step(pose,opening)
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                final=engine.snapshot()
                rows.append(np.r_[engine.data.time,engine.data.qpos,engine.data.qvel,engine.data.ctrl,
                    engine.data.xfrc_applied[engine.ids['tool']],float(bool(engine.safety))])
                if error:break
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        trace=np.asarray(rows)
        np.savez_compressed(output/'stress_actual.npz',rows=trace,nq=nq,nv=nv)
        q=trace[:,1+engine.finger_q] if len(trace) else np.empty((0,2))
        return dict(case=case,driver_version=spec['version'],driver_spec_sha256=sha(spec_path),
            status='VALID_SAFETY_FAIL_CANDIDATE_RETIRED' if engine.safety else 'ENGINEERING_FAILURE_PRESERVED' if error else 'STRESS_CASE_PASS',
            error=error,safety=list(engine.safety),physics_steps=engine.steps,physical_starts=1,
            initial_state=initial_state,final_finger_q=engine.data.qpos[engine.finger_q].tolist(),
            finger_minimum_m=q.min(0).tolist() if len(q) else None,finger_maximum_m=q.max(0).tolist() if len(q) else None,
            final_contacts=[] if final is None else final['contacts'],
            finger_ctrl_saturation_rows=int((np.abs(trace[:,1+nq+nv+engine.finger_ctrl])>=8-1e-12).any(1).sum()) if len(q) else 0,
            RGB_pairs=0,neural_queries=0,old_safety_and_arm_controller_unchanged=True,
            explicit_last_reference_hold_after_2_4s=case=='failure_prefix_hold',task_success_claim=False)
    finally:engine.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--phase',choices=['G0','G1','G2'],required=True)
    ap.add_argument('--case');ap.add_argument('--driver-spec');ap.add_argument('--attempt',type=int,default=1);a=ap.parse_args()
    if a.attempt<1:ap.error('positive source-bound engineering attempt required')
    if a.phase=='G1' and a.case is None:ap.error('G1 requires one fixed --case')
    if a.phase=='G2' and (a.case is None or a.driver_spec is None):ap.error('G2 requires fixed case and driver spec')
    root=Path(a.root);manifest,binding=admitted(root)
    from scripts.pi0292.durable import OwnerLock,write
    lock=OwnerLock(root/'GRIPPER_STAGE_OWNER.lock')
    name=a.phase+('-'+a.case if a.case else '')
    if a.phase=='G2':name+='-'+read(a.driver_spec)['version']
    if a.attempt>1:name+=f'-ENG{a.attempt:02d}'
    output=root/'evidence'/name
    output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();cpu=time.process_time()
    state=dict(run_uuid=RUN,phase=a.phase,case=a.case,pid=os.getpid(),
        proc_start=Path('/proc/self/stat').read_text().split()[21],status='RUNNING',source_sha=manifest['code']['vla_commit'],
        T0_epoch=T0,deadline_epoch=min(SAFETY_DEADLINE,time.time()+1200),physics_steps=0)
    write(output/'STATE.json',state)
    ledger=read(root/'RESOURCE_LEDGER.json')
    if ledger.get('Core_CPU_s',0)+1200>8*3600 or ledger.get('physical_starts',0)+1>32:
        raise PermissionError('SAFETY_CHILD_CPU_OR_START_CAP')
    def guard():
        if time.time()>=state['deadline_epoch'] or time.process_time()-cpu>=1200:
            raise RuntimeError('FROZEN_CHILD_CPU_OR_WALL_LIMIT')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4*1024**3:
            raise RuntimeError('FOUR_GIB_PROCESS_MEMORY_LIMIT')
    result=None
    try:
        if a.phase=='G0':result=force_balance(root,binding,output,guard)
        elif a.phase=='G1':result=causal_replay(root,binding,output,guard,a.case)
        else:result=stress(root,binding,output,guard,a.case,a.driver_spec)
        write(output/'RESULT.json',result)
        state['status']='TERMINAL_RECORDED'
    except Exception as exc:
        import traceback
        state.update(status='ENGINEERING_FAILURE_PRESERVED',error=f'{type(exc).__name__}: {exc}')
        write(output/'ERROR.json',dict(error=state['error'],traceback=traceback.format_exc()))
        raise
    finally:
        used=time.process_time()-cpu
        state.update(cpu_s=used,wall_s=time.monotonic()-began)
        write(output/'STATE.json',state)
        ledger['Core_CPU_s']=ledger.get('Core_CPU_s',0)+used
        ledger['physical_starts']=ledger.get('physical_starts',0)+(1 if a.phase in ('G1','G2') else 0)
        bound=1200 if a.phase=='G1' else 2000
        charge=bound if a.phase in ('G1','G2') and result is None else 0 if result is None else result.get('physics_steps',0)
        ledger['physics_integrations']=ledger.get('physics_integrations',0)+charge
        if a.phase in ('G1','G2') and result is None:state['integration_charge']=f'CONSERVATIVE{bound}_EXACT_COUNT_UNOBSERVED'
        ledger.setdefault('stages',[]).append(dict(phase=name,CPU_s=used,status=state['status']))
        write(root/'RESOURCE_LEDGER.json',ledger)
        lock.close()
    print(json.dumps(dict(status=state['status'],phase=name,CPU_s=state['cpu_s']),allow_nan=False),flush=True)


if __name__=='__main__':
    main()
