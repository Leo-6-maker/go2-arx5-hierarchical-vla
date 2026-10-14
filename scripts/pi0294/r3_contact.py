"""Saved contact identity first; reconstructed fields are not actual old forces."""
import argparse,hashlib,json,os,resource,sys,time
from pathlib import Path
if not __package__:
    sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[2]/'src')]
    __package__='scripts.pi0294'
from scripts.pi0292.durable import OwnerLock,read,write
from .continuation import validate_entry,RUN
from .gripper_safety import saved_inputs,create_original


def saved_contacts(root,output,binding):
    import mujoco
    import numpy as np
    card,initial,old,nq,nv,commands=saved_inputs(binding)
    engine=create_original(card,initial,binding)
    def annotate(c):
        names=[mujoco.mj_id2name(engine.model,mujoco.mjtObj.mjOBJ_GEOM,c[k]) for k in ('geom0','geom1')]
        bodies=[mujoco.mj_id2name(engine.model,mujoco.mjtObj.mjOBJ_BODY,int(engine.model.geom_bodyid[c[k]])) for k in ('geom0','geom1')]
        return dict(c,geom_names=names,body_names=bodies)
    try:
        retired=read(Path(binding['R2_root'])/'evidence/G2-failure_prefix_hold-R2_VEL_C01/RESULT.json')
        observed=[annotate(c) for c in retired['final_contacts']]
        path=Path(binding['R2_root'])/'evidence/G1-B/actual_replay.npz'
        with np.load(path,allow_pickle=False) as f:row=f['rows'][-1]
        engine.data.qpos[:]=row[1:1+nq];engine.data.qvel[:]=row[1+nq:1+nq+nv]
        engine.data.ctrl[:]=row[1+nq+nv:1+nq+nv+engine.model.nu];engine.data.time=row[0]
        mujoco.mj_forward(engine.model,engine.data)
        _,_,_,illegal,contacts=engine.contacts()
        result=dict(run_uuid=RUN,status='SAVED_CONTACT_IDENTITY_AND_RECONSTRUCTED_B_COMPLETE',
            original_model_xml_sha256=initial['model_xml_sha256'],
            retired_candidate_final_contacts_observed=observed,
            G1_B=dict(time_s=float(row[0]),recorded_safety_flag=float(row[-1]),
                contacts_RECONSTRUCTED=[annotate(c) for c in contacts],illegal_RECONSTRUCTED=illegal,
                original_warmstart='UNOBSERVED',original_contact_force='UNOBSERVED',
                replay_npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            model_constructions=1,reconstruction_forward_calls=1,physics_integrations=0,RGB_pairs=0,
            actual_contact_force_causal_claim=False,selected='NONE',deployment='NONE')
        write(output/'CONTACT_EVENT_ATLAS_INITIAL.json',result)
        return result
    finally:engine.close()


def instrumented(root,output,binding,case):
    import mujoco
    import numpy as np
    from . import gripper_safety as replay
    from scripts.pi0291_arx5_data.engine import Engine as BaseEngine
    if case not in ('A','B'):raise ValueError('FIXED_A_OR_B_ONLY')
    holder={};events=[];first_pairs={};max_sum_error=0.;stop_detail=None
    original_create=replay.create_original;original_step=mujoco.mj_step;original_control=BaseEngine.step
    log=(output/'native_events.jsonl').open('x')
    def contacts(m,d):
        rows=[];jac=np.asarray(d.efc_J).reshape(d.nefc,m.nv)
        types=np.asarray(d.efc_type);ids=np.asarray(d.efc_id);forces=np.asarray(d.efc_force)
        for i in range(d.ncon):
            c=d.contact[i];wrench=np.zeros(6);mujoco.mj_contactForce(m,d,i,wrench)
            selection=(ids==i)&(types>=int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS))
            generalized=jac[selection].T@forces[selection]
            names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)]
            rows.append(dict(geom_names=names,geom_ids=[int(c.geom1),int(c.geom2)],
                point_world_m=c.pos.tolist(),frame_world=c.frame.tolist(),distance_m=float(c.dist),
                wrench_contact_frame=wrench.tolist(),finger_generalized_force_N=generalized[holder['engine'].finger_v].tolist()))
        return rows
    def created(*args):
        engine=original_create(*args);holder['engine']=engine
        if hasattr(engine,'finger_driver'):raise ValueError('D0_NO_RETIRED_HOOK_REQUIRED')
        original_snapshot=engine.snapshot
        def observed_snapshot():
            nonlocal stop_detail
            state=original_snapshot()
            record=dict(phase='EXISTING_SNAPSHOT_AFTER_ORIGINAL_FORWARD',time_s=state['time_s'],
                actual_TCP=state['tcp'].tolist(),contacts=contacts(engine.model,engine.data),
                joint_limit_penetration=state['joint_limit_penetration'],illegal_contact=state['illegal_contact'])
            if state['illegal_contact'] or state['joint_limit_penetration']:stop_detail=record
            log.write(json.dumps(record)+'\n')
            return state
        engine.snapshot=observed_snapshot
        return engine
    def control(engine,target,opening):
        holder['target']=np.asarray(target).copy();holder['opening']=float(opening)
        return original_control(engine,target,opening)
    def step(m,d):
        nonlocal max_sum_error
        engine=holder['engine']
        pre=dict(time_s=float(d.time),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),ctrl=d.ctrl.tolist(),
            arm_targets=engine.arm_targets.tolist(),finger_targets=engine.finger_targets.tolist(),
            target_TCP=holder['target'].tolist(),requested_opening=holder['opening'],actual_TCP_cached=engine.pose('tool').tolist())
        original_step(m,d)
        if mujoco.mj_isSparse(m):raise ValueError('DENSE_NV14_LOGGER_CONTRACT_ONLY')
        jac=np.asarray(d.efc_J).reshape(d.nefc,m.nv);force=np.asarray(d.efc_force);types=np.asarray(d.efc_type)
        reconstructed=jac.T@force
        error=float(np.max(np.abs(reconstructed-d.qfrc_constraint)));max_sum_error=max(max_sum_error,error)
        by_type={str(mujoco.mjtConstraint(int(k))):(jac[types==k].T@force[types==k])[engine.finger_v].tolist() for k in np.unique(types)}
        contact=contacts(m,d)
        for row in contact:
            key='|'.join(row['geom_names'])
            if key not in first_pairs:first_pairs[key]=dict(time_s=float(d.time),**row)
        record=dict(phase='NATIVE_STEP_RETURN_BEFORE_EXISTING_FORWARD',pre=pre,post_time_s=float(d.time),
            post_qpos=d.qpos.tolist(),post_qvel=d.qvel.tolist(),native_qacc=d.qacc.tolist(),
            bias=d.qfrc_bias[engine.finger_v].tolist(),passive=d.qfrc_passive[engine.finger_v].tolist(),
            actuator=d.qfrc_actuator[engine.finger_v].tolist(),constraint=d.qfrc_constraint[engine.finger_v].tolist(),
            constraint_type_breakdown_N=by_type,J_transpose_force_residual=error,contacts=contact,
            solver_fields_not_relabelled_as_post_integrated_kinematics=True)
        log.write(json.dumps(record)+'\n');events.append(record)
    replay.create_original=created;mujoco.mj_step=step;BaseEngine.step=control
    began=time.monotonic()
    def guard():
        if time.monotonic()-began>1200:raise TimeoutError('SHORT_LOGGER1200S')
    try:
        result=replay.causal_replay(root,binding,output,guard,case)
        expected=1081 if case=='A' else 1118
        if result['physics_steps']!=expected or case=='A' and not result['original_reproduced']:
            raise ValueError('LOGGER_CHANGED_FIXED_BASELINE_NO_FORCE_CAUSAL_CLAIM')
        write(output/'LOGGER_EQUIVALENCE.json',dict(status='PASS_A_EXACT_SAVED_ROWS' if case=='A' else 'PASS_B_ORIGINAL_STOP_TICK',
            case=case,steps=result['physics_steps'],max_original_row_difference=result['maximum_original_row_abs_difference'],
            added_dynamics_forward_calls=0,changed_step_or_solver=False,first_safety=result['safety']))
        write(output/'RESULT.json',result)
        detail=dict(status='ACTUAL_INSTRUMENTED_CONTACT_AND_CONSTRAINT',case=case,
            first_pair_events=list(first_pairs.values()),terminal_snapshot=stop_detail,
            maximum_actual_J_transpose_force_residual=max_sum_error,last_native_step=events[-1],
            source_identity_from_runtime_manifest=True,model_queries=0,RGB_pairs=0)
        write(output/'CONSTRAINT_FORCE_BREAKDOWN.json',detail)
        return dict(status=detail['status'],case=case,physics_steps=result['physics_steps'],
            first_pair_events=detail['first_pair_events'],terminal_snapshot=stop_detail,
            maximum_actual_J_transpose_force_residual=max_sum_error)
    finally:
        replay.create_original=original_create;mujoco.mj_step=original_step;BaseEngine.step=original_control;log.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',choices=['A0','A1'],default='A0');p.add_argument('--case',choices=['A','B']);a=p.parse_args()
    root=Path(a.root);validate_entry(root,profile='4080-simulation')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU_ONLY')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    owner=OwnerLock(root/'CONTACT_OWNER.lock');output=root/'evidence'/('contact-A0' if a.phase=='A0' else 'contact-A1-'+str(a.case))
    output.mkdir(exist_ok=False);started=time.process_time()
    result=None
    try:
        if a.phase=='A0':result=saved_contacts(root,output,read(root/'INPUT_BINDINGS.private.json'))
        else:result=instrumented(root,output,read(root/'INPUT_BINDINGS.private.json'),a.case)
    finally:
        cpu=time.process_time()-started;ledger=read(root/'RESOURCE_LEDGER.json')
        ledger['Core_CPU_s']+=cpu;ledger['stages'].append(dict(phase=output.name,CPU_s=cpu,status=result['status'] if result else 'ENGINEERING_FAILURE_PRESERVED'))
        if a.phase=='A1':
            ledger['main_or_diagnostic_starts']+=1;ledger['main_or_diagnostic_integrations']+=result['physics_steps'] if result else 1200
        write(root/'RESOURCE_LEDGER.json',ledger);owner.close()
    print(json.dumps(dict(result,CPU_s=cpu)))


if __name__=='__main__':main()
