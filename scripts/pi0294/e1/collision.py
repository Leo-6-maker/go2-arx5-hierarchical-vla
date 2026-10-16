"""Exact TRAIN replay reconstruction and read-only original-prefix collision logs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0294.scene import make_engine
from scripts.pi0294.teacher import make_teacher
from scripts.pi0294.episode import _robot_contact
from scripts.pi0294.s1_preflight import clone,data_state,fingerprint,observed_state
from scripts.pi0294.native_dataset import sha
from scripts.pi0291_arx5_data.config import PARAMETERS as P
from .scope import validate,Ledger
from go2arx5_vla.trajectory.so3 import geodesic_angle


def component_errors(model_poses,model_gripper,expert):
    xyz=np.linalg.norm(model_poses[:,:3,3]-expert['poses'][:,:3,3],axis=1)
    rotation=geodesic_angle(model_poses[:,:3,:3],expert['poses'][:,:3,:3])
    grip=np.abs(model_gripper-expert['opening'])
    return {view:{name:dict(mean=float(np.mean(values[:n])),maximum=float(np.max(values[:n])))
        for name,values in [('XYZ_m',xyz),('SO3_rad',rotation),('gripper_absolute',grip)]}
        for view,n in [('prefix8',8),('H40',40)]}


def collision_pairs(engine):
    import mujoco
    rows=[]
    for c in engine.data.contact:
        g=[int(c.geom1),int(c.geom2)];b=[int(engine.model.geom_bodyid[x]) for x in g]
        bodies=[mujoco.mj_id2name(engine.model,mujoco.mjtObj.mjOBJ_BODY,x) for x in b]
        arm=any(name.startswith('x5_') for name in bodies)
        arm_surface=arm and any(engine.geoms[x] in g for x in ('table','floor'))
        penetration=float(c.dist)<-P['penetration_stop_m']
        if arm_surface or penetration:
            rows.append(dict(geom_ids=g,geom_names=[mujoco.mj_id2name(engine.model,mujoco.mjtObj.mjOBJ_GEOM,x) for x in g],
                body_ids=b,body_names=bodies,position_world=np.asarray(c.pos).tolist(),normal_world=np.asarray(c.frame[:3]).tolist(),
                signed_distance_m=float(c.dist),penetration_m=max(0.,-float(c.dist)),
                original_rule='ARM_TABLE_OR_FLOOR' if arm_surface else 'PENETRATION_STOP',
                penetration_stop_m=P['penetration_stop_m']))
    return rows


def inspect_prefix(engine,poses,opening,ledger,identifier):
    before=fingerprint(engine);shadow=clone(engine);error=None;rows=[]
    with ledger.steps('shadow_integrations',200,identifier) as meter:
        try:
            for i in range(200):
                ledger.guard()
                if i and i%25==0:shadow.snapshot()
                previous=shadow.steps
                try:shadow.step(poses[i//25],float(opening[i//25]))
                except Exception as exc:error=type(exc).__name__+': '+str(exc)
                finally:meter['actual']+=shadow.steps-previous
                shadow.snapshot();rows.append(observed_state(shadow))
                if error:break
            state=shadow.snapshot()
            result=dict(decision='REJECT' if error or shadow.safety else 'PASS',error=error,
                predicted_native_safety=list(shadow.safety),shadow_integrations=meter['actual'],
                first_violation_offset_ticks=meter['actual'] if shadow.safety else None,
                first_violation_offset_s=meter['actual']*.002 if shadow.safety else None,
                illegal_pairs=collision_pairs(shadow),actual_TCP_at_end=state['tcp'].tolist(),
                actual_object_pose_at_end=state['object_pose'].tolist(),actual_joint_q_at_end=state['arm_q'].tolist(),
                requested_TCP_at_end=poses[(meter['actual']-1)//25].tolist(),
                requested_gripper_at_end=float(opening[(meter['actual']-1)//25]),
                live_state_unchanged=fingerprint(engine)==before,actions_modified=False,
                diagnostic_shadow_not_live=True)
            if not result['live_state_unchanged']:raise RuntimeError('DIAGNOSTIC_SHADOW_MUTATED_SOURCE')
            return result,np.asarray(rows)
        finally:shadow.close()


def save_state(folder,engine,teacher,case,sequence,observation):
    folder.mkdir(mode=0o700,parents=True,exist_ok=False)
    python={k:copy.deepcopy(v) for k,v in engine.__dict__.items() if k not in ('model','data','ik_data','renderer')}
    np.savez_compressed(folder/'integration.npz',live=data_state(engine.model,engine.data),ik=data_state(engine.model,engine.ik_data),observed=observed_state(engine))
    private=dict(engine_python=python,teacher=copy.deepcopy(teacher),python_rng=random.getstate(),numpy_rng=np.random.get_state())
    with (folder/'history.private.pkl').open('xb') as f:pickle.dump(private,f)
    for p in folder.iterdir():p.chmod(0o600)
    write(folder/'observation.json',observation)
    result=dict(id=case['attempt_id']+'-S'+str(sequence).zfill(3),original_attempt_id=case['attempt_id'],
        original_episode_uuid=case['episode_uuid'],task=case['card']['task'],split='TRAIN',sequence=sequence,tick=sequence*200,
        time_s=float(engine.data.time),card=case['card'],model_xml_sha256=hashlib.sha256(engine.xml.encode()).hexdigest(),
        observation_sha256=observation['sha256'],snapshot_relative=str(folder.name),
        full_integration_state_sha256=sha(folder/'integration.npz'),history_sha256=sha(folder/'history.private.pkl'),
        restoration='DETERMINISTIC_ORIGINAL_INITIAL_STATE_AND_ALL_ACTUAL_SUBMITTED_COMMANDS_REPLAY_NOT_QPOS_INJECTION',
        process_RNG='NEW_E1_CAPTURED_RNG_E0_PROCESS_RNG_NOT_RECORDED_OR_USED_BY_FROZEN_D0_ENGINE',
        physically_safety_valid=not bool(engine.safety))
    write(folder/'STATE.json',result)
    return result


def reconstruct_case(root,case,ledger,*,save=True,diagnose=True):
    inputs=read(root/'E1_INPUTS.private.json');old=Path(inputs['E0_root']);folder=old/'episodes'/case['attempt_id']
    card=read(folder/'CARD.json');assert card['split']=='TRAIN'
    binding=read(root/'INPUT_BINDINGS.private.json');engine=make_engine(card,binding['asset_root'])
    first=engine.snapshot();teacher=make_teacher(card,first['object_pose'],engine)
    initial=read(folder/'INITIAL_STATE.json')
    if hashlib.sha256(engine.xml.encode()).hexdigest()!=initial['model_xml_sha256']:
        raise ValueError('ORIGINAL_REPLAY_XML_CHANGED')
    if not np.array_equal(first['qpos'],np.asarray(initial['qpos'])) or not np.array_equal(first['qvel'],np.asarray(initial['qvel'])):
        raise ValueError('ORIGINAL_INITIAL_STATE_NOT_EXACT')
    observations=sorted((folder/'relay/observations').glob('*.json'));last=len(observations)-1
    selected=sorted({0,last//2,last});snapshots={};states=[];maximum=0.;model_checks=[];comparison=None
    with np.load(folder/'highrate.npz',allow_pickle=False) as z:recorded=z['rows'].copy()
    try:
        for sequence in range(last+1):
            state=engine.snapshot();state['robot_object_contact']=_robot_contact(engine,state)
            if sequence==0:
                from .initial_history import qualify_recorded_initial
                qualify_recorded_initial(engine,folder,ledger.guard)
            teacher.observe(state)
            packet=read(folder/'relay/observations'/f'{sequence:03d}.json');body=unseal(packet)
            assert body['sequence']==sequence and body['tick']==engine.steps
            if sequence in selected:
                snapshots[sequence]=(clone(engine),copy.deepcopy(teacher))
                if save:states.append(save_state(root/'evidence/TRAIN_STATES'/(case['attempt_id']+'-S'+str(sequence).zfill(3)),engine,teacher,case,sequence,packet))
            command_path=folder/'relay/commands'/f'{sequence:03d}.json'
            if not command_path.exists():break
            bundle=read(command_path);action=np.asarray(unseal(bundle['action'])['payload']['trajectory_eef10'])
            poses,opening=matrix10(action),action[:,9]
            ack=folder/'relay/acks'/f'{sequence:03d}.json'
            if not ack.exists():
                if diagnose:
                    expert=teacher.plan(state);candidates=[('MODEL',poses,opening),('SAME_STATE_EXPERT',expert['poses'],expert['opening'])]
                    comparison=dict(component_errors=component_errors(poses,opening,expert),expert_stage=expert['stage'],
                        source_state_time_s=float(engine.data.time),source_actual_TCP=state['tcp'].tolist(),
                        source_actual_object_pose=state['object_pose'].tolist(),source_actual_gripper_fraction=float(state['gripper_fraction']),
                        model_H40_poses=poses.tolist(),model_H40_gripper=opening.tolist(),
                        expert_H40_poses=expert['poses'].tolist(),expert_H40_gripper=expert['opening'].tolist(),
                        reference_frame='PI0291_MUJOCO_WORLD',TCP='ARX5_NATIVE_SITE_x5_ee',
                        hybrid_actions_never_installed=True,unique_cause_not_claimed=True)
                    for component in ['XYZ','SO3','GRIPPER']:
                        p,g=poses.copy(),opening.copy()
                        if component=='XYZ':p[:,:3,3]=expert['poses'][:,:3,3]
                        elif component=='SO3':p[:,:3,:3]=expert['poses'][:,:3,:3]
                        else:g=expert['opening'].copy()
                        candidates.append(('EXPERT_'+component+'_ONLY',p,g))
                    for label,p,g in candidates:
                        check,_=inspect_prefix(engine,p[:8],g[:8],ledger,'A-'+case['attempt_id']+'-'+label)
                        model_checks.append(dict(candidate=label,**check))
                    original=read(folder/'S1_PREFLIGHTS.json')[-1]
                    if model_checks[0]['first_violation_offset_ticks']!=original['first_predicted_violation_offset_tick']:
                        raise ValueError('ORIGINAL_REJECTION_OFFSET_NOT_REPRODUCED')
                break
            with ledger.steps('shadow_integrations',200,'REPLAY-'+case['attempt_id']+'-'+str(sequence)) as meter:
                for i in range(200):
                    ledger.guard()
                    if i and i%25==0:
                        current=engine.snapshot();current['robot_object_contact']=_robot_contact(engine,current);teacher.observe(current)
                    previous=engine.steps
                    try:engine.step(poses[i//25],float(opening[i//25]))
                    finally:meter['actual']+=engine.steps-previous
                    current=engine.snapshot()
                    actual=np.r_[engine.data.time,engine.data.qpos,engine.data.qvel,engine.data.ctrl,engine.data.xfrc_applied[engine.ids['tool']],float(not current['safety_valid'])]
                    error=float(np.max(np.abs(actual-recorded[engine.steps-1])));maximum=max(maximum,error)
                    if error>1e-10 or engine.safety:
                        raise ValueError('RECONSTRUCTED_ACTUAL_HISTORY_MISMATCH')
        if engine.steps!=read(folder/'RESULT.json')['physics_steps']:raise ValueError('REPLAY_ACTUAL_PROGRESS_MISMATCH')
        return dict(attempt_id=case['attempt_id'],task=card['task'],original_main_steps=engine.steps,replay_integrations=engine.steps,
            maximum_actual_highrate_difference=maximum,states=states,collision_checks=model_checks,same_state_comparison=comparison,
            original_initial_state_exact=True,control_targets_and_IK_history_reconstructed=True,
            recorded_tick_fields='TIME_QPOS_QVEL_CTRL_ZERO_WRENCH_NATIVE_SAFETY',old_process_RNG_unobserved=True),snapshots
    except BaseException:
        for shadow,_ in snapshots.values():shadow.close()
        raise
    finally:engine.close()


def run(root):
    root=Path(root);validate(root,'4080-simulation',physical=True);ledger=Ledger(root,'core')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:3]));start=time.process_time()
    lock=OwnerLock(root/'E1_SOURCE_OWNER.lock');results=[]
    old=Path(read(root/'E1_INPUTS.private.json')['E0_root']);registry=read(old/'evidence/model-M2048/REGISTRY.json')
    cases=sorted([c for c in registry['cases'].values() if c['card']['split']=='TRAIN'],key=lambda c:c['panel_index'])
    try:
        assert len(cases)==4
        for case in cases:
            result,states=reconstruct_case(root,case,ledger)
            results.append(result)
            for engine,_ in states.values():engine.close()
            write(root/'evidence/TRAIN_COLLISION_ATTRIBUTION.json',dict(status='ACTUAL_TRAIN_REPLAY_AND_COLLISION_DIAGNOSTIC',cases=results,completed=len(results),planned=4))
        pool=[s for r in results for s in r['states']];assert len(pool)<=12
        write(root/'plans/TRAIN_STATE_POOL.json',dict(schema='E1_TRAIN_STATE_POOL_V1',selection='INITIAL_MIDDLE_ACK_LAST_SAFE_PER_ORIGINAL_TRAIN_EPISODE_DEDUP',states=pool,count=len(pool),DEV_access=False))
        return dict(status='TRAIN_COLLISION_AND_STATE_POOL_ACTUAL_COMPLETE',cases=4,states=len(pool),CPU_s=time.process_time()-start)
    except Exception as exc:
        if any(key in str(exc) for key in ['STATE_NOT_EXACT','XML_CHANGED','HISTORY_MISMATCH','OFFSET_NOT_REPRODUCED','PROGRESS_MISMATCH','MUTATED_SOURCE']):
            ledger.stop_physics('REPLAY_STATE_OR_SOURCE_INTEGRITY',type(exc).__name__+': '+str(exc))
        write(root/'evidence/A_ENGINEERING_FAILURE.json',dict(error=type(exc).__name__+': '+str(exc),completed=len(results)))
        raise
    finally:ledger.cpu(time.process_time()-start);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(run(a.root)),flush=True)
