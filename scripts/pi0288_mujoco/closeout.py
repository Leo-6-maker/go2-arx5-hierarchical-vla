"""Recompute the bounded startup outcome from saved original evidence."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from go2arx5_vla.target_mujoco.reference import interface_contract,CONDITIONS


def main():
    out=ROOT/'reports/PI0288/execution'
    episode='55f23c661cdc4c8aa50a8731928dc101'
    raw=out/'evidence'/episode
    model=json.loads((out/'evidence/DEFAULT_MODEL_CPU_AUDIT.json').read_text())
    rows=[json.loads(x) for x in (raw/'physics.jsonl').read_text().splitlines()]
    loaded=next(x for x in rows if x['kind']=='model')
    samples=[x for x in rows if x['kind']=='sample']
    stops=[x for x in rows if x['kind']=='stop']
    assert len(samples)==1 and samples[0]['step']==1 and samples[0]['joint_stop']
    assert stops[0]['reason']=='REGISTERED_JOINT_SAFETY'
    assert model['invalid_default_joint_count']==4 and model['physics_integrations']==0
    assert model['private_snapshot_original_integration_state_unchanged']
    original=json.loads((raw/'RESULT.json').read_text())
    joints=[]
    for joint in loaded['joints']:
        if not joint['limited']:continue
        q=samples[0]['qpos'][joint['qposadr']];lo,hi=joint['range'];v=max(0,lo-q,q-hi)
        if v>0:joints.append(dict(joint=joint['name'],q=q,lower=lo,upper=hi,positive_violation=v,over_hinge_epsilon=v>1e-4,over_hinge_immediate=v>1e-3))
    def write(name,data):(out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    write('INTERFACE_CONTRACT.json',dict(status='SOURCE_BOUND_NOT_LIVE_QUALIFIED',contract=interface_contract(),TCP=dict(MuJoCo_site='x5_ee',URDF_frame='gripper_center',link6_xyz=[.1424,0,.0001057],binding='actual delivered MJCF/URDF CPU parse'),policy_target_consumer='/mobile_manipulator_mpc_policy.plan_target_trajectories'))
    write('CLOCK_CONTRACT.json',dict(status='BLOCKED_STARTUP_SAFETY',loaded_physics_step_s=loaded['timestep'],native_clock_source='wall/ROS MPC; MuJoCo integration clock separate',observed_sim_monotonic_wall_pairs=len(samples),MPC_time_mapping='UNOBSERVED',reset_expiry_negatives='NOT_RUN',fit_performed=False))
    write('SAFETY_LIMITS.json',dict(hinge_epsilon_rad=1e-4,slide_epsilon_m=1e-6,hinge_immediate_rad=1e-3,slide_immediate_m=1e-4,sustained_epsilon_stop_s=.020,source='PI0288 fixed release section3',loaded_joints=loaded['joints'],first_step_positive_violations=joints,unmodified_raw_limits=True,contact_whitelist=['FL-floor','FR-floor','RL-floor','RR-floor']))
    write('ADJUDICATION.json',dict(status='ENV_READY_INTERFACE_BLOCKED',scope='native build/environment available; task executor, clock, reference and data NOT qualified',root_cause='INVALID_NATIVE_DEFAULT_INITIAL_STATE_BEFORE_CONTROLLER_READY',episode=episode,original_result_status=original['status'],original_error=original['error'],original_result_sha256=hashlib.sha256((raw/'RESULT.json').read_bytes()).hexdigest(),actual_cause='observer invokes registered native SIGINT after first integration safety violation; ROS feedback unavailable is consequent',default_qpos0_calf_violation_rad=.83776,first_step_violations=joints,rule='PI0288 section3 invalid initial condition rejected; no temporary pose replacement',not_inferred=['MPC motion tracking failure','RL weakness','grasp capability','safe home-keyframe replacement'],home_keyframe='present but not applied by delivered native startup; not tested as substitute',core_unchanged=True))
    write('GATE_TABLE.json',dict(G0='BLOCKED_INVALID_NATIVE_INITIAL_STATE',G1=[dict(condition=x,status='NOT_RUN_G0_DEPENDENCY') for x in CONDITIONS],G2='CPU_IMPLEMENTED_NOT_RUNTIME_QUALIFIED_AND_GRIPPER_SPEC_MISSING',G3='NOT_RUN',G4='NOT_RUN_NO_REAL_DATA',native_reproduction_parent='PASS_UNCHANGED_BOUNDED_HOLD_ONLY',target_executor_freeze=False,teacher_freeze=False,selected='NONE',deployment='NONE'))
    ledger=json.loads((out/'evidence/BUDGET.json').read_text())
    write('ACCOUNTING.json',dict(env_episodes_observed=len(ledger['episodes']),setup_slots_used=1,native_reference_slots_used=0,technical_replacement_slots_used=0,physics_integrations_observed=1,physics_sim_s_observed=.002,leg_RL_forwards=dict(observed='UNOBSERVED',reserved_upper_bound=6000),MPC_solve_attempts=dict(observed='UNOBSERVED',reserved_upper_bound=12000),dispatch=dict(received_count=0,direct_actual='UNOBSERVED',reserved_upper_bound=36000),GPU_service_interval_upper_bound_s=original['service_s'],prelaunch_reservation=ledger['episodes'][0]['reservation'],CPU_core_time_total='UNOBSERVED; test wall16.57s and compilation timings retained; no all-cap PASS claim',new_output_bytes='see source readback and server inventory receipt',teacher_plans=0,RGB_pairs=0,target_teacher_episodes=0,GR00T_forwards=0,optimizer_updates=0,hardware=0,budget_reset=False))
    write('CPU_RECEIPT.json',dict(local_suite=dict(command='py -3.12 -m pytest tests/target_mujoco -q',passed=13,skipped=1,wall_s=16.57,skip_reason='private source not available locally'),actual_private_callback_test=dict(status='PASS',calls=dict(native_accept=2,native_send=1,publisher=1,arm=1),trace_sha256='05913d0d1c314c6cfffc261a617dc13a560b796b29a4dd701ce7a925c71bc788'),loaded_model_zero_step_audit=dict(status='PASS_FOR_MEASUREMENT_NOT_SAFETY',snapshot_original_state_unchanged=True,invalid_default_joints=4),official_GR00T_processor_batch='NOT_RUN_NO_ADMITTED_REAL_DATA'))
    print(json.dumps(dict(terminal='ENV_READY_INTERFACE_BLOCKED',physics_integrations=1,reference_conditions_run=0,episodes_collected=0)))


if __name__=='__main__':main()
