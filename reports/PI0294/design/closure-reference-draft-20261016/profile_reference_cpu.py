"""NOT_AUTHORIZED: CPU-only requested-setpoint proposal, never native physics."""
import json,math
MID=.022
CLOSED=.002
SPEED=.04
CMD_DT=.05
PHYSICS_DT=.002
ACK=.05


def profile(at,start,duration):
    if at<=start:return MID
    if at>=start+duration:return CLOSED
    u=(at-start)/duration
    return MID-(MID-CLOSED)*(10*u**3-15*u**4+6*u**5)


def check():
    # Single analytic profile, duration comes from its exact maximum slope.
    d=math.ceil(((15/8)*(MID-CLOSED)/SPEED)/CMD_DT-1e-12)*CMD_DT
    old_start=6.45
    old_travel=(MID-CLOSED)/SPEED
    old_ACK=old_start+old_travel+ACK
    start=old_ACK-ACK-d
    n=round(d/CMD_DT)
    requested=[profile(start+i*CMD_DT,start,d) for i in range(n+1)]
    assert n==19 and abs(d-.95)<1e-12 and abs(start-6.)<1e-12
    assert requested[0]==MID and requested[-1]==CLOSED
    assert all(CLOSED<=q<=MID for q in requested)
    assert all(b<=a for a,b in zip(requested,requested[1:]))
    slopes=[(b-a)/CMD_DT for a,b in zip(requested,requested[1:])]
    assert max(abs(x) for x in slopes)<=SPEED+1e-12
    assert max(abs((CLOSED+(.042-CLOSED)*((q-CLOSED)/(.042-CLOSED)))-q) for q in requested)<1e-15
    def replay(values):
        target=MID;rows=[]
        for k in range(round((d+ACK)/PHYSICS_DT)):
            index=min(int((k*PHYSICS_DT+1e-12)/CMD_DT),len(values)-1)
            wanted=values[index]
            target+=max(-SPEED*PHYSICS_DT,min(SPEED*PHYSICS_DT,wanted-target))
            rows.append(target)
        return rows
    target=replay(requested)
    assert all(CLOSED-1e-12<=x<=MID+1e-12 for x in target)
    assert abs(target[-1]-CLOSED)<1e-12
    return dict(status='PASS_CPU_REQUESTED_CLOSURE_PROFILE_ONLY',launch_allowed=False,authorizes_execution=False,
        scientific_factor='ONLY_T2_ENTRY_20HZ_GRIPPER_SETPOINT_PROFILE',original_midpoint_m=MID,original_closed_m=CLOSED,
        duration_s=d,start_s=start,target_arrival_s=start+d,original_ACK_deadline_s=old_ACK,
        original_nominal_target_arrival_s=old_start+old_travel,
        continuous_design_peak_speed_m_s=(15/8)*(MID-CLOSED)/d,
        continuous_design_peak_acceleration_m_s2=(10/math.sqrt(3))*(MID-CLOSED)/d**2,
        sampled_20Hz_peak_difference_speed_m_s=max(abs(x) for x in slopes),
        current_slew_adapter_replay_points=len(target),
        continuous_design_C2_is_not_a_claim_about_sampled_or_actual_finger_acceleration=True,
        original_slew_cap_m_s=SPEED,gain_changes=0,endpoint_changes=0,WBC_changes=0,physics_changes=0,
        dynamic_safety='UNPROVEN',task_success='UNPROVEN',new_physics=0,new_IK=0,new_RRT=0,new_neural_queries=0)


if __name__=='__main__':print(json.dumps(check(),indent=2))
