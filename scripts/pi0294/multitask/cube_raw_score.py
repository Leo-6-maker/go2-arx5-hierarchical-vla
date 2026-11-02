"""Map recorded cube states into the unchanged T1/T2 physical scorer.

No simulation, Teacher or policy metadata participates in the score. The
source capture audit separately checks the native clock, safety, commands
and original RGB bytes before a result can count as complete success.
"""
import numpy as np


def score(rows,card,*,complete):
    from scripts.pi0294.score import score as original_score
    from go2arx5_vla.target_sim.scorer import BOOL_FIELDS
    obs=[r['value']for r in rows if r['kind']=='observation']
    ticks=[r['value']for r in rows if r['kind']=='native_tick']
    if not obs:return dict(persistent_success=False,complete_32s=False,reason='NO_ACTUAL_TASK_OBSERVATION')
    states=[r['actual_state']for r in obs]
    data={key:np.asarray([s[key]for s in states],dtype=bool)for key in BOOL_FIELDS}
    data.update(time_s=np.asarray([r['time_s']-5. for r in obs]),
        object_pose=np.asarray([s['object_pose']for s in states]),
        tcp_world=np.asarray([s['tcp']for s in states]),
        object_linear_velocity=np.asarray([s['object_linear_velocity']for s in states]),
        object_angular_velocity=np.asarray([s['object_angular_velocity']for s in states]),
        robot_object_contact=np.asarray([s['robot_object_contact']for s in states],dtype=bool),
        contact_time_s=np.asarray([r['actual_state']['time_s']-5. for r in ticks]),
        contact_audit=np.asarray([[int(s['robot_object_contact']),int(s['native_grasp']),
            int(s['left_object_contact']and s['right_object_contact']),s['object_pose'][2][3],
            int(not s['safety_valid'])]for s in(r['actual_state']for r in ticks)]).reshape(-1,5))
    return original_score(data,card,complete=complete)
