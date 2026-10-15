"""Exact saved-query control-flow replay; no modified candidate or rollout.

Separate an observed position veto, an angular veto, and legitimate orientation
work. This does not infer how a changed Teacher would perform in physics.
"""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .persistent_approach_draft import PersistentApproachDraft
from .task_capture_runtime import sha,read
from go2arx5_vla.trajectory.so3 import geodesic_angle,matrix_to_quat,quat_to_matrix,quat_slerp


def classify(core,actual,dt):
    cursor=core.cursor;segment=core.segment
    if segment==len(core.route):return dict(reason='ROUTE_COMPLETE')
    goal=core.route[segment];delta=goal[:3,3]-cursor[:3,3];distance=np.linalg.norm(delta)
    candidate=cursor.copy();candidate[:3,3]+=delta*min(1.,core.config.linear_speed_m_s*dt/max(distance,1e-15))
    gap=float(np.linalg.norm(candidate[:3,3]-actual[:3,3]));bound=core.config.linear_speed_m_s*core.config.query_s
    detail=dict(position_candidate_actual_gap_m=gap,position_bound_m=bound,
        candidate_translation_increment_m=float(np.linalg.norm(candidate[:3,3]-cursor[:3,3])),
        segment_goal_position_residual_m=float(distance),
        segment_goal_angle_residual_rad=float(geodesic_angle(cursor[:3,:3],goal[:3,:3])))
    if gap>bound+1e-10:return dict(detail,reason='POSITION_VETO_BEFORE_ROTATION_OR_WAYPOINT_COMPLETION')
    angle=float(geodesic_angle(cursor[:3,:3],goal[:3,:3]))
    q=quat_slerp(matrix_to_quat(cursor[:3,:3]),matrix_to_quat(goal[:3,:3]),
        min(1.,core.config.angular_speed_rad_s*dt/max(angle,1e-15)))
    candidate[:3,:3]=quat_to_matrix(q)
    angle_gap=float(geodesic_angle(candidate[:3,:3],actual[:3,:3]))
    detail['candidate_actual_angle_rad']=angle_gap;detail['angular_bound_rad']=core.config.angular_speed_rad_s*core.config.query_s
    if angle_gap>detail['angular_bound_rad']+1e-10:return dict(detail,reason='ANGULAR_VETO')
    if detail['candidate_translation_increment_m']>1e-10:return dict(detail,reason='ADVANCE_TRANSLATION')
    if geodesic_angle(candidate[:3,:3],cursor[:3,:3])>1e-7:return dict(detail,reason='ADVANCE_ROTATION_ONLY')
    return dict(detail,reason='ZERO_MOTION_WAYPOINT_COMPLETION')


def run(root):
    from scripts.pi0292.durable import write
    root=Path(root);series=root/'teacher-qualification-v1'
    runtime_source='4a12eef87c10d437b4c204e9b6abbd83abefa1f0'
    original=root/'capsules'/runtime_source/'code/scripts/pi0294/persistent_approach_draft.py'
    if sha(original)!=sha(Path(__file__).with_name('persistent_approach_draft.py')):
        raise ValueError('ACTUAL_RUNTIME_COMPONENT_SOURCE_IDENTITY_REQUIRED')
    output=[]
    for folder in sorted((series/'episodes').iterdir()):
        if read(folder/'RESULT.json')['source_sha']!=runtime_source:raise ValueError('UNKNOWN_EXECUTION_SOURCE')
        samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
        intents=[r for r in (json.loads(s) for s in (folder/'committed_intents.jsonl').read_text().splitlines())
            if r['phase']=='SUBMITTED_REQUIRES_INSTALL_ACK']
        transitions=[]
        for previous,current in zip(intents,intents[1:]):
            core=PersistentApproachDraft.restore(previous['persistent'])
            actual=np.asarray(samples[current['index']]['tcp']);dt=current['persistent']['last_time']-previous['persistent']['last_time']
            if abs(dt-.4)>1e-8:raise ValueError('SAVED_QUERY_PERIOD')
            replay,segment=core._advance(core.cursor.copy(),core.segment,actual,dt)
            expected=np.asarray(current['persistent']['cursor'])
            if not np.array_equal(replay,expected) or segment!=current['persistent']['segment']:
                raise ValueError('CPU_REPLAY_DIFFERS_FROM_ACTUAL_SAVED_CURSOR')
            detail=classify(core,actual,dt)
            transitions.append(dict(query_index=current['sequence'],task_time_s=current['index']*.05,
                previous_segment=core.segment,saved_segment=segment,exact_cursor_matrix_replay=True,**detail))
        counts=dict(Counter(t['reason'] for t in transitions))
        zero_veto=sum(t['reason']=='POSITION_VETO_BEFORE_ROTATION_OR_WAYPOINT_COMPLETION' and
            t['candidate_translation_increment_m']<=1e-10 for t in transitions)
        output.append(dict(id=folder.name,source_intents_sha256=sha(folder/'committed_intents.jsonl'),
            source_samples_sha256=sha(folder/'samples.jsonl'),transitions=len(transitions),reason_counts=counts,
            position_veto_despite_zero_requested_translation=zero_veto,details=transitions))
    receipt=dict(status='PASS_EXACT_SAVED_TRANSITION_REPLAY_NOT_PHYSICAL_SUCCESS',runtime_source_sha=runtime_source,
        component_sha256=sha(original),total_transitions=sum(r['transitions'] for r in output),cases=output,
        counterfactual_state_changes=0,new_candidate_queries=0,new_physics=0,new_render=0,new_GR00T_queries=0,
        WBC_changes=0,old_results_modified=False,scope='CONTROL_FLOW_CAUSE_WITHIN_TEACHER_NOT_UNIQUE_PHYSICAL_CAUSE',
        next_physics='NOT_AUTHORIZED_TERMINAL_SHORT_SERIES_PRESERVED')
    path=series/'EXACT_CURSOR_TRANSITION_CPU.json'
    if path.exists():raise FileExistsError('FIRST_EXACT_REPLAY_RECEIPT_PRESERVED')
    write(path,receipt)
    return receipt
