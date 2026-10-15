"""Bounded CPU fixtures and single-step proposals on saved actual inputs."""
import json
from pathlib import Path
import numpy as np
from .component_progress_draft import ComponentProgressDraft,VERSION,LAUNCH_ALLOWED
from .persistent_approach_draft import PersistentApproachDraft
from .task_capture_runtime import sha
from go2arx5_vla.trajectory.so3 import geodesic_angle,quat_to_matrix


def restored(snapshot):
    original=PersistentApproachDraft.restore(snapshot)
    candidate=ComponentProgressDraft(task=original.task,actual_tcp=original.initial,stage_target=original.target,
        actual_BASE=original.initial_base,station_world=original.station,pad_radius_m=original.pad_radius_m)
    candidate.cursor=original.cursor.copy();candidate.route=[p.copy() for p in original.route];candidate.segment=original.segment
    return candidate


def check_changed_components(before,after,actual,core):
    dp=float(np.linalg.norm(after[:3,3]-before[:3,3]));dr=float(geodesic_angle(after[:3,:3],before[:3,:3]))
    if dp>1e-10 and np.linalg.norm(after[:3,3]-actual[:3,3])>.024+1e-10:raise ValueError('CHANGED_POSITION_ENVELOPE')
    if dr>1e-7 and geodesic_angle(after[:3,:3],actual[:3,:3])>.12+1e-10:raise ValueError('CHANGED_ROTATION_ENVELOPE')
    return dp,dr


def run(root):
    from scripts.pi0292.durable import write
    series=Path(root)/'teacher-qualification-v1';fixtures=[]
    initial=np.eye(4);initial[2,3]=.72;goal=initial.copy();goal[:3,:3]=quat_to_matrix([0,0,np.sin(.5),np.cos(.5)])
    for task in ('T1','T2'):
        core=ComponentProgressDraft(task=task,actual_tcp=initial,stage_target=goal,actual_BASE=np.eye(4),
            station_world=np.eye(4),pad_radius_m=.05)
        actual=initial.copy();actual[0,3]+=.04
        cursor,segment=core._advance(initial.copy(),0,actual,.4)
        assert np.array_equal(cursor,initial) and segment==1
        after,segment=core._advance(cursor,segment,actual,.4)
        dp,dr=check_changed_components(cursor,after,actual,core)
        assert dp==0 and dr>.1 and segment==1
        fixtures.append(dict(task=task,case='POSITION_LAG_DOES_NOT_VETO_ROTATION',rotation_rad=dr))
        core.observe(time_s=0.,actual_tcp=actual,actual_BASE=np.eye(4),native_safe=False)
        try:core.plan(time_s=0.,actual_tcp=actual,actual_BASE=np.eye(4))
        except PermissionError:pass
        else:raise ValueError('ORIGINAL_NATIVE_SAFETY_MUST_STOP_DRAFT')
    output=[]
    for folder in sorted((series/'episodes').iterdir()):
        samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
        intents=[r for r in (json.loads(s) for s in (folder/'committed_intents.jsonl').read_text().splitlines())
            if r['phase']=='SUBMITTED_REQUIRES_INSTALL_ACK']
        steps=[]
        for old,current in zip(intents,intents[1:]):
            core=restored(old['persistent']);actual=np.asarray(samples[current['index']]['tcp'])
            before=core.cursor.copy();state_before=actual.copy();candidate,segment=core._advance(before.copy(),core.segment,actual,.4)
            dp,dr=check_changed_components(before,candidate,actual,core)
            assert np.array_equal(actual,state_before)
            steps.append(dict(query=current['sequence'],proposed_position_increment_m=dp,proposed_rotation_increment_rad=dr,
                previous_segment=core.segment,proposed_segment=segment,
                actual_saved_segment=current['persistent']['segment'],actual_state_modified=False))
        output.append(dict(id=folder.name,finite_single_step_proposals=len(steps),
            rotation_progress_proposals=sum(s['proposed_rotation_increment_rad']>1e-7 for s in steps),
            waypoint_completion_proposals=sum(s['proposed_segment']>s['previous_segment'] for s in steps),steps=steps,
            raw_intents_sha256=sha(folder/'committed_intents.jsonl')))
    receipt=dict(status='PASS_CPU_ONLY_SINGLE_INTERVENTION_DRAFT_NOT_ADOPTED',version=VERSION,launch_allowed=LAUNCH_ALLOWED,
        scientific_change='WHOLE_POSE_VETO_TO_CHANGED_COMPONENT_ADMISSION',fixtures=fixtures,saved_input_cases=output,
        sequential_counterfactual_rollouts=0,actual_state_edits=0,new_physics=0,new_render=0,new_GR00T_queries=0,
        changed_components_envelope_validated=True,whole_pose_absolute_envelope_claimed=False,
        does_not_prove_actual_motion_contact_or_task_success=True)
    path=series/'COMPONENT_PROGRESS_V2_DRAFT_CPU.json'
    if path.exists():raise FileExistsError('FIRST_DRAFT_CPU_RECEIPT_PRESERVED')
    write(path,receipt)
    return receipt
