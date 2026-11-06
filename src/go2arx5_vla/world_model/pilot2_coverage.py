"""Randomized, preregistered coverage using the already-qualified four actions."""
from pathlib import Path
import json
import numpy as np

PLAN='6d82fa23c892f167001abe743dd5eadfca6fd777'
MANIFEST='WM02_PILOT2_COVERAGE_REPLAY_MANIFEST_V1'


def coverage_contract(spec,episode):
    if (spec.get('pilot2_plan_sha')!=PLAN or episode.get('schema')!='WM02_PILOT2_COVERAGE_EPISODE_V1'
            or type(episode.get('forced_action'))is not int or episode['forced_action']not in range(4)
            or type(episode.get('repeat_index'))is not int or episode['repeat_index']not in (0,1)
            or episode.get('selection_source')!='PREREGISTERED_RANDOMIZED_ACTION_MANIFEST'
            or episode.get('split')not in ('TRAIN','DEV')
            or episode.get('purpose')not in ('ACTION_COVERAGE','SCENE_QUALIFICATION')
            or episode.get('purpose')=='SCENE_QUALIFICATION'and episode['forced_action']!=0):
        raise PermissionError('FROZEN_PILOT2_FOUR_ACTION_PARENT_SPLIT_AND_RANDOMIZED_ORDER_REQUIRED')
    return episode


def choose_predeclared(adapter,episode):
    from .pilot1_episode import OFFSETS
    from .next2_episode import state_vector
    from .target_reference import require_install_clock
    from .storage import digest
    from scripts.pi0292.durable import write
    coverage_contract(adapter.spec,episode)
    adapter.wait_source();packet=adapter.observe()
    if packet['step']!=3703:raise ValueError('ORIGINAL_REAL_DECISION_BOUNDARY_REQUIRED')
    source=float(adapter.latest['ee'].time);require_install_clock(source,packet['time_s'])
    initial=np.r_[adapter.active_reference['action_EEF10'][-1],adapter.base_pose_hold]
    tcp=adapter.measured_task_state()['tcp'][:3,3].copy();delta=np.asarray(episode['task_goal_offset_xy_m'])
    selected=episode['forced_action']
    value=dict(schema='WM02_PILOT2_PREREGISTERED_CHOICE_BEFORE_ACTION_V1',source_sha=adapter.spec['source_sha'],
        native_step=packet['step'],method='PREDECLARED_ACTION',purpose=episode['purpose'],split=episode['split'],
        scene_id=episode['scene_id'],selection_source=episode['selection_source'],repeat_index=episode['repeat_index'],
        goal_offset_xy_m=delta.tolist(),actual_decision_TCP_m=tcp.tolist(),task_goal_xy_m=(tcp[:2]+delta).tolist(),
        actual_history=np.asarray([state_vector(s)for s in adapter.next1_samples[-9:]]).tolist(),
        past_reference=initial.tolist(),source_age_s=packet['time_s']-source,selected_action=selected,
        selected_offset_xyz_m=OFFSETS[selected].tolist(),future_response_observed=False,
        predictions=None,actor_logits=None,teacher_or_actor_not_used_to_select_collection_action=True,
        task_sha256=adapter.spec['task_sha256'])
    path=adapter.output/'PILOT1_DECISION.private.json';write(path,value)
    if adapter.observe()['step']!=packet['step']:raise RuntimeError('PHYSICS_ADVANCED_DURING_COVERAGE_CHOICE')
    adapter.command_event(dict(phase='pilot1-choice-before-install',native_step=packet['step'],
        decision_sha256=digest(path),selected_action=selected,selection_source=episode['selection_source']))
    return OFFSETS[selected].copy()
