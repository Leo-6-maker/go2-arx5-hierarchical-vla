"""One differentiable reward implementation for real and predicted states.

Units/weights are frozen before optimization; this never supplies task PASS.
"""
import torch
from .features import SLICES

RECIPE = dict(version='WM01_T1_POTENTIAL_V1', gamma=.99, reach_weight=.15,
    grasp_lift_weight=.15, carry_weight=.30, released_goal_weight=.60,
    reach_scale_m=.05, carry_scale_m=.08, goal_scale_m=.04,
    height_scale_m=.01, speed_scale_m_s=.02, clearance_m=.08,
    action_cost=.01, failure_penalty=1.)


def potential(state, goal, events):
    tcp = state[..., SLICES['tcp_position_m']]
    obj = state[..., SLICES['object_position_m']]
    distance = torch.linalg.vector_norm(tcp-obj, dim=-1)
    goal_distance = torch.linalg.vector_norm(obj[..., :2]-goal[..., :2], dim=-1)
    velocity = torch.linalg.vector_norm(state[..., SLICES['object_velocity']][..., :3], dim=-1)
    opening = state[..., SLICES['finger_q_m']].mean(-1)/.044
    grasp = events[..., 2]
    lift = torch.sigmoid((obj[..., 2]-.48)/.01)
    reach = .15*torch.exp(-distance/.05)
    grasp_lift = .15*grasp*lift
    carry = .30*grasp*lift*torch.exp(-goal_distance/.08)
    released_goal = (.60*torch.exp(-goal_distance/.04)*torch.exp(-torch.abs(obj[..., 2]-.46)/.01)
        *torch.exp(-velocity/.02)*(1-grasp)*torch.sigmoid((opening-.7)/.05)
        *torch.sigmoid((distance-.08)/.01))
    return reach+grasp_lift+carry+released_goal


def reward(before, after, action, goal, before_events, after_events, failure):
    scale = action.new_tensor([.01]*3+[.05]*3+[1.]).repeat(8)
    cost = .01*(action/scale).square().mean(-1)
    return .99*potential(after, goal, after_events)-potential(before, goal, before_events)-cost-failure
