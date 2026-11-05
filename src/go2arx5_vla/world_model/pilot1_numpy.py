"""Portable trained Actor inference without torch in the frozen ROS Python."""
import numpy as np
from .target_arx import CausalARX


def goal_offsets(task):
    rng=np.random.default_rng(task['goal_seed'])
    jitter=rng.uniform(-task['jitter_m'],task['jitter_m'],(4,task['goals_per_center'],2))
    return(np.asarray(task['mean_response_xy_m'])[:,None]+jitter).reshape(-1,2)


def actor_features(data,meta,ids,offsets_xy):
    layout=CausalARX(meta).layout
    cubes=[j for j in meta['joints']if j['name']=='task_cube_free']
    if len(cubes)!=1 or(cubes[0]['qpos_address'],cubes[0]['qvel_address'])!=(27,26):raise ValueError('ACTUAL_ROBOT_OBJECT_BOUNDARY_REQUIRED')
    columns=list(range(layout['qpos'].start,layout['qpos'].start+27))+list(range(layout['qvel'].start,layout['qvel'].start+26))
    for name in('ctrl','tcp_position_m','tcp_rotation6d','finger_position_m'):columns+=list(range(layout[name].start,layout[name].stop))
    prefix=data['state'][ids,:9][:,:,columns].reshape(len(ids),-1);past=data['history_goal'][ids,:8].reshape(len(ids),-1)
    return np.concatenate((prefix,past,np.asarray(offsets_xy)/.001,data['source_age'][ids,8:9],CausalARX.remaining(data,ids)[:,None]),axis=1)


def logits(arrays,x):
    value=(np.asarray(x,dtype=np.float32)-arrays['mean'])/arrays['scale']
    for layer in(0,2,4):
        value=value@arrays[f'w{layer}'].T+arrays[f'b{layer}']
        if layer<4:value=np.maximum(value,0.)
    return value
