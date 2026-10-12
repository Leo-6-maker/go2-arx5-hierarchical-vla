from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from ..target_sim.scorer import BOOL_FIELDS,score_episode
from .config import PARAMETERS as P,TASK
from .io import file_hash,write


def saved_score(folder):
    root=Path(folder);card=json.loads((root/'CARD.json').read_text())
    with np.load(root/'physics.npz',allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    complete=bool(len(arrays['time_s'])==641)
    result=score_episode(arrays,goal_center=[*card['goal_xy'],P['table_top']+.02],
        initial_object_pose=arrays['object_pose'][0],table_height_m=P['table_top'],complete=complete,
        evidence_source='REAL_MUJOCO_INDEPENDENT_DYNAMIC_TOOL')
    result['task_id']=TASK;result['domain']='INDEPENDENT_DYNAMIC_TOOL_SIM_ONLY_NON_REALTIME'
    result['physics_sha256']=file_hash(root/'physics.npz')
    result['raw_sample_count']=len(arrays['time_s'])
    result['maximum_actual_lift_m']=float(np.max(arrays['object_pose'][:,2,3]-arrays['object_pose'][0,2,3]))
    result['maximum_actual_carry_xy_m']=float(np.max(np.linalg.norm(arrays['object_pose'][:,:2,3]-arrays['object_pose'][0,:2,3],axis=1)))
    write(root/'SCORE.json',result);return result
