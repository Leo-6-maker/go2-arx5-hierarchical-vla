from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from go2arx5_vla.target_sim.scorer import BOOL_FIELDS,score_episode
from .config import PARAMETERS as P,TASK
from go2arx5_vla.task_data.io import file_hash,write


def saved_score(folder):
    root=Path(folder);card=json.loads((root/'CARD.json').read_text())
    with np.load(root/'physics.npz',allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    complete=bool(len(arrays['time_s'])==641)
    if card['mode']=='empty_grasp':
        t=arrays['time_s'];grasp=arrays['native_grasp'];run=0;maximum=0
        for value in grasp:
            run=run+1 if value else 0;maximum=max(maximum,run)
        result=dict(task_id=TASK,domain='FIXED_BASE_ARX5_TASK_SIM',
            fixture='EMPTY_GRASP_CONTACT_NEGATIVE_NOT_A_T1_QUALITY_LAYOUT',
            original_source_shift_m=.18,complete_32s=complete and abs(float(t[-1])-32)<1e-8,
            safety_valid=bool(arrays['safety_valid'].all()),
            history=dict(sustained_grasp=bool(maximum>=7),sustained_lift=False,horizontal_carry=False),
            actual_grasp_samples=int(grasp.sum()),left_contact_samples=int(arrays['left_object_contact'].sum()),
            right_contact_samples=int(arrays['right_object_contact'].sum()),persistent_success=False,
            physics_sha256=file_hash(root/'physics.npz'),raw_sample_count=len(t),selected='NONE',deployment='NONE')
        write(root/'SCORE.json',result);return result
    result=score_episode(arrays,goal_center=[*card['goal_xy'],P['table_top']+.02],
        initial_object_pose=arrays['object_pose'][0],table_height_m=P['table_top'],complete=complete,
        evidence_source='REAL_MUJOCO_FIXED_BASE_ARX5')
    result['task_id']=TASK;result['domain']='FIXED_BASE_ARX5_TASK_SIM'
    result['physics_sha256']=file_hash(root/'physics.npz')
    result['raw_sample_count']=len(arrays['time_s'])
    result['maximum_actual_lift_m']=float(np.max(arrays['object_pose'][:,2,3]-arrays['object_pose'][0,2,3]))
    result['maximum_actual_carry_xy_m']=float(np.max(np.linalg.norm(arrays['object_pose'][:,:2,3]-arrays['object_pose'][0,:2,3],axis=1)))
    write(root/'SCORE.json',result);return result
