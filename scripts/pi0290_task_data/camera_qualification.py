"""New optics on saved C2903 dynamic state: diagnostic render, no integration."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
from go2arx5_vla.task_data.engine import Engine
from go2arx5_vla.task_data.config import layout
from go2arx5_vla.task_data.io import write,file_hash
from go2arx5_vla.task_data.resource import Budget

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));budget=Budget(root)
folder=root/'episodes/C2903';high=np.load(folder/'highrate.npz')['rows']
engine=Engine(layout('C2903'));records=[]
for frame in [0,56,80,115,160,240,272,300,560,640]:
    if frame:
        row=high[frame*25-1];engine.data.qpos[:]=row[1:17];engine.data.qvel[:]=row[17:31];engine.data.time=row[0]
    engine.steps=frame*25
    mujoco.mj_forward(engine.model,engine.data)
    images,geometry=engine.render(engine.snapshot());paths={}
    for camera,image in images.items():
        path=root/'evidence/camera_x_side'/f'{frame:04d}_{camera}.png';path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(image).save(path);paths[camera]=dict(path=path.relative_to(root).as_posix(),sha256=file_hash(path),
            visible_object_vertices=int(sum(geometry[camera]['object_point_visible'])),
            visible_red_pixels=int(np.sum((image[:,:,0].astype(float)>1.5*image[:,:,1])&(image[:,:,0].astype(float)>1.5*image[:,:,2])&(image[:,:,0]>80))))
    records.append(dict(frame_index=frame,source='RERENDER_OF_SAVED_STATE_NOT_ORIGINAL_EPISODE_OBSERVATION',cameras=paths))
engine.close()
write(root/'evidence/CAMERA_QUALIFICATION_X_SIDE.json',dict(status='PASS' if all(r['cameras']['wrist_rgb']['visible_red_pixels']>=20 for r in records) else 'FAIL',
    optics_only_change=True,physical_controller_or_teacher_changes=False,new_integrations=0,
    diagnostic_pairs=len(records),records=records,source_physics_sha256=file_hash(folder/'physics.npz')))
budget.finish_cpu('CAMERA_V2_QUALIFICATION');print(json.dumps(records))
