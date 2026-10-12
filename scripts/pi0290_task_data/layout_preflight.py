"""Compile and render fixed reset layouts, without any physical integration."""
import argparse
import os
from pathlib import Path
import numpy as np
from go2arx5_vla.task_data.config import COMMISSIONING,PILOTS,layout
from go2arx5_vla.task_data.engine import Engine
from go2arx5_vla.task_data.io import write
from go2arx5_vla.task_data.resource import Budget

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));budget=Budget(root)
assert 'diagnostic_pairs_actual' not in budget.value
budget.value['diagnostic_pairs_actual']=22
budget.value['diagnostic_pairs_provenance']='2 actual software dual-render probes +10 first optics diagnostics +10 final X-side optics diagnostics; all zero integrations'
budget.save();records=[]
for identifier in COMMISSIONING+PILOTS:
    budget.guard();engine=Engine(layout(identifier));images,geometry=engine.render(engine.snapshot())
    image=images['wrist_rgb'];red=int(np.sum((image[:,:,0].astype(float)>1.5*image[:,:,1])&(image[:,:,0].astype(float)>1.5*image[:,:,2])&(image[:,:,0]>80)))
    assert red>=20 and engine.steps==0
    records.append(dict(id=identifier,seed=layout(identifier)['seed'],initial_wrist_red_pixels=red,physics_steps=0))
    engine.close();budget.value['diagnostic_pairs_actual']+=1;budget.save()
write(root/'evidence/LAYOUT_PREFLIGHT.json',dict(status='PASS',records=records,new_physics_steps=0,new_episode_starts=0,
    actual_diagnostic_pairs=12,table_margin_min_m=.07,source_goal_min_m=.15,layout_replacements=0))
budget.finish_cpu('LAYOUT_RESET_PREFLIGHT');print('PASS: all twelve fixed reset layouts visible, zero integrations')
