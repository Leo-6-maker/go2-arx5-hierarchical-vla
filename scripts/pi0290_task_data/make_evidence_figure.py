"""Actual saved P2900 RGB and measured physical signals, CPU only."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from go2arx5_vla.task_data.io import write,file_hash
from go2arx5_vla.task_data.resource import Budget

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));budget=Budget(root)
folder=root/'episodes/P2900';samples=json.loads((folder/'samples.json').read_text());d=np.load(folder/'physics.npz')
image=Image.new('RGB',(1024,640),'white');draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=16)
draw.text((12,8),'PI0290 P2900 | ACTUAL RGB / MEASURED PHYSICS | SIM_ONLY, NOT_WBC',fill='#152528',font=font)
for k,camera in enumerate(['body_rgb','wrist_rgb']):
    draw.text((k*256+12,38),camera+' at sim t=4.00s',fill='#152528',font=font)
    image.paste(Image.open(folder/camera/'0080.png').convert('RGB'),(k*256,64))
labels=['Dynamic tool: free body, no arm joints','TCP: tool body origin at pad center','left_slide: qpos14 / qvel12 / motor0','right_slide: qpos15 / qvel13 / motor1','Finger travel 0..44 mm; actual contact','Motor force <=8 N, finite pad friction','Cartesian force <=40 N per component','Cartesian torque <=2 Nm per component','Body camera: fixed world rig','Wrist camera: rigid tool X-side','K + world optical transform each tick','No WBC, GR00T or robot mesh used']
for j,label in enumerate(labels):draw.text((528,64+j*21),label,fill='#152528',font=font)
t=d['time_s'];q=np.asarray([s['execution']['actual_finger_q'] for s in samples])*1000
series=[('Measured finger opening (mm)',[q[:,0],q[:,1]],['left','right'],44.),
    ('Object displacement (cm)',[(d['object_pose'][:,2,3]-.42)*100,np.linalg.norm(d['object_pose'][:,:2,3]-d['object_pose'][0,:2,3],axis=1)*100],['lift Z','carry XY'],30.),
    ('Same-object contact / actual grasp',[d['left_object_contact'].astype(float),d['right_object_contact'].astype(float)+1.3,d['native_grasp'].astype(float)+2.6],['left','right','grasp'],4.)]
colors=['#d52a35','#198c9e','#6b4d99']
for j,(title,ys,names,scale) in enumerate(series):
    x0=j*340+42;y0=390;w=280;h=180
    draw.text((j*340+10,344),title,fill='#152528',font=font)
    draw.line([(x0,y0),(x0,y0+h),(x0+w,y0+h)],fill='#657276',width=1)
    for tick in [0,8,16,24,32]:
        x=x0+tick/32*w;draw.line([(x,y0),(x,y0+h)],fill='#e8ebed')
        draw.text((x-6,y0+h+6),str(tick),fill='#657276',font=font)
    draw.text((x0+w-55,y0+h+28),'sim s',fill='#657276',font=font)
    for k,(y,name) in enumerate(zip(ys,names)):
        points=list(zip(x0+t/32*w,y0+h-np.clip(y,0,scale)/scale*h))
        draw.line(points,fill=colors[k],width=2)
        draw.text((x0+k*86,y0-24),name,fill=colors[k],font=font)
out=root/'evidence/PILOT_CONTACT_AND_CAMERA.png';image.save(out)
write(root/'evidence/FIGURE_RECEIPT.json',dict(source='actual saved P2900 raw frames and measured signals',
    rgb_frame_index=80,physics_sha256=file_hash(folder/'physics.npz'),figure_sha256=file_hash(out),
    new_integrations=0,new_render=0,actual_obs_not_synthetic=True))
budget.finish_cpu('D4_ACTUAL_EVIDENCE_FIGURE');print(file_hash(out))
