from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
from go2arx5_vla.task_data.io import write,file_hash,manifest
from go2arx5_vla.task_data.resource import Budget


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--id',required=True);a=p.parse_args()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    root=Path(a.root);folder=root/'episodes'/a.id;budget=Budget(root)
    frames=sorted((folder/'body_rgb').glob('*.png'));out=folder/'ACTUAL_DUAL_RGB.mp4'
    if out.exists():raise RuntimeError('Video already exists; preserve it')
    with imageio.get_writer(out,fps=20,codec='libx264',quality=7,macro_block_size=16,ffmpeg_params=['-threads','2','-pix_fmt','yuv420p']) as writer:
        for i,path in enumerate(frames):
            base=Image.open(path).convert('RGB');wrist=Image.open(folder/'wrist_rgb'/path.name).convert('RGB')
            canvas=Image.new('RGB',(512,288),'#202528');canvas.paste(base,(0,32));canvas.paste(wrist,(256,32))
            draw=ImageDraw.Draw(canvas);draw.text((6,2),f'PI0290 {a.id}  t={i/20:05.2f}s  SIM_ONLY / NOT_WBC',fill='white')
            draw.text((6,18),'BODY RGB',fill='white');draw.text((262,18),'WRIST RGB',fill='white');writer.append_data(np.asarray(canvas))
    decoded=sum(1 for _ in imageio.get_reader(out));assert decoded==len(frames)
    write(folder/'VIDEO_RECEIPT.json',dict(status='PASS_ACTUAL_SAVED_RGB_FULL_DECODE',frames=decoded,fps=20,
        first_sim_time_s=0,last_sim_time_s=(decoded-1)/20,sha256=file_hash(out),bytes=out.stat().st_size,
        new_render=0,new_physics=0,domain='INDEPENDENT_DYNAMIC_TOOL_SIM_ONLY_NON_REALTIME'))
    write(folder/'FILE_MANIFEST.json',manifest(folder));budget.finish_cpu('VIDEO_'+a.id)
    print(json.dumps({'path':str(out),'frames':decoded,'sha256':file_hash(out),'bytes':out.stat().st_size}))


if __name__=='__main__':main()
