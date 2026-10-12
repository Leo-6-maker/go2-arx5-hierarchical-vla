"""D0 supplement: actual dual software-rendered images, zero integrations."""
import argparse
import os
from pathlib import Path
import time

from PIL import Image
from go2arx5_vla.task_data.config import layout
from go2arx5_vla.task_data.engine import Engine
from OpenGL import GL
from go2arx5_vla.task_data.io import write,file_hash
from go2arx5_vla.task_data.resource import Budget

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
root=Path(a.root);budget=Budget(root);began=time.monotonic()
engine=Engine(layout('C2900'));images,geometry=engine.render(engine.snapshot())
vendor=GL.glGetString(GL.GL_VENDOR).decode();renderer=GL.glGetString(GL.GL_RENDERER).decode()
assert vendor=='Mesa' and 'llvmpipe' in renderer and engine.steps==0
hashes={}
for name,array in images.items():
    assert array.shape==(256,256,3) and float(array.std())>8
    path=root/'evidence'/f'D0_SOFTWARE_{name}.png';Image.fromarray(array).save(path)
    hashes[name]=dict(sha256=file_hash(path),standard_deviation=float(array.std()))
engine.close()
write(root/'evidence/SOFTWARE_RENDER_PROBE.json',dict(status='PASS',backend='mesa_cpu',GL_VENDOR=vendor,
    GL_RENDERER=renderer,physics_steps=0,episode_starts=0,standalone_probe_images=2,
    images=hashes,geometry=geometry,wall_s=time.monotonic()-began,GPU_s=0))
budget.finish_cpu('D0_SOFTWARE_RENDER_PROBE')
print('PASS: actual Mesa llvmpipe dual RGB, zero physics steps')
