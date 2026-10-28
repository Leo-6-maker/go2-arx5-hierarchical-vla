"""D0 import/compile/one-frame render. No mj_step or episode start."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from go2arx5_vla.task_data.config import AUTHORITY, layout
from go2arx5_vla.task_data.scene import xml_scene


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    root=Path(a.root);began=time.time();cpu=resource.getrusage(resource.RUSAGE_SELF)
    import mujoco
    import numpy as np
    from PIL import Image
    assert mujoco.__version__=='3.2.7'
    xml=xml_scene(layout('C2900'));model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    assert data.time==0 and model.neq==0 and model.nmocap==0
    renderer=mujoco.Renderer(model,256,256)
    renderer.update_scene(data,camera='body_rgb');frame=renderer.render().copy();renderer.close()
    assert frame.shape==(256,256,3) and frame.dtype==np.uint8 and float(frame.std())>8
    path=root/'evidence/D0_BODY_RENDER.png';Image.fromarray(frame).save(path)
    after=resource.getrusage(resource.RUSAGE_SELF)
    receipt=dict(stage='D0',status='PASS_ISOLATED_MUJOCO_ENGINE_AND_HEADLESS_RENDER',authority=AUTHORITY,
        source_sha=a.source_sha,engine=mujoco.__version__,backend=os.environ.get('MUJOCO_GL'),
        EGL_device=os.environ.get('MUJOCO_EGL_DEVICE_ID'),model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        nq=model.nq,nv=model.nv,nu=model.nu,neq=model.neq,nmocap=model.nmocap,
        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),image_std=float(frame.std()),
        physics_steps=0,episode_starts=0,paired_observations=0,standalone_probe_images=1,
        GPU_service_s=time.time()-began,CPU_s=after.ru_utime+after.ru_stime-cpu.ru_utime-cpu.ru_stime,
        packages={n:metadata.version(n) for n in ['mujoco','numpy','pillow','imageio','imageio-ffmpeg','pytest']},
        GR00T=0,WBC=0,selected='NONE',deployment='NONE')
    (root/'evidence/D0_RUNTIME.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
