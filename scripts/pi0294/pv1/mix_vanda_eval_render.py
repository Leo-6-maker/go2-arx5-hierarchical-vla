"""The two bounded original source constructions, inside the C/D PBS job.

The old failed standalone qualification is never reopened. These two actual
constructors consume the original panel's +2 constructor allowance; no episode,
integration, Teacher or model is run. The original physical runtime validates
before any native imports, and every intent is charged before construction.
"""
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read
from . import mix_vanda_eval_context as context,mix_accounting as a,mix_vanda_cohost as storage


def need(ok,why):
    if not ok:raise PermissionError('V5_COLOCATED_RENDER_'+why)


def run(root):
    root=Path(root);m,assets,*_=context.validate(root)
    from .mix_vanda_eval_identity import PHYSICAL_ROLE
    need(m['scope']==PHYSICAL_ROLE,'OWN_BOUND_PHYSICAL_ROLE_BEFORE_CONSTRUCTION')
    from . import core
    from scripts.pi0294 import scene
    from scripts.pi0294.s1_preflight import fingerprint
    output=root/'evidence/COLOCATED_ZERO_INTEGRATION_SOURCE';output.mkdir(mode=0o700)
    def put(name,value):
        p=output/name
        with p.open('xb') as f:f.write(a.raw(value));f.flush();os.fsync(f.fileno())
        p.chmod(0o400)
    put('INTENT.json',dict(job_id=os.environ['PBS_JOBID'],source_sha=m['code']['vla_commit'],
        operation='ORIGINAL_COUNTER_PARITY_TWO_CONSTRUCTORS_NOT_TWO_NEW_SCENES',
        constructors_upper=2,physics_ticks=0,observed_epoch=time.time()))
    import mujoco
    import numpy as np
    from PIL import Image
    need(mujoco.__version__=='3.2.7' and np.__version__=='1.26.4','ORIGINAL_NATIVE_PACKAGE_VERSIONS')
    original=scene.make_engine;records=[]
    def observed_engine(*args,**kwargs):
        engine=original(*args,**kwargs);render=engine.render
        def observed_render(state):
            before=fingerprint(engine);images,geometry=render(state)
            from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER
            vendor=glGetString(GL_VENDOR).decode();renderer=glGetString(GL_RENDERER).decode()
            need('Mesa' in vendor and 'llvmpipe' in renderer.lower(),'ACTUAL_ORIGINAL_CPU_RENDERER')
            need(engine.steps==0 and engine.data.time==0 and before==fingerprint(engine)
                 and set(images)=={'body_rgb','wrist_rgb'},'FULL_STATE_UNCHANGED_ZERO_INTEGRATION')
            pins={};i=len(records)
            for camera,pixels in images.items():
                need(pixels.shape==(256,256,3) and pixels.dtype==np.uint8 and np.isfinite(pixels).all()
                     and float(pixels.std())>8,'REAL_ORIGINAL_NONBLANK_DUAL_RGB')
                p=output/(str(i)+'-'+camera+'.png');need(not p.exists(),'PRESERVE_EACH_FRAME')
                Image.fromarray(pixels).save(p);p.chmod(0o400)
                from scripts.pi0294.native_dataset import sha
                pins[camera]=dict(sha256=sha(p),shape=list(pixels.shape),geometry=geometry[camera])
            records.append(dict(state_before_sha256=before,state_after_sha256=fingerprint(engine),
                time_s=float(engine.data.time),physics_ticks=engine.steps,GL_VENDOR=vendor,GL_RENDERER=renderer,images=pins))
            return images,geometry
        engine.render=observed_render
        return engine
    try:
        scene.make_engine=observed_engine
        counter=core.qualify(root,m['candidate']+'-DEV-N294101')
    finally:scene.make_engine=original
    need(len(records)==2 and counter['model_constructor_attempts']==2 and counter['integrations']==0,
         'ONLY_ORIGINAL_PARITY_COUNTER_CONSTRUCTORS')
    retained=storage.enforce_output(output,8*1024**2)
    result=dict(status='PASS_CURRENT_PBS_ORIGINAL_SOURCE_DUAL_RGB_ZERO_INTEGRATION',job_id=os.environ['PBS_JOBID'],
        source_sha=m['code']['vla_commit'],records=records,constructors=2,physics_ticks=0,episode_starts=0,
        model_loads=0,queries=0,retained_output=retained,Eval_conditions_completed=0,task_or_Teacher_qualification=False,
        original_counter_qualification=counter)
    put('RESULT.json',result);return result



if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();run(args.root)
