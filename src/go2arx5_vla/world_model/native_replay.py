"""Full-mesh saved-target replay on PC: pose reconstruction, never integration/control."""
from pathlib import Path
from contextlib import contextmanager
import json,struct,time,threading,bisect,os
import numpy as np
from .storage import digest,write_json
from scripts.pi0294.task_state import parse_state


def packets(path):
    out=[]
    with Path(path).open('rb')as f:
        while size:=f.read(4):
            if len(size)!=4:raise ValueError('Truncated journal length')
            n,=struct.unpack('<I',size)
            if not 40<=n<=2*1024**2:raise ValueError('Invalid packet length')
            blob=f.read(n)
            if len(blob)!=n:raise ValueError('Truncated saved state packet')
            row=parse_state(blob)
            if out and(row['step']<=out[-1]['step']or row['time_s']<=out[-1]['time_s']):raise ValueError('Nonmonotone native packet')
            if abs(row['time_s']-row['step']*.002)>1e-8:raise ValueError('Original 2ms clock differs')
            for c in row['contacts']:
                if not np.isfinite([c['distance_m'],*c['wrench'],*c['frame'].flat,*c['point']]).all():raise ValueError('Nonfinite saved contact')
            out.append(row)
    if not out:raise ValueError('No original packets')
    return out


@contextmanager
def forbid_integration(mujoco):
    """Fail immediately if any Python replay path tries an integration API."""
    originals={name:getattr(mujoco,name)for name in('mj_step','mj_step1','mj_step2')};calls=[]
    def denied(*args,**kwargs):calls.append(True);raise RuntimeError('REPLAY_FORBIDS_INTEGRATION')
    for name in originals:setattr(mujoco,name,denied)
    try:yield calls
    finally:
        for name,value in originals.items():setattr(mujoco,name,value)


def load(config):
    import mujoco
    if mujoco.__version__!='3.2.7':raise ValueError('Exact saved-model MuJoCo3.2.7 required')
    if config['label']not in('TARGET_FAILED_REPLAY','SOURCE_REPLAY'):raise PermissionError('Saved replay labels only; live target requires separate owner')
    for key in('scene','journal'):
        b=config[key]
        if digest(b['path'])!=b['sha256']:raise ValueError('Original replay bytes changed:'+key)
    if Path(config['scene']['path']).suffix!='.mjb':raise ValueError('Exact compiled original model required')
    m=mujoco.MjModel.from_binary_path(config['scene']['path'])
    if (m.nq,m.nv,m.nu,m.na,m.nplugin)!=(34,32,20,0,0):raise ValueError('Native replay ABI/plugin differs')
    if mujoco.get_mjcb_control()is not None:raise PermissionError('No controller callback in passive replay')
    rows=packets(config['journal']['path'])
    for row in rows:
        if any(c[k]>=m.ngeom for c in row['contacts']for k in('geom0','geom1')):raise ValueError('Saved contact/model identity differs')
    d=mujoco.MjData(m);return mujoco,m,d,rows


def apply_frame(mujoco,m,d,row):
    d.qpos[:]=row['qpos'];d.qvel[:]=row['qvel'];d.ctrl[:]=row['ctrl'];d.time=row['time_s'];before=(d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),float(d.time))
    mujoco.mj_forward(m,d)
    if not all(np.array_equal(a,b)for a,b in zip(before[:3],(d.qpos,d.qvel,d.ctrl)))or d.time!=before[3]:raise RuntimeError('Forward changed recorded state/time')


def render_check(config,output,frames=120):
    from PIL import Image,ImageDraw
    start=time.monotonic();cpu=time.process_time();output=Path(output);mujoco,m,d,rows=load(config);timings=[]
    # Only offscreen framebuffer size changes; no physical parameters/file edits.
    m.vis.global_.offwidth=960;m.vis.global_.offheight=640
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera);camera.lookat[:]=[.28,0,.45];camera.distance=2.6;camera.azimuth=135;camera.elevation=-23
    with forbid_integration(mujoco)as calls,mujoco.Renderer(m,height=640,width=960)as renderer:
        saved=[]
        for i in range(frames):
            row=rows[min(len(rows)-1,int(i*(len(rows)-1)/max(1,frames-1)))];tick=time.monotonic();apply_frame(mujoco,m,d,row);renderer.update_scene(d,camera=camera);rgb=renderer.render();timings.append(time.monotonic()-tick)
            if i in(0,frames//2,frames-1):
                im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,34),fill=(20,20,20));draw.text((10,8),f"{config['label']} | {config['episode_id']} | saved step {row['step']} / t={row['time_s']:.3f}s | NO INTEGRATION",fill='white');name=f'frame-{i:04d}.private.png';im.save(output/name);saved.append(name)
        if calls:raise RuntimeError('Integration attempt during replay')
    result=dict(status='FULL_MESH_REPLAY_RENDER_CHECK_COMPLETE',source_sha=config['source_sha'],scene_sha256=config['scene']['sha256'],journal_sha256=config['journal']['sha256'],episode_id=config['episode_id'],label=config['label'],nq=m.nq,nv=m.nv,nu=m.nu,nmesh=m.nmesh,nbody=m.nbody,ngeom=m.ngeom,saved_packets=len(rows),first_step=rows[0]['step'],last_step=rows[-1]['step'],missing_steps=sum(b['step']-a['step']-1 for a,b in zip(rows,rows[1:])),recorded_interval_s=[rows[0]['time_s'],rows[-1]['time_s']],render_frames=frames,resolution=[960,640],render_elapsed_s=sum(timings),render_FPS=frames/sum(timings),frame_time_p95_s=float(np.quantile(timings,.95)),actual_forward_calls=frames,integration_calls=0,target_controller_starts=0,new_native_T_integrations=0,original_files_unchanged=all(digest(config[k]['path'])==config[k]['sha256']for k in('scene','journal')),screenshots=saved,restoration='Recorded qpos/qvel/ctrl/time geometry only, not controller or solver-state restore',contact_scope='Saved contacts validated; forward recomputes geometry but is not original force/safety evaluation',CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(output/'RESULT.json',result);return result


def passive(config,output,max_wall=60,rate=1.,*,transport_self_check=False,replay_feed=None):
    """MuJoCo native orbit/zoom + small Tk transport. No new 3D engine."""
    import tkinter as tk
    from tkinter import ttk
    from mujoco import viewer
    start=time.monotonic();cpu=time.process_time();output=Path(output);mujoco,m,d,rows=load(config);times=[r['time_s']for r in rows];paused=False;position=times[0];speed=rate;last=time.monotonic();displayed=0;seeks=0
    ui=tk.Tk();ui.title(config['label']+' · saved full-body replay · no integration');ui.geometry('740x150+20+20');label=tk.StringVar();tk.Label(ui,textvariable=label).pack();slider=tk.Scale(ui,from_=times[0],to=times[-1],resolution=.002,orient='horizontal',length=700);slider.set(position);slider.pack();controls=tk.Frame(ui);controls.pack();state={'seek':None,'pause':False,'closed':False}
    def seek(value):state['seek']=float(value)
    # Ignore programmatic slider updates; seek only while pointer is dragging.
    slider.bind('<B1-Motion>',lambda event:seek(slider.get()));slider.bind('<ButtonRelease-1>',lambda event:seek(slider.get()))
    def toggle():state['pause']=not state['pause']
    ttk.Button(controls,text='Pause / Play',command=toggle).pack(side='left');speed_var=tk.StringVar(value=str(rate));ttk.Combobox(controls,textvariable=speed_var,values=['0.05','0.1','0.25','0.5','1.0','2.0'],width=6,state='readonly').pack(side='left');tk.Label(controls,text='Arrow keys: ±one saved packet; mouse: orbit / zoom. End of recording holds.').pack(side='left')
    ui.protocol('WM_DELETE_WINDOW',lambda:state.update(closed=True))
    control_checks=[]
    if transport_self_check and replay_feed is None:
        ui.after(300,lambda:(toggle(),control_checks.append('pause')))
        ui.after(600,lambda:(seek(times[len(times)//2]),control_checks.append('seek_saved_packet')))
        ui.after(900,lambda:(speed_var.set('0.1'),toggle(),control_checks.append('rate_change_and_resume')))
    feed_sequence=0;feed_updates=0;stale_frames=0;feed_last=time.monotonic();feed_row=rows[0]
    def key(k):
        if k==32:toggle()
        elif k in(262,263):
            i=max(0,min(len(times)-1,bisect.bisect_right(times,position)-1+(1 if k==262 else-1)));state['seek']=times[i];state['pause']=True
        elif k==268:state['seek']=times[0]
        elif k==269:state['seek']=times[-1]
    apply_frame(mujoco,m,d,rows[0]);samples=[]
    with forbid_integration(mujoco)as calls,viewer.launch_passive(m,d,key_callback=key,show_left_ui=False,show_right_ui=False)as handle:
        with handle.lock():handle.cam.lookat[:]=[.28,0,.45];handle.cam.distance=2.6;handle.cam.azimuth=135;handle.cam.elevation=-23
        while handle.is_running()and not state['closed']and time.monotonic()-start<max_wall:
            now=time.monotonic();elapsed=now-last;last=now;speed=float(speed_var.get())
            if state['seek']is not None:position=max(times[0],min(times[-1],state.pop('seek')));state['seek']=None;seeks+=1
            elif not state['pause']:position=min(times[-1],position+elapsed*speed)
            i=max(0,bisect.bisect_right(times,position)-1);row=rows[i]
            if replay_feed is not None:
                from .replay_feed import read_snapshot
                incoming=read_snapshot(replay_feed,config,feed_sequence)
                if incoming is not None:feed_row=incoming;feed_sequence=incoming['sequence'];feed_updates+=1;feed_last=now
                row=feed_row;position=row['time_s'];stale_frames+=int(now-feed_last>1.)
            with handle.lock():
                handle.perturb.active=0;apply_frame(mujoco,m,d,row)
            handle.sync();displayed+=1;slider.set(position);label.set(f"{config['label']} | original packets {len(rows)} | saved step {row['step']} | physics t={row['time_s']:.3f}s | no interpolation / no mj_step");
            if replay_feed is not None:label.set(label.get()+(' | STALE: holding last recorded packet'if now-feed_last>1. else' | PRIVATE REPLAY FILE FEED, NOT LIVE TARGET'))
            ui.update();samples.append(now);time.sleep(max(0,1/30-(time.monotonic()-now)))
        if calls:raise RuntimeError('Integration attempted')
    ui.destroy();result=dict(status='ACTUAL_PASSIVE_VIEWER_FINITE_EXIT',source_sha=config['source_sha'],scene_sha256=config['scene']['sha256'],journal_sha256=config['journal']['sha256'],episode_id=config['episode_id'],label=config['label'],display_frames=displayed,seek_events=seeks,gui_wall_s=time.monotonic()-start,display_mean_FPS=(len(samples)-1)/(samples[-1]-samples[0])if len(samples)>1 else 0,rate=rate,transport_control_checks=control_checks,replay_feed_updates=feed_updates,replay_feed_last_sequence=feed_sequence,stale_display_frames=stale_frames,integration_calls=0,new_native_T_integrations=0,actual_forward_calls=displayed+1,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start);write_json(output/'RESULT.json',result);return result
