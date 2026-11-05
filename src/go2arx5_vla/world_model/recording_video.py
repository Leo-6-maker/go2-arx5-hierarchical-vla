"""Private native-mesh recording export: saved poses only, never integration."""
from pathlib import Path
import bisect,json,time
import numpy as np
from .native_replay import load,apply_frame,forbid_integration
from .storage import digest,write_json


def render(config,output,*,fps=30,max_wall=90):
    from PIL import Image,ImageDraw
    import imageio_ffmpeg
    from scripts.pi0294.task_state import ActualState
    if fps not in(15,30)or not 1<=max_wall<=180:raise ValueError('FINITE_RECORDED_VIDEO_ENDPOINT_REQUIRED')
    if config['label']not in('TARGET_QUALIFIED_RECORDING_REPLAY','TARGET_FAILED_REPLAY'):raise ValueError('TARGET_RECORDED_VIDEO_LABEL_REQUIRED')
    out=Path(output);out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time()
    mujoco,m,d,rows=load(config);actual=ActualState(m);times=[r['time_s']for r in rows]
    entry=config['native_result']
    if digest(entry['path'])!=entry['sha256']:raise ValueError('ORIGINAL_NATIVE_RESULT_CHANGED')
    native=json.loads(Path(entry['path']).read_text());view_class='FREE_SPACE'if config['label']=='TARGET_QUALIFIED_RECORDING_REPLAY'else'FAILED'
    if view_class=='FAILED'and(native.get('status')!='NATIVE_STAGE_FAILED_PRESERVED'or native.get('scene_sha256')!=config['scene']['sha256']or native.get('source_sha')!=config['original_native_source_sha']or native.get('actual_target_integrations')!=len(rows)):raise ValueError('FAILED_VIDEO_REQUIRES_SAME_ORIGINAL_FAILED_NATIVE_EPISODE')
    commands=[]
    if config.get('commands'):
        entry=config['commands']
        if digest(entry['path'])!=entry['sha256']:raise ValueError('ORIGINAL_COMMAND_RECORDING_CHANGED')
        raw=[json.loads(line)for line in Path(entry['path']).read_text().splitlines()]
        installed={r['wire_sha256']for r in raw if r.get('phase')=='installed'}
        commands=[r for r in raw if r.get('phase')=='proposed'and r['wire_sha256']in installed]
    count=int(np.ceil((times[-1]-times[0])*fps))+1
    if count>900:raise ValueError('FINITE_VIDEO_MAX900_FRAMES')
    m.vis.global_.offwidth=960;m.vis.global_.offheight=640
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.lookat[:]=[.20,0,.43];cam.distance=1.8;cam.azimuth=135;cam.elevation=-23
    path=out/'recorded-native.private.mp4';writer=imageio_ffmpeg.write_frames(str(path),(960,640),fps=fps,codec='libx264',output_params=['-crf','20','-threads','2'],ffmpeg_log_level='error');writer.send(None)
    samples=[];forward=0
    try:
        with forbid_integration(mujoco)as calls,mujoco.Renderer(m,height=640,width=960)as renderer:
            for i,t in enumerate(np.linspace(times[0],times[-1],count)):
                if time.monotonic()-start>=max_wall:raise TimeoutError('FINITE_RECORDED_VIDEO_WALL')
                row=rows[max(0,bisect.bisect_right(times,t)-1)];apply_frame(mujoco,m,d,row);state=actual.update(row);forward+=2
                renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,94),fill=(20,20,20))
                past=[c for c in commands if c['native_step']<=row['step']];goal=np.asarray(past[-1]['action_EEF10'][-1])[:3]if past else None
                xyz=lambda v:', '.join(f'{x:+.4f}'for x in v)
                lines=[f"REPLAY / {view_class} | {config['episode_id']} | saved step {row['step']} / t={row['time_s']:.3f}s | NO INTEGRATION",
                       'TCP actual m: '+xyz(state['tcp'][:3,3])+' | installed command goal m: '+('not installed'if goal is None else xyz(goal)),
                       'BASE actual m: '+xyz(row['qpos'][:3])+' | finger L/R m: '+xyz(state['finger_q']),
                       f"Recorded pad contacts L/R: {state['left_object_contact']}/{state['right_object_contact']} | grasp: {state['native_grasp']} | finite collection, no full-task success claim"]
                if view_class=='FAILED':lines[-1]='Original failure / STOP: '+str(native.get('physics_stops')or native.get('trace_audit',{}).get('errors')or native.get('failure'))[:120]
                for j,line in enumerate(lines):draw.text((8,5+22*j),line,fill='white')
                writer.send(np.ascontiguousarray(im).tobytes())
                if i in(0,count//2,count-1):im.save(out/f'frame-{i:04d}.private.png')
                samples.append(dict(frame=i,native_step=row['step'],time_s=row['time_s']))
            if calls:raise RuntimeError('REPLAY_ATTEMPTED_INTEGRATION')
    finally:writer.close()
    pins={key:config[key]['sha256']for key in('scene','journal','native_result','commands')if key in config}
    if any(digest(config[key]['path'])!=pin for key,pin in pins.items()):raise ValueError('RECORDING_INPUT_CHANGED')
    result=dict(status='PRIVATE_FULL_MESH_RECORDED_VIDEO_COMPLETE',source_sha=config['source_sha'],episode_id=config['episode_id'],label=config['label'],view_class=view_class,video_sha256=digest(path),input_sha256=pins,frames=count,fps=fps,resolution=[960,640],native_interval_s=[times[0],times[-1]],actual_forward_calls=forward,integration_calls=0,target_controller_starts=0,WM_updates=0,Actor_updates=0,original_files_unchanged=True,restoration='Recorded geometry qpos/qvel/ctrl/time only; no hidden-controller restoration',contact_scope='Saved native contact forces/flags plus FK, no new contact qualification',CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(out/'FRAME_INDEX.private.json',samples);write_json(out/'RESULT.json',result);return result
