"""Encode both original saved camera views; no simulator/model imports."""
import json,os,subprocess
from pathlib import Path
import cv2
import numpy as np
from closed_loop_audit import ROOT,sha


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    folder=ROOT/'source/episodes/ACTION2048-TRAIN-10001/attempt-000'
    result=json.loads((folder/'RESULT.json').read_text())
    assert result['status']=='VALID_SOURCE_COMPLETE' and not result['persistent_success']
    with np.load(folder/'observations.npz') as f:base=f['base_camera_raw'];wrist=f['hand_camera_raw']
    with np.load(folder/'physics.npz') as f:flags={k:f[k] for k in ('is_cubeA_grasped','is_cubeA_on_cubeB','is_cubeA_static','success')}
    out=ROOT/'videos/ACTION2048_TRAIN10001_FIRST_SUCCESS_THEN_FAIL.mp4'
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x592','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    child=subprocess.Popen(command,stdin=subprocess.PIPE)
    for i in range(0,len(base),2):
        canvas=np.zeros((592,1024,3),dtype=np.uint8)
        canvas[:512]=np.concatenate([cv2.resize(base[i],(512,512),interpolation=cv2.INTER_NEAREST),cv2.resize(wrist[i],(512,512),interpolation=cv2.INTER_NEAREST)],axis=1)
        labels=[f'PI0282 ACTION2048 TRAIN10001  actual task t={i*.05-1:.2f}s  BASE | WRIST',
            'SOURCE PANDA native PD / non-realtime / NOT WBC / persistent FAIL 0/41',
            'native: '+ ' '.join(k.replace('is_cubeA_','')+'='+str(bool(v[i*5])) for k,v in flags.items())]
        for j,label in enumerate(labels):cv2.putText(canvas,label,(12,532+24*j),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        child.stdin.write(canvas.tobytes())
    child.stdin.close();assert child.wait()==0
    value=dict(status='PASS_ACTUAL_SAVED_TWO_VIEW_VIDEO',point='ACTION2048',condition=10001,
        first_success_s=result['first_success_s'],persistent=False,hold_success_samples=0,
        frames=(len(base)+1)//2,original_frames=len(base),bytes=out.stat().st_size,sha256=sha(out),
        observation_sha256=sha(folder/'observations.npz'),physics_sha256=sha(folder/'physics.npz'),
        new_model_queries=0,new_render=0,new_physics=0,scope='SOURCE_PANDA_NATIVE_PD_NOT_WBC_NOT_REALTIME_NOT_TASK_PASS')
    (ROOT/'videos/ACTION2048_VIDEO.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
