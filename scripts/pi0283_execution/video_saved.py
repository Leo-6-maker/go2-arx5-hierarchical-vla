"""Encode both original saved camera views; no simulator/model imports."""
import argparse,json,os,subprocess
from pathlib import Path
import cv2
import numpy as np
from closed_loop_audit import ROOT,sha


def main(point,condition):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    folder=ROOT/'source/episodes'/f'{point}-TRAIN-{condition}'/'attempt-000'
    result=json.loads((folder/'RESULT.json').read_text())
    assert result['status'].startswith('VALID')
    (ROOT/'videos').mkdir(exist_ok=True)
    with np.load(folder/'observations.npz') as f:base=f['base_camera_raw'];wrist=f['hand_camera_raw']
    with np.load(folder/'physics.npz') as f:flags={k:f[k] for k in ('is_cubeA_grasped','is_cubeA_on_cubeB','is_cubeA_static','success')}
    out=ROOT/'videos'/f'{point}_TRAIN{condition}_ACTUAL_SOURCE.mp4'
    receipt=out.with_suffix('.json')
    if receipt.exists():assert sha(out)==json.loads(receipt.read_text())['sha256'];return
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x592','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    child=subprocess.Popen(command,stdin=subprocess.PIPE)
    for i in range(0,len(base),2):
        canvas=np.zeros((592,1024,3),dtype=np.uint8)
        canvas[:512]=np.concatenate([cv2.resize(base[i],(512,512),interpolation=cv2.INTER_NEAREST),cv2.resize(wrist[i],(512,512),interpolation=cv2.INTER_NEAREST)],axis=1)
        labels=[f'PI0283 {point} TRAIN{condition} actual task t={i*.05-1:.2f}s BASE | WRIST',
            f'SOURCE PANDA native PD / non-realtime / NOT WBC / persistent={result["persistent_success"]} {result["persistent_pass_samples"]}/41',
            'native: '+ ' '.join(k.replace('is_cubeA_','')+'='+str(bool(v[i*5])) for k,v in flags.items())]
        for j,label in enumerate(labels):cv2.putText(canvas,label,(12,532+24*j),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        child.stdin.write(canvas.tobytes())
    child.stdin.close();assert child.wait()==0
    value=dict(status='PASS_ACTUAL_SAVED_TWO_VIEW_VIDEO',point=point,condition=condition,
        first_success_s=result['first_success_s'],persistent=result['persistent_success'],hold_success_samples=result['persistent_pass_samples'],
        frames=(len(base)+1)//2,original_frames=len(base),bytes=out.stat().st_size,sha256=sha(out),
        observation_sha256=sha(folder/'observations.npz'),physics_sha256=sha(folder/'physics.npz'),
        new_model_queries=0,new_render=0,new_physics=0,scope='SOURCE_PANDA_NATIVE_PD_NOT_WBC_NOT_REALTIME')
    receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');p.add_argument('--condition',type=int,default=10000);a=p.parse_args();main(a.point,a.condition)
