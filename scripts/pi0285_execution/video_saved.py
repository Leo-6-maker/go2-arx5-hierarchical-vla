"""Actual teacher/model saved frames only; zero new rendering or physics."""
import argparse,json,os,subprocess
from pathlib import Path
import cv2
import numpy as np
from contract import ROOT,sha,write


def identity(card,result):
    if 'point' in card:
        assert result['point']==card['point'] and result['model_queries']>0 and result['teacher_passes']==0
        return 'MODEL',card['point']
    assert card['teacher_sha256'] and result['model_queries']==0 and result['teacher_passes']==1
    return 'TEACHER',''


def video(folder):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    folder=Path(folder);card=json.loads((folder/'EPISODE_CARD.json').read_text());result=json.loads((folder/'RESULT.json').read_text())
    assert result['status'].startswith('VALID')
    role,point=identity(card,result)
    out=(folder/f'{point}-ACTUAL_MODEL.mp4') if role=='MODEL' else ROOT/'videos'/f"{card['task']}-{card['split']}-{card['layout']}-ACTUAL_TEACHER.mp4";receipt=out.with_suffix('.json');out.parent.mkdir(exist_ok=True)
    if receipt.exists():assert sha(out)==json.loads(receipt.read_text())['sha256'];return
    with np.load(folder/'observations.npz') as f:base=f['base_camera_raw'];wrist=f['hand_camera_raw']
    with np.load(folder/'physics.npz') as f:grasp=f['is_cubeA_grasped'];red=f['cubeA_pose'][:,:3];tcp=f['tcp'][:,:3,3]
    partial=out.with_suffix('.partial.mp4')
    child=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x624','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],stdin=subprocess.PIPE)
    for i in range(0,len(base),2):
        canvas=np.zeros((624,1024,3),dtype=np.uint8);canvas[:512]=np.concatenate([cv2.resize(base[i],(512,512),interpolation=cv2.INTER_NEAREST),cv2.resize(wrist[i],(512,512),interpolation=cv2.INTER_NEAREST)],axis=1)
        labels=[f"PI0285 {point} {card['task']} {card['split']} layout {card['layout']} t={i*.05-1:.2f}s BASE | WRIST",f'ACTUAL {role} / SOURCE PANDA native PD / NON-REALTIME / NOT TARGET WBC',f"full32s={result['task_score']['complete_32s']} persistent={result['persistent_success']} hold={result['task_score']['hold_pass']}/41",f'actual grasp={bool(grasp[i*5])} TCP-red={np.linalg.norm(tcp[i*5]-red[i*5]):.3f}m']
        for j,label in enumerate(labels):cv2.putText(canvas,label,(12,534+25*j),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),1,cv2.LINE_AA)
        child.stdin.write(canvas.tobytes())
    child.stdin.close();assert child.wait()==0;partial.replace(out)
    value=dict(status=f'PASS_ACTUAL_SAVED_{role}_TWO_VIEW_VIDEO_NOT_POLICY_QUALIFICATION',role=role,point=point,task=card['task'],split=card['split'],layout=card['layout'],sha256=sha(out),bytes=out.stat().st_size,frames=(len(base)+1)//2,original_frames=len(base),source_result_sha256=sha(folder/'RESULT.json'),source_observation_sha256=sha(folder/'observations.npz'),source_physics_sha256=sha(folder/'physics.npz'),persistent=result['persistent_success'],new_render=0,new_physics=0,new_model_calls=0)
    write(receipt,value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');video(p.parse_args().folder)
