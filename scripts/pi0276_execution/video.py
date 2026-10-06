"""Twelve registered saved-frame videos; no renderer, teacher or extra physics."""
import json,os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import cv2,numpy as np
from run import ROOT,write,sha
from report import valid,RAW


def main():
    slots=[f'TEST-{seed}-{point}' for seed in range(30000,30004) for point in ('BASE0','TASK8192')]
    slots+=[f'DEV-{seed}-open-control' for seed in range(4)]
    rows=[];dst=RAW/'delivery';dst.mkdir(exist_ok=True)
    for slot in slots:
        folder=valid(slot);rec=json.loads((folder/'RESULT.json').read_text())
        source=folder/'observations.npz';out=dst/f'PI0276_{slot}.mp4'
        with np.load(source) as z:
            left=z['base_camera_96'];right=z['hand_camera_96'];assert len(left)==len(right)==rec['RGB_pairs'] and len(left)>0
            stream=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),20,(512,320));assert stream.isOpened()
            try:
                for a,b in zip(left,right):
                    rgb=np.concatenate([cv2.resize(v,(256,256),interpolation=cv2.INTER_NEAREST) for v in (a,b)],axis=1)
                    frame=np.zeros((320,512,3),np.uint8);frame[64:]=rgb[:,:,::-1]
                    for line,text in enumerate((slot+' persistent '+('PASS' if rec['persistent_success'] else 'FAIL'),
                        'Saved actual physics RGB | NON_REALTIME | source Panda',rec['status'])):
                        cv2.putText(frame,text,(8,16+19*line),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
                    stream.write(frame)
            finally:stream.release()
        capture=cv2.VideoCapture(str(out));assert capture.isOpened()
        count=0
        while True:
            ok,frame=capture.read()
            if not ok:break
            assert frame.shape==(320,512,3);count+=1
        capture.release();assert count==rec['RGB_pairs']
        rows.append(dict(slot=slot,path=str(out),frames=count,sha256=sha(out),source_sha256=sha(source),
            persistent_success=rec['persistent_success'],status=rec['status'],new_renders=0,new_physics_steps=0))
    write(ROOT/'evidence/VIDEOS.json',dict(status='PASS_ALL12_REGISTERED_SAVED_RGB_VIDEOS',rows=rows))


if __name__=='__main__':main()
