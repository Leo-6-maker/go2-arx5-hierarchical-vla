"""Recorded RGB/actual arrays only. No re-render or model call for prettier video."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
sys.path[:0]=[str(BASE/'PI0266/code'),str(BASE/'PI0250/runtime/reference-317a/src')]
from pi0255_motion3 import write,sha


def read(p):return json.loads(Path(p).read_text())


def video(mode):
    import cv2
    out=RAW/'delivery';out.mkdir(exist_ok=True);path=out/('PI0267_'+mode.upper()+'.mp4')
    if (out/(mode+'-VIDEO_RECEIPT.json')).exists():assert sha(path)==read(out/(mode+'-VIDEO_RECEIPT.json'))['sha256'];return
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1024,640));assert writer.isOpened();count=0;chapters=[]
    def append(frames,labels):
        nonlocal count
        canvas=np.full((640,1024,3),18,np.uint8)
        for i,key in enumerate(('body_rgb','wrist_rgb')):
            image=cv2.resize(frames[key],(480,480),interpolation=cv2.INTER_NEAREST)
            canvas[110:590,16+i*512:496+i*512]=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        for y,text in zip((24,49,74,99),labels):cv2.putText(canvas,text,(18,y),cv2.FONT_HERSHEY_SIMPLEX,.46,(240,240,240),1)
        cv2.putText(canvas,'Recorded96x96 RGB / no outcome resampling / selected=NONE',(18,624),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,230,230),1)
        writer.write(canvas);count+=1
    if mode=='virtual':
        verdict=read(out/'PRIMARY_VERDICT.json');assert verdict['virtual_episodes']==192
        family=read(ROOT/'design/SCENE_CARD.json')['families'][28]['family_id']
        for point in ('donor','N4096','R4096'):
            for task in ('M1','M3'):
                for command in (0,1):
                    for backend in ('E0','E1'):
                        folder=RAW/'virtual'/point/f'{family}-{task}-c{command}-{backend}';result=read(folder/'result.json');start=count
                        frames=sorted((folder/'queries').glob('*/observation.npz'))
                        for p in frames:
                            z=np.load(p);append(z,['VIRTUAL_EEF_KINEMATIC / NOT_WBC / NON_REALTIME_LOCKSTEP',f'{point} {family} {task} c{command} {backend}',
                                f"actual virtual t={float(z['timestamp']):.2f}s /2x playback / success={result['success']}",f"stop={result['stop_reason']} / all192 outcomes in fixed quality table"])
                        chapters.append(dict(point=point,case=folder.name,first_frame=start,last_frame=count-1,success=result['success']))
    elif mode=='teacher':
        admission=read(R1/'design/DATA_ADMISSION.json');fid=admission['families_admitted'][0];folder=RAW/'S2'/fid
        rows=read(folder/'rows.json');frames=np.load(folder/'policy_frames.npz');z=np.load(folder/'proposals.npz')
        for task in ('M1','M3'):
            for command in (0,1):
                start=count
                own=[r for r in rows if r['task']==task and r['command']==command and r['nominal']]
                for row in own:
                    j=row['frame_index'];append(dict(body_rgb=frames['body_rgb'][j],wrist_rgb=frames['wrist_rgb'][j]),
                        ['VIRTUAL_EEF_KINEMATIC / NOT_WBC / TRAIN ANCHOR MONTAGE',f'{fid} {task} c{command} nominal anchor{row["anchor_index"]}/15',
                         'Actual existing rendered TRAIN states / NOT an executed teacher episode',f"matched actual state/label / ell={row['label']['ell_m']:.5f}m"])
                chapters.append(dict(task=task,command=command,first_frame=start,last_frame=count-1))
    else:
        card=read(R1/'design/TEACHER_CARD.json')
        for case in card['diagnostics']:
            attempts=sorted((RAW/'diagnostics'/case['case_id']).glob('attempt-*/result.json'));valid=[p for p in attempts if read(p)['validity']=='VALID']
            if not valid:chapters.append(dict(case=case['case_id'],status='NOT_RUN_OR_INVALID',frames=0));continue
            p=valid[0];result=read(p);start=count
            for frame in sorted((p.parent/'frames').glob('*.npz')):
                z=np.load(frame);append(z,['ACTUAL GO2/ARX5 SIMULATOR WBC / NON_REALTIME_LOCKSTEP',case['case_id'],
                    f"physical t={float(z['time']):.2f}s /2x playback / success={result['success']}",f"failure={result['failure']} / conservative local-H40 witness"])
            chapters.append(dict(case=case['case_id'],first_frame=start,last_frame=count-1,success=result['success'],failure=result['failure']))
    writer.release()
    if count:
        cap=cv2.VideoCapture(str(path));decoded=0
        while True:
            okay,frame=cap.read()
            if not okay:break
            assert frame.shape==(640,1024,3);decoded+=1
        cap.release();assert decoded==count
    write(out/(mode+'-VIDEO_RECEIPT.json'),dict(status='PASS_RECORDED_RGB_VIDEO_FULL_DECODE' if count else 'NO_VALID_ACTUAL_CAMERA_FRAMES',
        mode=mode,file=str(path),sha256=sha(path),frames=count,fps=5.,chapters=chapters,model_queries=0,new_renders=0,new_physics=0))
    print(json.dumps(dict(mode=mode,frames=count,bytes=path.stat().st_size)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('teacher','virtual','wbc'));video(p.parse_args().mode)
