"""Fixed first-family montage from saved observations, including failures and late drift."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from infer_worker import ROOT,BASE,read,sha,write,admitted
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0269')


def main():
    admitted();assert read(RAW/'delivery/PRIMARY_VERDICT.json')['virtual_episodes']==192
    families=[f for f in read(BASE/'PI0267/design/SCENE_CARD.json')['families'] if f['split']=='NEW_SIM_EVAL']
    fid=families[0]['family_id'];out=RAW/'delivery'
    with (out/'VIRTUAL192.csv').open() as stream:
        reliability={(r['point'],r['family'],r['task'],r['command'],r['backend']):r['reliable']=='True' for r in csv.DictReader(stream)}
    assert len(reliability)==192
    for point in ('UNIFORM4096','PREFIX4096'):
        path=out/('PI0269_'+point+'_VIRTUAL.mp4');receipt=out/(point+'-VIDEO_RECEIPT.json')
        if receipt.exists():assert sha(path)==read(receipt)['sha256'];continue
        source=RAW/'virtual'/point
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),5.,(1024,640));assert writer.isOpened()
        frames=0;chapters=[]
        try:
            for task in ('M1','M3'):
                for command in (0,1):
                    for backend in ('E0','E1'):
                        folder=source/f'{fid}-{task}-c{command}-{backend}';result=read(folder/'result.json');start=frames
                        stable=reliability[point,fid,task,str(command),backend]
                        assert result['validity']=='VALID'
                        for p in sorted((folder/'queries').glob('*/observation.npz')):
                            with np.load(p) as z:
                                canvas=np.full((640,1024,3),18,np.uint8)
                                for i,key in enumerate(('body_rgb','wrist_rgb')):
                                    image=cv2.resize(z[key],(480,480),interpolation=cv2.INTER_NEAREST)
                                    canvas[110:590,16+i*512:496+i*512]=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                                labels=['VIRTUAL_EEF_KINEMATIC / NOT_WBC / NON_REALTIME_LOCKSTEP',f'{point} {fid} {task} c{command} {backend}',
                                    f"actual t={float(z['timestamp']):.2f}s /2x playback / V3={result['success']} / reliable40s={stable}",
                                    f"stop={result['stop_reason']} / fixed first family, all8 cases, no score-based selection"]
                            for y,text in zip((24,49,74,99),labels):cv2.putText(canvas,text,(18,y),cv2.FONT_HERSHEY_SIMPLEX,.46,(240,240,240),1)
                            cv2.putText(canvas,'Actual96x96 RGB / no re-render / all192 original-start outcomes retained / selected=NONE',(18,624),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1)
                            writer.write(canvas);frames+=1
                        chapters.append(dict(case=folder.name,first_frame=start,last_frame=frames-1,original_success=result['success'],reliable_40s=stable,stop_reason=result['stop_reason'],result_sha256=sha(folder/'result.json')))
        finally:writer.release()
        assert frames>0 and len(chapters)==8
        capture=cv2.VideoCapture(str(path));decoded=0
        while True:
            okay,frame=capture.read()
            if not okay:break
            assert frame.shape==(640,1024,3);decoded+=1
        capture.release();assert frames==decoded
        write(receipt,dict(status='PASS_RECORDED_RGB_VIDEO_FULL_DECODE',point=point,file=str(path),sha256=sha(path),frames=frames,fps=5.,chapters=chapters,
            qualification_summary_csv_sha256=sha(out/'VIRTUAL192.csv'),
            selection='First frozen EVAL family, both tasks/commands/backends, entire recorded observation stream',model_queries=0,new_renders=0,physics_steps=0,
            scope='VIRTUAL / NOT_WBC / NON_REALTIME',selected_checkpoint='NONE'))
        print(point,frames,'SAVED_RGB_ONLY',flush=True)


if __name__=='__main__':main()
