"""Videos from saved actual fixture RGB only; no renderer or simulation calls."""
import hashlib,json,os
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0281/wbc')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
    out=ROOT/'videos';out.mkdir(exist_ok=True);receipts=[];samples=[]
    for folder in sorted(ROOT.iterdir()):
        if not (folder/'frames.npz').exists():continue
        result=json.loads((folder/'RESULT.json').read_text())
        with np.load(folder/'frames.npz',allow_pickle=False) as data:body=data['body'];wrist=data['wrist'];t=data['time']
        assert len(body)==len(wrist)==len(t)==result['counts']['RGB_pairs']
        h=max(body.shape[1],wrist.shape[1]);w=body.shape[2]+wrist.shape[2]
        target=out/(folder.name+'.mp4');writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),10.,(w,h+48));assert writer.isOpened()
        invalid='REPAIR_NOT_INSTALLED' if folder.name.startswith('repair') and '-engineering-' not in folder.name else result['status']
        for a,b,time in zip(body,wrist,t):
            frame=np.zeros((h+48,w,3),dtype=np.uint8)
            frame[48:48+a.shape[0],:a.shape[1]]=cv2.cvtColor(a,cv2.COLOR_RGB2BGR)
            frame[48:48+b.shape[0],a.shape[1]:]=cv2.cvtColor(b,cv2.COLOR_RGB2BGR)
            for text,y in [(folder.name,12),('FROZEN WBC FIXTURE | NON_REALTIME',25),(invalid+' t=%.2fs'%time,39)]:cv2.putText(frame,text,(4,y),cv2.FONT_HERSHEY_SIMPLEX,.30,(255,255,255),1,cv2.LINE_AA)
            writer.write(frame)
        writer.release();cap=cv2.VideoCapture(str(target));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(t)
        selected=[]
        for index in (0,len(t)//2,len(t)-1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();assert ok;selected.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((512,224)))
        cap.release();samples.append((folder.name,selected))
        receipts.append(dict(condition=folder.name,status=result['status'],annotation=invalid,video=str(target),sha256=sha(target),frames=len(t),fps=10,actual_t_s=[float(t[0]),float(t[-1])],source_frames_sha256=sha(folder/'frames.npz'),new_render_calls=0,new_physics=0,new_actor_calls=0))
    sheet=Image.new('RGB',(1536,250*len(samples)),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,images) in enumerate(samples):
        draw.text((4,i*250+4),name+' first / middle / last',fill='black')
        for j,im in enumerate(images):sheet.paste(im,(512*j,i*250+26))
    p=out/'WBC_CONTACT_SHEET.png';sheet.save(p)
    (ROOT/'WBC_VIDEOS.json').write_text(json.dumps(dict(videos=receipts,contact_sheet=str(p),sheet_sha256=sha(p),new_model_queries=0,new_render_calls=0,new_physics=0),indent=2)+'\n')
    print(json.dumps(dict(videos=len(receipts),new_render_calls=0)))
if __name__=='__main__':main()
