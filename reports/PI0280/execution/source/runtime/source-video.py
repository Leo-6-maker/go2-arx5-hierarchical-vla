from pathlib import Path
import cv2,json,hashlib,numpy as np
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0280')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for result in sorted((r/'source/episodes').glob('*/attempt-*/RESULT.json')):
 folder=result.parent;source=folder/'observations.npz';out=folder/'source_ood_nonrealtime.mp4';receipt=folder/'VIDEO_OOD.json'
 if not source.exists():continue
 if receipt.exists():
  value=json.loads(receipt.read_text());assert digest(out)==value['video_sha256'] and digest(source)==value['original_observations_sha256'];rows.append(value);continue
 with np.load(source) as saved:
  a=saved['base_camera_raw'];b=saved['hand_camera_raw'];timestamps=saved['time_s']
 assert len(a)==len(b)==len(timestamps) and a.shape[1:]==b.shape[1:]==(128,128,3)
 writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),20.,(256,176));assert writer.isOpened()
 for base,wrist,t in zip(a,b,timestamps):
  frame=np.zeros((176,256,3),dtype=np.uint8);frame[48:]=cv2.cvtColor(np.concatenate([base,wrist],axis=1),cv2.COLOR_RGB2BGR)
  cv2.putText(frame,'OOD source-Panda | original frozen GR00T',(4,12),cv2.FONT_HERSHEY_SIMPLEX,.30,(255,255,255),1,cv2.LINE_AA)
  cv2.putText(frame,'NON_REALTIME | not WBC',(4,25),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1,cv2.LINE_AA)
  cv2.putText(frame,'base | wrist  task t=%+.2fs'%(float(t)-1.),(4,39),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1,cv2.LINE_AA)
  writer.write(frame)
 writer.release();cap=cv2.VideoCapture(str(out));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
 assert count==len(a) and abs(fps-20)<1e-6;cap.set(cv2.CAP_PROP_POS_FRAMES,count-1);okay,last=cap.read();cap.release();assert okay and last.shape==(176,256,3)
 value=dict(folder=str(folder),video_path=str(out),video_sha256=digest(out),original_observations_path=str(source),original_observations_sha256=digest(source),frames=count,fps=fps,native_times_s=[float(timestamps[0]),float(timestamps[-1])],task_time_origin_offset_s=1.,source='preserved raw128 real base/wrist RGB side by side',new_RGB_pairs=0,new_physics_steps=0,new_model_queries=0,labels=['OOD source-Panda','original frozen GR00T','NON_REALTIME','not WBC'])
 receipt.write_text(json.dumps(value,indent=2)+'\n');rows.append(value)
(r/'evidence/SOURCE_VIDEOS.json').write_text(json.dumps(dict(rows=rows,new_model_queries=0,new_physics_steps=0,new_RGB_pairs=0),indent=2)+'\n')
print(json.dumps(dict(videos=len(rows),frames=[v['frames'] for v in rows])))
