import json,time,resource,hashlib
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0281/source');video=r/'videos';video.mkdir(exist_ok=True)
started=time.monotonic();cpu=resource.getrusage(resource.RUSAGE_SELF);rows=[]
labels={'ORIGINAL_BASE_JOINT':'SOURCE JOINT','FK_OF_JOINT_COMMAND_REPLAY':'FK REPLAY','EXPERT_DROID_BRIDGE_ROUNDTRIP':'EXPERT CONTROL'}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for lane,label in labels.items():
 for seed in range(10000,10004):
  folder=r/'episodes'/(lane+'-'+str(seed))/'attempt-000';result=folder/'RESULT.json'
  if not result.exists():continue
  record=json.loads(result.read_text());source=folder/'observations.npz';target=video/(lane+'-'+str(seed)+'.mp4');receipt=video/(lane+'-'+str(seed)+'.json')
  if receipt.exists():
   value=json.loads(receipt.read_text());assert digest(target)==value['video_sha256'] and digest(source)==value['source_observation_sha256'];rows.append(value);continue
  with np.load(source) as saved:a=saved['base_camera_raw'];b=saved['hand_camera_raw'];ts=saved['time_s']
  assert a.shape==b.shape==(len(ts),128,128,3) and len(ts)==record['RGB_pairs']
  writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),20.,(256,176));assert writer.isOpened()
  for base,wrist,t in zip(a,b,ts):
   frame=np.zeros((176,256,3),dtype=np.uint8);frame[48:]=cv2.cvtColor(np.concatenate([base,wrist],axis=1),cv2.COLOR_RGB2BGR)
   for text,y in [(label+' | OOD SOURCE-PANDA',12),('NON_REALTIME | not WBC | no takeover',25),('base | wrist task t=%+.2fs'%(float(t)-1),39)]:cv2.putText(frame,text,(4,y),cv2.FONT_HERSHEY_SIMPLEX,.31,(255,255,255),1,cv2.LINE_AA)
   writer.write(frame)
  writer.release();cap=cv2.VideoCapture(str(target));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(ts) and cap.get(cv2.CAP_PROP_FPS)==20.;samples=[0,len(ts)//2,len(ts)-1]
  for index in samples:
   cap.set(cv2.CAP_PROP_POS_FRAMES,index);okay,frame=cap.read();assert okay and frame.shape==(176,256,3)
  cap.release();value=dict(lane=lane,label=label,condition=seed,status=record['status'],persistent_success=record['persistent_success'],video_path=str(target),video_sha256=digest(target),source_observations=str(source),source_observation_sha256=digest(source),frames=len(ts),fps=20,native_time_s=[float(ts[0]),float(ts[-1])],task_origin_offset_s=1,labels=['OOD SOURCE-PANDA',label,'NON_REALTIME','not WBC','no takeover'],new_RGB_pairs=0,new_physics_steps=0,new_model_queries=0)
  receipt.write_text(json.dumps(value,indent=2)+'\n');rows.append(value)
sheet=Image.new('RGB',(768,len(rows)*202),'white');draw=ImageDraw.Draw(sheet)
for row,value in enumerate(rows):
 cap=cv2.VideoCapture(value['video_path']);count=value['frames'];draw.text((4,row*202+5),value['label']+' '+str(value['condition'])+' | first / middle / last | '+value['status'],fill='black')
 for column,index in enumerate([0,count//2,count-1]):
  cap.set(cv2.CAP_PROP_POS_FRAMES,index);okay,frame=cap.read();assert okay;sheet.paste(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),(column*256,row*202+26))
 cap.release()
sheet_path=video/'SOURCE_THREE_ROUTE_CONTACT_SHEET.png';sheet.save(sheet_path);after=resource.getrusage(resource.RUSAGE_SELF)
receipt=dict(rows=rows,contact_sheet=str(sheet_path),contact_sheet_sha256=digest(sheet_path),elapsed_s=time.monotonic()-started,user_CPU_s=after.ru_utime-cpu.ru_utime,system_CPU_s=after.ru_stime-cpu.ru_stime,maxrss_KiB=after.ru_maxrss,new_model_queries=0,new_physics_steps=0,new_RGB_pairs=0,new_render_calls=0,source='CPU decode preserved real128 base/wristRGB, all videos NON_REALTIME/no takeover')
(r/'evidence/SOURCE_VIDEOS.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(videos=len(rows),elapsed_s=receipt['elapsed_s'],user_CPU_s=receipt['user_CPU_s'],system_CPU_s=receipt['system_CPU_s'])))

