"""PI0257: stored evidence in, presentation out. Default mode cannot load a model."""
import argparse
import hashlib
import json
import math
import subprocess
import os
import shutil
import re
from pathlib import Path

AUTH = 'PI0257_CORELAB_VIDEO_WORKFLOW_AND_EVIDENCE_EXPORT_GO_V1'
DEFAULT_ROOT = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0257_v2')
VIDEO_BIN = Path('/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/stage13c/runtime_env/public_native/python_env/bin')
FFMPEG = Path(os.environ.get('PI0257_FFMPEG','/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0257/deps/ffmpeg'))
if not FFMPEG.exists():
    found=shutil.which('ffmpeg')
    if not found:
        try:
            import imageio_ffmpeg
            found=imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError: pass
    if found: FFMPEG=Path(found)

def saved_path(out, source):
    """Archived absolute provenance stays intact; resolve saved bytes after PC transfer."""
    value=str(source).replace('\\','/')
    return out/value.split('/delivery/',1)[1] if '/delivery/' in value else Path(source)

def policy_class(queries):
    return 'ONE_CHUNK_POLICY_EXECUTION' if queries==1 else ('RECEDING_POLICY_EXECUTION' if queries>1 else 'EXPERT_REFERENCE')

def scientific_key(tree, processor, statistics, decoder, task, contract):
    return hashlib.sha256(json.dumps([tree,processor,statistics,decoder,task,*contract],sort_keys=True).encode()).hexdigest()

def admit_query(attempts, valid, heads):
    assert 0<=valid<=attempts and attempts<104 and valid<96
    assert attempts-valid<8 and heads+4<=416,'Recorded-input technical-attempt/head budget exhausted'

def matching_metrics(achieved,reference,target,actual_time,reference_time,actual_frame,reference_frame):
    import numpy as np
    assert actual_frame==reference_frame=='world' and np.array_equal(actual_time,reference_time),'UNAVAILABLE: unmatched frames or clocks'
    assert achieved.shape==reference.shape and achieved.shape[-1]==3
    return np.linalg.norm(achieved-target,axis=1),np.linalg.norm(achieved-reference,axis=1)

def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def frame_index(times, now):
    import numpy as np
    assert len(times) and np.all(np.diff(times)>0)
    return max(0,min(len(times)-1,int(np.searchsorted(times,now,side='right')-1)))
def text(canvas, words, x, y, scale=.55, color=(220,230,240)):
    import cv2
    cv2.putText(canvas,str(words),(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
def encoder(path, fps=30):
    return subprocess.Popen([str(FFMPEG),'-v','error','-y','-f','rawvideo','-pixel_format','bgr24',
        '-video_size','1280x720','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2',
        '-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
def decode(path):
    import cv2
    ffprobe=os.environ.get('PI0257_FFPROBE') or (str(VIDEO_BIN/'ffprobe') if (VIDEO_BIN/'ffprobe').exists() else shutil.which('ffprobe'))
    if ffprobe:
        probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_streams','-show_format','-of','json',str(path)]))
        stamps=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]))['frames']
        pts=[float(f['best_effort_timestamp_time']) for f in stamps]
        subprocess.run([str(FFMPEG),'-v','error','-xerror','-i',str(path),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
    else:
        process=subprocess.run([str(FFMPEG),'-v','info','-xerror','-i',str(path),'-vf','showinfo','-f','null','-'],check=True,capture_output=True,text=True)
        pts=[float(x) for x in re.findall(r'\bpts_time:([0-9.eE+-]+)',process.stderr)]
        probe={'timestamp_backend':'ffmpeg showinfo; ffprobe unavailable'}
    assert pts and all(b>a for a,b in zip(pts,pts[1:]))
    cap=cv2.VideoCapture(str(path));declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);count=0;shape=None
    while True:
        ok,img=cap.read()
        if not ok:break
        shape=list(img.shape);count+=1
    cap.release();assert count>0 and count==declared==len(pts)
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,decoded_frames=count,shape=shape,fps=fps,duration_s=count/fps,first_pts_s=pts[0],last_pts_s=pts[-1],monotonic_pts=True,probe=probe)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',nargs='?',default='plan',choices=['plan','export','infer-recorded','render','verify','verify-portable','verify-corelab','package','run-episode']);p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--playback',type=float,default=.25);p.add_argument('--authorization');p.add_argument('--mode',default='archived');p.add_argument('--kind',choices=['all','actual','predictions','preparation'],default='all');a=p.parse_args()
    if a.command=='plan': print(json.dumps(dict(authorization=AUTH,commands=['plan','export','infer-recorded','render','verify','package'],maximum_new_query_attempts=104,new_physics_allowed=0,selected_checkpoint='NONE')))
    elif a.command=='run-episode': p.error('GUARD_STUB_ONLY: physical runner NOT_CONNECTED; PI0256/PI0257 do not authorize NEW learned-policy physics; separate qualified episode authority and read-only observer proof required')
    elif a.command=='infer-recorded':
        from recorded_adapter import infer
        infer(a.root,a.authorization)
    elif a.command=='export':
        assert a.mode=='archived'
        from bundle_adapter import export
        export(a.root)
    elif a.command=='render':
        assert 0<a.playback<=4
        from bundle_adapter import render
        render(a.root,a.kind,a.playback)
    elif a.command in ('verify','verify-portable'):
        from portable_verify import verify
        verify(a.root)
    elif a.command=='verify-corelab':
        from bundle_adapter import verify_corelab
        verify_corelab(a.root)
    elif a.command=='package':
        from bundle_adapter import package
        package(a.root)

if __name__=='__main__': main()
