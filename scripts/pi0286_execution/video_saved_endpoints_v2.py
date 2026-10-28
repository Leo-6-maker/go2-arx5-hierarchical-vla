"""Closed endpoint saved RGB evidence: existing PI0285 numpy/cv2/ffmpeg route, no simulation."""
import argparse,hashlib,html,json,os,resource,subprocess,time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
OUT=ROOT/'evidence/video/endpoint-saved-rgb/V2-complete-text'
GROUPS=('LEARNER16','CORRECTION03S16','CORRECTION12S16')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)


def plan(point,panel):
    assert point in ('C0_REPLAY1024-1024','C1_CORRECT1024-1024') and panel in ('TRAIN','DEV','CONFIRM_C')
    group=point+'_'+panel
    source=ROOT/'source/evidence'/(group+'.json');cpu=ROOT/'audit'/(group+'_CPU.json')
    s=json.loads(source.read_text());a=json.loads(cpu.read_text())
    assert a['source_sha256']==sha(source) and a['status'].startswith('PASS_INDEPENDENT_CPU_')
    requested={'TRAIN':8,'DEV':16,'CONFIRM_C':32}[panel]
    expected=list(range({'TRAIN':285100,'DEV':285200,'CONFIRM_C':285500}[panel],{'TRAIN':285100,'DEV':285200,'CONFIRM_C':285500}[panel]+requested))
    assert len(s['rows'])==requested and sorted(r['layout'] for r in s['rows'])==expected
    proofs={r['source_result_sha256']:r for r in a['rows']};rows=[]
    for i,row in enumerate(sorted(s['rows'],key=lambda r:r['layout'])):
        folder=Path(row['folder']);result=folder/'RESULT.json';card=folder/'EPISODE_CARD.json'
        assert json.loads(result.read_text())==row and row['status']=='VALID_SOURCE_COMPLETE'
        h=sha(result);assert h in proofs and proofs[h]['persistent_success']==row['persistent_success']
        cv=json.loads(card.read_text());assert cv['task']=='T1' and cv['layout']==row['layout']
        rows.append(dict(point=point,panel=panel,task='T1',layout=row['layout'],folder=str(folder),status=row['status'],
          physical_persistent=bool(row['persistent_success']),hold_pass=row['task_score']['hold_pass'],original_stack_hold_pass=row['task_score']['original_stack_hold_pass'],
          source_result_sha256=h,source_card_sha256=sha(card),source_observation_sha256=row['files']['observations.npz'],
          source_physics_sha256=row['files']['physics.npz'],source_file_hashes=row['files'],CPU_proof_sha256=sha(cpu),segment=i//4,tile=i%4,
          segment_start_video_s=i//4*33,NON_REALTIME=True,domain='SOURCE_PANDA_NATIVE_PD_NOT_TARGET_WBC'))
    v=dict(status='ACTUAL_INDEPENDENT_CPU_CLOSED_FIXED_PANEL_SAVED_RGB_INDEX',point=point,panel=panel,requested=requested,
      physical_success=sum(r['persistent_success'] for r in s['rows']),rows=rows,source_path=str(source),source_sha256=sha(source),
      CPU_path=str(cpu),CPU_sha256=sha(cpu),model_view_identity=json.loads((ROOT/'design/MODEL_POINTS.json').read_text())[point],
      selected='NONE',deployment='NONE',new_render=0,new_physics=0,new_model_calls=0)
    OUT.mkdir(parents=True,exist_ok=True);index=OUT/(group+'_SOURCE_INDEX.json');write(index,v);return group,rows,index

def text(canvas,label,xy,color=(255,255,255),scale=.52):
    cv2.putText(canvas,label,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def encode(group,rows,index):
    out=OUT/(group+'.mp4');receipt=OUT/(group+'_VIDEO.json')
    if receipt.exists():
        v=json.loads(receipt.read_text());assert sha(out)==v['sha256'];return v
    assert not out.exists(),'Preserve existing unresolved encoder output'
    partial=out.with_suffix('.partial.mp4');assert not partial.exists()
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x720','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','3','-filter_threads','1','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE);segments=[]
    try:
        for start in range(0,len(rows),4):
            batch=rows[start:start+4];frames=[]
            for row in batch:
                path=Path(row['folder'])/'observations.npz';assert sha(path)==row['source_observation_sha256']
                with np.load(path,allow_pickle=False) as f:base=f['base_camera_raw'].copy();wrist=f['hand_camera_raw'].copy();clock=f['time_s'].copy()
                assert base.shape==wrist.shape==(660,128,128,3) and base.dtype==wrist.dtype==np.uint8
                assert np.allclose(clock,np.arange(660)*.05,rtol=0,atol=1e-8)
                frames.append((base,wrist))
            for i in range(0,660,2):
                canvas=np.zeros((720,1024,3),dtype=np.uint8)
                text(canvas,'PI0286 T1 | SOURCE PANDA native PD | NON-REALTIME | saved BASE + WRIST', (12,23),scale=.59)
                text(canvas,f'{group} | all{len(rows)} fixed layouts | page{start//4+1}/{len(rows)//4} | Actual saved outcome | NOT qualification',(12,48),scale=.51)
                for tile,(row,(base,wrist)) in enumerate(zip(batch,frames)):
                    x=(tile%2)*512;y=64+(tile//2)*328
                    canvas[y:y+256,x:x+256]=cv2.resize(base[i],(256,256),interpolation=cv2.INTER_NEAREST)
                    canvas[y:y+256,x+256:x+512]=cv2.resize(wrist[i],(256,256),interpolation=cv2.INTER_NEAREST)
                    color=(80,240,100) if row['physical_persistent'] else (255,100,100)
                    text(canvas,f'T1 layout{row["layout"]} MODEL t={i*.05-1:.1f}s',(x+8,y+275))
                    text(canvas,f'Physical {"PASS" if row["physical_persistent"] else "FAIL"} T1 hold{row["hold_pass"]}/41',(x+8,y+296),color)
                    text(canvas,f'Original stack hold{row["original_stack_hold_pass"]}/41 | Saved {row["status"]}',(x+8,y+317),scale=.45)
                process.stdin.write(canvas.tobytes())
            segments.append(dict(segment=start//4,start_video_s=start//4*33,end_video_s=(start//4+1)*33,tiles=batch))
            write(OUT/'ENCODER_PROGRESS.json',dict(group=group,completed_segments=len(segments),source_episodes_seen=(start+4),epoch=time.time(),new_model_calls=0,new_physics=0))
        process.stdin.close();assert process.wait()==0;partial.replace(out)
    except BaseException:
        if process.stdin and not process.stdin.closed:process.stdin.close()
        process.wait();raise
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate:format=duration','-of','json',str(out)],text=True))
    stream=probe['streams'][0];assert stream['width']==1024 and stream['height']==720 and int(stream['nb_frames'])==len(rows)//4*330 and abs(float(probe['format']['duration'])-len(rows)//4*33)<.01
    value=dict(status='PASS_ALL_FIXED_CONDITIONS_ACTUAL_TWO_VIEW_SAVED_VIDEO_NOT_QUALIFICATION',group=group,sha256=sha(out),bytes=out.stat().st_size,
        frames=len(rows)//4*330,fps=10,duration_s=len(rows)//4*33,width=1024,height=720,segments=segments,source_index_path=str(index),source_index_sha256=sha(index),new_render=0,new_physics=0,new_model_calls=0)
    write(receipt,value);return value


def main(point,panel):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    # Owner calls only after scientific evaluation; this helper never starts itself.
    for endpoint in ('C0_REPLAY1024-1024','C1_CORRECT1024-1024'):
        assert (ROOT/'runtime'/(endpoint+'_endpoint_EVALUATION_CLOSED.json')).is_file()
    cv2.setNumThreads(1);resource.setrlimit(resource.RLIMIT_AS,(16<<30,16<<30));os.nice(10)
    assert (ROOT/'runtime'/(point+'_endpoint_EVALUATION_CLOSED.json')).is_file()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:argv=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        assert not any(str(ROOT/'code'/n).encode() in argv for n in ('evaluate.py','policy.py','model_worker.py')),'Do not compete with scientific mainline'
    group,rows,index=plan(point,panel);v=encode(group,rows,index)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',str(OUT/(group+'.mp4')),'-f','null','-'],check=True)
    result=dict(status='PASS_ALL_FIXED_PANEL_SAVED_RGB_FULL_DECODE_ACTUAL_CPU_BOUND_OUTCOMES',point=point,panel=panel,
      video_sha256=v['sha256'],source_index_sha256=sha(index),conditions=len(rows),new_render=0,new_physics=0,new_model_calls=0)
    write(OUT/(group+'_DELIVERY.json'),result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',required=True);p.add_argument('--panel',choices=('TRAIN','DEV','CONFIRM_C'),required=True);a=p.parse_args();main(a.point,a.panel)
