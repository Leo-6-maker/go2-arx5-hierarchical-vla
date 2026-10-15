"""Private annotated playback from actual saved RGB only; no new rendering."""
import json
from pathlib import Path
import subprocess
from .task_capture_runtime import cpu_stage,read,sha


def run(root,episode):
    from PIL import Image,ImageDraw,ImageFont
    from scripts.pi0292.durable import write
    root=Path(root);folder=root/'episodes'/episode
    with cpu_stage(root,'VIDEO_'+episode,cores=1,cpu_s_cap=600):
        audit=read(folder/'FULLBODY_CPU_AUDIT.json')
        target=folder/'ACTUAL_FULLBODY_TASK.mp4'
        if target.exists():raise FileExistsError('ACTUAL_VIDEO_EXISTS')
        samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
        log=(folder/'VIDEO_ENCODE.log').open('x')
        proc=subprocess.Popen(['/usr/bin/ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','512x332','-r','20','-i','pipe:0',
            '-an','-c:v','libx264','-threads','1','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE,stderr=log)
        try:
            for i,s in enumerate(samples):
                frame=Image.new('RGB',(512,332),'black')
                for x,cam in ((0,'body_rgb'),(256,'wrist_rgb')):
                    with Image.open(folder/'images'/f'{i:06d}-{cam}.png') as im:frame.paste(im,(x,76))
                d=ImageDraw.Draw(frame)
                d.text((6,4),'ACTUAL FULL Go2+ARX5 | FROZEN WBC | NON REALTIME',fill='white')
                d.text((6,19),episode,fill='white')
                d.text((6,34),audit['status']+' | t=%.2fs'%s['time_s'],fill=(255,140,80))
                xyz=s['object_pose'];p=[xyz[k][3] for k in range(3)]
                d.text((6,49),'Actual cube xyz: %.3f %.3f %.3f m'%tuple(p),fill='white')
                d.text((6,63),'BODY RGB',fill='white');d.text((262,63),'WRIST RGB',fill='white')
                proc.stdin.write(frame.tobytes())
            proc.stdin.close();status=proc.wait(timeout=60)
            if status:raise RuntimeError('PRIVATE_ACTUAL_VIDEO_ENCODING_FAILED')
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            log.close()
        value=dict(status='ACTUAL_SAVED_RGB_PLAYBACK_COMPLETE',frames=len(samples),playback_hz=20,wall_clock_realtime=False,
            quality=audit['status'],video_sha256=sha(target),bytes=target.stat().st_size,
            actual_RGB_hash_index_sha256=audit['RGB_index_sha256'],public=False,new_physics=0,new_render=0,new_model_queries=0)
        write(folder/'ACTUAL_VIDEO_RECEIPT.json',value);print(json.dumps(value))
