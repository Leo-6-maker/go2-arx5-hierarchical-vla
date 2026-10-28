"""Render recorded synthetic cameras only; never launch a simulator."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from pi0255_motion3 import sha,write


def main(root):
    report=json.loads((root/'evidence/pilot_cpu_replay.json').read_text());assert report['status']=='PILOTS_COMPLETE'
    chosen=[]
    # Fixed IDs include each actual failure class and both recording versions.
    ids=('pi0255-family-000-M1-c0','pi0255-family-002-M1-c0','pi0255-family-004-M1-c0','pi0255-family-005-M1-c0')
    output=root/'evidence/videos';output.mkdir(exist_ok=True)
    for identity in ids:
        directory=root/'episodes'/identity/'attempt-00';result=json.loads((directory/'result.json').read_text())
        frames=sorted((directory/'settling_frames').glob('*.npz'))
        terminal=directory/'terminal_images.npz'
        if terminal.exists():frames.append(terminal)
        assert frames
        path=output/(identity+'.mp4');writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10.,(576,340));assert writer.isOpened()
        timestamps=[]
        for source in frames:
            with np.load(source,allow_pickle=False) as data:
                image=np.concatenate([data['body_rgb'],data['wrist_rgb']],axis=1);now=float(data['actual_sim_time_s'])
            assert image.shape==(96,192,3) and image.dtype==np.uint8
            canvas=np.zeros((340,576,3),np.uint8);canvas[26:314]=cv2.resize(cv2.cvtColor(image,cv2.COLOR_RGB2BGR),(576,288),interpolation=cv2.INTER_NEAREST)
            label='%s t=%.3fs'%(identity,now);cv2.putText(canvas,label,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
            status=result['failure']+(' | terminal may be post-reset' if source==terminal else '')
            cv2.putText(canvas,status,(4,332),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
            writer.write(canvas);timestamps.append(dict(source=str(source.relative_to(root)),sha256=sha(source),actual_sim_time_s=now,terminal=source==terminal))
        writer.release();capture=cv2.VideoCapture(str(path));assert capture.isOpened()
        decoded=0
        while True:
            ok,frame=capture.read()
            if not ok:break
            assert frame.shape==(340,576,3);decoded+=1
        capture.release();assert decoded==len(frames)
        chosen.append(dict(episode=identity,failure=result['failure'],path=str(path.relative_to(root)),sha256=sha(path),decoded_frames=decoded,
            playback_fps=10,actual_timestamps=timestamps,task_phase_entered=False,terminal_post_reset_caveat=True))
    assert {x['failure'] for x in chosen}==set(report['failure_classes'])
    write(root/'evidence/video_manifest.json',dict(status='PASS_RECORDED_CAMERA_VIDEO_DECODE',examples=chosen,new_physics=0,new_GR00T_queries=0))
    print(json.dumps(dict(status='PASS_RECORDED_CAMERA_VIDEO_DECODE',videos=len(chosen))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
