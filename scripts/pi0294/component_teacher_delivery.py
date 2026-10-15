"""Derived saved-array diagnostics and labelled actual RGB playback only."""
import json,subprocess
from pathlib import Path
import numpy as np
from .task_capture_runtime import read,sha
from .component_teacher_audit import rows
from go2arx5_vla.trajectory.so3 import geodesic_angle


def diagnostics(root):
    from scripts.pi0292.durable import write
    series=Path(root)/'teacher-qualification-v2';cases=[];plot=[]
    for folder in sorted((series/'episodes').iterdir()):
        audit=read(folder/'INDEPENDENT_CPU_AUDIT.json');samples=rows(folder/'samples.jsonl')
        if not samples:
            cases.append(dict(id=folder.name,status=audit['status'],queries=0,
                diagnosis='PRETASK_READINESS_NO_TEACHER_ACTION_NOT_TASK_QUALITY'))
            continue
        intents=[r for r in rows(folder/'committed_intents.jsonl') if r['phase']=='SUBMITTED_REQUIRES_INSTALL_ACK']
        t=np.array([s['time_s'] for s in samples]);tcp=np.array([s['tcp'] for s in samples])
        target=np.asarray(intents[0]['persistent']['target']);queries=[];same_time=[]
        for j,intent in enumerate(intents):
            i=intent['index'];snap=intent['persistent'];cursor=np.asarray(snap['cursor']);actual=tcp[i]
            route=np.asarray(snap['route']);segment=snap['segment'];candidate=cursor.copy()
            blocked_position=False;blocked_angle=False
            if j and segment<len(route):
                delta=route[segment,:3,3]-cursor[:3,3];dist=np.linalg.norm(delta)
                candidate[:3,3]+=delta*min(1.,.06*.4/max(dist,1e-15))
                blocked_position=np.linalg.norm(candidate[:3,3]-actual[:3,3])>.024+1e-10
                blocked_angle=float(geodesic_angle(cursor[:3,:3],actual[:3,:3]))>.12+1e-10
            queries.append(dict(time_s=float(t[i]),segment=segment,cursor_world_xyz=cursor[:3,3].tolist(),
                actual_world_xyz=actual[:3,3].tolist(),cursor_actual_gap_m=float(np.linalg.norm(cursor[:3,3]-actual[:3,3])),
                next_candidate_outside_absolute_position_bound=bool(blocked_position),
                cursor_outside_angular_bound=bool(blocked_angle)))
        for i in range(1,len(samples)):
            intent=intents[(i-1)//8];action=np.asarray(intent['action_EEF13']);elapsed=i-intent['index']
            # H40 pose grid starts at query time. Prefix endpoint is action[8],
            # while each interval's preceding row is labelled indices0..7.
            ref=action[min(elapsed,39),:3];same_time.append(float(np.linalg.norm(ref-tcp[i,:3,3])))
        cursor_xyz=np.asarray([q['cursor_world_xyz'] for q in queries])
        qdistance=np.linalg.norm(np.diff(cursor_xyz,axis=0),axis=1)
        cursor_rotation=np.asarray([r['persistent']['cursor'] for r in intents])[:,:3,:3]
        angular_steps=[float(geodesic_angle(a,b)) for a,b in zip(cursor_rotation,cursor_rotation[1:])]
        case=dict(id=folder.name,status=audit['status'],queries=len(queries),
            query_route_segments=sorted({q['segment'] for q in queries}),
            cursor_nonzero_position_advances=int((qdistance>1e-10).sum()),
            cursor_total_position_advance_m=float(qdistance.sum()),
            cursor_total_rotation_advance_rad=float(sum(angular_steps)),
            queries_next_candidate_outside_absolute_position_bound=sum(q['next_candidate_outside_absolute_position_bound'] for q in queries),
            cursor_actual_gap_max_m=max(q['cursor_actual_gap_m'] for q in queries),
            previous_installed_reference_at_actual_sample_position_error_median_m=float(np.median(same_time)),
            previous_installed_reference_at_actual_sample_position_error_max_m=float(np.max(same_time)),
            local_reference_difference_is_not_final_goal_distance=True,
            new_physics=0,new_model_queries=0,reference_consumption='UNKNOWN',query_details=queries)
        cases.append(case)
        plot.append(dict(id=folder.name,status=audit['status'],time_s=t.tolist(),actual_TCP_xyz=tcp[:,:3,3].tolist(),
            goal_xyz=target[:3,3].tolist(),goal_distance_m=np.linalg.norm(tcp[:,:3,3]-target[:3,3],axis=1).tolist(),
            query_time_s=[q['time_s'] for q in queries],cursor_xyz=cursor_xyz.tolist(),
            previous_reference_error_m=same_time,object_xyz=[np.asarray(s['object_pose'])[:3,3].tolist() for s in samples]))
    report=dict(status='SAVED_ARRAY_DIAGNOSTICS_NOT_NEW_INTERVENTION',cases=cases,
        claim='BOUNDED_REFERENCE_CURSOR_STALL_IS_OBSERVED_NOT_UNIQUE_WBC_CAUSATION',new_physics=0,new_model_queries=0)
    write(series/'PERSISTENT_REFERENCE_DIAGNOSTICS.json',report)
    write(series/'PLOT_DATA.private.json',plot)
    return report,plot


def video(root,episode):
    from PIL import Image,ImageDraw
    from scripts.pi0292.durable import write
    folder=Path(root)/'teacher-qualification-v2/episodes'/episode
    audit=read(folder/'INDEPENDENT_CPU_AUDIT.json');samples=rows(folder/'samples.jsonl')
    target=folder/'ACTUAL_TEACHER_DEVELOPMENT.mp4'
    if target.exists():raise FileExistsError('FIRST_ACTUAL_VIDEO_PRESERVED')
    with (folder/'VIDEO_ENCODE.log').open('x') as log:
        p=subprocess.Popen(['/usr/bin/ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','512x348','-r','20','-i','pipe:0',
            '-an','-c:v','libx264','-threads','1','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE,stderr=log)
        try:
            for i,s in enumerate(samples):
                frame=Image.new('RGB',(512,348),'black');draw=ImageDraw.Draw(frame)
                for x,camera in ((0,'body_rgb'),(256,'wrist_rgb')):
                    with Image.open(folder/'images'/f'{i:06d}-{camera}.png') as image:frame.paste(image,(x,92))
                draw.text((5,3),'ACTUAL Go2+ARX5 | FROZEN WBC | NON REALTIME',fill='white')
                draw.text((5,19),episode,fill='white')
                label='ENGINEERING INCOMPLETE' if audit['engineering_incomplete'] else 'SHORT MILESTONES PASS' if audit['full_confirmation_eligible'] else 'VALID SHORT MILESTONE FAIL'
                draw.text((5,35),label+' | NOT A SUCCESS DEMO',fill=(255,140,80))
                draw.text((5,51),'t=%.2fs | stage=%s | SHORT DEVELOPMENT'%(s['time_s'],s['teacher_stage']),fill='white')
                draw.text((5,74),'BODY RGB',fill='white');draw.text((261,74),'WRIST RGB',fill='white')
                p.stdin.write(frame.tobytes())
            p.stdin.close()
            if p.wait(timeout=60):raise RuntimeError('ACTUAL_VIDEO_ENCODING_FAILURE')
        finally:
            if p.poll() is None:p.kill();p.wait()
    receipt=dict(status='ACTUAL_SAVED_RGB_ONLY',id=episode,frames=len(samples),hz=20,non_realtime=True,
        result_label=audit['status'],video_sha256=sha(target),bytes=target.stat().st_size,
        RGB_index_sha256=sha(folder/'RGB_HASHES.json'),public=False,new_physics=0,new_render=0,new_model_queries=0)
    write(folder/'ACTUAL_VIDEO_RECEIPT.json',receipt)
    return receipt
