"""Historical world telemetry plots; never correct or overwrite the old TCP record."""
import math
import cv2
import numpy as np
from eef_video_workflow import read,write,rows,text,encoder,decode,frame_index,matching_metrics
from plotting import plot,padded,COLORS,GREY

def render(root,playback=.25):
    out=root/'delivery';base=out/'actual_evidence';gm=read(base/'G/A/output/PI0252_G_ONE_CHUNK_MANIFEST.json');em=read(base/'controls/A_attempt2/control.json');request=read(base/'G/A/mailbox/requests/model-0000.json');clips=[];new=[]
    for label,relative,clock,ap,rp,target,verdict,gate in [('G2000','G/A/output','native_reference_local_time','achieved_pos','desired_pos',gm['protocol']['target_position_world_m'],gm['qualification'],gm['model_reference']['safety_gate']['decision']),('EXPERT A_attempt2','controls/A_attempt2','reference_time_s','achieved_world_m','desired_world_m',em['target_world_m'],em['status'],em['gate']['decision'])]:
        rr=rows(base/relative/('model_receding_rows.jsonl' if label=='G2000' else 'rows.jsonl'))
        with np.load(base/relative/'actual_camera_frames.npz',allow_pickle=False) as z:frames=np.concatenate([z['body_rgb'],z['wrist_rgb']],axis=2)
        times=np.r_[0.,[r[clock] for r in rr]];assert np.all(np.diff(times)>0) and len(frames)==len(times) and not any(r['reset_buf'] for r in rr)
        achieved=np.array([r[ap] for r in rr]);reference=np.array([r[rp] for r in rr]);target=np.array(target)
        distance,tracking=matching_metrics(achieved,reference,target,times[1:],times[1:],'world','world');refdistance=np.linalg.norm(reference-target,axis=1)
        item=dict(model=label,sampled_native_reference_source=relative+'/rows: desired world at control ticks',sampled_native_reference_count=len(rr),sampled_native_reference_time_range_s=[float(times[1]),float(times[-1])],sampled_native_reference_minimum_target_m=float(refdistance.min()),sampled_native_reference_final_target_m=float(refdistance[-1]),sampled_native_reference_max_linear_speed_mps=float((np.linalg.norm(np.diff(reference,axis=0),axis=1)/np.diff(times[1:])).max()),achieved_minimum_target_m=float(distance.min()),achieved_final_target_m=float(distance[-1]),same_tick_tracking_mean_m=float(tracking.mean()),first_hit_35mm_s=float(times[1:][distance<=.035][0]) if np.any(distance<=.035) else None,origin='HISTORICAL_AS_RECORDED; no cache-corrected measurement')
        if label=='G2000':
            raw=gm['model_reference']['decoded_world_target_stats'];item.update(raw_h40_source='Original PI0252 model_reference.decoded_world_target_stats; original 40-point output, not control-tick resampling',raw_h40_count=raw['count'],raw_h40_time_range_s=[0.,1.95],raw_h40_minimum_target_m=raw['min_distance_m'],raw_h40_final_target_m=raw['endpoint_distance_m'],raw_h40_max_linear_speed_mps=raw['max_linear_speed_mps'])
        item['sampled_native_reference_source']=relative+('/model_receding_rows.jsonl' if label=='G2000' else '/rows.jsonl');item['sampled_native_reference_field']=rp
        new.append(item);clips.append(dict(label=label,relative=relative,frames=frames,times=times,achieved=achieved,reference=reference,target=target,distance=distance,refdistance=refdistance,tracking=tracking,verdict=verdict,gate=gate))
    xyz=np.concatenate([c[k] for c in clips for k in ('reference','achieved')]+[c['target'][None] for c in clips]);limits=[padded(xyz[:,i]) for i in range(3)];end=max(c['times'][-1] for c in clips);ymax=math.ceil(max(np.max(c[k])*1000 for c in clips for k in ('distance','refdistance','tracking'))/50)*50
    file=out/('01_G_vs_expert_actual_one_chunk.mp4' if playback==.25 else '01_G_vs_expert_actual_1x.mp4');n=math.ceil(end/playback*30)+1;proc=encoder(file);mapping=[]
    try:
        for k in range(n):
            now=k/30*playback;canvas=np.full((720,1280,3),(26,21,17),np.uint8)
            text(canvas,'ARCHIVED ACTUAL EXECUTION | reference vs achieved TCP | historical telemetry',20,30,.66)
            text(canvas,f"{request['language']} | TRAIN A / C0 | {playback:g}x | reference clock {now:.3f}s",20,58,.55)
            text(canvas,'Separate runs aligned at initial reference bind. Complete stored interval; earlier preparation unavailable.',20,82,.47)
            entry=dict(output_frame=k,output_time_s=k/30,reference_elapsed_s=now,panels=[])
            for j,c in enumerate(clips):
                x=20+640*j;ix=frame_index(c['times'],now);text(canvas,c['label']+' / historical '+c['verdict'],x,111,.58,COLORS[j])
                canvas[132:292,x:x+320]=cv2.resize(cv2.cvtColor(c['frames'][ix],cv2.COLOR_RGB2BGR),(320,160),interpolation=cv2.INTER_NEAREST)
                labels=['BODY / WRIST: 96x96 each',f"Source t={c['times'][ix]:.4f}s"+(' END' if now>=c['times'][-1] else ''),'G@efecea485ae7 / H40 dt=.05' if j==0 else 'Expert + WBC / H40 dt=.05','Noise 22301 / SHA585570a70d' if j==0 else 'Noise N/A / reset seed22301','Query1 / heads4 / handoff1' if j==0 else 'Query0 / expert initial bind1','Updates0 / Gate '+c['gate']]
                for line,label in enumerate(labels):text(canvas,label,x+333,145+line*27,.40)
                value='Pre-control: no telemetry sample' if not ix else f"Target {c['distance'][ix-1]*1000:.1f}mm | tracking {c['tracking'][ix-1]*1000:.1f}mm"
                text(canvas,value,x,319,.48)
                upto=ix
                for a,v in enumerate((1,2)):
                    series=[(c['reference'][:,0],c['reference'][:,v],GREY),(c['achieved'][:upto,0],c['achieved'][:upto,v],COLORS[j]),([c['target'][0]],[c['target'][v]],(80,100,255))]
                    plot(canvas,(x+a*300,334,290,166),series,limits[0],limits[v],'World XY (m)' if a==0 else 'World XZ (m)','X m','Y m' if a==0 else 'Z m')
                ts=c['times'][1:]
                plot(canvas,(x,511,600,145),[(ts,c['distance']*1000,COLORS[j]),(ts,c['refdistance']*1000,GREY),(ts,c['tracking']*1000,(165,235,130))],(0,end),(0,ymax),'Target: actual=color / reference=grey; tracking=green; line=35mm','time s','mm',cursor=now,threshold=35)
                entry['panels'].append(dict(source=c['relative']+'/actual_camera_frames.npz',source_frame=ix,source_timestamp_s=float(c['times'][ix]),row_index=ix-1 if ix else None,end=bool(now>=c['times'][-1])))
            text(canvas,'Plots use 97 desired/achieved control ticks, NOT raw H40 samples. Red point=target. Same scales for both runs.',20,678,.47)
            text(canvas,'HISTORICAL_AS_RECORDED: derived TCP cache caveat applies. Gate legality / tracking / task success are separate.',20,706,.47,(140,180,255))
            proc.stdin.write(canvas.tobytes());mapping.append(entry)
            if k in (0,n//2,n-1):cv2.imwrite(str(out/f'01_v2_{playback:g}x_{k:04d}.png'),canvas)
    finally:proc.stdin.close()
    assert proc.wait()==0;qa=decode(file);assert qa['decoded_frames']==n
    suffix='' if playback==.25 else '_1x'
    write(out/f'01_frame_mapping{suffix}.json',dict(alignment='initial reference bind; no matched-reset claim',playback_rate=playback,frames=mapping));write(out/f'01_qa{suffix}.json',qa)
    write(out/'01_metrics_v2.json',dict(legacy='01_metrics.json retained unchanged; its reference_* names denote sampled_native_reference_* only',records=new))
    print('PASS_ACTUAL_WORLD_TRACKING_VIDEO '+str(playback),flush=True)
