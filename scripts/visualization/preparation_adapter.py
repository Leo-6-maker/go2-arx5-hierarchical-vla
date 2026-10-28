"""Read sealed producer evidence; never import a collector or native environment."""
from pathlib import Path
import json,math
from eef_video_workflow import read,write,sha,DEFAULT_ROOT,text,encoder,decode,frame_index,saved_path
from recorded_adapter import copy_bound,BASE

def stage(root=DEFAULT_ROOT):
    import numpy as np
    out=root/'delivery';manifest=BASE/'PI0255/evidence/video_manifest.json'
    assert sha(manifest)=='f72b5dd4f72a64c0d0a0832f8a8615200b92696939dd7eaac18357f5ebde9cf5'
    cards=read(manifest);examples=[];bindings=[copy_bound(manifest,out/'provenance/PI0255_video_manifest.json')]
    for c in cards['examples']:
        assert not c['task_phase_entered']
        bindings.append(copy_bound(BASE/'PI0255'/c['path'],out/'originals'/Path(c['path']).name,c['sha256']))
        result_path=BASE/'PI0255/episodes'/c['episode']/'attempt-00/result.json'; result=read(result_path)
        bindings.append(copy_bound(result_path,out/'preparation/PI0255'/c['episode']/'result.json'))
        frames=[]
        for f in c['actual_timestamps']:
            dst=out/'preparation/PI0255'/f['source']; bindings.append(copy_bound(BASE/'PI0255'/f['source'],dst,f['sha256']))
            frames.append(dict(path=str(dst),time_s=f['actual_sim_time_s'],terminal=f['terminal'],usable_as_pre_failure=not f['terminal']))
        examples.append(dict(id=c['episode'],group='PI0255',failure=c['failure'],frames=frames,verdict='FAIL',scope='PREPARATION_ONLY; task ticks 0',end_time_s=result['end_actual_sim_time_s'],revision='c6bc2a041323788a4b19b7b8ef11227f9950a0a6',queries=0,handoffs=len(result['reference_load_events'])))
    producer=BASE/'PI0256';info=read(producer/'evidence/HF_PUBLIC_SUMMARY.json')
    assert info['status']=='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK'
    mf=producer/'archive/anonymous-MEMBER_MANIFEST.json';expected=next(x for x in info['HF_files'] if x['path']=='MEMBER_MANIFEST.json')
    assert sha(mf)==expected['sha256'];entries={x['path']:x for x in read(mf)['files']}
    bindings.append(copy_bound(producer/'evidence/HF_PUBLIC_SUMMARY.json',out/'provenance/PI0256_HF_PUBLIC_SUMMARY.json'))
    bindings.append(copy_bound(mf,out/'provenance/PI0256_MEMBER_MANIFEST.json',expected['sha256']))
    for name in ('preparation_stage_summary.json','preparation_replay_tcp_sync.json'):
        rel='evidence/'+name;bindings.append(copy_bound(producer/rel,out/'provenance'/('PI0256_'+name),entries[rel]['sha256']))
    summary=read(out/'provenance/PI0256_preparation_stage_summary.json')
    assert summary['status']=='PREPARATION_V2_BLOCKED'
    for case in ('pi0256-prep-A00','pi0256-prep-B00','pi0256-prep-Bfix00'):
        rel=f'episodes/{case}/result.json';result=read(producer/rel)
        bindings.append(copy_bound(producer/rel,out/'preparation/PI0256'/rel,entries[rel]['sha256']))
        assert result['state']=='TERMINAL' and result['validity']=='VALID' and result['task_ticks']==0
        frames=[]
        for relative,entry in sorted(result['files'].items()):
            if not relative.startswith('frames/') or not relative.endswith('.npz'):continue
            rel=f'episodes/{case}/{relative}';dst=out/'preparation/PI0256'/rel
            assert entry['sha256']==entries[rel]['sha256']
            bindings.append(copy_bound(producer/rel,dst,entry['sha256']))
            with np.load(dst,allow_pickle=False) as z:now=float(z['time_s'])
            frames.append(dict(path=str(dst),time_s=now,terminal=False,usable_as_pre_failure=True))
        frames.sort(key=lambda f:f['time_s'])
        for name in ('raw_ticks.jsonl','reference_loads.jsonl'):
            rel=f'episodes/{case}/{name}';bindings.append(copy_bound(producer/rel,out/'preparation/PI0256'/rel,entries[rel]['sha256']))
        loads=[json.loads(x) for x in (producer/f'episodes/{case}/reference_loads.jsonl').read_text().splitlines()]
        examples.append(dict(id=case,group='PI0256',failure=result['failure'],frames=frames,verdict='PASS' if result['success'] else 'FAIL',scope='PREPARATION_ONLY; task ticks 0',end_time_s=result['end_time_s'],revision=info['revision'],queries=0,handoffs=len(loads),execution_source_commit=result['execution_seal']['commit']))
    write(out/'preparation_manifest.json',dict(evidence_class='ARCHIVED_PHYSICS_RECORDING',case_selection='PI0255 fixed four manifest IDs; PI0256 fixed matched reset family 00 in A/B/Bfix; C not released',examples=examples,bindings=bindings,PI0256_status=summary['status'],new_physics=0,new_model_queries=0))
    print('PASS_SEALED_PREPARATION_EXPORT',flush=True)

def render(root=DEFAULT_ROOT,playback=.25):
    import cv2
    import numpy as np
    from plotting import plot,COLORS
    from portable_numeric import preparation_series
    out=root/'delivery';manifest=read(out/'preparation_manifest.json');suffix='' if playback==.25 else '_1x';path=out/('03_preparation_failures_and_repairs'+suffix+'.mp4');proc=encoder(path);mapping=[];ordinal=0;diagnostics={}
    for c in manifest['examples']:
        if c['group']=='PI0256':
            log=out/'preparation/PI0256/episodes'/c['id']/'raw_ticks.jsonl'
            diagnostics[c['id']]=preparation_series([json.loads(line) for line in log.read_text().splitlines()])
    write(out/'03_preparation_metrics_v2.json',dict(origin='Independent CPU recomputation of sealed historical post-step telemetry; no new physics',thresholds=dict(rotation_error_rad=.10,angular_speed_radps=.20,continuous_qualified_s=.5,speed_mps=.02,position_error_m=.035,other='fresh, clean, no joint_limit_violation'),series=diagnostics))
    try:
        for chapter,group in enumerate(('PI0255','PI0256')):
            cases=[c for c in manifest['examples'] if c['group']==group];end=max(c['end_time_s'] for c in cases)
            for c in cases:
                valid=[f for f in c['frames'] if f['usable_as_pre_failure']];c['valid']=valid
                c['images']=[]
                for f in valid:
                    with np.load(saved_path(out,f['path']),allow_pickle=False) as z:c['images'].append(np.concatenate([z['body_rgb'],z['wrist_rgb']],axis=1))
            for k in range(math.ceil(end/playback*30)+1):
                now=k/30*playback;canvas=np.full((720,1280,3),(26,21,17),np.uint8)
                text(canvas,f'ARCHIVED PREPARATION | {group} | complete recorded intervals | {playback:g}x',24,32,.7)
                text(canvas,'Instruction: prepare/settle before task | learned-model queries 0 | no new physical execution',24,60,.52)
                text(canvas,f'Actual simulation clock {now:.3f}s | source frame hold; no time warp | selection/deployment NONE',24,86,.53)
                entry=dict(output_frame=ordinal,chapter=group,simulation_time_s=now,panels=[])
                for j,c in enumerate(cases):
                    if group=='PI0255':x=24+(j%2)*632;y=116+(j//2)*276;w,h=368,184
                    else:x=20+j*420;y=121;w,h=288,144
                    ts=[f['time_s'] for f in c['valid']];idx=frame_index(ts,now);f=c['valid'][idx]
                    text(canvas,c['id'],x,y,.47)
                    if now>=ts[0]:canvas[y+14:y+14+h,x:x+w]=cv2.resize(cv2.cvtColor(c['images'][idx],cv2.COLOR_RGB2BGR),(w,h),interpolation=cv2.INTER_NEAREST)
                    else:text(canvas,'MISSING: before first stored frame',x,y+80,.45)
                    note='END / '+c['verdict'] if now>=c['end_time_s'] else ('MISSING tail; last stored image held' if now>ts[-1] else f"source t={ts[idx]:.3f}s")
                    text(canvas,note,x,y+h+34,.43,(100,170,255))
                    text(canvas,c['failure'],x,y+h+55,.42,(100,170,255))
                    if group=='PI0255':
                        for line,label in enumerate(('EXPERT / WBC','Noise / sampler: N/A','H40 / dt 0.05s','Source '+c['revision'][:12],'All reference loads: '+str(c['handoffs']),'Includes reset binds','Gate / failure below')):text(canvas,label,x+380,y+32+line*24,.4)
                    if group=='PI0256':
                        series=diagnostics[c['id']];ts2=np.array([r['time_s'] for r in series]);ix2=frame_index(ts2,now);current=series[ix2] if now>=ts2[0] else None
                        for row,(key,limit,unit,title) in enumerate([('rotation_error_rad',.10,'rad','SO3'),('angular_speed_radps',.20,'rad/s','Angular speed'),('qualified_continuous_s',.5,'s','Qualified interval')]):
                            maximum=max(max(v[key] for values in diagnostics.values() for v in values)*1.05,limit*1.2)
                            label=title+(' MISSING' if current is None else f" {current[key]:.3f} {unit}")+(' / need >= ' if key=='qualified_continuous_s' else ' / need <= ')+f'{limit:g}'
                            plot(canvas,(x,333+row*101,395,98),[(ts2,[r[key] if not r['post_reset'] else np.nan for r in series],COLORS[j])],(0,end),(0,maximum),label,'time s',unit,cursor=now,threshold=limit)
                        text(canvas,'Loads '+str(c['handoffs'])+' | noise N/A | source '+c['execution_source_commit'][:12],x,652,.37)
                    entry['panels'].append(dict(case=c['id'],source_frame=f['path'],source_timestamp_s=ts[idx],missing_before=now<ts[0],missing_tail=now>ts[-1],end=now>=c['end_time_s'],failure=c['failure']))
                if group=='PI0255':
                    text(canvas,'Fixed four archived examples. Terminal/post-autoreset images are preserved separately, never shown as failure poses.',24,666,.48)
                else:
                    text(canvas,'Cache-copy source/CPU proof: PASS | Physical qualification: A/B/Bfix all 0/6; C NOT RELEASED.',24,675,.49)
                text(canvas,'Historical CPU metrics. Native 96x96 cameras. Expert preparation / model queries 0; '+('H40 dt=.05' if group=='PI0255' else 'B/Bfix H101 dt=.05'),24,701,.46)
                proc.stdin.write(canvas.tobytes());mapping.append(entry);ordinal+=1
                if k in (0,int(end/playback*30)//2,math.ceil(end/playback*30)):
                    cv2.imwrite(str(out/f'03_preview_{group}_{k:04d}.png'),canvas)
    finally:proc.stdin.close()
    assert proc.wait()==0;qa=decode(path);assert qa['decoded_frames']==ordinal
    write(out/f'03_frame_mapping{suffix}.json',dict(policy='Latest stored pre-failure RGB at or before real source time; initial/tail gaps labelled; terminal post-reset images excluded',frames=mapping))
    write(out/f'03_qa{suffix}.json',qa);print(json.dumps(dict(status='PASS_SEALED_PREPARATION_VIDEO',frames=ordinal)),flush=True)

if __name__=='__main__':stage();render()
