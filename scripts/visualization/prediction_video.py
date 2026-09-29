"""Fixed physical scales and orthographic/per-axis diagnostics, saved arrays only."""
import math
import cv2
import numpy as np
from eef_video_workflow import read,write,text,encoder,decode,DEFAULT_ROOT,saved_path
from plotting import plot,COLORS,GREY

def render(root=DEFAULT_ROOT):
    out=root/'delivery';matrix=read(out/'display_matrix.json');complete=read(out/'completeness.json');qa=read(out/'recorded_cpu_qa.json')
    cells={c['id']:c for c in complete['cells']};metrics={c['id']:c for c in qa['cells']};models=list(matrix['models']);clips=[]
    for window in matrix['windows']:
        wid=f"{window['cohort']}_{window['episode_index']:06d}"
        with np.load(saved_path(out,complete['inputs'][wid])/'tensors.npz',allow_pickle=False) as z:image=np.concatenate([z['body_rgb'],z['wrist_rgb']],axis=1)
        for noise in matrix['noises']:
            data=[]
            for model in models:
                c=cells[f'{wid}_{model}_{noise}']
                with np.load(saved_path(out,c['path'])/'tensors.npz',allow_pickle=False) as z:prediction=z['decoded'][0,:,:3].copy();expert=z['expert'][:,:3].copy()
                data.append((c,metrics[c['id']],prediction,expert))
            clips.append((window,wid,noise,image,data))
    extent=math.ceil(max(max(np.abs(d[2]-d[3][0]).max(),np.abs(d[3]-d[3][0]).max()) for c in clips for d in c[4])*10)/10
    error_extent=math.ceil(max(np.abs(d[2]-d[3]).max() for c in clips for d in c[4])*10)*100
    assert extent>0 and error_extent>0
    path=out/'02_G_U_FULL_recorded_input_predictions.mp4';proc=encoder(path);mapping=[];chapters=[];ordinal=0;(out/'cards').mkdir(exist_ok=True)
    try:
        for chapter,(window,wid,noise,image,data) in enumerate(clips):
            chapter_card=f'cards/{wid}_{noise}.png'
            chapters.append(dict(chapter=chapter+1,start_s=chapter*2,end_s=(chapter+1)*2,source=window['source_episode_id'],cohort=window['cohort'],noise=noise,card=chapter_card,zoom_card=f'cards/{wid}_{noise}_zoom.png',cells=[dict(id=d[0]['id'],card=f"cards/{d[0]['id']}.png") for d in data]))
            for k in range(60):
                horizon=min(1.95,k/30);tick=min(39,int((horizon+1e-8)/.05));canvas=np.full((720,1280,3),(26,21,17),np.uint8)
                text(canvas,'OFFLINE PREDICTIONS | FIXED PHYSICAL SCALE across all 32 chapters',22,30,.70)
                text(canvas,f"Chapter {chapter+1}/32 | {wid} / TRAIN-{window['target_family'].upper()} | {noise} | source t={window['observation_timestamp']:.3f}s | horizon {horizon:.2f}s / 1x",22,58,.51)
                text(canvas,'Instruction: move the end effector to the red target | H40, dt 0.05s, S4 | no new model-driven motion',22,82,.48)
                canvas[101:229,22:278]=cv2.resize(cv2.cvtColor(image,cv2.COLOR_RGB2BGR),(256,128),interpolation=cv2.INTER_NEAREST)
                for i,line in enumerate([window['source_episode_id'],'Static original BODY/WRIST: 96x96 per sensor; no trajectory overlay on RGB.',f'XY / XZ: native-axis offsets from expert first point; fixed +/-{extent:g} m. No auto-zoom.',f'Per-axis error = model - expert, mm; fixed +/-{error_extent:g} mm. X orange / Y blue / Z green.',f"Noise SHA {data[0][0]['historical_record']['noise_source_sha256'][:20]}; matched model contracts."]):text(canvas,line,300,114+i*26,.41)
                entry=dict(output_frame=ordinal,source_episode_id=window['source_episode_id'],noise=noise,prediction_horizon_s=horizon,prediction_index=tick,panels=[])
                for j,(cell,metric,prediction,expert) in enumerate(data):
                    x=20+420*j;color=COLORS[j];anchor=expert[0];a=prediction-anchor;e=expert-anchor;record=cell['historical_record'];gate=record['gate'];v=metric['metrics']
                    text(canvas,cell['model']+' @'+matrix['models'][cell['model']][:12],x,252,.52,color)
                    for axis,vertical in enumerate((1,2)):
                        plot(canvas,(x+axis*202,262,195,164),[(e[:,0],e[:,vertical],GREY),(a[:tick+1,0],a[:tick+1,vertical],color)],(-extent,extent),(-extent,extent),'XY native (m)' if axis==0 else 'XZ native (m)','X m','Y m' if axis==0 else 'Z m')
                    error=(prediction-expert)*1000;t=np.arange(40)*.05
                    plot(canvas,(x,434,395,113),[(t,error[:,axis],COLORS[axis]) for axis in range(3)],(0,1.95),(-error_extent,error_extent),'Per-axis error (mm); cursor = displayed horizon','time s','mm',cursor=horizon)
                    text(canvas,f"XYZ RMSE {v['xyz_rmse_m']*1000:.2f} mm | SO3 {np.degrees(v['so3_rmse_rad']):.2f} deg",x,572,.46)
                    text(canvas,f"Initial rotation jump {np.degrees(gate['metrics']['initial_rotation_discontinuity_rad']):.2f} deg",x,596,.45)
                    text(canvas,f"Peak linear {v['max_linear_speed_mps']:.3f} m/s | Gate {metric['gate']}",x,620,.45)
                    text(canvas,';'.join(g['code'] for g in gate['issues']) or 'NONE',x,643,.35,(100,170,255))
                    text(canvas,'Queries 1 / S4 heads 4 / WBC handoffs 0',x,664,.43)
                    text(canvas,'NEW offline inference (v1)' if cell['status']=='NEW_RECORDED_INPUT_INFERENCE' else 'Exact archived output reused',x,685,.43)
                    entry['panels'].append(dict(model=cell['model'],cell=cell['id'],cache_key=cell['cache_key'],noise_sha256=record['noise_source_sha256'],gate_issues=gate['issues']))
                text(canvas,'CPU metrics | grey=expert | SafetyGate != target success. Historical G FAIL / U old-support FAIL / FULL NONE.',22,710,.44)
                proc.stdin.write(canvas.tobytes());mapping.append(entry)
                if k==59:
                    cv2.imwrite(str(out/chapter_card),canvas)
                    zoom=canvas.copy();zoom[0:38]=(26,21,17);zoom[148:200,298:1270]=(26,21,17)
                    local=max(.005,max(max(np.abs(d[2]-d[3][0]).max(),np.abs(d[3]-d[3][0]).max()) for d in data)*1.1)
                    error_local=max(5.,max(np.abs(d[2]-d[3]).max()*1000 for d in data)*1.1)
                    text(zoom,'OFFLINE STATIC CARD | LOCAL ZOOM, common within this chapter only',22,30,.68)
                    text(zoom,f'LOCAL ZOOM: +/-{local:.4f} m. Not the fixed scale used in the main video.',300,166,.43)
                    text(zoom,f'Per-axis error LOCAL ZOOM: +/-{error_local:.1f} mm. Numeric metrics unchanged.',300,192,.43)
                    for j,(c,m,p,e) in enumerate(data):
                        x=20+j*420;zoom[234:548,x:x+398]=(26,21,17);anchor=e[0]
                        text(zoom,c['model']+' / LOCAL ZOOM',x,252,.50,COLORS[j])
                        for axis,v in enumerate((1,2)):
                            plot(zoom,(x+axis*202,262,195,164),[((e-anchor)[:,0],(e-anchor)[:,v],GREY),((p-anchor)[:,0],(p-anchor)[:,v],COLORS[j])],(-local,local),(-local,local),'XY zoom (m)' if axis==0 else 'XZ zoom (m)','X m')
                        plot(zoom,(x,434,395,113),[(np.arange(40)*.05,(p-e)[:,a]*1000,COLORS[a]) for a in range(3)],(0,1.95),(-error_local,error_local),'Per-axis error LOCAL ZOOM (mm)','time s')
                        cv2.imwrite(str(out/'cards'/f"{c['id']}.png"),zoom[233:695,x:x+395])
                    cv2.imwrite(str(out/chapters[-1]['zoom_card']),zoom)
                if ordinal in (0,750,990,1710,1919):cv2.imwrite(str(out/f'02_preview_{ordinal:04d}.png'),canvas)
                ordinal+=1
    finally:proc.stdin.close()
    assert proc.wait()==0;result=decode(path);assert result['decoded_frames']==1920
    write(out/'02_frame_mapping.json',dict(evidence_class='OFFLINE_PREDICTION_ON_RECORDED_OBSERVATIONS',native_offset_origin='expert[0] per cell; prediction and expert translated together',fixed_native_extent_m=extent,fixed_error_extent_mm=error_extent,frames=mapping));write(out/'02_chapters.json',chapters);write(out/'02_qa.json',result)
    print('PASS_FIXED_SCALE_32_CHAPTERS_96_STATIC_CARDS',flush=True)
