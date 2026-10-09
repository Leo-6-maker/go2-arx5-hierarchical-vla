"""All48 saved RGB evidence: existing PI0285 numpy/cv2/ffmpeg route, no simulation."""
import argparse,hashlib,html,json,os,resource,subprocess,time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')
OUT=ROOT/'evidence/video/V2-native-contract'
GROUPS=('LEARNER16','CORRECTION03S16','CORRECTION12S16')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)


def plan(selected=GROUPS):
    parent=ROOT/'source/evidence/PARENT_DOSE6144_LEARNER_TRAIN16.json'
    collection=ROOT/'source/evidence/CORRECTIONS32.json';bank=ROOT/'correction/BANK_NATIVE_CONTRACT_V2.json'
    cpu=ROOT/'audit/PARENT_DOSE6144_LEARNER_TRAIN16_CPU.json'
    p=json.loads(parent.read_text());c=json.loads(collection.read_text());b=json.loads(bank.read_text());a=json.loads(cpu.read_text())
    assert a['source_sha256']==sha(parent) and a['status'].startswith('PASS_INDEPENDENT_CPU_')
    assert len(p['rows'])==16 and len(c['slots'])==len(c['rows'])==32
    assert b['collection_sha256']==sha(collection) and b['status']=='PASS_ACTUAL_BRANCH_RESTORE_FRESH_CORRECTIONS_FIXED_CODEC' and b['valid_branches']==27
    reports={v['slot']:v for v in b['all_requested_slot_reports']}
    proofs={v['source_result_sha256']:v['status'] for v in b['CPU_readbacks']}
    parent_proofs={v['source_result_sha256']:v['persistent_success'] for v in a['rows']}
    groups={g:[] for g in selected}
    for role,rows in [('LEARNER',p['rows']),('CORRECTION',c['rows'])]:
        for row in rows:
            desired='LEARNER16' if role=='LEARNER' else ('CORRECTION03S16' if row['takeover_tick']==80 else 'CORRECTION12S16')
            if desired not in selected:continue
            folder=Path(row['folder']);result=folder/'RESULT.json';card=folder/'EPISODE_CARD.json'
            current=json.loads(result.read_text());assert current==row and current['status']=='VALID_SOURCE_COMPLETE'
            card_value=json.loads(card.read_text());assert card_value['task']=='T1' and 285108<=card_value['layout']<=285123
            result_sha=sha(result)
            if role=='LEARNER':
                assert result_sha in parent_proofs and parent_proofs[result_sha]==row['persistent_success']
                group='LEARNER16';takeover=None;codec='NOT_APPLICABLE';proof=sha(cpu)
            else:
                assert result_sha in proofs and proofs[result_sha]=='PASS_INDEPENDENT_CPU_CORRECTION_CONTACT_CLOCK_STATE_GRIPPER_SCORE'
                takeover=card_value['task_time_s'];assert takeover in (3,12)
                assert card_value['takeover_tick']==20+takeover*20
                group='CORRECTION03S16' if takeover==3 else 'CORRECTION12S16'
                codec=reports[row['slot']]['status'];proof=sha(bank)
            groups[group].append(dict(role=role,group=group,task='T1',layout=row['layout'],takeover_s=takeover,
                takeover_tick=None if takeover is None else 20+20*takeover,slot=folder.parent.name,folder=str(folder),
                status=row['status'],physical_persistent=bool(row['persistent_success']),hold_pass=row['task_score']['hold_pass'],
                supervision_codec=codec,branch_admitted=role=='CORRECTION' and row['slot'] in b['valid_slots'],
                source_result_sha256=result_sha,source_card_sha256=sha(card),source_observation_sha256=row['files']['observations.npz'],
                source_physics_sha256=row['files']['physics.npz'],CPU_proof_sha256=proof,
                montage_path=str(OUT/(group+'.mp4')),NON_REALTIME=True,domain='SOURCE_PANDA_NATIVE_PD_NOT_TARGET_WBC'))
    for group,rows in groups.items():
        rows.sort(key=lambda x:x['layout']);assert [r['layout'] for r in rows]==list(range(285108,285124))
        for j,row in enumerate(rows):row.update(segment=j//4,tile=j%4,segment_start_video_s=(j//4)*33.,sampled_source_frames='0,2,...,658',saved_input_Hz=20,video_fps=10)
    value=dict(status='ACTUAL_SAVED_RGB_VIDEO_PLAN_NOT_POLICY_QUALIFICATION',groups=groups,covered_episodes=16*len(selected),
        source_panel_sha256=sha(parent),correction_collection_sha256=sha(collection),correction_bank_sha256=sha(bank),
        learner_physical_success=sum(x['persistent_success'] for x in p['rows']),
        correction_physical_success=sum(x['persistent_success'] for x in c['rows']),
        correction_codec_admitted=b['valid_branches'],C1='NO_TRAINING_DEADLINE_PARTIAL',selected='NONE',deployment='NONE',
        new_render=0,new_physics=0,new_model_calls=0)
    name='ALL48_SOURCE_INDEX.json' if len(selected)==3 else selected[0]+'_SOURCE_INDEX.json'
    write(OUT/name,value);return value


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
        for start in range(0,16,4):
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
                text(canvas,f'{group} | all16 layouts; page{start//4+1}/4 | V2 physical codec27/32 | NO TRAINING (deadline partial) | NOT qualification',(12,48),scale=.51)
                for tile,(row,(base,wrist)) in enumerate(zip(batch,frames)):
                    x=(tile%2)*512;y=64+(tile//2)*328
                    canvas[y:y+256,x:x+256]=cv2.resize(base[i],(256,256),interpolation=cv2.INTER_NEAREST)
                    canvas[y:y+256,x+256:x+512]=cv2.resize(wrist[i],(256,256),interpolation=cv2.INTER_NEAREST)
                    color=(80,240,100) if row['physical_persistent'] else (255,100,100)
                    takeover=row['takeover_s'];phase='LEARNER MODEL' if takeover is None else ('SAVED LEARNER REPLAY' if i<row['takeover_tick'] else 'ORIGINAL TEACHER')
                    label='MODEL' if takeover is None else f'TEACHER @ {takeover}s'
                    text(canvas,f'T1 layout{row["layout"]} {label} t={i*.05-1:.1f}s',(x+8,y+275))
                    text(canvas,f'Physical {"PASS" if row["physical_persistent"] else "FAIL"} hold{row["hold_pass"]}/41 | {phase}',(x+8,y+296),color)
                    codec='N/A' if takeover is None else ('PASS' if row['branch_admitted'] else ('FAIL' if row['supervision_codec'].startswith('FAIL_') else 'NOT ADMITTED'))
                    text(canvas,f'Original physical codec {codec} | NO TRAINING',(x+8,y+317),scale=.49)
                process.stdin.write(canvas.tobytes())
            segments.append(dict(segment=start//4,start_video_s=start//4*33,end_video_s=(start//4+1)*33,tiles=batch))
            write(OUT/'ENCODER_PROGRESS.json',dict(group=group,completed_segments=len(segments),source_episodes_seen=(start+4),epoch=time.time(),new_model_calls=0,new_physics=0))
        process.stdin.close();assert process.wait()==0;partial.replace(out)
    except BaseException:
        if process.stdin and not process.stdin.closed:process.stdin.close()
        process.wait();raise
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate:format=duration','-of','json',str(out)],text=True))
    stream=probe['streams'][0];assert stream['width']==1024 and stream['height']==720 and int(stream['nb_frames'])==1320 and abs(float(probe['format']['duration'])-132)<.01
    value=dict(status='PASS_ALL16_FIXED_CONDITIONS_ACTUAL_TWO_VIEW_SAVED_VIDEO_NOT_QUALIFICATION',group=group,sha256=sha(out),bytes=out.stat().st_size,
        frames=1320,fps=10,duration_s=132,width=1024,height=720,segments=segments,source_index_path=str(index),source_index_sha256=sha(index),new_render=0,new_physics=0,new_model_calls=0)
    write(receipt,value);return value


def main(selected=GROUPS):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    cv2.setNumThreads(1);resource.setrlimit(resource.RLIMIT_AS,(16<<30,16<<30));os.nice(10)
    OUT.mkdir(parents=True,exist_ok=True);assert OUT.resolve().is_relative_to(ROOT.resolve())
    # Only this group's already-offloaded arrays may be opened.
    for group in selected:
        slots=([f'PARENT_DOSE6144-T1-LEARNER_TRAIN16-{i}' for i in range(285108,285124)] if group=='LEARNER16' else
               [f'T1-TRAIN-{i}-TAKEOVER-{"03" if group=="CORRECTION03S16" else "12"}s' for i in range(285108,285124)])
        for slot in slots:
            source=ROOT/'source/episodes'/slot;proof=ROOT/'evidence/offload'/(slot+'.json')
            receipt=json.loads(proof.read_text())
            assert receipt['status']=='PASS_CLOSED_SLOT_DURABLE_READBACK_AND_EXACT_SSD_OFFLOAD_LOGICAL_PATH_PRESERVED'
            assert source.is_symlink() and source.resolve()==(ROOT/'durable/episodes'/slot).resolve()
    v=plan(selected);index=OUT/('ALL48_SOURCE_INDEX.json' if len(selected)==3 else selected[0]+'_SOURCE_INDEX.json')
    videos=[encode(g,v['groups'][g],index) for g in selected]
    # Also fully decode each encoded stream with the existing ffmpeg CPU decoder.
    for group in selected:subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',str(OUT/(group+'.mp4')),'-f','null','-'],check=True)
    if len(selected)<3:
        result=dict(status='PASS_OFFLOADED_FIXED16_SAVED_RGB_GROUP_FULL_DECODE',group=selected[0],video_sha256=videos[0]['sha256'],source_index_sha256=sha(index),covered_episodes=16,new_render=0,new_physics=0,new_model_calls=0)
        write(OUT/(selected[0]+'_DELIVERY.json'),result);print(json.dumps(result));return
    result=dict(status='PASS_ALL48_SAVED_RGB_MONTAGES_FULL_DECODE_AND_COMPLETE_CONDITION_INDEX',videos=[{k:x for k,x in a.items() if k!='segments'} for a in videos],
        covered_episodes=48,learner_physical_success=v['learner_physical_success'],correction_physical_success=v['correction_physical_success'],correction_codec_admitted=v['correction_codec_admitted'],
        C1='NO_TRAINING_DEADLINE_PARTIAL',source_index_sha256=sha(OUT/'ALL48_SOURCE_INDEX.json'),helper_sha256=sha(__file__),selected='NONE',deployment='NONE',new_render=0,new_physics=0,new_model_calls=0)
    write(OUT/'VIDEO_DELIVERY.json',result)
    table=[]
    for group in GROUPS:
        for row in v['groups'][group]:
            values=(group,row['layout'],row['takeover_s'] if row['takeover_s'] is not None else 'learner',
                    'PASS' if row['physical_persistent'] else 'FAIL',row['supervision_codec'],
                    f"segment{row['segment']+1}, tile{row['tile']+1}",row['source_result_sha256'])
            table.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in values)+'</tr>')
    cards=''.join(f'<h2>{g}</h2><video controls preload="metadata" width="1024" src="{g}.mp4"></video>' for g in GROUPS)
    page='''<!doctype html><meta charset="utf-8"><title>PI0286 all48 saved RGB evidence</title>
<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;background:#10151b;color:#e6edf5}video{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:8px;border:1px solid #445;word-break:break-all}a{color:#84bfff}</style>
<h1>PI0286: all48 saved RGB episodes</h1><p>T1, source Panda native PD, NON-REALTIME. Both saved cameras; each video contains four successive pages of four conditions. Playback follows recorded simulation time, not measured execution speed.</p>
<p>Parent learner:12/16 physical success. Teacher corrections:28/32 physical success; original physical codec27/32 (V2). NO TRAINING: optimizer start deadline partial. V1 extra-preclip15/32 is superseded and retained separately. These videos do not establish policy qualification or deployment.</p>
<p><a href="ALL48_SOURCE_INDEX.json">Complete source/hash index</a> | <a href="VIDEO_DELIVERY.json">Video verification receipt</a></p>'''+cards+'''<h2>Every fixed condition</h2><table><tr><th>Group</th><th>Layout</th><th>Takeover</th><th>Physical</th><th>Supervision codec</th><th>Video position</th><th>Actual RESULT SHA256</th></tr>'''+''.join(table)+'</table>'
    (OUT/'index.html').write_text(page,encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--group',choices=GROUPS);a=p.parse_args()
    selected=(a.group,) if a.group else GROUPS
    plan(selected) if a.plan else main(selected)
