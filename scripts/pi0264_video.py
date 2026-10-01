"""Prospective old display cases plus all M1 wording/noise crossings, saved arrays only."""
import argparse
import json
import math
import textwrap

import cv2
import numpy as np
import pi0264_eval as evaluation
from pi0264_plan import read,write,sha
import pi0260_r3_video as vis

ROOT=evaluation.ROOT
vis.COLORS.update({'X2000':(70,225,245),'B2048':(225,170,70),'L2048':(100,220,95)})


def freeze():
    card=read(ROOT/'UNION_EVAL_CARD.json')
    old=read(evaluation.r3.R3/'S4_FIXED_DISPLAY_SELECTION.json')
    cases=read(evaluation.r3.R3/'motion2_eval_cases.json')['cases']
    chosen=[]
    for spec in old['semantic_cases']:
        assert spec['noise']=='B0'
        for index in spec['case_indices']:
            alias=next(a for a in card['legacy_aliases'] if a['panel']=='legacy' and a['source_row']['cohort']=='Motion2'
                       and a['source_row']['case_index']==index and a['source_row']['noise']=='B0')
            task=cases[index]['task']
            family=cases[index]['sequence'].rsplit(f'-{task}-c',1)[0]
            chosen.append(dict(uid=alias['uid'],task=task,family=family,
                command=cases[index]['command'],noise='B0',template='ORIGINAL',language=cases[index]['language']))
    assert len(chosen)==4
    family=next(r['family'] for r in chosen if r['task']=='M1')
    for alias in card['legacy_aliases']:
        if alias['panel']=='crossed' and alias['source_row']['family']==family:
            r=alias['source_row']
            if any(c['uid']==alias['uid'] for c in chosen):
                continue
            chosen.append(dict(uid=alias['uid'],task='M1',family=family,command=r['command'],
                noise=r['noise'],template=r['template'],language=r['language']))
    assert len(chosen)==18 and len({c['uid'] for c in chosen})==18, (len(chosen),family,chosen[:4])
    for panel in ('B','C'):
        family=min(r['source_row']['family'] for r in card['rows'] if r['panel']==panel)
        for row in card['rows']:
            if row['panel']!=panel or row['source_row']['family']!=family:continue
            if panel=='B' and row['noise']!='B0':continue
            src=row['source_row']
            chosen.append(dict(uid=row['uid'],task='M1',family=family,command=src['command'],
                noise=row['noise'],template=src['template'],language=src['text']))
    assert len(chosen)==54 and len({r['uid'] for r in chosen})==54
    write(ROOT/'DISPLAY_CARD.json',dict(schema='PI0264_SAVED_ARRAY_DISPLAY_V1',chapters=chosen,
        source_selection_sha256=sha(evaluation.r3.R3/'S4_FIXED_DISPLAY_SELECTION.json'),
        union_card_sha256=sha(ROOT/'UNION_EVAL_CARD.json'),basis='original18 chapters plus earliest TRAIN family all six new texts/both commands/B0 and earliest KIN-VAL family all Q/anchor texts/both commands/N0/N1; no endpoint scores used',
        failure_coverage_rule='After CPU replay append first uid with incorrect M1 target or REJECT for each point/panel if absent; no new queries or favorable-result selection.',
        points=['X2000','B2048','L2048'],frames_per_chapter=40,fps=20,model_queries=0))
    print(json.dumps(dict(status='PI0264_DISPLAY_FROZEN',chapters=54)))


def render():
    evaluation.assert_release()
    import csv
    display=read(ROOT/'DISPLAY_CARD.json');card=read(ROOT/'UNION_EVAL_CARD.json')
    assert read(ROOT/'CPU_REPLAY.json')['status']=='PASS_INDEPENDENT_COMPUTATION_PATH_CPU_6336_REPLAY'
    with (ROOT/'M1_ALL_PANELS_ROWS.csv').open(newline='') as stream:scored=list(csv.DictReader(stream))
    chapters=list(display['chapters']);existing={r['uid'] for r in chapters};failures=[]
    rows_by_uid={r['uid']:r for r in card['rows']}
    for point in display['points']:
        for panel in ('A_CROSSED','B','C'):
            vals=sorted((r for r in scored if r['point']==point and r['panel']==panel and (r['correct']=='False' or r['reject']=='True')),key=lambda r:r['uid'])
            if not vals:continue
            r=vals[0];failures.append(dict(point=point,panel=panel,uid=r['uid'],reason='incorrect target or REJECT'))
            if r['uid'] in existing:continue
            row=rows_by_uid[r['uid']];src=row['source_row']
            chapters.append(dict(uid=row['uid'],task='M1',family=src['family'],command=src['command'],noise=row['noise'],
                template=src['template'],language=src['text'] if row['panel']!='A' else src['language']))
            existing.add(r['uid'])
    write(ROOT/'DISPLAY_FAILURE_COVERAGE.json',dict(rule=display['failure_coverage_rule'],failures=failures,chapters=chapters,
        source_csv_sha256=sha(ROOT/'M1_ALL_PANELS_ROWS.csv'),model_queries=0))
    assert sha(ROOT/'UNION_EVAL_CARD.json')==display['union_card_sha256']
    rows={r['uid']:r for r in card['rows']};items=[]
    for chapter in chapters:
        packet,reference=evaluation.source(rows[chapter['uid']])
        rec,_,labels=evaluation.r3.sequence(f"{chapter['family']}-{chapter['task']}-c{1-chapter['command']}")
        j=next(j for j,r in enumerate(rec['rows']) if r['index']==0)
        paths={'reference':reference,'opposite':labels[j]};scores={}
        for point in display['points']:
            cell=ROOT/'eval'/point/'cells'/chapter['uid'];record=read(cell/'record.json')
            assert record['status']=='VALID' and sha(cell/'tensors.npz')==record['tensor_sha256']
            with np.load(cell/'tensors.npz',allow_pickle=False) as z:
                assert np.array_equal(z['expert'],reference);paths[point]=z['decoded'][0].copy()
            scores[point]=(record['metrics']['xyz_rmse_m'],record['metrics']['so3_rmse_rad'],record['gate']['decision'])
        items.append((chapter,packet,paths,scores))
    extent=math.ceil(max(float(np.max(np.abs(p[:,:3]-packet['state_v1'][:3]))) for _,packet,paths,_ in items for p in paths.values())*100)/100+.01
    out=ROOT/'eval/media';out.mkdir(parents=True,exist_ok=True);video=out/'PI0264_X_B_L_fixed_arrays.mp4'
    assert not video.exists()
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20.,(1280,720));assert writer.isOpened()
    for number,(chapter,packet,paths,scores) in enumerate(items,1):
        for tick in range(40):
            canvas=np.full((720,1280,3),(28,25,22),np.uint8)
            vis.line(canvas,'KINEMATIC RENDERED REFERENCE / OFFLINE PREDICTION / NO PHYSICS OR WBC',18,28,(100,205,255),.56)
            vis.line(canvas,f"Fixed chapter {number}/{len(chapters)} | {chapter['task']} c{chapter['command']} {chapter['template']} {chapter['noise']} | H40 dt=.05s | t={tick*.05:.2f}s",18,57,size=.48)
            for name,x in (('body_rgb',20),('wrist_rgb',275)):
                canvas[100:340,x:x+240]=cv2.resize(cv2.cvtColor(packet[name],cv2.COLOR_RGB2BGR),(240,240),interpolation=cv2.INTER_NEAREST)
            vis.line(canvas,'Saved BODY 96x96',20,365,size=.43);vis.line(canvas,'Saved WRIST 96x96',275,365,size=.43)
            for n,text in enumerate(textwrap.wrap(chapter['language'],width=65)):
                vis.line(canvas,text,20,400+22*n,size=.43)
            vis.plot(canvas,(595,125),'XY',(0,1),paths,packet['state_v1'][:3],extent,tick)
            vis.plot(canvas,(930,125),'XZ',(0,2),paths,packet['state_v1'][:3],extent,tick)
            for n,name in enumerate(paths):
                y=445+n*34;cv2.line(canvas,(600,y-5),(630,y-5),vis.COLORS[name],3)
                metric=scores.get(name);text=name if metric is None else f'{name}: XYZ {metric[0]:.4f}m, SO3 {metric[1]:.3f}rad, gate {metric[2]}'
                vis.line(canvas,text,640,y,size=.42)
            vis.line(canvas,'Existing-scene research only. L T0-T7 supervised; Q/V excluded. No physical generalization. selected=NONE.',18,690,size=.42)
            writer.write(canvas)
    writer.release();capture=cv2.VideoCapture(str(video));frames=0
    while capture.read()[0]:frames+=1
    capture.release();assert frames==40*len(chapters)
    write(out/'video_receipt.json',dict(status='PASS_FIXED_ARRAY_VIDEO_DECODE',video=video.name,
        sha256=sha(video),frames=frames,fps=20,chapters=len(chapters),failure_coverage_sha256=sha(ROOT/'DISPLAY_FAILURE_COVERAGE.json'),display_card_sha256=sha(ROOT/'DISPLAY_CARD.json'),
        model_queries=0,physics_steps=0))
    print(json.dumps(dict(status='PASS_PI0264_VIDEO',frames=frames)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');freeze() if p.parse_args().freeze else render()
