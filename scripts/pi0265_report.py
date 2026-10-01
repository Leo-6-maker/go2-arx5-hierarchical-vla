"""Read-only independent CPU world-FK replay, actual-camera video and public archive."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pi0265_source import ROOT,BASE,sha,write
from pi0265_metrics import score,so3

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0265')
sys.path.insert(0,str(BASE/'PI0250/runtime/reference-317a/src'))


def report():
    from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics
    verdict=json.loads((ROOT/'evidence/G1_VERDICT.json').read_text())
    source=json.loads((ROOT/'evidence/SOURCE_IDENTITY.json').read_text())
    kin=NativeArmKinematics.from_urdf(next(row['path'] for row in source['files'] if row['name']=='native_urdf'))
    all_results=verdict['initial_results']+verdict['confirmation'];rows=[];replay=[];chapters=[]
    for result in all_results:
        case=result['case'];folder=RAW/'controls'/case['case_id']
        valid=[p.parent for p in folder.glob('attempt-*/result.json') if json.loads(p.read_text())['validity']=='VALID' and not (p.parent/'INVALIDATION.json').exists()]
        assert len(valid)==1;attempt=valid[0]
        data=json.loads((attempt/'rows.json').read_text())
        row=dict(case=case['case_id'],stage=case['stage'],version=case['version'],family=case.get('family',{}).get('family_id','C0'),
            task=case['task'],command=case['command'],success=result['success'],failure=result.get('failure') or ('TASK_CRITERIA_FAIL' if not result['success'] else ''),
            safety_failure=result.get('safety_failure',False),validity=result['validity'],attempt=str(attempt),
            recorded_world_rows=len(data),WBC_actor_calls=result.get('WBC_actor_calls',0))
        row['terminal_collision_pairs']=json.dumps([hit['pair'] for hit in data[-1].get('collision',{}).get('hits',[])]) if data else 'UNAVAILABLE'
        row['terminal_hard_unsafe_joints']=json.dumps(data[-1].get('hard_unsafe',[])) if data else 'UNAVAILABLE'
        if data and (attempt/'task_spec.json').exists():
            position=[];rotation=[];max_xyz=max_rot=0.
            for state in data:
                root=np.array(state['root']);r=Rotation.from_quat(root[3:7]).as_matrix()
                q=np.array(state['telemetry']['arm_q']);local=kin.tcp_pose(q)
                p=r@local[:3,3]+root[:3];rr=r@local[:3,:3];position.append(p);rotation.append(rr)
                max_xyz=max(max_xyz,float(np.linalg.norm(p-state['TCP'])));max_rot=max(max_rot,float(so3(rr,np.array(state['rotation']))))
            assert max_xyz<=1e-5 and max_rot<=1e-4
            spec=json.loads((attempt/'task_spec.json').read_text())
            independent=score([x['elapsed'] for x in data],position,rotation,[x['installed'] for x in data],
                [x['installed_rotation'] for x in data],spec,result['safety_failure'])
            for key in ('success','time_to_success_s','final_xyz_m','final_so3_rad','tracking_xyz_rms_m','tracking_xyz_p95_m','tracking_so3_rms_rad','ordered_bins','reverse','gap_shortcut'):
                value,actual=independent[key],result[key]
                assert value==actual if not isinstance(value,float) else abs(value-actual)<=1e-10,(case['case_id'],key,value,actual)
            requested=dict(np.load(attempt/'reference.npz',allow_pickle=False))
            row.update({key:independent[key] for key in ('final_xyz_m','final_so3_rad','min_xyz_m','tracking_xyz_rms_m','tracking_xyz_p95_m','tracking_so3_rms_rad','time_to_success_s')})
            row['ordered_bins']=len(independent['ordered_bins']);row['observed_task_s']=data[-1]['elapsed']
            row['requested_final_target_xyz_m']=float(np.linalg.norm(requested['position'][-1]-spec['goal']))
            row['installed_final_target_xyz_m']=float(np.linalg.norm(np.array(data[-1]['installed'])-spec['goal']))
            table=dict(np.load(attempt/'installed_table.npz',allow_pickle=False))
            row['canonical_terminal_to_requested_xyz_m']=float(np.linalg.norm(table['gamma_p'][0,-1]-requested['position'][-1]))
            row['canonical_terminal_to_requested_so3_rad']=float(so3(Rotation.from_quat(table['gamma_quat'][0,-1]).as_matrix(),Rotation.from_quat(requested['quaternion'][-1]).as_matrix()))
            replay.append(dict(case=case['case_id'],rows=len(data),max_reconstructed_FK_xyz_m=max_xyz,max_reconstructed_FK_so3_rad=max_rot,
                independent_world_FK_and_metric_match=True,raw_source_sha256=sha(attempt/'rows.json')))
        rows.append(row);chapters.append(dict(case=case,result=result,attempt=str(attempt),row=row))
    out=RAW/'delivery';out.mkdir(exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (out/'CONTROL_QUALITY.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write(out/'CPU_REPLAY.json',dict(status='PASS_INDEPENDENT_WORLD_FK_AND_METRIC_REPLAY',episodes=len(rows),
        rows=sum(x['rows'] for x in replay),checks=replay,physics_steps=0,GR00T_queries=0))
    write(out/'VIDEO_CARD.json',dict(chapters=chapters,playback='4x, every fourth actual10Hz frame at10fps',
        no_generated_or_reconstructed_camera=True,freeze_frames_explicit=True))
    lines=['# PI0265 — actual controller comparison','',
        '**Overall: '+verdict['overall']+'; '+verdict['status']+'.**',
        'Historical PI0264 COMPLETE/VALID_FAIL is unchanged. Formal selected/deployment: NONE.','',
        'G0 source/CPU checks passed. Checkpoint training-time installed PyTorch3D remains UNKNOWN. '
        'LEGACY and ROW6D alone were compared; native quaternion-vector feature, weights, gains, dynamics and gate were preserved.','',
        'C0 is diagnostic, not general-controller admission. Source reset22301 and original reference attitude were retained. '
        'Actual direct/provider reset/RNG and all installed canonical-table arrays match exactly for each version.','',
        '| Case | Task success | Safety fail | Final XYZ m | Final SO3 rad | Tracking RMS / p95 m | Tracking SO3 RMS rad | Arc bins |','|---|---|---|---|---|---|---|---|']
    for row in rows:
        fmt=lambda key:format(row[key],'.6f') if key in row else 'UNAVAILABLE'
        lines.append('| %s | %s | %s | %s | %s | %s / %s | %s | %s |'%(row['case'],row['success'],row['safety_failure'],
            fmt('final_xyz_m'),fmt('final_so3_rad'),fmt('tracking_xyz_rms_m'),fmt('tracking_xyz_p95_m'),fmt('tracking_so3_rms_rad'),row.get('ordered_bins','—')))
    lines+=['','## Registered qualification','',json.dumps(verdict['qualification'],sort_keys=True),'',
        'Every fixed independent case and every failed start is retained. No score-based retry, fourth-DEV rescue, new pose/seed, gains or settling search. '
        'A source-bound contact-classification bug initially treated normal foot/ground support as forbidden self/obstacle contact. '
        'Seven zero-actor starts were marked ENGINEERING_INVALID with raw receipts unchanged and budget retained; this is not a scientific-failure reclassification.','',
        'Raw budget: '+json.dumps(json.loads((ROOT/'evidence/RAW_BUDGET.json').read_text()),sort_keys=True)+'.','',
        'Requested, installed and achieved are distinct world-coordinate columns. Tracking uses the same native tick, not differences of minimum distances. '
        'Native termination state comes from pre-autoreset receipts when available; unavailable terminal camera views are not reconstructed.','']
    unsafe=[row for row in rows if row['stage']=='DEV' and row['safety_failure']]
    lines+=['## Actual safety and task failures','',
        '| Case | Stop reason | Actual forbidden collision pairs | Hard-bound joints |','|---|---|---|---|']
    for row in unsafe:lines.append('| %s | %s | %s | %s |'%(row['case'],row['failure'],row['terminal_collision_pairs'],row['terminal_hard_unsafe_joints']))
    lines+=['','Safety stops and complete but unsuccessful tracking episodes are separate outcomes. '
        'A reset-dependent forbidden collision shared by commands/representations is not evidence that either6D packing caused it. '
        'C0 or a single successful M1 case cannot replace7/8 complete successes,3/4 per behavior and every successful episode\'s tracking requirements.','']
    if verdict['status']=='PI0265_CONTROLLER_ENVELOPE_BLOCKED':
        lines += ['Neither version completed the registered dynamic qualification. G2 feedback collection, R/D training and learned-policy closed loop were not started. '
            'This is a controller-envelope stop and overall PARTIAL, not an executed feedback-training pilot or evidence of a unique controller root cause.']
    else:lines += ['Controller stage passed; automatically continue the authorized measured-feedback pipeline. This report closes only G1.']
    (out/'CONTROL_REPORT.md').write_text('\n'.join(lines)+'\n')
    for name in ('CONTROL_QUALITY.csv','CPU_REPLAY.json','CONTROL_REPORT.md','VIDEO_CARD.json'):shutil.copy2(out/name,ROOT/'evidence'/name)
    print(json.dumps(dict(status='PASS_ACTUAL_CONTROL_DELIVERY_CPU',episodes=len(rows),overall=verdict['overall'])),flush=True)


def video():
    import cv2
    out=RAW/'delivery';card=json.loads((out/'VIDEO_CARD.json').read_text());path=out/'PI0265_ACTUAL_WBC_CONTROLS.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10.,(1024,640));assert writer.isOpened()
    chapters=[];count=0
    for chapter in card['chapters']:
        frames=sorted((Path(chapter['attempt'])/'frames').glob('*.npz'));start=count;case=chapter['case'];row=chapter['row']
        selected=frames[::4]
        for frame_path in selected:
            frame=dict(np.load(frame_path,allow_pickle=False));canvas=np.full((640,1024,3),18,np.uint8)
            for i,key in enumerate(('body_rgb','wrist_rgb')):
                image=cv2.resize(frame[key],(480,480),interpolation=cv2.INTER_NEAREST)
                canvas[90:570,16+i*512:496+i*512]=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            cv2.putText(canvas,'PI0265 ACTUAL SIMULATOR / EXPERT WBC / '+case['version'],(18,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1)
            cv2.putText(canvas,case['case_id'],(18,55),cv2.FONT_HERSHEY_SIMPLEX,.52,(230,230,230),1)
            cv2.putText(canvas,'physical t=%.3fs | 4x playback | success=%s | safety=%s'%(float(frame['time']),row['success'],row['safety_failure']),
                (18,82),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,210,80),1)
            cv2.putText(canvas,'Fresh96x96 camera pixels; no pose placement, expert replay images or synthetic replacement',(18,602),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
            writer.write(canvas);count+=1
        if selected:
            cv2.putText(canvas,'LAST RECORDED FRAME / PAUSED: '+row['failure'],(18,627),cv2.FONT_HERSHEY_SIMPLEX,.45,(100,170,250),1)
            for _ in range(10):writer.write(canvas);count+=1
        chapters.append(dict(case=case['case_id'],actual_captured_frames=len(frames),displayed_frames=len(selected),first_video_frame=start,
            last_video_frame=count-1,no_camera_capture=len(frames)==0))
    writer.release();assert path.stat().st_size>0
    cap=cv2.VideoCapture(str(path));decoded=0
    while True:
        okay,frame=cap.read()
        if not okay:break
        assert frame.shape==(640,1024,3);decoded+=1
    cap.release();assert decoded==count
    write(ROOT/'evidence/VIDEO_RECEIPT.json',dict(status='PASS_ACTUAL_CAMERA_VIDEO_FULL_DECODE',file=str(path),sha256=sha(path),frames=decoded,
        chapters=chapters,width=1024,height=640,fps=10.,physics=0,GR00T_queries=0))
    print(json.dumps(dict(status='PASS_ACTUAL_VIDEO',frames=count,bytes=path.stat().st_size)),flush=True)


def archive():
    import requests
    from huggingface_hub import HfApi,hf_hub_url
    existing=ROOT/'evidence/PUBLIC_RECEIPT.json'
    if existing.exists():
        saved=json.loads(existing.read_text());assert saved['status']=='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK'
        print(json.dumps(saved),flush=True);return
    assert json.loads((ROOT/'evidence/CPU_REPLAY.json').read_text())['status']=='PASS_INDEPENDENT_WORLD_FK_AND_METRIC_REPLAY'
    assert json.loads((ROOT/'evidence/VIDEO_RECEIPT.json').read_text())['status']=='PASS_ACTUAL_CAMERA_VIDEO_FULL_DECODE'
    out=RAW/'archive/public';out.mkdir(parents=True,exist_ok=True)
    files=[]
    for base,folders in ((ROOT,('code','design','evidence','logs','noise')),(RAW,('controls','delivery'))):
        for folder in folders:
            for p in sorted((base/folder).rglob('*')):
                if not p.is_file() or p.suffix not in ('.py','.json','.jsonl','.npz','.csv','.md','.log','.mp4'):continue
                assert not p.is_symlink() and p.resolve().is_relative_to(base.resolve())
                files.append((p,('small/' if base==ROOT else 'raw/')+p.relative_to(base).as_posix()))
    rows=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for p,name in files]
    tree=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write(out/'MEMBER_MANIFEST.json',dict(files=rows,tree_sha256=tree,members=len(rows),private_WBC_assets_or_weights=False))
    package=out/'PI0265_ACTUAL_CONTROL_EVIDENCE.tar'
    with tarfile.open(package,'w') as tar:
        for (p,name),row in zip(files,rows):
            info=tarfile.TarInfo(name);info.size=row['bytes'];info.mode=0o644;info.mtime=0
            with p.open('rb') as stream:tar.addfile(info,stream)
    for name in ('CONTROL_REPORT.md','PI0265_ACTUAL_WBC_CONTROLS.mp4'):shutil.copy2(RAW/'delivery'/name,out/name)
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0265/new-scene-controller-fixed-20261001'
    api=HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip());assert api.model_info(repo).private is False
    receipt=RAW/'archive/UPLOAD_COMMIT.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text());assert saved['tree_sha256']==tree;revision=saved['revision']
    else:
        revision=api.upload_folder(repo_id=repo,folder_path=str(out),path_in_repo=prefix,commit_message='PI0265 actual frozen WBC controls and all raw failures').oid
        write(receipt,dict(repo_id=repo,prefix=prefix,revision=revision,tree_sha256=tree))
    session=requests.Session();session.trust_env=False;checked=[]
    for p in sorted(out.iterdir()):
        destination=RAW/'archive'/('anonymous-'+p.name)
        with session.get(hf_hub_url(repo,prefix+'/'+p.name,revision=revision),stream=True,timeout=(30,180)) as response:
            response.raise_for_status();assert 'Authorization' not in response.request.headers
            with destination.open('wb') as stream:
                for block in response.iter_content(8<<20):stream.write(block)
        assert sha(destination)==sha(p) and destination.stat().st_size==p.stat().st_size
        checked.append(dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size,anonymous=True))
    remaining={row['path']:row for row in rows}
    with tarfile.open(RAW/'archive'/('anonymous-'+package.name)) as tar:
        for member in tar:
            assert member.isfile() and member.name in remaining;row=remaining.pop(member.name)
            h=hashlib.sha256();size=0
            with tar.extractfile(member) as stream:
                for block in iter(lambda:stream.read(8<<20),b''):h.update(block);size+=len(block)
            assert h.hexdigest()==row['sha256'] and size==row['bytes']
    assert not remaining
    result=dict(status='PASS_PUBLIC_ANONYMOUS_FULL_FILE_AND_MEMBER_READBACK',repo_id=repo,prefix=prefix,revision=revision,
        tree_sha256=tree,members=len(rows),source_bytes=sum(x['bytes'] for x in rows),files=checked,GR00T_queries=0,new_physics=0)
    write(ROOT/'evidence/PUBLIC_RECEIPT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',action='store_true');parser.add_argument('--video',action='store_true');parser.add_argument('--archive',action='store_true');args=parser.parse_args()
    if args.report:report()
    if args.video:video()
    if args.archive:archive()
