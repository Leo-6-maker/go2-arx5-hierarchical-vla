"""Read-only actual teacher comparison and independent full-tree FK replay."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from pi0266_common import ROOT,RAW,REF,read,write,sha
import sys
sys.path.insert(0,str(REF/'src'))
from pi0258_geometry import full_fk,homogeneous
from go2arx5_vla.wbc.native.kinematics import NativeArmKinematics
from pi0266_metrics import score,so3


def report(stage):
    verdict=read(ROOT/'evidence'/('DIAGNOSTIC_VERDICT.json' if stage=='diagnostics' else 'COLLECTION_VERDICT.json'))
    assert verdict['all_registered_cases_valid'] and len(verdict['rows'])==(16 if stage=='diagnostics' else 96)
    source=read(ROOT/'evidence/SOURCE_IDENTITY.json')
    urdf_path=next(x['path'] for x in source['files'] if x['name']=='native_urdf')
    urdf=ET.parse(urdf_path).getroot();kin=NativeArmKinematics.from_urdf(urdf_path)
    rows=[];checks=[];phase=Counter();chapters=[]
    for parent in verdict['rows']:
        out=Path(parent['attempt']);result=parent['result'];case=result['case'];states=read(out/'rows.json')
        assert read(out/'result.json')==result and result['validity']=='VALID'
        row=dict(case=case['case_id'],family=case['family']['family_id'],task=case['task'],command=case['command'],expert=case['expert'],
            task_success=result['success'],safety_failure=result['safety_failure'],failure=result['failure'],
            actual_task_s=result.get('actual_task_s',0.),proposed_windows=result.get('proposed_windows',0),valid_windows=result.get('valid_windows',0),
            attempt=str(out),rows=len(states),actor_calls=result.get('WBC_actor_calls',0))
        if (out/'world_tcp.npz').exists():
            actual=dict(np.load(out/'world_tcp.npz',allow_pickle=False));positions=[];rotations=[]
            for state in states:
                root=np.asarray(state['root']);dof=np.asarray(state['DOF']).reshape(-1,2)
                joints=dict(zip(result['dof_names'],dof[:,0]))
                transform=homogeneous(Rotation.from_quat(root[3:7]).as_matrix(),root[:3])
                pose=full_fk(urdf,joints,transform)['x5_link6']@homogeneous(np.eye(3),kin.tcp_offset_xyz)
                positions.append(pose[:3,3]);rotations.append(pose[:3,:3])
            positions=np.asarray(positions);rotations=np.asarray(rotations)
            pe=float(np.linalg.norm(positions-actual['position'],axis=-1).max());re=float(so3(rotations,actual['rotation']).max())
            assert pe<=1e-5 and re<=1e-4
            recomputed=score(actual['elapsed'],positions,rotations,actual['installed_position'],actual['installed_rotation'],read(out/'task_spec.json'),result['safety_failure'])
            saved=result['metrics']
            for key in ('success','ordered_bins','reverse','gap_shortcut','entry_time_s','time_to_success_s'):
                if key=='success' and result['failure'] in ('ORIGINAL_RAW_GATE_REJECT','CLOCK_GATED_REFERENCE_REJECT'):continue
                assert recomputed[key]==saved[key],(case['case_id'],key,recomputed[key],saved[key])
            for key in ('final_xyz_m','final_so3_rad','total_negative_variation_m','max_backward_excursion_m'):
                assert abs(recomputed[key]-saved[key])<=1e-6,(case['case_id'],key)
            for key in ('xyz_rms_m','xyz_p95_m','so3_rms_rad'):
                assert abs(recomputed['qualification_tracking'][key]-saved['qualification_tracking'][key])<=1e-6
            row.update(final_xyz_m=saved['final_xyz_m'],final_so3_rad=saved['final_so3_rad'],ordered_bins=len(saved['ordered_bins']),
                reverse=saved['reverse'],gap_shortcut=saved['gap_shortcut'],time_to_success_s=saved['time_to_success_s'],
                tracking_xyz_rms_m=saved['qualification_tracking']['xyz_rms_m'],tracking_xyz_p95_m=saved['qualification_tracking']['xyz_p95_m'],
                tracking_so3_rms_rad=saved['qualification_tracking']['so3_rms_rad'],legacy_v1_success=saved['legacy_v1_first20s']['success'])
            checks.append(dict(case=case['case_id'],rows=len(states),full_tree_fk_xyz_m=pe,full_tree_fk_so3_rad=re,
                same_owner_separate_full_URDF_computation=True,scorer_code_shared=True,metric_replay_match=True,source_sha256=sha(out/'rows.json')))
        windows=read(out/'windows.json') if (out/'windows.json').exists() else []
        for w in windows:
            phase[row['family'],row['task'],row['command'],w['admitted'],4 if w['metadata']['terminal_hold'] else w['metadata']['quartile'],w['label_class']]+=1
        rows.append(row);chapters.append(dict(case=case,attempt=str(out),row=row))
    destination=RAW/'delivery'/stage;destination.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (destination/'TEACHER_QUALITY.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write(destination/'CPU_REPLAY.json',dict(status='PASS_SAME_OWNER_FULL_TREE_FK_AND_SHARED_SCORER_CPU_REPLAY',stage=stage,episodes=len(rows),checks=checks,
        GR00T_queries=0,new_physics=0,optimizer_updates=0,qualification='No independent reviewer signature claimed.'))
    write(destination/'WINDOW_COUNTS.json',dict(rows=[dict(family=f,task=t,command=c,admitted=a,bucket=b,label_class=l,n=n) for (f,t,c,a,b,l),n in sorted(phase.items())]))
    write(destination/'VIDEO_CARD.json',dict(chapters=chapters,playback='Every fourth actual .4-second observation at2.5fps: 4x. Explicit paused last frames.'))
    lines=['# PI0266 actual '+stage,'','selected/deployment: NONE. Teacher-panel GR00T queries / optimizer updates: 0 / 0.',
        'Complete task outcomes and valid local-window counts are separate. No failed scenario is replaced. Historical PI0265 remains PARTIAL/FAIL.','',
        '| Case | Task success | Safety | Valid/proposed windows | Final XYZ m | Final SO3 rad |', '|---|---|---|---|---|---|']
    for row in rows:
        lines.append('| %s | %s | %s | %s/%s | %s | %s |'%(row['case'],row['task_success'],row['safety_failure'],row['valid_windows'],row['proposed_windows'],row.get('final_xyz_m','UNAVAILABLE'),row.get('final_so3_rad','UNAVAILABLE')))
    lines+=['','The CPU replay uses a distinct full-URDF traversal on actual root/DOFs and the shared prospectively fixed V2 scorer. '
        'Tracking is actual versus installed reference on the same native tick, through first success/failure. V1 is retained for each first20s prefix. '
        'This teacher report grants no model deployment or full-task controller qualification.','']
    (destination/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS_ACTUAL_TEACHER_CPU_REPORT',stage=stage,episodes=len(rows),task_successes=sum(r['task_success'] for r in rows),valid_windows=sum(r['valid_windows'] for r in rows))),flush=True)


def video(stage):
    import cv2
    destination=RAW/'delivery'/stage;card=read(destination/'VIDEO_CARD.json');path=destination/'PI0266_ACTUAL_TEACHERS.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),2.5,(1024,640));assert writer.isOpened()
    count=0;chapters=[]
    for chapter in card['chapters']:
        frames=sorted((Path(chapter['attempt'])/'frames').glob('*.npz'));start=count;row=chapter['row']
        for frame_path in frames[::4]:
            frame=dict(np.load(frame_path,allow_pickle=False));canvas=np.full((640,1024,3),18,np.uint8)
            for i,key in enumerate(('body_rgb','wrist_rgb')):
                image=cv2.resize(frame[key],(480,480),interpolation=cv2.INTER_NEAREST)
                canvas[90:570,16+i*512:496+i*512]=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            labels=['PI0266 ACTUAL SIMULATOR / EXPERT WBC / ROW6D',row['case'],
                'physical t=%.3fs | 4x | task success=%s | valid windows=%d'%(float(frame['time']),row['task_success'],row['valid_windows'])]
            for y,text in zip((26,55,82),labels):cv2.putText(canvas,text,(18,y),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
            cv2.putText(canvas,'Actual fresh96x96 pixels / no GR00T / no pose-placement playback',(18,602),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
            writer.write(canvas);count+=1
        if frames:
            cv2.putText(canvas,'PAUSED LAST RECORDED FRAME / '+str(row['failure']),(18,627),cv2.FONT_HERSHEY_SIMPLEX,.45,(100,170,250),1)
            for _ in range(3):writer.write(canvas);count+=1
        chapters.append(dict(case=row['case'],source_frames=len(frames),first_frame=start,last_frame=count-1))
    writer.release();cap=cv2.VideoCapture(str(path));decoded=0
    while True:
        okay,frame=cap.read()
        if not okay:break
        assert frame.shape==(640,1024,3);decoded+=1
    cap.release();assert decoded==count and count>0
    write(destination/'VIDEO_RECEIPT.json',dict(status='PASS_ACTUAL_CAMERA_VIDEO_FULL_DECODE',file=str(path),sha256=sha(path),frames=count,
        fps=2.5,chapters=chapters,physics=0,GR00T_queries=0))
    print(json.dumps(dict(status='PASS_ACTUAL_TEACHER_VIDEO',stage=stage,frames=count,bytes=path.stat().st_size)),flush=True)


def gate_replay(stage):
    import ast
    from go2arx5_vla.contracts import TrajectoryChunkV1
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    api_path=REF/'scripts/stage21_native_receding_horizon.py'
    constants={node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(api_path.read_text()).body
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('SOURCE_FRAME','TCP_SEMANTIC_ID')}
    verdict=read(ROOT/'evidence'/('DIAGNOSTIC_VERDICT.json' if stage=='diagnostics' else 'COLLECTION_VERDICT.json'))
    assert verdict['all_registered_cases_valid'] and len(verdict['rows'])==(16 if stage=='diagnostics' else 96)
    gate=TrajectorySafetyGateV1();checked=[]
    def difference(a,b):
        if isinstance(a,dict):
            assert set(a)==set(b)
            return max((difference(a[k],b[k]) for k in a),default=0.)
        if isinstance(a,(int,float)):
            assert np.isfinite(a) and np.isfinite(b);return abs(a-b)
        assert a==b;return 0.
    for parent in verdict['rows']:
        case=parent['result']['case'];out=Path(parent['attempt'])
        if case['expert']!='OBSERVABLE_CORRECTION_V1':continue
        binding=read(out/'task_binding.json');anchor=np.asarray(binding['native_anchor'])
        states=read(out/'rows.json');times=np.asarray([r['time'] for r in states]);maximum=0.;windows=read(out/'windows.json')
        for w in windows:
            assert sha(w['proposal'])==w['proposal_sha256']
            z=dict(np.load(w['proposal'],allow_pickle=False));now=float(z['time']);row=states[int(np.argmin(abs(times-now)))];assert abs(row['time']-now)<=1e-6
            ts=now+np.arange(40)*.05
            chunk=TrajectoryChunkV1(chunk_id=w['id'],source='PI0266_'+case['expert'],observation_timestamp=now,point_timestamps=ts,
                eef_position=z['position_world']-anchor,eef_orientation=Rotation.from_matrix(z['rotation_world']).as_quat(),
                source_frame=constants['SOURCE_FRAME'],tcp_semantic_id=constants['TCP_SEMANTIC_ID'],validity_start=now,validity_end=float(ts[-1]),gripper=np.full(40,.5))
            actual=gate.evaluate(chunk,now=now,achieved_position=np.asarray(row['TCP'])-anchor,achieved_orientation=Rotation.from_matrix(row['rotation']).as_quat()).as_dict()
            for key in ('decision','issues','provenance'):assert actual[key]==w['gate'][key],(w['id'],key)
            error=difference(actual['metrics'],w['gate']['metrics']);assert error<=1e-6,(w['id'],error);maximum=max(maximum,error)
            if not w['admitted']:assert not (Path(w['proposal']).parent/'reference_only_install.json').exists(),'rejected window installed'
        checked.append(dict(case=case['case_id'],windows=len(windows),maximum_absolute_metric_difference=maximum,source_window_manifest_sha256=sha(out/'windows.json')))
    destination=RAW/'delivery'/stage;destination.mkdir(parents=True,exist_ok=True);receipt=destination/'CPU_GATE_REPLAY.json';assert not receipt.exists()
    report=dict(status='PASS_SAVED_ACTUAL_H40_ORIGINAL_GATE_CPU_REPLAY',stage=stage,checks=checked,windows=sum(r['windows'] for r in checked),
        same_owner_shared_gate_implementation=True,gate_modified=False,source_api_sha256=sha(api_path),code_sha256=sha(__file__),new_physics=0,model_queries=0,optimizer_updates=0)
    write(receipt,report);print(json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)


def witness_replay(stage):
    verdict=read(ROOT/'evidence'/('DIAGNOSTIC_VERDICT.json' if stage=='diagnostics' else 'COLLECTION_VERDICT.json'))
    assert verdict['all_registered_cases_valid'] and len(verdict['rows'])==(16 if stage=='diagnostics' else 96)
    source=read(ROOT/'evidence/SOURCE_IDENTITY.json');path=next(x['path'] for x in source['files'] if x['name']=='native_urdf')
    urdf=ET.parse(path).getroot();kin=NativeArmKinematics.from_urdf(path)
    velocity=np.asarray([float(next(j for j in urdf.findall('joint') if j.attrib['name']==name).find('limit').attrib['velocity']) for name in kin.joint_names])
    checks=[]
    for parent in verdict['rows']:
        result=parent['result'];out=Path(parent['attempt'])
        if result['case']['expert']!='OBSERVABLE_CORRECTION_V1':continue
        states=read(out/'rows.json');times=np.asarray([r['time'] for r in states]);count=0;max_pe=0.;max_re=0.
        for w in read(out/'windows.json'):
            if not w['admitted']:continue
            assert sha(w['proposal'])==w['proposal_sha256']
            z=dict(np.load(w['proposal'],allow_pickle=False));row=states[int(np.argmin(abs(times-w['timestamp'])))];assert abs(row['time']-w['timestamp'])<=1e-6
            root=np.asarray(row['root']);transform=homogeneous(Rotation.from_quat(root[3:7]).as_matrix(),root[:3])
            joints=dict(zip(result['dof_names'],np.asarray(row['DOF']).reshape(-1,2)[:,0]));previous=np.asarray([joints[n] for n in kin.joint_names])
            witness=w['checks']['rows'];assert len(witness)==40 and w['checks']['complete_H40']
            for index,entry in enumerate(witness):
                assert entry['index']==index and entry['accepted'] and entry['IK_ok'] and entry['hard_ok'] and entry['velocity_ok'] and not entry['hits']
                q=np.asarray(entry['candidate']);assert np.array_equal(previous,np.asarray(entry['start']))
                assert np.all(q>=kin.joint_limits[:,0]-1e-3) and np.all(q<=kin.joint_limits[:,1]+1e-3)
                assert index==0 or np.all(abs(q-previous)/.05<=velocity+1e-8)
                values=dict(joints);values.update(zip(kin.joint_names,q))
                pose=full_fk(urdf,values,transform)['x5_link6']@homogeneous(np.eye(3),kin.tcp_offset_xyz)
                pe=float(np.linalg.norm(pose[:3,3]-z['position_world'][index]));re=float(so3(pose[:3,:3],z['rotation_world'][index]))
                assert pe<=.002 and re<=.01,(w['id'],index,pe,re)
                max_pe=max(max_pe,pe);max_re=max(max_re,re);previous=q
            count+=1
        checks.append(dict(case=result['case']['case_id'],windows=count,points=count*40,maximum_xyz_m=max_pe,maximum_so3_rad=max_re,window_manifest_sha256=sha(out/'windows.json')))
    destination=RAW/'delivery'/stage;destination.mkdir(parents=True,exist_ok=True);receipt=destination/'CPU_LOCAL_WITNESS_REPLAY.json';assert not receipt.exists()
    report=dict(status='PASS_SAVED_LOCAL_H40_FULL_URDF_FK_WITNESS_REPLAY',stage=stage,checks=checks,windows=sum(x['windows'] for x in checks),points=sum(x['points'] for x in checks),
        distinction='Separate full-tree FK of saved joint solutions; retained producer collision checks. No new IK, collision search or dynamics guarantee.',
        actual_root_fixed_only_for_conservative_local_witness=True,failed_solution_propagation=False,URDF_sha256=sha(path),code_sha256=sha(__file__),new_physics=0,model_queries=0,optimizer_updates=0)
    write(receipt,report);print(json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)


def attempt_accounting(out,stage,caps):
    result=read(out/'result.json');started=(out/'physics_started.json').exists()
    if result['validity']!='VALID':
        assert not started,'physical engineering-invalid attempt requires separate raw reconciliation'
        return dict(attempt=str(out),validity=result['validity'],starts=0,control_ticks=0,actor_calls=0,result_sha256=sha(out/'result.json'))
    assert started
    implicit=round(result['initialization_sim_s']/caps['control_dt']);assert implicit>=1
    ticks=result.get('task_ticks',0)+implicit;actors=result['WBC_actor_calls']
    if stage!='resets':
        with (out/'physics_steps.jsonl').open('rb') as stream:assert sum(1 for _ in stream)==ticks
        assert len(read(out/'actor_calls.json'))==actors==2*result['task_ticks']
        assert sum(1 for _ in (out/'actor_inputs').glob('*.npz'))==actors
    else:assert actors==0 and ticks*caps['control_dt']<=1+1e-6
    return dict(attempt=str(out),validity='VALID',ordinal=read(out/'physics_started.json')['ordinal'],starts=1,
        implicit_ticks=implicit,control_ticks=ticks,actor_calls=actors,result_sha256=sha(out/'result.json'))


def accounting():
    import fcntl
    assert read(ROOT/'evidence/COLLECTION_VERDICT.json')['all_registered_cases_valid']
    assert len(read(ROOT/'evidence/COLLECTION_VERDICT.json')['rows'])==96
    caps=read(ROOT/'design/BUDGET_CARD.json');rows=[];ordinals=[]
    for stage in ('resets','diagnostics','collection'):
        for out in sorted((RAW/stage).glob('*/attempt-*')):
            row=attempt_accounting(out,stage,caps);rows.append(row)
            if row['starts']:ordinals.append(row['ordinal'])
    assert sorted(ordinals)==list(range(1,len(ordinals)+1)) and len(ordinals)==116
    totals={key:sum(r[key] for r in rows) for key in ('starts','control_ticks','actor_calls')}
    totals['simulate_calls']=totals['control_ticks']*caps['decimation']
    with (ROOT/'physics.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH);ledger=read(ROOT/'evidence/RAW_BUDGET.json')
    for key,value in totals.items():assert ledger[key]==value,(key,ledger[key],value)
    assert abs(ledger['simulated_seconds']-totals['control_ticks']*float(np.float32(caps['control_dt'])))<=1e-5
    for key in ('starts','control_ticks','actor_calls','simulate_calls','simulated_seconds'):assert ledger[key]<=caps['hard_'+key]
    destination=ROOT/'evidence/TEACHER_RAW_ACCOUNTING.json';assert not destination.exists()
    receipt=dict(status='PASS_ALL_TEACHER_ATTEMPTS_RAW_COUNTER_RECONCILIATION',attempts=rows,ledger=ledger,
        integration_substeps=totals['simulate_calls']*caps['sim_substeps'],caps_sha256=sha(ROOT/'design/BUDGET_CARD.json'),code_sha256=sha(__file__),
        implicit_initialization_counted=True,engineering_invalid_retained=True,model_queries=0,optimizer_updates=0,new_physics=0)
    write(destination,receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='attempts'}),flush=True)


def archive_members(out,files):
    import hashlib
    manifest=out/'MEMBER_MANIFEST.json'
    current={name:p for p,name in files}
    if manifest.exists():
        saved=read(manifest);members=saved['files']
        assert len(members)==saved['members'] and len({r['path'] for r in members})==len(members)
        files=[(current[r['path']],r['path']) for r in members]
        for (p,_),row in zip(files,members):
            assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
    else:
        members=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for p,name in files]
    tree=hashlib.sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if manifest.exists():assert saved['tree_sha256']==tree
    else:write(manifest,dict(files=members,tree_sha256=tree,members=len(members),private_WBC_assets_or_weights=False))
    return files,members,tree


def archive():
    import tarfile
    import hashlib
    import requests
    from huggingface_hub import HfApi,hf_hub_url
    existing=ROOT/'evidence/COLLECTION_PUBLIC_RECEIPT.json'
    if existing.exists():
        saved=read(existing);assert saved['status']=='PASS_PUBLIC_FIXED_REVISION_ALL_FILES_AND_MEMBERS'
        print(json.dumps(saved),flush=True);return
    assert read(ROOT/'evidence/COLLECTION_VERDICT.json')['all_registered_cases_valid']
    for stage in ('diagnostics','collection'):
        assert read(RAW/'delivery'/stage/'CPU_REPLAY.json')['status']=='PASS_SAME_OWNER_FULL_TREE_FK_AND_SHARED_SCORER_CPU_REPLAY'
        assert read(RAW/'delivery'/stage/'VIDEO_RECEIPT.json')['status']=='PASS_ACTUAL_CAMERA_VIDEO_FULL_DECODE'
    assert (ROOT/'evidence/WINDOW_ADMISSION.json').exists()
    out=RAW/'archive/collection';out.mkdir(parents=True,exist_ok=True);files=[]
    for base,folders in ((ROOT,('code','design','evidence','logs','noise')),(RAW,('resets','diagnostics','collection','delivery'))):
        for folder in folders:
            for path in sorted((base/folder).rglob('*')):
                if not path.is_file() or path.suffix not in ('.py','.json','.jsonl','.npz','.csv','.md','.log','.mp4'):continue
                assert not path.is_symlink() and path.resolve().is_relative_to(base.resolve())
                files.append((path,('small/' if base==ROOT else 'raw/')+path.relative_to(base).as_posix()))
    files,members,tree=archive_members(out,files)
    package=out/'PI0266_ACTUAL_TEACHER_AND_FEEDBACK_EVIDENCE.tar'
    if not package.exists():
        partial=package.with_suffix('.tar.partial')
        with tarfile.open(partial,'w') as tar:
            for (p,name),row in zip(files,members):
                info=tarfile.TarInfo(name);info.size=row['bytes'];info.mode=0o644;info.mtime=0
                with p.open('rb') as stream:tar.addfile(info,stream)
        partial.replace(package)
    import shutil
    for stage in ('diagnostics','collection'):
        for name in ('REPORT.md','TEACHER_QUALITY.csv','PI0266_ACTUAL_TEACHERS.mp4'):
            shutil.copy2(RAW/'delivery'/stage/name,out/(stage+'-'+name))
    shutil.copy2(ROOT/'evidence/WINDOW_ADMISSION.json',out/'WINDOW_ADMISSION.json')
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0266/state-conditioned-feedback-20261001'
    api=HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip());assert api.model_info(repo).private is False
    upload=RAW/'archive/COLLECTION_UPLOAD_COMMIT.json'
    if upload.exists():
        previous=read(upload);assert previous['tree_sha256']==tree;revision=previous['revision']
    else:
        revision=api.upload_folder(repo_id=repo,folder_path=str(out),path_in_repo=prefix,allow_patterns=[p.name for p in out.iterdir() if p.is_file()],
            commit_message='PI0266 actual fixed teacher feedback windows and all failures').oid
        write(upload,dict(repo_id=repo,prefix=prefix,revision=revision,tree_sha256=tree))
    session=requests.Session();session.trust_env=False;checked=[]
    for p in sorted(p for p in out.iterdir() if p.is_file()):
        h=hashlib.sha256();size=0
        with session.get(hf_hub_url(repo,prefix+'/'+p.name,revision=revision),stream=True,timeout=(30,180)) as response:
            response.raise_for_status();assert 'Authorization' not in response.request.headers
            for block in response.iter_content(8<<20):h.update(block);size+=len(block)
        assert size==p.stat().st_size and h.hexdigest()==sha(p),p.name
        checked.append(dict(path=p.name,bytes=size,sha256=h.hexdigest(),anonymous=True))
    remaining={row['path']:row for row in members}
    # ponytail: stream the verified public tar; no second local download of the evidence tree.
    with session.get(hf_hub_url(repo,prefix+'/'+package.name,revision=revision),stream=True,timeout=(30,180)) as response:
        response.raise_for_status()
        response.raw.decode_content=True
        with tarfile.open(fileobj=response.raw,mode='r|') as tar:
            for member in tar:
                assert member.isfile() and member.name in remaining;row=remaining.pop(member.name);h=hashlib.sha256();size=0
                with tar.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(8<<20),b''):h.update(block);size+=len(block)
                assert size==row['bytes'] and h.hexdigest()==row['sha256'],member.name
    assert not remaining
    receipt=dict(status='PASS_PUBLIC_FIXED_REVISION_ALL_FILES_AND_MEMBERS',repo_id=repo,prefix=prefix,revision=revision,tree_sha256=tree,
        members=len(members),source_bytes=sum(r['bytes'] for r in members),files=checked,new_physics=0,GR00T_queries=0)
    write(existing,receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('diagnostics','collection'));parser.add_argument('--video',action='store_true');parser.add_argument('--gate',action='store_true');parser.add_argument('--witness',action='store_true');parser.add_argument('--accounting',action='store_true');parser.add_argument('--archive',action='store_true')
    args=parser.parse_args()
    if args.accounting:accounting()
    elif args.archive:archive()
    else:
        assert args.stage
        witness_replay(args.stage) if args.witness else gate_replay(args.stage) if args.gate else video(args.stage) if args.video else report(args.stage)
