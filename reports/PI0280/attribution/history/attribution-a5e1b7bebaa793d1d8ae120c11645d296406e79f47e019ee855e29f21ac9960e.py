"""CPU arithmetic on completed PI0280 raw evidence; never imports a model or simulator."""
import argparse,collections,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def stats(values):
    a=np.asarray(values,dtype=float)
    assert len(a) and np.isfinite(a).all()
    return dict(mean=float(a.mean()),rms=float(np.sqrt(np.mean(a*a))),p95=float(np.quantile(a,.95)),max=float(a.max()),final=float(a[-1]))


def quat_matrix(q):
    x,y,z,w=np.asarray(q,dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def orientation_error(actual,reference):
    # ponytail: fixture references have a fixed quaternion; moving orientation needs logged native targets.
    return np.arccos(np.clip((np.einsum('nij,ij->n',actual,reference)-1)/2,-1,1))


def file_row(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def video(directory,out,name):
    from PIL import Image,ImageDraw
    path=directory/'frames.npz'
    if not path.exists():return 'UNOBSERVED_NO_SAVED_FRAMES'
    with np.load(path,allow_pickle=False) as data:
        body,wrist,t=data['body'],data['wrist'],data['time']
    assert len(body)==len(wrist)==len(t) and len(t)>0
    fps=10 if len(t)<2 else 1/float(np.median(np.diff(t)))
    destination=out/(name+'.mp4');temporary=destination.with_suffix('.tmp.mp4')
    process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','768x448','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264',
        '-threads','4','-pix_fmt','yuv420p',str(temporary)],stdin=subprocess.PIPE)
    try:
        for left,right,stamp in zip(body,wrist,t):
            image=Image.new('RGB',(768,448),'black');draw=ImageDraw.Draw(image)
            draw.text((8,8),'DELIVERED_OLD_WBC / actual simulation / fixture / NON_REALTIME',fill='white')
            draw.text((8,30),name+' | saved sim time %.3fs'%stamp,fill='white')
            draw.text((8,52),'Body camera',fill='white');draw.text((392,52),'Wrist camera',fill='white')
            image.paste(Image.fromarray(left).resize((384,384),Image.Resampling.NEAREST),(0,64))
            image.paste(Image.fromarray(right).resize((384,384),Image.Resampling.NEAREST),(384,64))
            process.stdin.write(image.tobytes())
        process.stdin.close();assert process.wait()==0
    except BaseException:
        process.kill();process.wait();raise
    temporary.replace(destination)
    return dict(**file_row(destination),source=file_row(path),saved_frames=len(t),
        first_sim_time_s=float(t[0]),last_sim_time_s=float(t[-1]),fps=fps,
        rendering='Only actual saved body/wrist RGB; nearest pixel enlargement; timestamps preserved')


def fixture(directory,out,ledger,make_video):
    result=json.loads((directory/'RESULT.json').read_text());name=directory.name
    assert result['status']!='RUNNING'
    rows=json.loads((directory/'rows.json').read_text())
    actors=[json.loads(s) for s in (directory/'actor_outputs.jsonl').read_text().splitlines()] if (directory/'actor_outputs.jsonl').exists() else []
    origin=float(result.get('initialization_s',rows[0]['time'] if rows else 0))
    controlled=[r for r in rows if r['time']>origin+1e-7]
    report=dict(fixture=name,execution_status=result['status'],execution_error=result.get('error'),
        controller='DELIVERED_OLD_WBC',lane='WBC_NATIVE_FIXTURE',task_success='NOT_AN_OBJECT_TASK',
        selected='NONE',deployment='NONE',raw_result=result,initialization_sim_time_s=origin,
        saved_rows=len(rows),saved_controlled_rows=len(controlled),saved_actor_output_rows=len(actors),
        ledger_counts=dict(collections.Counter(r['kind'] for r in ledger if r.get('fixture')==name)),
        installed_orientation_tracking='UNOBSERVED_NATIVE_INSTALLED_ROTATION_NOT_RECORDED',
        object_contacts='UNOBSERVED_NOT_AN_OBJECT_TASK',object_completion='NOT_AN_OBJECT_TASK',
        metric_semantics='Saved installed=p_at(s_ref(traj_sim_time)) after step; pre-step dispatched nine-point preview is not saved. Timestamped raw reference is a distinct clock from spatial progress.',
        raw_inputs=[file_row(p) for p in directory.iterdir() if p.is_file() and p.name!='frames.npz'])
    if rows:
        report['initialization_row']=dict(sim_time_s=rows[0]['time'],FK_cache_error_m=rows[0]['FK_cache_error_m'],
            hard_max_penetration=max(rows[0]['hard_penetration'].values()),
            soft_margin_min=min(rows[0]['telemetry']['joint_limit_margin']))
    if controlled:
        t=np.array([r['time'] for r in controlled]);actual=np.array([r['TCP'] for r in controlled]);installed=np.array([r['installed'] for r in controlled])
        with np.load(directory/'reference.npz',allow_pickle=False) as reference:
            p,q,tref=reference['position'],reference['quaternion'],reference['timestamp']
        assert np.isfinite(actual).all() and np.isfinite(installed).all() and np.all(np.diff(t)>0)
        assert np.allclose(np.abs(q@q[0]),1,atol=1e-6),'Only fixed-orientation fixtures are admitted here'
        relative=t-origin;assert relative.min()>=tref[0]-1e-6 and relative.max()<=tref[-1]+1e-6
        raw=np.stack([np.interp(relative,tref,p[:,j]) for j in range(3)],axis=1)
        names=list(controlled[0]['hard_penetration']);margins=np.array([r['telemetry']['joint_limit_margin'] for r in controlled]);hard=np.array([[r['hard_penetration'][j] for j in names] for r in controlled])
        assert margins.shape==hard.shape and margins.shape[1]==len(names)
        report.update(last_saved_sim_time_s=float(t[-1]),saved_controlled_elapsed_s=float(t[-1]-origin),
            actual_to_same_tick_installed_position_m=stats(np.linalg.norm(actual-installed,axis=1)),
            actual_to_timestamped_raw_position_m=stats(np.linalg.norm(actual-raw,axis=1)),
            same_tick_installed_to_timestamped_raw_position_m=stats(np.linalg.norm(installed-raw,axis=1)),
            actual_to_raw_fixed_orientation_rad=stats(orientation_error(np.array([r['rotation'] for r in controlled]),quat_matrix(q[0]))),
            FK_cache_error_m=stats([r['FK_cache_error_m'] for r in controlled]),
            hard_penetration=dict(unit='rad_for_revolute_m_for_prismatic',rows_with_any=int(np.any(hard>0,axis=1).sum()),per_joint_max=dict(zip(names,hard.max(axis=0).tolist()))),
            inherited_soft_margins=dict(rows_with_any_negative=int(np.any(margins<0,axis=1).sum()),per_joint_min=dict(zip(names,margins.min(axis=0).tolist())),
                per_joint_negative_rows=dict(zip(names,(margins<0).sum(axis=0).tolist())),meaning='Inherited 0.9 range soft margins; negative does not imply URDF hard penetration'),
            gripper_actual_final_m=dict(zip(names[-2:],controlled[-1]['q'][-2:])),
            telemetry_flags={k:sum(bool(r['telemetry'].get(k)) for r in controlled) for k in ('invalid_state','fall_flag','joint_limit_violation')})
    if make_video:report['video']=video(directory,out,name)
    write(out/(name+'.json'),report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0280'));parser.add_argument('--video',action='store_true');parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:
        assert np.allclose(quat_matrix([0,0,0,1]),np.eye(3))
        assert np.allclose(orientation_error(np.array([np.eye(3),quat_matrix([0,0,1,0])]),np.eye(3)),[0,np.pi])
        assert stats([3,4])['rms']==np.sqrt(12.5)
        print('PASS_CPU_ARITHMETIC_SELF_CHECK');return
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','Hide CUDA explicitly for this CPU-only reader'
    root=args.root;out=root/'attribution';out.mkdir(exist_ok=True)
    binding=json.loads((root/'evidence/WBC_BINDING.json').read_text())
    assert sha(root/'design/WBC_CONFIG.json')==binding['configuration_sha256']
    identity=[dict(**r,current_sha256=sha(r['path'])) for r in binding['files']]
    assert all(r['current_sha256']==r['sha256'] for r in identity),'Frozen WBC identity changed'
    ledger=[json.loads(s) for s in (root/'evidence/wbc_budget.jsonl').read_text().splitlines()]
    reports=[fixture(p,out,ledger,args.video) for p in sorted((root/'wbc').iterdir()) if (p/'RESULT.json').is_file()]
    summary=dict(schema='PI0280_CPU_RAW_ATTRIBUTION_V1',observed_unix_s=time.time(),implementation=file_row(Path(__file__)),
        status='COMPLETED_RAW_AUDITED_WITH_EXECUTION_OUTCOMES_RETAINED',binding=file_row(root/'evidence/WBC_BINDING.json'),
        configuration=file_row(root/'design/WBC_CONFIG.json'),frozen_WBC_identity_files=identity,
        completed_or_partial_fixtures=[r['fixture'] for r in reports],
        original_wbc_ledger=file_row(root/'evidence/wbc_budget.jsonl'),cumulative_wbc_ledger_counts=dict(collections.Counter(r['kind'] for r in ledger)),
        ledger_semantics='Reservation is written before invocation. Failed operations count; saved returned rows independently report completed telemetry. Existing ledgers untouched.',
        new_policy_calls=0,new_WBC_actor_calls=0,new_physics_steps=0,new_training_updates=0,selected='NONE',deployment='NONE')
    write(out/'SUMMARY.json',summary)
    lines=['| Fixture | Status | Saved control ticks | Installed tracking final/max (mm) | Raw orientation final/max (rad) | Hard penetration rows |','|---|---|---:|---:|---:|---:|']
    for r in reports:
        position=r.get('actual_to_same_tick_installed_position_m',{});rotation=r.get('actual_to_raw_fixed_orientation_rad',{})
        lines.append('| %s | %s | %d | %.3f / %.3f | %.4f / %.4f | %s |'%(r['fixture'],r['execution_status'],r['saved_controlled_rows'],position.get('final',0)*1000,position.get('max',0)*1000,rotation.get('final',0),rotation.get('max',0),r.get('hard_penetration',{}).get('rows_with_any','UNOBSERVED')))
    (out/'FIXTURES.md').write_text('\n'.join(lines)+'\n\nFixtures do not test object success. Same-tick installed tracking and timestamped raw-reference tracking use different reference clocks. Native installed orientation is UNOBSERVED.\n')
    print(json.dumps(dict(summary=str(out/'SUMMARY.json'),fixtures=[{k:r[k] for k in ('fixture','execution_status','saved_controlled_rows','actual_to_same_tick_installed_position_m','actual_to_raw_fixed_orientation_rad','hard_penetration','inherited_soft_margins')} for r in reports])))


if __name__=='__main__':main()
