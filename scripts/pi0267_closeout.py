"""Read-only CPU replay and compact public seal for the authorized S1 stop."""
import argparse
import ast
import csv
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267')
sys.path.insert(0,str(BASE/'PI0250/runtime/reference-317a/src'))
sys.path.append(str(BASE/'PI0266/code'))
from pi0266_metrics import score
from pi0266_expert import reference
from pi0265_metrics import so3
from pi0255_motion3 import sha,write


def read(p):return json.loads(Path(p).read_text())


def interp_saved(p,r,t):
    grid=np.arange(40)*.05;i=np.searchsorted(grid,t,side='right')-1;i=np.clip(i,0,38)
    u=(t-grid[i])/.05;positions=p[i]*(1-u[:,None])+p[i+1]*u[:,None]
    q=Rotation.from_matrix(r).as_quat();a=q[i];b=q[i+1].copy();dot=np.sum(a*b,axis=1)
    b[dot<0]*=-1;dot=abs(dot);theta=np.arccos(np.clip(dot,-1,1));s=np.sin(theta)
    out=np.empty_like(a);near=abs(s)<1e-8
    out[near]=a[near]*(1-u[near,None])+b[near]*u[near,None]
    far=~near
    out[far]=(np.sin((1-u[far])*theta[far])[:,None]*a[far]+np.sin(u[far]*theta[far])[:,None]*b[far])/s[far,None]
    out/=np.linalg.norm(out,axis=1)[:,None]
    x,y,z,w=out.T
    matrices=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]).transpose(2,0,1)
    return positions,matrices


def numpy_lag_rotation(desired,current,alpha):
    delta=desired@current.T
    v=np.array([delta[2,1]-delta[1,2],delta[0,2]-delta[2,0],delta[1,0]-delta[0,1]])*.5
    sine=np.linalg.norm(v);theta=np.arctan2(sine,(np.trace(delta)-1)*.5)
    if sine<1e-14:return current.copy()
    axis=v/sine;x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return (np.eye(3)+np.sin(alpha*theta)*K+(1-np.cos(alpha*theta))*(K@K))@current


def replay():
    verdict=read(ROOT/'evidence/S1_TEACHER_LIVENESS.json');assert len(verdict['episodes'])==64
    card=read(ROOT/'design/SCENE_CARD.json');reg=card['registry'];lookup={f['family_id']:f for f in card['families']}
    checks=[];entry=[]
    for rec in verdict['episodes']:
        path=RAW/'S1'/(rec['key']+'.npz');z=np.load(path);detail=read(path.with_suffix('.json'))
        assert len(z['time'])==2001 and z['label_p'].shape==(100,40,3) and z['label_r'].shape==(100,40,3,3)
        assert np.max(abs(np.diff(z['time'])-.02))<1e-12
        assert all(np.isfinite(z[k]).all() for k in ['actual_p','actual_r','installed_p','installed_r','label_p','label_r'])
        p=z['actual_p'][0].copy();r=z['actual_r'][0].copy();ps=[p.copy()];rs=[r.copy()]
        max_ip=0.;max_ir=0.;label_error=0.;label_rot_error=0.
        if rec['teacher']=='NEW':
            end=6. if rec['task']=='M1' else 16.;grid=np.arange(round(end/.01)+1)*.01
            cp,cr=reference(reg,lookup[rec['family']],rec['task'],rec['command'],grid)
            yaw=np.arctan2(cr[:,1,0],cr[:,0,0]);metric=np.c_[cp,.05*yaw]
            lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(metric,axis=0),axis=1))]
            keep=np.r_[True,np.diff(lengths)>0];cp,cr,yaw,metric,lengths=cp[keep],cr[keep],yaw[keep],metric[keep],lengths[keep]
            seg=np.diff(metric,axis=0);seg2=np.sum(seg**2,axis=1)
        for j,(lp,lr) in enumerate(zip(z['label_p'],z['label_r'])):
            assert np.linalg.norm(lp[0]-p)<1e-10 and so3(lr[0],r)<1e-9
            if rec['teacher']=='NEW':
                current=np.r_[p,.05*np.arctan2(r[1,0],r[0,0])]
                u=np.clip(np.sum((current-metric[:-1])*seg,axis=1)/seg2,0,1)
                projected=metric[:-1]+u[:,None]*seg;cost=np.sum((projected-current)**2,axis=1)
                index=int(np.argmin(cost));ell=lengths[index]+u[index]*(lengths[index+1]-lengths[index])
                future=np.minimum(lengths[-1],ell+.02*np.arange(40)*.05)
                bar=np.column_stack([np.interp(future,lengths,cp[:,axis]) for axis in range(3)])
                decay=np.exp(-2*np.arange(40)*.05)
                formula=bar+decay[:,None]*(p-bar[0]);label_error=max(label_error,float(np.max(np.linalg.norm(lp-formula,axis=1))))
                bar_yaw=np.interp(future,lengths,yaw)
                formula_yaw=bar_yaw+decay*(np.arctan2(r[1,0],r[0,0])-bar_yaw[0])
                label_rot_error=max(label_rot_error,float(np.max(abs(np.arctan2(lr[:,1,0],lr[:,0,0])-formula_yaw))))
            ip,ir=interp_saved(lp,lr,np.arange(1,21)*.02)
            for k,(dp,dr) in enumerate(zip(ip,ir)):
                at=j*20+k+1
                max_ip=max(max_ip,float(np.linalg.norm(dp-z['installed_p'][at])))
                max_ir=max(max_ir,float(so3(dr,z['installed_r'][at])))
                if rec['backend']=='E0':p,r=dp.copy(),dr.copy()
                else:
                    alpha=1-np.exp(-.02/.15);p+=alpha*(dp-p);r=numpy_lag_rotation(dr,r,alpha)
                ps.append(p.copy());rs.append(r.copy())
        xyz=float(np.max(np.linalg.norm(np.asarray(ps)-z['actual_p'],axis=1)))
        rotation=float(np.max(so3(np.asarray(rs),z['actual_r'])))
        assert xyz<1e-10 and rotation<1e-9 and max_ip<1e-10 and max_ir<1e-9
        assert label_error<=1e-5 and label_rot_error<=1e-4,(rec['key'],label_error,label_rot_error)
        spec=json.loads(str(z['spec_json']));result=score(z['time'],z['actual_p'],z['actual_r'],z['installed_p'],z['installed_r'],spec)
        assert {k:v for k,v in result.items() if k!='progress_rad'}==rec['result'],rec['key']
        checks.append(dict(key=rec['key'],raw_sha256=sha(path),executor_xyz_max_m=xyz,executor_so3_max_rad=rotation,
            interpolation_xyz_max_m=max_ip,interpolation_so3_max_rad=max_ir,
            independent_arclength_label_xyz_max_m=label_error,independent_arclength_label_yaw_max_rad=label_rot_error,
            shared_fixed_scorer_exact_result=True))
        if rec['teacher']=='NEW' and rec['task']=='M3':
            progress=z['actual_directed_progress_rad'];valid=np.flatnonzero(np.isfinite(progress));start=int(valid[0])
            th=progress[valid];greatest=np.maximum.accumulate(np.maximum(th,0));draw=(greatest-th)*spec['radius']
            bad=np.flatnonzero(draw>.001+1e-12);gap=np.flatnonzero((th*spec['radius']<-.001-1e-12)|((th-1.5*np.pi)*spec['radius']>.001+1e-12))
            first=None if not len(bad) else int(valid[bad[0]])
            first_gap=None if not len(gap) else int(valid[gap[0]])
            entry.append(dict(key=rec['key'],success=rec['success'],entry_s=float(z['time'][start]),
                source_phase_at_entry_s=detail['replans'][start//20]['source_phase_s'],
                reverse_first_s=None if first is None else float(z['time'][first]),
                gap_first_s=None if first_gap is None else float(z['time'][first_gap]),
                max_backward_excursion_m=float(draw.max()),endpoint_xyz_m=rec['result']['final_xyz_m'],
                actual_bins=rec['result']['ordered_bins'],result_source='Saved actual state; no new teacher/executor episode'))
    write(ROOT/'evidence/CPU_REPLAY.json',dict(status='PASS_64_SAVED_EPISODES_INDEPENDENT_EXECUTOR_AND_NEW_LABEL_REPLAY',
        episodes=64,executor_interpolation='Independent quaternion interpolation / NumPy Rodrigues lag; not producer Slerp/Rotation lag.',
        teacher_formula='Independent 4D polyline segment projection / scalar yaw formula on sealed source curves; no ArclengthCorrection import.',
        formula_scope='Canonical initial R=I and source yaw-only/constant rotations of the fixed 64 episodes; general noncommuting rotations checked separately in producer fixtures.',
        scorer='Shared unchanged PI0266 V2, explicitly not independent scoring implementation.',rows=checks,new_conditions=0,model_queries=0,training_updates=0,robot_physics_steps=0))
    write(ROOT/'evidence/M3_ENTRY_DECOMPOSITION.json',dict(rows=entry,parameters_changed=False,
        caveat='Source-time phase is metadata. Entry/reversal come from actual mathematical states, not projection differences. No post-hoc gate change.'))
    print(json.dumps(dict(status='CPU_REPLAY_PASS',episodes=64,max_xyz=max(x['executor_xyz_max_m'] for x in checks),
        max_rotation=max(x['executor_so3_max_rad'] for x in checks),max_label_xyz=max(x['independent_arclength_label_xyz_max_m'] for x in checks))),flush=True)


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verdict=read(ROOT/'evidence/S1_TEACHER_LIVENESS.json');replay_receipt=read(ROOT/'evidence/CPU_REPLAY.json')
    assert verdict['status']=='PRIMARY_BLOCKED_TEACHER_LIVENESS' and replay_receipt['episodes']==64
    prefix=[]
    for record in verdict['episodes']:
        if record['teacher']!='NEW':continue
        replans=read(RAW/'S1'/(record['key']+'.json'))['replans']
        active=[(a,b) for a,b in zip(replans,replans[1:]) if a['ell_m']<a['ell_end_m']-.008]
        advances=[b['ell_m']-a['ell_m'] for a,b in active]
        assert active and min(advances)>0 and min(a['actual_prefix_displacement_m'] for a,b in active)>0
        prefix.append(dict(key=record['key'],nonterminal_prefixes=len(active),minimum_observable_projection_advance_m=min(advances),
            minimum_actual_displacement_m=min(a['actual_prefix_displacement_m'] for a,b in active)))
    write(ROOT/'evidence/S1_PREFIX_PROGRESS.json',dict(status='PASS_ACTUAL_NONTERMINAL_PREFIX_PROGRESS_WITHOUT_HIDDEN_HISTORY',rows=prefix,
        bound='Preterminal replans at least .008 metric m from endpoint; terminal formula fixtures are separate.',
        caveat='Projection advance is not actual directed path progress. Neither is used to clamp teacher phase.',new_episodes=0,model_queries=0))
    out=RAW/'delivery';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for row in verdict['episodes']:
        r=row['result'];rows.append(dict(key=row['key'],family=row['family'],task=row['task'],command=row['command'],
            teacher=row['teacher'],backend=row['backend'],success=row['success'],elapsed_s=row['actual_elapsed_s'],
            entry_s=r['entry_time_s'],ordered_bins=len(r['ordered_bins']),reverse=r['reverse'],gap=r['gap_shortcut'],
            backward_m=r['max_backward_excursion_m'],final_xyz_m=r['final_xyz_m'],final_so3_rad=r['final_so3_rad'],
            maximum_projection_source_phase_jump_s=row['maximum_projection_source_phase_jump_s']))
    with (out/'TEACHER_QUALITY.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for col,family in enumerate([f'pi0265-family-{j:03d}' for j in range(4)]):
        for cmd in (0,1):
            ax=axes[cmd,col]
            for backend,color in [('E0','#2878b5'),('E1','#dc8325')]:
                rec=next(x for x in verdict['episodes'] if (x['family'],x['task'],x['command'],x['teacher'],x['backend'])==(family,'M3',cmd,'NEW',backend))
                z=np.load(RAW/'S1'/(rec['key']+'.npz'));spec=json.loads(str(z['spec_json']));p=z['actual_p']
                ax.plot(p[:,0],p[:,1],color=color,linewidth=1.6,label=f"{backend}: {'PASS' if rec['success'] else 'FAIL'}")
                start=np.flatnonzero(np.isfinite(z['actual_directed_progress_rad']))[0]
                ax.scatter(p[start,0],p[start,1],color=color,s=32,marker='x',zorder=4)
                if backend=='E0':
                    ax.scatter(p[0,0],p[0,1],color='black',s=20,marker='s');ax.scatter(*spec['goal'][:2],color='green',s=32)
            ax.set_aspect('equal');ax.set_title(f'{family[-3:]} / command {cmd}');ax.legend(fontsize=8);ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.grid(alpha=.2)
    fig.suptitle('PI0267 CPU mathematical teacher paths — NOT WBC / NOT learned policy\nNEW teacher; crosses mark V2 entry admission, green marker is endpoint',fontsize=13)
    fig.savefig(out/'PI0267_CPU_TEACHER_PATHS.png',dpi=150);plt.close(fig)
    old=read(ROOT/'evidence/S0_PREFIX_AUDIT.json')
    maxima={teacher:max(x['requested_installed_xyz_max_m'] for c in old['cases'] if c['case']['expert']==teacher for x in c['replans'])
        for teacher in ('CLOCK_DRIVEN','OBSERVABLE_CORRECTION_V1')}
    text='''# PI0267 fixed CPU liveness return — PRIMARY_PARTIAL

Authority: `PI0267_EXECUTOR_DECOUPLED_RECOVERY_LEARNING_GO_V1`, release `f21bca4ec5b004e968f285032fc57df904deebef`, canonical 5937193116.
Branch: `pi0267/executor-decoupled-recovery`, parent `48048ae37b76534bbbe8a0ba76a9c84ed793e7ea`.

## Fixed results

| Teacher | Executor | M1 | M3 | Total |
|---|---|---:|---:|---:|
| OLD | E0 exact | 8/8 | 1/8 | 9/16 |
| OLD | E1 lag 0.15 s | 8/8 | 1/8 | 9/16 |
| NEW | E0 exact | 8/8 | 1/8 | 9/16 |
| NEW | E1 lag 0.15 s | 8/8 | 1/8 | 9/16 |

All 64 original conditions completed 40 s with 100 repeated 0.4 s prefixes. First-valid results are retained, including every failure. No parameter/seed/geometry/score search followed the result.

Required NEW E0 16/16 is not met; required NEW E1 >=12/16 and each task >=6/8 is not met. Verdict: **PRIMARY_BLOCKED_TEACHER_LIVENESS / PRIMARY_PARTIAL**. This is a teacher/path-contract result, not a GR00T fine-tuning quality failure.

## What failed

Only family-000 / M3 command-0 passes, in both executors. The other seven M3 conditions fail in both OLD and NEW matrices. All NEW M3 endpoints are reached within 15 nm at 40 s, but endpoint accuracy cannot replace entry/order/direction qualification.

In five failing M3 geometries/commands per NEW backend, V2 first admits entry with directed angle already beyond its -1 mm path deadband; reverse/gap becomes true at that admission. In two others, continued approach before the true arc start produces a later negative directed excursion. The saved actual-state decomposition reports 2.45–3.97 mm maximum backward excursion for failing NEW cases. For this fixed canonical virtual initial pose, entry is admitted during the source curve's approach stage, with current-replan source phase <4 s. It is not evidence of a unique WBC/neural root cause. The original scorer and source geometry are unchanged; no entry rule, pose, path, phase history, or tolerance is revised after observing this result.

Large changes in source-time projection are separately saved. Source time is quintic-paced metadata; its difference is not physical directed arc progress or a measured tracking error. Neither executor uses task geometry or the scorer to modify execution. E0 tracks the linearly interpolated H40 / quaternion shortest interpolation at dt .02; E1 uses exactly the registered first-order law. Neither is robot dynamics or a second embodiment.

## Source and finite S0 closure

The fixed four DEV geometries and source canonical initial TCP are sealed before S1. The canonical initial rotation is identity. This is the declared virtual design; it is not the measured PI0266 WBC reset. The historical post-reset pose remains bound separately in the S0 raw evidence. General nonidentity/noncommuting rotation, anchor, repeated-input identity and terminal contraction are checked in formula fixtures.

S0 inspects 16 historical DEV diagnostics, first/middle/final frame ticks (0/1000/1980), 48 prefixes total. Requested H40 or once-installed CLOCK references, native installed same-tick samples, actual current/next TCP, projection metadata and reference-only install events are recorded separately. Physical state/history is preserved by the retained install receipts. At OBS replanning boundaries the next chunk overwrites the rotation sample: the original previous-installed position is retained, but that boundary rotation is explicitly excluded rather than reconstructed. Source hashes bind every selected proposal, table and frame.

The maximum sampled requested-to-installed position difference is CLOCK %.9f m and OBS %.9f m. These are finite interpolation/binding/timing comparisons, not a blanket installation-defect verdict. Actual-to-installed tracking is calculated directly at matching ticks. The supplied 1-D toy is reexecuted and matches its frozen numerical JSON; its 0.5 mm bias is chosen, not identified empirical WBC dynamics.

Staged reference/scene wrapper byte hashes differ from the pinned Git versions; the actually consumed `reference`, `quintic` and `scene` function ASTs are identical and both byte identities are recorded. The remaining selected source files match the pinned release hashes. A NumPy bool JSON serialization repair occurred before any matrix episode; it changes neither the formula nor parameters. Original and repaired source bindings are preserved.

## CPU reproduction and counters

All 64 raw episode arrays replay through an independent quaternion interpolation and NumPy Rodrigues executor path. NEW labels also replay through an independent piecewise 4D metric projection and scalar yaw formula for the actual canonical/yaw-only matrix. Worst actual position replay difference %.3g m; worst rotation %.3g rad; independent new-label position difference %.3g m. These are numerical reproduction checks, not new episodes. The unchanged PI0266 V2 scorer is shared, and is explicitly not claimed independent. Original negative/positive scorer fixtures plus a missed-entry negative pass.

Completed CPU matrix: 64 episodes / 128000 dt-.02 mathematical updates / 6400 H40 plans / 2560 mathematical seconds. The constructed six toy traces and producer formula fixtures are separate CPU evidence. One serialization failure precedes episodes. Rendering proposal count=0; new policy-frame pairs=0; GR00T queries/heads=0; optimizer/canary updates=0; WBC starts/physics steps=0. Budget is not transferred to a new hypothesis.

## Conditional stop and omitted downstream work

S2 virtual rendering/dataset/statistics: NOT_RUN_FAILED_S1_ADMISSION.
S3/S4 canaries/N4096/R4096/checkpoints: NOT_RUN_FAILED_S1_ADMISSION.
S5 192 virtual model episodes / 768 offline / legacy new panel: NOT_RUN_FAILED_S1_ADMISSION.
S6 eight expert +24 model WBC transfer: NOT_RUN_FAILED_PRIMARY_DEPENDENCY.
No EVAL image, label or score payload is opened. No new donor/model materialization, training submission, camera tuning, replacement family, physical trial, cleanup, mail, monitor, PR77 pointer or merge.

PI0266 remains PARTIAL with its already verified public archive unchanged. Donor L2048 is unchanged/protected. Formal selected_checkpoint=NONE and deployment_selected_checkpoint=NONE. No virtual research candidate is qualified. The protocol-authorized early scientific stop is delivered; the primary learning study is not called COMPLETE. A changed entry/path contract would require new scientific authorization.
''' % (maxima['CLOCK_DRIVEN'],maxima['OBSERVABLE_CORRECTION_V1'],max(x['executor_xyz_max_m'] for x in replay_receipt['rows']),
       max(x['executor_so3_max_rad'] for x in replay_receipt['rows']),max(x['independent_arclength_label_xyz_max_m'] for x in replay_receipt['rows']))
    (out/'REPORT.md').write_text(text)
    print(json.dumps(dict(status='REPORT_AND_FIXED_ARRAY_FIGURE_READY',rows=len(rows))),flush=True)


def archive():
    import requests
    from huggingface_hub import HfApi,hf_hub_url
    assert read(ROOT/'evidence/CPU_REPLAY.json')['episodes']==64
    assert read(ROOT/'evidence/S1_TEACHER_LIVENESS.json')['status']=='PRIMARY_BLOCKED_TEACHER_LIVENESS'
    out=RAW/'archive/public';out.mkdir(parents=True,exist_ok=True)
    files=[]
    for base,folders in [(ROOT,['code','design','evidence']),(RAW,['S0','S1','delivery'])]:
        for folder in folders:
            for p in sorted((base/folder).rglob('*')):
                if p.is_file() and p.suffix in ('.json','.npz','.md','.csv','.py','.png','.mp4'):
                    assert not p.is_symlink() and p.resolve().is_relative_to(base.resolve()) and 'token' not in p.name.lower()
                    files.append((p,('small/' if base==ROOT else 'raw/')+p.relative_to(base).as_posix()))
    # Reuse the mature manifest helper without importing its unrelated robot/FK code.
    source=BASE/'PI0266/code/pi0266_report.py';tree=ast.parse(source.read_text())
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='archive_members')
    scope=dict(read=read,write=write,sha=sha,json=json)
    exec(compile(ast.Module(body=[helper],type_ignores=[]),str(source),'exec'),scope)
    files,members,digest=scope['archive_members'](out,files)
    package=out/'PI0267_FIXED_CPU_TEACHER_EVIDENCE.tar'
    if not package.exists():
        partial=package.with_suffix('.tar.partial')
        with tarfile.open(partial,'w') as tar:
            for (p,name),row in zip(files,members):
                info=tarfile.TarInfo(name);info.size=row['bytes'];info.mode=0o644;info.mtime=0
                with p.open('rb') as stream:tar.addfile(info,stream)
        partial.replace(package)
    for name in ['REPORT.md','TEACHER_QUALITY.csv','PI0267_CPU_TEACHER_PATHS.png']:
        shutil.copyfile(RAW/'delivery'/name,out/name)
    repo='leoyuer/pi0246-gate-pass-campaign';prefix='evidence/PI0267/executor-decoupled-recovery-20261002'
    api=HfApi(token=Path('/home/liuyu-nfs/.config/huggingface/token').read_text().strip());assert api.model_info(repo).private is False
    commit=RAW/'archive/UPLOAD_COMMIT.json'
    if commit.exists():
        prior=read(commit);assert prior['tree_sha256']==digest;revision=prior['revision']
    else:
        revision=api.upload_folder(repo_id=repo,folder_path=str(out),path_in_repo=prefix,
            commit_message='PI0267 fixed 64 CPU teacher liveness comparison, valid early stop, all raw negatives').oid
        write(commit,dict(repo_id=repo,prefix=prefix,revision=revision,tree_sha256=digest))
    session=requests.Session();session.trust_env=False;checked=[]
    for p in sorted(p for p in out.iterdir() if p.is_file()):
        h=hashlib.sha256();size=0
        with session.get(hf_hub_url(repo,prefix+'/'+p.name,revision=revision),stream=True,timeout=(30,180)) as response:
            response.raise_for_status();assert 'Authorization' not in response.request.headers
            for block in response.iter_content(8<<20):h.update(block);size+=len(block)
        assert size==p.stat().st_size and h.hexdigest()==sha(p),p.name
        checked.append(dict(path=p.name,bytes=size,sha256=h.hexdigest(),anonymous=True))
    remaining={r['path']:r for r in members}
    with session.get(hf_hub_url(repo,prefix+'/'+package.name,revision=revision),stream=True,timeout=(30,180)) as response:
        response.raise_for_status();response.raw.decode_content=True
        with tarfile.open(fileobj=response.raw,mode='r|') as tar:
            for member in tar:
                assert member.isfile() and member.name in remaining;row=remaining.pop(member.name);h=hashlib.sha256();size=0
                with tar.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(8<<20),b''):h.update(block);size+=len(block)
                assert size==row['bytes'] and h.hexdigest()==row['sha256'],member.name
    assert not remaining
    receipt=dict(status='PASS_PUBLIC_FIXED_REVISION_ALL_FILES_AND_MEMBERS',repo_id=repo,prefix=prefix,revision=revision,
        tree_sha256=digest,members=len(members),source_bytes=sum(r['bytes'] for r in members),files=checked,
        scientific_status='PRIMARY_PARTIAL / PRIMARY_BLOCKED_TEACHER_LIVENESS',model_queries=0,updates=0,robot_physics_steps=0)
    write(ROOT/'archive/PUBLIC_RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['replay','report','archive']);args=parser.parse_args()
    {'replay':replay,'report':report,'archive':archive}[args.action]()
