"""Independent immutable raw/score/control/video audit, including explicit takeover."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.score import score
from .scope import validate,Ledger


def video(folder,output,n):
    ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:raise RuntimeError('EXISTING_CPU_FFMPEG_REQUIRED')
    subprocess.run([ffmpeg,'-nostdin','-v','error','-n','-threads','1','-filter_complex_threads','1',
        '-framerate','20','-i',str(folder/'body_rgb/%04d.png'),'-framerate','20','-i',str(folder/'wrist_rgb/%04d.png'),
        '-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-frames:v',str(n),'-c:v','libx264',
        '-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(output)],check=True,timeout=240)
    output.chmod(0o600)
    info=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,nb_frames,duration','-of','json',str(output)],text=True))['streams'][0]
    decoded=subprocess.check_output([ffmpeg,'-nostdin','-v','error','-threads','1','-i',str(output),'-map','0:v:0','-f','framemd5','-'],text=True,timeout=240)
    count=sum(bool(line.strip()) and not line.startswith('#') for line in decoded.splitlines())
    if count!=n or int(info['nb_frames'])!=n or (info['width'],info['height'])!=(512,256):raise ValueError('ACTUAL_VIDEO_FULL_DECODE_FRAME_COUNT')
    return dict(sha256=sha(output),bytes=output.stat().st_size,actual_paired_frames=n,decoded_frames=count,fps=20,
        playback_duration_s=float(info['duration']),padding_or_interpolation=False,raw_video_private=True)


def episode(root,identifier,*,adjudication_name='E1_ADJUDICATION.json'):
    if adjudication_name not in ['E1_ADJUDICATION.json','MP1_ADJUDICATION.json','PV1_ADJUDICATION.json']:
        raise PermissionError('EXACT_VERSIONED_ADJUDICATION_ONLY')
    root=Path(root);folder=root/'episodes'/identifier;out=root/'evidence/audits'/identifier
    result_path=folder/'RESULT.json';adjudication_path=folder/adjudication_name
    if result_path.exists() and adjudication_path.exists():
        raw,decision=read(result_path),read(adjudication_path)
        if raw['physics_steps']==0 and decision['status']=='ENGINEERING_INCOMPLETE':
            from .zero_step_audit import audit
            return audit(root,identifier,video,adjudication_name=adjudication_name)
    if out.exists():return read(out/'AUDIT.private.json')
    out.mkdir(parents=True,mode=0o700);checks={};errors=[]
    def check(name,value):
        checks[name]=bool(value)
        if not value:errors.append(name)
    value=dict(attempt_id=identifier)
    try:
        card,result,adjudication=[read(folder/name) for name in ['CARD.json','RESULT.json',adjudication_name]]
        worker=read(folder/'WORKER_RECEIPT.json');reservation=read(folder/'RESERVATION.json');s1=read(folder/'S1_RESULT.json')
        manifest=read(folder/'FILE_MANIFEST.json')
        check('raw_manifest_all_sizes_and_hashes',all((folder/x['path']).is_file() and (folder/x['path']).stat().st_size==x['bytes'] and sha(folder/x['path'])==x['sha256'] for x in manifest))
        check('actual_execution_source_not_later_root_manifest',result['source_sha']==worker['source_sha']==reservation['source_sha'])
        check('adjudication_result_hash',adjudication['result_sha256']==sha(folder/'RESULT.json'))
        samples=read(folder/'samples.json');references=read(folder/'references.json');refs={x['ref_id']:x for x in references}
        maximum=0.;clock_maximum=0.;committed=0
        for index,sample in enumerate(samples):
            label=sample['supervision']
            if label is None:continue
            ref=refs[label['ref_id']];point=label['point_index'];planned=np.asarray(ref['planned_H40'][point]);opening=ref['gripper_H40'][point]
            for key in ['desired','accepted','installed','committed']:maximum=max(maximum,float(np.max(np.abs(np.asarray(label[key])-planned))))
            maximum=max(maximum,abs(label['committed_gripper']-opening));clock_maximum=max(clock_maximum,abs(label['command_time_s']-index*.05))
            committed+=1
        check('original_submitted_installed_committed_prefix_labels',maximum<=1e-12)
        check('original_20Hz_clock_tolerance',clock_maximum<=1e-8)
        with np.load(folder/'physics.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        recomputed=score(arrays,card,complete=len(arrays['time_s'])==641)
        recomputed.update(physics_sha256=sha(folder/'physics.npz'),raw_sample_count=len(arrays['time_s']))
        check('independent_saved_state_score_exact',recomputed==read(folder/'SCORE.json'))
        steps=result['physics_steps'];contact=arrays['contact_audit'];clock=arrays['contact_time_s']
        check('actual_contact_tick_coverage',contact.shape==(steps,5) and clock.shape==(steps,) and np.isfinite(contact).all() and np.allclose(clock,np.arange(1,steps+1)*.002,atol=1e-8,rtol=0))
        with np.load(folder/'highrate.npz',allow_pickle=False) as z:high=z['rows'].copy()
        check('actual_highrate_clock_and_count',len(high)==steps and np.isfinite(high).all() and np.allclose(high[:,0],np.arange(1,steps+1)*.002,atol=1e-8,rtol=0))
        flags=int(np.count_nonzero(high[:,-1]));forecasts=read(folder/'S1_PREFLIGHTS.json') if (folder/'S1_PREFLIGHTS.json').exists() else []
        check('original_S1_no_action_modification',s1['actions_modified'] is False)
        check('original_S1_live_shadow_parity',s1['maximum_live_shadow_difference']<=1e-10)
        check('shadow_did_not_mutate_live',all(x['live_state_unchanged'] for x in forecasts))
        check('S1_rejection_accounting',sum(x['decision']=='REJECT' for x in forecasts)==s1['pre_execution_rejections'])
        takeover=read(folder/'INTERVENTION_RESULT.json') if (folder/'INTERVENTION_RESULT.json').exists() else None
        if takeover and takeover['switches']:
            anchor=read(folder/'INTERVENTION.private.json');tick=anchor['tick'];acks=read(folder/'EXPERT_PREFIX_ACKS.json') if (folder/'EXPERT_PREFIX_ACKS.json').exists() else []
            check('student_rejected_before_takeover',anchor['original_student_S1_rejection']['decision']=='REJECT' and anchor['student_actual_prefixes']*200==tick)
            check('rejected_student_has_no_actual_ACK',not (folder/'relay/acks'/f"{anchor['sequence']:03d}.json").exists())
            for i,ack in enumerate(acks):
                check('expert_ACK_'+str(i),ack['tick_before']==tick and ack['tick_after']==tick+200 and sha(folder/('expert-prefix-'+str(i).zfill(3)+'.npz'))==ack['physical_receipt_sha256']);tick+=200
            check('expert_actual_ticks_accounted',tick<=steps and sum(a['expert_commands'] for a in acks)==takeover['expert_executed_commands'])
        elif s1['pre_execution_rejections']:
            intents=sorted((folder/'relay/intents').glob('*.json'));last=read(intents[-1]) if intents else {}
            check('rejected_prefix_zero_live_execution',steps==last.get('tick_before') and flags==0)
        paths=[sorted((folder/c).glob('*.png')) for c in ['body_rgb','wrist_rgb']];n=len(paths[0])
        check('paired_actual_camera_names',[p.name for p in paths[0]]==[p.name for p in paths[1]])
        check('actual_RGB_count',n==worker['rgb_pairs'] and n in [len(samples),len(samples)+1])
        for camera in range(2):
            for index in sorted({0,n-1}) if n else []:
                with Image.open(paths[camera][index]) as image:check('RGB_'+str(camera)+'_'+str(index),image.mode=='RGB' and image.size==(256,256) and any(hi>lo for lo,hi in image.getextrema()))
        movie=video(folder,out/'actual-body-wrist.mp4',n) if n else None
        prefix={'MP1_ADJUDICATION.json':'MP1_MAIN/','E1_ADJUDICATION.json':'E1_MAIN/','PV1_ADJUDICATION.json':'PV1_MAIN/'}[adjudication_name]
        if movie:movie.update(logical_id=prefix+identifier,relative_path=str((out/'actual-body-wrist.mp4').relative_to(root)),native_duration_s=steps*.002,initial_frame_included=True)
        value.update(status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT' if not errors else 'FAIL_PRESERVED_AUDIT',
            adjudication=adjudication,score=recomputed,video=movie,actual_native_safety_flag_rows=flags,
            committed_20Hz_command_rows=committed,maximum_reference_label_difference=maximum,
            maximum_20Hz_clock_difference_s=clock_maximum,
            robot_object_contact_ticks=int(np.count_nonzero(contact[:,0])),bilateral_grasp_ticks=int(np.count_nonzero(contact[:,1])),
            object_XY_maximum_displacement_m=float(np.max(np.linalg.norm(arrays['object_pose'][:,:2,3]-arrays['object_pose'][0,:2,3],axis=1))),
            object_maximum_lift_m=float(np.max(arrays['object_pose'][:,2,3]-arrays['object_pose'][0,2,3])),
            model_autonomous_evaluation=not card.get('e1_intervention_collection',False),execution_source_sha=result['source_sha'])
    except Exception as exc:value.update(status='ENGINEERING_AUDIT_INCOMPLETE',error=type(exc).__name__+': '+str(exc))
    value.update(checks=checks,errors=errors);write(out/'AUDIT.private.json',value);return value


def run(root):
    root=Path(root);validate(root,'4080-simulation');ledger=Ledger(root,'core');ledger.guard(closeout=True)
    if read(ledger.path)['reservations']:raise PermissionError('NO_EVIDENCE_CLOSEOUT_WITH_UNSETTLED_PHYSICS')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));started=time.process_time();child=resource.getrusage(resource.RUSAGE_CHILDREN);rows=[]
    try:
        for folder in sorted((root/'episodes').iterdir()):
            ledger.guard(closeout=True)
            if (folder/'E1_ADJUDICATION.json').exists():rows.append(episode(root,folder.name))
        value=dict(status='PASS_ALL_DERIVED_MAIN_EVIDENCE' if all(r['status'] in ['PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT','PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE'] for r in rows) else 'PRESERVED_AUDIT_GAPS',
            zero_integration_engineering_attempts=sum(r['status']=='PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE' for r in rows),
            all_audits_PASS_does_not_mean_all_attempts_have_task_results=True,episodes=len(rows),rows=rows)
        write(root/'evidence/MAIN_EVIDENCE_AUDIT_INDEX.private.json',value);return value
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN);ledger.cpu(time.process_time()-started+after.ru_utime+after.ru_stime-child.ru_utime-child.ru_stime)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();v=run(a.root);print(json.dumps(dict(status=v['status'],episodes=v['episodes'])),flush=True)
