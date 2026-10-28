"""Original zero-step IK/collision/FOV qualification, before any policy result."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

from PIL import Image

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.scene import make_engine,qualify_engine
from scripts.pi0294.scene_cards import task_card,near_duplicate_report,digest
from . import confirmation_config as config
from .science import Ledger,validate_runtime


def inspect(root,group,task,ledger,output,guard):
    root=Path(root);card=task_card(group,task);output=Path(output);output.mkdir(parents=True,mode=0o700)
    write(output/'CARD.json',card);ledger.charge('main_starts',1);engine=None
    try:
        guard();engine=make_engine(card,read(root/'INPUT_BINDINGS.private.json')['asset_root'])
        if engine.steps or engine.data.time:raise PermissionError('STATIC_QUALIFICATION_CANNOT_ADVANCE_PHYSICS')
        images,geometry=engine.render(engine.snapshot());files={}
        for camera,array in images.items():
            if camera not in ['body_rgb','wrist_rgb']:raise PermissionError('EXACT_ORIGINAL_TWO_STATIC_CAMERAS')
            p=output/(camera+'.png');Image.fromarray(array).save(p,format='PNG')
            files[camera]=dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size)
        if set(files)!={'body_rgb','wrist_rgb'}:raise PermissionError('BOTH_ACTUAL_STATIC_CAMERA_FRAMES_REQUIRED')
        result=qualify_engine(engine,rendered=(images,geometry),guard=guard)
        if engine.steps or engine.data.time or result['integration_steps']:
            raise PermissionError('ORIGINAL_ZERO_INTEGRATION_STATIC_RULES_ONLY')
        result.update(scene_group=group['group_id'],task=task,card_sha256=digest(card),actual_card_file_sha256=sha(output/'CARD.json'),
            model_xml_sha256=hashlib.sha256(engine.xml.encode()).hexdigest(),actual_static_images=files,
            engine_starts=1,Teacher_success_rollout=False,new_positive_labels=False)
        write(output/'STATIC_RESULT.json',result);return result
    finally:
        if engine is not None:engine.close()


def run(root):
    root=Path(root);m=validate_runtime(root,'4080-simulation')
    if m['execution'].get('confirmation_phase')!='STATIC_PREPARATION':
        raise PermissionError('NO_STATIC_SCENE_RESELECTION_AFTER_POLICY_FREEZE')
    q=read(root/config.STATIC_INPUTS)
    if (q['source_TRAIN_episodes']!=126 or q['source_DEV_episodes']!=32 or q['recovery_parents_verified']!=4
            or q['exact_raw_CARD_FILE_MANIFEST_verified'] is not True
            or q['original_generator_sha256']!='08530a62ff3f34509f3efb6459f20d5e0c600cdeb8dc31b01e7a766269a0dd91'
            or q['exclusion_population_sha256']!=digest(q['exclusion_layouts'])):
        raise PermissionError('ACTUAL_FROZEN_TRAIN_DEV_RECOVERY_AND_ORIGINAL_GENERATOR_EXCLUSIONS')
    window=read(root/'plans/CONFIRM_STATIC_WINDOW.json');now=time.time()
    if (window['maximum_CPU_s']!=3600 or window['maximum_candidates']!=40 or window['new_integrations']!=0
            or not window['created_epoch']<=now<window['deadline_epoch']<=min(window['created_epoch']+3600,1792372081)
            or type(window['CPU_id']) is not int or window['CPU_id'] not in os.sched_getaffinity(0)):
        raise PermissionError('FINITE_ONE_CPU_STATIC_WINDOW_BEFORE_CONFIRMATION_CUTOFF')
    if m['execution']['sealed_inputs'].get('plans/CONFIRM_STATIC_WINDOW.json')!=sha(root/'plans/CONFIRM_STATIC_WINDOW.json'):
        raise PermissionError('SEALED_STATIC_WINDOW_REQUIRED')
    os.sched_setaffinity(0,{window['CPU_id']});resource.setrlimit(resource.RLIMIT_CPU,(3600,3600))
    ledger=Ledger(root,'core');owner=OwnerLock(root/'CONFIRM_STATIC_OWNER.lock');started=time.process_time()
    output=root/'evidence/CONFIRM_STATIC';output.mkdir(mode=0o700);selected=[];receipts=[];trials=[];geometric_rejections=[]
    source=root.parent/'source-pair-01'
    def guard():
        ledger.guard()
        if (time.time()>=window['deadline_epoch'] or time.process_time()-started>=3600
                or any((source/name).exists() for name in ['SOURCE_PHYSICS_STOP.json','SOURCE_SAFETY_STOP_FROM_CORE.json'])):
            raise PermissionError('ORIGINAL_N1_STATIC_CLOCK_CPU_AND_SOURCE_STOP')
    def expire(signum,frame):raise TimeoutError('FINITE_STATIC_QUALIFICATION_WALL')
    signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(window['deadline_epoch']-now)))
    try:
        for family,count in config.plan.FAMILIES:
            for index in range(count):
                accepted=False
                for candidate in range(1024):
                    guard();group=config.plan.original_group(family,index,candidate)
                    duplicate=near_duplicate_report([group],[*q['exclusion_layouts'],*selected])
                    if duplicate['conflicts']:
                        geometric_rejections.append(dict(group_id=group['group_id'],candidate_index=candidate,conflicts=duplicate['conflicts']))
                        continue
                    if len(trials)>=40:raise PermissionError('FORTY_PREFREEZE_STATIC_CANDIDATES_EXHAUSTED_NO_POLICY')
                    folder=output/(family+'-'+str(index).zfill(3))/('candidate-'+str(candidate).zfill(3))
                    rows=[inspect(root,group,task,ledger,folder/task,guard) for task in ['T1','T2']]
                    passed=all(r['status']=='STATIC_PASS_PENDING_LIVE_CONTACT' for r in rows)
                    trials.append(dict(scene_group=group['group_id'],candidate_index=candidate,static_both_tasks_pass=passed,
                        task_results=[dict(task=r['task'],status=r['status'],card_sha256=r['card_sha256']) for r in rows]))
                    write(output/'PROGRESS.json',dict(qualified_groups=len(selected),static_candidates=len(trials),trials=trials))
                    if passed:selected.append(group);receipts+=rows;accepted=True;break
                if not accepted:raise PermissionError('PREDECLARED_GROUP_HAS_NO_ORIGINAL_STATIC_QUALIFICATION')
        candidate=config.plan.rank_candidate(read(root/config.MATRIX))
        plan=config.plan.evaluation_plan(selected,candidate,receipts,q['exclusion_layouts'])
        p=root/config.PLAN
        if p.exists():raise FileExistsError('NEVER_OVERWRITE_A_FROZEN_CONFIRMATION_PLAN')
        write(p,plan);p.chmod(0o400)
        value=dict(status='PASS_TEN_NEW_SCENES_TWENTY_ORIGINAL_STATIC_TASK_QUALIFICATIONS_NO_POLICY',
            source_sha=m['code']['vla_commit'],scene_groups=10,static_receipts=20,static_candidates=len(trials),
            actual_static_engine_starts=len(trials)*2,actual_physics_integrations=0,rendered_camera_pairs=len(trials)*2,
            plan_sha256=sha(p),exclusion_population_sha256=q['exclusion_population_sha256'],trials=trials,
            prefreeze_geometric_rejections=geometric_rejections,new_model_queries=0,new_positive_labels=0,
            Teacher_success_filtering=False,policy_opened=False)
        write(output/'RESULT.json',value);return value
    except BaseException as exc:
        write(output/'FAILURE.private.json',dict(error=type(exc).__name__,message=str(exc),qualified_groups=len(selected),
            static_candidates=len(trials),actual_physics_integrations=0,policy_opened=False));raise
    finally:signal.alarm(0);ledger.cpu(time.process_time()-started);owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    result=run(a.root);print(json.dumps(dict(status=result['status'],new_physics=0)),flush=True)
