"""Close the fixed evidence and literal budget; archival/publication are separate receipts."""
import argparse,csv,json
from collections import Counter
from pathlib import Path
from run import ROOT,write,sha
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0276')


def read(p):return json.loads(Path(p).read_text())


def main(verify=False):
    dst=RAW/'delivery';verdict=read(dst/'PHYSICAL_VERDICT.json')
    assert read(ROOT/'evidence/MODEL_CPU_REPLAY.json')['status']=='PASS_PI0276_ALL_SAVED_S4_DECODE_AND_OFFLINE_OUTPUTS'
    assert read(ROOT/'evidence/PHYSICS_CPU_REPLAY.json')['status']=='PASS_SAVED_PHYSICAL_PREDICATES'
    assert read(ROOT/'evidence/VIDEOS.json')['status']=='PASS_ALL12_REGISTERED_SAVED_RGB_VIDEOS'
    assert read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json')['status']=='PASS_PI0276_ACTUAL8192_TASK_TRAINING'
    if verify:
        archives={}
        for part in ('training_data','source','predictions','summary'):
            p=ROOT/f'evidence/RESULT_ARCHIVE_{part}.json';a=read(p)
            assert a['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS'
            archives[part]=dict(a,receipt_sha256=sha(p))
        checkpoints={}
        for step in (4096,8192):
            p=ROOT/f'evidence/TASK/checkpoint-{step}.full_readback.json';a=read(p)
            assert a['status']=='PASS_FULL_FILE_READBACK' and a['complete_trainer'];checkpoints[str(step)]=dict(a,receipt_sha256=sha(p))
        write(ROOT/'evidence/FINAL_ARTIFACT_AUDIT.json',dict(status='PASS_PI0276_COMPLETE_CPU_VIDEO_MODELS_AND_PUBLIC_ARCHIVES',
            archives=archives,complete_trainers=checkpoints,closeout_sha256=sha(dst/'CLOSEOUT.json'),
            report_sha256=sha(dst/'FINAL_RESEARCH_RETURN.md'),terminal_GitHub_publication='REQUIRED',
            selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
        return
    source=read(ROOT/'evidence/SOURCE.json');data=read(ROOT/'design/DATASET_CARD.json');index=[];completed=Counter()
    for p in sorted((RAW/'episodes').glob('*/attempt-*/RESULT.json')):
        r=read(p)
        for name,digest in r['files'].items():assert sha(p.parent/name)==digest
        index.append(dict(slot=p.parent.parent.name,attempt=p.parent.name,path=str(p.parent),result_sha256=sha(p),**r))
        completed.update(dict(control=r['control_calls'],physics=r['physics_completed'],rgb=r['RGB_pairs']))
    reserved=Counter(json.loads(l)['kind'] for l in (RAW/'evidence/budget.jsonl').open())
    limits=dict(starts=406,control=260000,physics=1300000,rgb=270000,teacher=110000)
    assert all(reserved[k]<=v for k,v in limits.items()) and reserved['starts']==len(index)
    model=read(ROOT/'evidence/MODEL_CPU_REPLAY.json');train=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json')
    assert model['attempts']<=8192 and model['heads']<=32768 and train['raw_completed_updates']<=8464 and train['raw_training_heads']<=155648
    write(dst/'SOURCE_INDEX.json',dict(rows=index,source_receipt_sha256=sha(ROOT/'evidence/SOURCE.json'),
        preserve_engineering_and_valid_quality_failures=True))
    with (dst/'OFFLINE384.csv').open() as f:offline=list(csv.DictReader(f))
    assert len(offline)==384
    table=[]
    for point in ('BASE0','TASK8192'):
        for split in ('TRAIN','VAL'):
            for group in range(4):
                own=[r for r in offline if r['point']==point and r['split']==split and int(r['group'])==group]
                assert len(own)==(32 if split=='TRAIN' else 16)
                row=dict(point=point,split=split,group=group,conditions=len(own),REJECT=sum(r['gate']=='REJECT' for r in own),
                    finite=sum(r['finite']=='True' for r in own))
                for name in ('XYZ_RMS_m','SO3_RMS_rad','prefix8_XYZ_RMS_m','prefix8_SO3_RMS_rad','gripper_MAE','prefix8_gripper_MAE','OPEN_MAE','CLOSE_MAE'):
                    values=[float(r[name]) for r in own if r[name]!=''];row[name+'_mean']=sum(values)/len(values) if values else None
                table.append(row)
    with (dst/'OFFLINE_GROUP16.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    budget=dict(reserved=dict(reserved),completed_saved_physics=dict(completed),limits=limits,
        formal_updates=8192,raw_updates=train['raw_completed_updates'],training_heads=train['raw_training_heads'],
        quality_attempts=model['attempts'],quality_heads=model['heads'],target_WBC=0,real_hardware=0)
    write(dst/'FINAL_BUDGET.json',budget)
    closeout=dict(status='PI0276_PHYSICS_TASK_FINETUNING_COMPLETE',execution='COMPLETE',
        physical_task_quality=verdict['physical_task_quality'],learning_effect_target_met=verdict['learning_effect_target_met'],
        next='STOP_FOR_FROZEN_WBC_COMPOSITION_OR_TARGETED_TASK_FAILURE_DECISION',
        archival_completeness='BOUND_BY_SEPARATE_FINAL_ARTIFACT_AUDIT',budget=budget,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    write(dst/'CLOSEOUT.json',closeout)
    lines=['# PI0276 — fixed StackCube task-learning result','',
        'Execution: COMPLETE. Physical task quality: '+verdict['physical_task_quality']+'.',
        'Source-Panda physical simulation; NON_REALTIME. No target WBC or hardware execution.','',
        '| Lane | First success | Persistent success | Wilson 95% |','|---|---:|---:|---|']
    for name,r in verdict['summary'].items():
        lines.append(f"| {name} | {r['first_success']}/{r['denominator']} | {r['persistent_success']}/{r['denominator']} | {r['wilson95']} |")
    lines+=['',f"Paired TEST: gained {verdict['gained']}, lost {verdict['lost']}, net gain {verdict['net_gain']}/32; planned net-gain target >=8: {verdict['learning_effect_target_met']}.",'',
        'Source admission retained DEV8/8, TRAIN93/96 and VAL15/16; all original failures and the complete VAL20009 horizon failure remain in SOURCE_INDEX.json.',
        'Offline384 and group16 tables report the fixed TRAIN/VAL panel, whole-H40 versus executed prefix8, gripper and original gate outcomes.',
        'Command-to-reference installation, reference-to-actual tracking and object-task completion are separate columns in PHYSICAL144.csv.',
        'Measured contacts, penetration, object transitions, joint-limit excursions and gripper saturation do not establish hardware safety.',
        'BASE0 is AGG4096 weights with this new task processor; it is not the historical AGG policy. Only TASK8192 is evaluated; no intermediate selection.',
        'Full4096/8192 Trainer and data/output/video archive identities are bound by FINAL_ARTIFACT_AUDIT.json after fixed-revision readback.',
        'Selected/deployment: NONE. Stop for frozen-WBC composition or a separately authorized targeted task-failure decision.','']
    (dst/'FINAL_RESEARCH_RETURN.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');main(p.parse_args().verify)
