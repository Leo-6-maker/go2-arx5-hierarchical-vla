"""Close actual artifacts and budgets; publication remains a separate final receipt."""
import csv
from pathlib import Path
from quality_worker import ROOT,read,sha,write,admitted

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0273')

def main():
    admitted()
    inventory=read(ROOT/'evidence/QUALITY_OUTPUT_INVENTORY.json')
    assert inventory['status']=='PASS_EXACT_ALL_REGISTERED_RESULTS_AND_LITERAL_BUDGETS'
    for name,digest in inventory['journals'].items():assert sha(ROOT/'evidence'/f'{name}.jsonl')==digest
    for r in inventory['outputs']:
        assert sha(r['path'])==r['sha256'] and sha(Path(r['path']).with_suffix('.receipt.json'))==r['receipt_sha256']
    for r in inventory['records']:assert sha(r['path'])==r['sha256']
    cpu=read(RAW/'delivery/CPU_REPLAY.json');assert cpu['status']=='PASS_PI0273_ALL_NEW_S4_DECODE_AND_TRAJECTORIES'
    assert {r['file']:r['sha256'] for r in cpu['predictions']}=={r['path']:r['sha256'] for r in inventory['outputs']}
    for name,proof in cpu['tables'].items():
        path=RAW/'delivery'/name;assert sha(path)==proof['sha256']
        with path.open() as f:assert len(list(csv.DictReader(f)))==proof['rows']
    assert {n:cpu['tables'][n]['rows'] for n in ('OFFLINE2904.csv','NEW_FIT576.csv','LATE_CONTEXT2368.csv','VIRTUAL192.csv','COMPONENT96.csv')}==dict(
        [('OFFLINE2904.csv',2904),('NEW_FIT576.csv',576),('LATE_CONTEXT2368.csv',2368),('VIRTUAL192.csv',192),('COMPONENT96.csv',96)])
    transitions=read(RAW/'delivery/TRANSITION_ANALYSIS.json');assert transitions['status']=='PASS_SAVED_TRANSITIONS_AND_POST_SUCCESS_DIAGNOSIS'
    assert len(transitions['episodes'])==128 and len(transitions['M3_entry_and_coverage'])==64
    for name,digest in transitions['tables'].items():assert sha(RAW/'delivery'/name)==digest
    videos=[]
    for point in ('BROAD5120','TARGET5120'):
        receipt=read(RAW/'delivery'/f'{point}-VIDEO_RECEIPT.json')
        assert receipt['status']=='PASS_RECORDED_RGB_VIDEO_FULL_DECODE' and len(receipt['chapters'])==8
        assert receipt['point']==point and sha(receipt['file'])==receipt['sha256']
        assert receipt['model_queries']==receipt['new_renders']==receipt['physics_steps']==0;videos.append(receipt)
    archive=read(ROOT/'evidence/RESULT_ARCHIVE_predictions.json')
    assert archive['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS'
    mp=RAW/'archive/results-predictions/MEMBER_MANIFEST.json';manifest=read(mp)
    assert manifest['tree_sha256']==archive['tree_sha256'] and len(manifest['files'])==archive['members']
    assert sha(mp)==next(r['sha256'] for r in archive['files'] if r['path']=='MEMBER_MANIFEST.json')
    kept={str(RAW/r['path']):r['sha256'] for r in manifest['files']}
    for r in inventory['outputs']:
        assert kept[r['path']]==r['sha256'] and kept[str(Path(r['path']).with_suffix('.receipt.json'))]==r['receipt_sha256']
    for r in inventory['records']:assert kept[r['path']]==r['sha256']
    training=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json');trainers=[]
    for cp in training['complete_checkpoints']:
        r=read(ROOT/'evidence'/cp['arm']/f"checkpoint-{cp['step']}.full_readback.json")
        assert r['status']=='PASS_FULL_FILE_READBACK' and r['files']==cp['entries'];trainers.append(r)
    assert len(trainers)==4 and 10272<=training['raw_completed_updates']<=10528 and training['raw_training_heads']<=196608
    write(ROOT/'evidence/FINAL_ARTIFACT_AUDIT.json',dict(status='PASS_PI0273_ALL_FIXED_RESULTS_TRAINERS_AND_BUDGETS',
        formal_updates=dict(BROAD=5120,TARGET=5120),raw_completed_updates=training['raw_completed_updates'],raw_training_heads=training['raw_training_heads'],
        new_quality_outputs=len(inventory['outputs']),quality_attempts=inventory['attempts'],quality_heads=inventory['heads'],
        technical_quality_attempts=inventory['technical_attempts'],new_virtual_starts=inventory['starts'],new_frame_attempts=inventory['frame_attempts'],
        complete_trainer_readbacks=trainers,predictions_archive=archive,videos=videos,new_teacher_labels=0,new_WBC=0,new_dynamic_physics=0,new_hardware=0,
        summary_archive_and_terminal_publication='STILL_REQUIRED',selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PASS_PI0273_ALL_FIXED_RESULTS_TRAINERS_AND_BUDGETS',flush=True)

if __name__=='__main__':main()
