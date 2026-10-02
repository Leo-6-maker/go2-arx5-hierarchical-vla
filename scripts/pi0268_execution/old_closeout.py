"""Terminal evidence reconciliation only, never a rerun or quality-selection path."""
import json
import sys
from pathlib import Path
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0267';R1=ROOT/'R1'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0267/R1')
sys.path[:0]=[str(BASE/'PI0266/code'),str(BASE/'PI0250/runtime/reference-317a/src')]
from pi0255_motion3 import write,sha


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]


def main():
    training=read(R1/'evidence/MAIN_TRAINING_AUDIT.json');assert training['status']=='PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY'
    primary=read(RAW/'delivery/PRIMARY_VERDICT.json');assert primary['status']=='PRIMARY_COMPLETE' and primary['virtual_episodes']==192 and primary['offline_rows']==2088
    secondary=read(BASE/'PI0268/R1/evidence/PI0267_SECONDARY_CONTINUATION.json');
    assert secondary['original_secondary_verdict_sha256']==sha(RAW/'delivery/SECONDARY_VERDICT.json')
    assert secondary['status']=='SECONDARY_PARTIAL'
    assert sum(r['validity']=='VALID' for r in secondary['outcomes'])==8
    assert sum(r['validity']=='ENGINEERING_INCOMPLETE' for r in secondary['outcomes'])==1
    assert sum(r['validity']=='DEFERRED_BY_PI0268_PRIMARY_PRIORITY' for r in secondary['outcomes'])==23
    assert secondary['status'] in ('SECONDARY_COMPLETE','SECONDARY_PARTIAL') and len(secondary['outcomes'])==32
    checkpoints=[]
    for arm in ('N','R'):
        for step in (2048,4096):
            source=R1/'evidence'/arm/f'checkpoint-{step}.full_readback.json';receipt=read(source)
            assert receipt['status']=='PASS_FULL_FILE_READBACK' and receipt['complete_trainer'] and receipt['step']==step
            checkpoints.append(dict(arm=arm,step=step,HF_revision=receipt['revision'],HF_prefix=receipt['prefix'],tree=receipt['tree_sha256'],receipt_sha256=sha(source)))
    archives=[]
    for part in ('donor','N4096','R4096','secondary','summary'):
        path=R1/'evidence'/('RESULT_ARCHIVE_'+part+'.json');r=read(path);assert r['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS';archives.append(r)
    attempts=lines(R1/'evidence/model_attempts.jsonl');heads=lines(R1/'evidence/model_heads.jsonl');starts=lines(R1/'evidence/virtual_starts.jsonl')
    assert [x['ordinal'] for x in attempts]==list(range(1,len(attempts)+1)) and len(attempts)<=24576
    assert [x['ordinal'] for x in heads]==list(range(1,len(heads)+1)) and len(heads)<=98304 and len(starts)<=208
    actual_virtual=sum(len(read(p)['queries']) for p in (RAW/'virtual').glob('*/*/result.json'));assert actual_virtual<=19200
    actual_WBC=secondary['actual_model_attempts_including_invalid'];assert actual_WBC<=2400
    camera_actual=sum(1 for _ in (RAW/'virtual').glob('*/*/queries/*/observation.npz'))+sum(1 for _ in (RAW/'diagnostics').glob('*/attempt-*/frames/*.npz'))
    actual_frames=10248+6+256+camera_actual;assert actual_frames<=40000
    value=dict(status='PI0267_PRIMARY_COMPLETE_SECONDARY_PARTIAL_ARCHIVES_COMPLETE',stop='SECONDARY_INTEGRITY_GAP_DEFERRED_UNDER_PI0268_R1_SECTION_3',
        release='a0f10306419d77242ba31c84d34ce7ce3c59f419',continuation_release='e5ec6d52100f6bb2e32be5b9661c756b0adb61f4',secondary_continuation=secondary,original_V2_PRIMARY_PARTIAL_preserved=True,
        V3_readmission_separate_from_model_learning=True,optimizer_updates={'N':4096,'R':4096},full_Trainer_checkpoints=checkpoints,
        training_raw_completed_updates=training['raw_completed_updates'],technical_recovery_updates=training['technical_recovery_completed_updates'],
        primary_quality=primary['quality'],virtual_research_candidates=primary['virtual_research_candidates'],virtual_episodes=192,virtual_actual_queries=actual_virtual,
        offline_logical_rows=2088,legacy_donor_cache_reuse=440,secondary_status=secondary['status'],secondary_WBC_model_queries=actual_WBC,
        cumulative_model_attempts=len(attempts),inference_heads=len(heads),virtual_starts=len(starts),actual_saved_frame_pairs=actual_frames,
        fixed_HF_result_archives=archives,TRAIN_data_archive=read(R1/'evidence/DATA_PUBLIC_RECEIPT.json'),
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',hardware_steps=0,protected_TEST_DEV_F_reads=0,monitor_mail_merge=False)
    write(BASE/'PI0268/R1/evidence/PI0267_CLOSEOUT_WITH_SECONDARY_PARTIAL.json',value);print(json.dumps({k:v for k,v in value.items() if k not in ('fixed_HF_result_archives','TRAIN_data_archive','secondary_continuation')}),flush=True)


if __name__=='__main__':main()
