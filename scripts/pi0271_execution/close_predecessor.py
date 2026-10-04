"""Reconcile PI0270's actual complete work before summary archival and publication."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from support import OLD as ROOT, read, sha, write

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0270')


def journal(path,cap):
    rows=[json.loads(s) for s in path.read_text().splitlines() if s]
    assert len(rows)<=cap and [r['ordinal'] for r in rows]==list(range(1,len(rows)+1)),str(path)
    return rows


def ready():
    assert read(ROOT/'evidence/QUALITY_QUERY_LANES_COMPLETE.json')['status']=='ALL_FIXED_NEW_QUALITY_QUERY_LANES_COMPLETE'
    assert read(RAW/'delivery/PRIMARY_VERDICT.json')['status']=='PRIMARY_EVALUATION_COMPLETE'
    assert read(RAW/'delivery/CPU_REPLAY.json')['status']=='PASS_SAVED_ARRAY_REPLAY'
    assert (RAW/'delivery/AGG4096-VIDEO_RECEIPT.json').exists()
    for part in ('collection','AGG4096','newfit','component'):
        assert read(ROOT/'evidence'/('RESULT_ARCHIVE_'+part+'.json'))['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS'


def main():
    ready()
    evidence=[]
    def record(path,status=None):
        value=read(path)
        if status is not None:assert value['status']==status,str(path)
        evidence.append(dict(path=str(path),sha256=sha(path),status=value.get('status')));return value
    record(ROOT/'evidence/A0_PREDECESSOR_CLOSEOUT.json','PASS_CONTINUE_PI0270_A1')
    audit=record(ROOT/'evidence/MAIN_TRAINING_AUDIT.json','PASS_AGG4096_REUSED_CONTROL_TRAINING_INTEGRITY')
    assert audit['optimizer_updates']==4096 and audit['raw_completed_updates']==4112 and audit['raw_training_heads']==74040
    assert all(audit[k] for k in ('actual_inputs_match_full_CPU_plan','all4096_COMMON_input_label_bit_identities',
        'all8192_actual_random_packets_equal_control','actual_S4_loss_gradient_mode_checks'))
    transfer=record(ROOT/'evidence/TRAINING_EVIDENCE_TRANSFER.json','PASS_COMPLETE_TRAINING_JOURNALS_TRANSFER_AND_MEMBER_HASHES')
    harvest=ROOT/'evidence/training_hopper';manifest=record(harvest/'TRAINING_EVIDENCE_MANIFEST.json','PASS_COMPLETE_SMALL_TRAINING_EVIDENCE_SEAL')
    assert sha(harvest/'TRAINING_EVIDENCE_MANIFEST.json')==transfer['source_manifest_sha256'] and len(manifest['files'])==100
    for row in manifest['files']:
        path=harvest/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    updates=journal(harvest/'raw_completed_updates.jsonl',4368);trainheads=journal(harvest/'raw_training_heads.jsonl',80000)
    assert len(updates)==audit['raw_completed_updates'] and len(trainheads)==audit['raw_training_heads']
    canary=record(harvest/'AGG/canary_compare.json','PASS_PI0270_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY')
    assert canary['units']==8 and canary['model_max_abs']==canary['optimizer_max_abs']==canary['scheduler_lr_abs']==0
    assert canary['packet_equal'] and canary['mode_restored'] and all(canary['saved_RNG_states_equal'].values())
    checkpoints=[]
    for source in audit['complete_checkpoints']:
        step=source['step'];receipt=record(ROOT/'evidence/AGG'/f'checkpoint-{step}.full_readback.json','PASS_FULL_FILE_READBACK')
        assert receipt['complete_trainer'] and receipt['files']==source['entries'] and len(receipt['files'])==18
        checkpoints.append({k:receipt[k] for k in ('step','revision','prefix','tree_sha256')})
    assert {r['step'] for r in checkpoints}=={2048,4096}
    coverage=record(ROOT/'evidence/DATA_COVERAGE_ADMISSION.json','PASS_ALL96_FIXED_NEW_SUPPORT_CELLS')
    assert coverage['proposals']==6784 and coverage['eligible']+coverage['excluded_or_missing']==6784
    assert len(coverage['coverage'])==96 and all(r['PASS'] for r in coverage['coverage'])
    record(ROOT/'evidence/TRAINING_DATA_ARCHIVE.json','PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS')
    collection=record(ROOT/'evidence/COLLECTION_S4_REPLAY_FAMILIES_24.json','PASS_COMPLETED_FIXED_COLLECTION_FAMILY_S4_REPLAY')
    assert collection['all_collection_complete'] and collection['cases']==192 and collection['predictions']==19200
    predictions={}
    for item in collection['receipts']:
        assert sha(item['path'])==item['sha256'];receipt=read(item['path'])
        for row in receipt['predictions']:
            assert row['file'] not in predictions;predictions[row['file']]=row['sha256']
    trajectory=record(ROOT/'evidence/TRAJECTORY_CPU_collection_192.json','PASS_SAVED_EXECUTOR_SAME_TICK_V3_RELIABILITY_BATCH')
    assert trajectory['cases']==192 and trajectory['collection_all192_complete']
    for row in trajectory['checks']:assert sha(row['path'])==row['sha256']
    record(ROOT/'evidence/FULL_COLLECTION_AND_SUPPORT_CPU_RETURN.json','FULL_FIXED_COLLECTOR_AND_NEW_SUPPORT_CPU_RETURN_NOT_AGG_QUALITY')
    reuse=record(ROOT/'evidence/EVALUATION_EXACT_REUSE.json','PASS_EXACT_CLOSED_1936_OFFLINE_128_MAIN_64_COMPONENT_REUSE')
    assert reuse['counts']==dict(component=64,offline=1936,virtual=128)
    verdict=record(RAW/'delivery/PRIMARY_VERDICT.json','PRIMARY_EVALUATION_COMPLETE')
    cpu=record(RAW/'delivery/CPU_REPLAY.json','PASS_SAVED_ARRAY_REPLAY')
    for row in cpu['predictions']:
        assert row['file'] not in predictions;predictions[row['file']]=row['sha256']
    for row in cpu['trajectory_receipts']:assert sha(row['path'])==row['sha256']
    attempts=journal(ROOT/'evidence/model_attempts.jsonl',30720);heads=journal(ROOT/'evidence/model_heads.jsonl',122880)
    starts=journal(ROOT/'evidence/virtual_starts.jsonl',304);frames=journal(ROOT/'evidence/virtual_frame_starts.jsonl',28672)
    teachers=journal(ROOT/'evidence/teacher_calls.jsonl',7168);virtual=journal(ROOT/'evidence/virtual_model_attempts.jsonl',30720)
    head_counts=Counter(r['attempt'] for r in heads);assert set(head_counts)<=set(r['ordinal'] for r in attempts)
    assert all(0<=head_counts[r['ordinal']]<=4 for r in attempts)
    assert set(r['request']['output'] for r in attempts)==set(predictions)
    technical=[]
    for attempt in attempts:
        path=Path(attempt['request']['output']);receipt=read(path.with_suffix('.receipt.json'))
        assert receipt['point']==attempt['point'] and receipt['output_sha256']==predictions[str(path)]
        assert receipt['input_sha256']==attempt['input_sha256'] and receipt['noise_sha256']==attempt['noise_sha256']
        if receipt['attempt']!=attempt['ordinal']:
            assert receipt['attempt']>attempt['ordinal']
            technical.append(dict(attempt=attempt['ordinal'],saved_receipt_attempt=receipt['attempt'],heads=head_counts[attempt['ordinal']],
                state='NO_FINAL_OUTPUT_FOR_THIS_ATTEMPT',request=attempt['request']))
        else:assert receipt['native_S4'] and receipt['heads']==head_counts[attempt['ordinal']]==4
    # Retain every earlier attempt and its literal head count; completed first-valid outputs stay unique.
    assert all(attempts[r['global_attempt']-1]['request']==r['request'] for r in virtual)
    lane_attempts=Counter(Path(r['request']['output']).relative_to(RAW).parts[0] for r in attempts)
    lane_counts=Counter(Path(path).relative_to(RAW).parts[0] for path in predictions)
    assert lane_counts['collection']==19200 and lane_counts['offline']==968 and lane_counts['newfit']==384
    assert lane_counts['virtual']<=6400 and lane_counts['component']<=640
    assert len(virtual)==lane_attempts['virtual']+lane_attempts['component']
    component_counts=Counter()
    for lane,expected in [('collection',192),('virtual',64),('component',32)]:
        point='PREFIX4096' if lane=='collection' else 'AGG4096';paths=list((RAW/lane/point).glob('*/result.json'));assert len(paths)==expected
        queries=0
        for path in paths:
            row=read(path);assert row['point']==point and row['validity'] in ('VALID','PREQUERY_EXCLUDED')
            if row['validity']=='PREQUERY_EXCLUDED':assert lane=='component' and not row['queries']
            else:assert row['physics_steps']==0 and sha(path.parent/'trajectory.npz')==row['arrays_sha256']
            queries+=len(row['queries'])
            if lane=='component':component_counts[row['validity']]+=1
        assert queries==lane_counts[lane]
    assert len({r['proposal'] for r in teachers})<=6784
    video=record(RAW/'delivery/AGG4096-VIDEO_RECEIPT.json','PASS_RECORDED_RGB_VIDEO_FULL_DECODE')
    assert len(video['chapters'])==8 and sha(Path(video['file']))==video['sha256']
    assert video['model_queries']==video['new_renders']==video['physics_steps']==0
    tables={'OFFLINE2904.csv':2904,'NEW_FIT384.csv':384,'VIRTUAL192.csv':192,'COMPONENT96.csv':96,
        'PAIRED_OFFLINE_CHANGES.csv':968,'PAIRED_NEW_FIT_CHANGES.csv':192}
    for name,count in tables.items():
        path=RAW/'delivery'/name
        with path.open() as stream:assert len(list(csv.DictReader(stream)))==count
        evidence.append(dict(path=str(path),sha256=sha(path)))
    archives=[record(ROOT/'evidence'/('RESULT_ARCHIVE_'+part+'.json'),'PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS') for part in ('collection','AGG4096','newfit','component')]
    preserved={}
    for receipt in archives:
        part=receipt['part'];path=RAW/'archive'/('results-'+part)/'MEMBER_MANIFEST.json';members=read(path)
        assert members['tree_sha256']==receipt['tree_sha256'] and len(members['files'])==receipt['members']
        assert sha(path)==next(f['sha256'] for f in receipt['files'] if f['path']=='MEMBER_MANIFEST.json')
        preserved.update({str(RAW/r['path']):r['sha256'] for r in members['files']})
    assert all(preserved[path]==digest for path,digest in predictions.items())
    value=dict(status='PASS_COMPLETE_PI0270_PRIMARY_ARTIFACTS_AND_RAW_BUDGETS',source_sha256=sha(__file__),
        final_summary_archive_and_publication='STILL_REQUIRED',optimizer_updates=4096,canary_updates=16,
        raw_training_updates=len(updates),raw_training_heads=len(trainheads),checkpoints=checkpoints,
        inference_attempts=len(attempts),inference_heads=len(heads),lane_query_counts=dict(lane_counts),
        lane_attempt_counts=dict(lane_attempts),
        virtual_starts=len(starts),new_frame_pairs=len(frames),teacher_calls=len(teachers),teacher_proposals=6784,
        technical_inference_attempts=technical,component_new_valid=component_counts['VALID'],component_new_excluded=component_counts['PREQUERY_EXCLUDED'],
        offline_logical_rows=2904,newfit_logical_rows=384,main_logical_episodes=192,component_logical_proposals=96,
        exact_reuse=reuse['counts'],closed_collection_replay_reused=True,new_saved_prediction_replays=len(cpu['predictions']),
        quality=verdict['interpretation'],reliable_candidates=verdict['reliable_virtual_research_candidates'],
        evidence=evidence,archives=archives,model_queries_by_audit=0,new_WBC_physics_hardware=0,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    write(ROOT/'evidence/PRIMARY_COMPLETION_AUDIT.json',value)
    print('PASS_COMPLETE_PI0270_PRIMARY_ARTIFACTS_AND_RAW_BUDGETS',len(attempts),len(heads),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--admission-only',action='store_true');args=parser.parse_args()
    if args.admission_only:ready()
    else:main()
