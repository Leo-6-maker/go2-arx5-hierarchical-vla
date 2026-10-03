"""Reconcile completed artifacts and cumulative budgets from saved evidence only."""
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0269'
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0269')
POINTS=('ROT2048','UNIFORM4096','PREFIX4096')


def read(path):return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def lines(path,cap):
    rows=[json.loads(s) for s in path.read_text().splitlines() if s]
    assert len(rows)<=cap and [r['ordinal'] for r in rows]==list(range(1,len(rows)+1)),path
    return rows


def main():
    evidence=[]
    def record(path,status=None):
        value=read(path)
        if status is not None:assert value['status']==status,path
        evidence.append(dict(path=str(path),sha256=sha(path),status=value.get('status')))
        return value
    card=record(ROOT/'design/APPROVED_RUN_CARD.json')
    assert card['release']=='e77ad092406555645fb8c68ba34cc8594ef18ee6' and card['updates_per_arm']==4096
    assert card['optimizer']['accumulation']==2
    diagnosis=record(ROOT/'evidence/SAVED_STATE_PREFIX_AUDIT.json','PASS_BOUNDED_SAME_STATE_LABEL_DIAGNOSIS_NO_NEW_CORRECTNESS_DEFECT')
    assert diagnosis['proposals']==len(diagnosis['rows'])==96 and diagnosis['label_calls']<=96
    assert diagnosis['model_queries']==diagnosis['render_calls']==diagnosis['physics_calls']==diagnosis['training_examples_added']==0
    record(ROOT/'evidence/DONOR_EXACT_REUSE.json','PASS_EXISTING_968_AND_64_EXACT_REUSE')
    training=record(ROOT/'evidence/MAIN_TRAINING_AUDIT.json','PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY')
    assert training['optimizer_updates']=={'UNIFORM':4096,'PREFIX':4096}
    assert all(training[k] for k in ('actual_inputs_match_Hopper_CPU_plan','actual_random_packets_equal','actual_S4_gradient_loss_mode_checks'))
    assert len(training['FIT112_consumption'])==224 and all(r['actual_seen'] for r in training['FIT112_consumption'])
    harvested=ROOT/'evidence/training_hopper'
    transfer=record(ROOT/'evidence/TRAINING_EVIDENCE_TRANSFER.json','PASS_COMPLETE_TRAINING_JOURNALS_TRANSFER_AND_MEMBER_HASHES')
    manifest=read(harvested/'TRAINING_EVIDENCE_MANIFEST.json')
    assert sha(harvested/'TRAINING_EVIDENCE_MANIFEST.json')==transfer['source_manifest_sha256']
    assert len(manifest['files'])==transfer['files']
    for row in manifest['files']:
        p=harvested/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
    updates=lines(harvested/'raw_completed_updates.jsonl',8480)
    trainheads=lines(harvested/'raw_training_heads.jsonl',160000)
    assert len(updates)==training['raw_completed_updates'] and len(trainheads)==training['raw_training_heads']
    assert sha(harvested/'raw_completed_updates.jsonl')==training['completed_updates_sha256']
    assert sha(harvested/'raw_training_heads.jsonl')==training['training_heads_sha256']
    assert 0<=len(updates)-8224<=256
    for arm in ('UNIFORM','PREFIX'):
        c=record(harvested/arm/'canary_compare.json','PASS_PI0269_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY')
        assert c['units']==8 and c['model_max_abs']==c['optimizer_max_abs']==c['scheduler_lr_abs']==0
        assert c['packet_equal'] and c['mode_restored'] and all(c['saved_RNG_states_equal'].values())
    checkpoints=[]
    for source in training['complete_checkpoints']:
        arm,step=source['arm'],source['step']
        receipt=record(ROOT/'evidence'/arm/f'checkpoint-{step}.full_readback.json','PASS_FULL_FILE_READBACK')
        assert receipt['complete_trainer'] and receipt['files']==source['entries'] and len(receipt['files'])==18
        checkpoints.append(dict(arm=arm,step=step,revision=receipt['revision'],prefix=receipt['prefix'],tree_sha256=receipt['tree_sha256']))
    assert {(r['arm'],r['step']) for r in checkpoints}==set(itertools.product(('UNIFORM','PREFIX'),(2048,4096)))
    verdict=record(RAW/'delivery/PRIMARY_VERDICT.json','PRIMARY_EVALUATION_COMPLETE')
    cpu=record(RAW/'delivery/CPU_REPLAY.json','PASS_SAVED_ARRAY_REPLAY')
    assert verdict['offline_rows']==len(cpu['metrics'])==2904 and verdict['virtual_episodes']==len(cpu['episodes'])==192
    assert cpu['model_queries']==cpu['gradient_queries']==cpu['physics_steps']==0
    attempts=lines(ROOT/'evidence/model_attempts.jsonl',18432)
    heads=lines(ROOT/'evidence/model_heads.jsonl',73728)
    virtual=lines(ROOT/'evidence/virtual_model_attempts.jsonl',16496)
    starts=lines(ROOT/'evidence/virtual_starts.jsonl',240)
    frames=lines(ROOT/'evidence/virtual_frame_starts.jsonl',17000)
    assert Counter(r['attempt'] for r in heads)==Counter({r['ordinal']:4 for r in attempts})
    predictions={r['file']:r['sha256'] for r in cpu['predictions']}
    assert len(predictions)==len(cpu['predictions'])==len(attempts)
    assert {r['request']['output'] for r in attempts}==set(predictions)
    for a in attempts:
        receipt=read(Path(a['request']['output']).with_suffix('.receipt.json'))
        assert receipt['attempt']==a['ordinal'] and receipt['heads']==4 and receipt['native_S4']
        assert receipt['point']==a['point'] and receipt['output_sha256']==predictions[a['request']['output']]
        assert receipt['input_sha256']==a['input_sha256'] and receipt['noise_sha256']==a['noise_sha256']
        assert receipt['no_teacher_or_privileged_condition']
    assert all(attempts[r['global_attempt']-1]['request']==r['request'] for r in virtual)
    assert len(attempts)-len(virtual)==1936
    families=[f['family_id'] for f in read(BASE/'PI0267/design/SCENE_CARD.json')['families'] if f['split']=='NEW_SIM_EVAL']
    expected={f'{f}-{t}-c{c}-{b}' for f,t,c,b in itertools.product(families,('M1','M3'),(0,1),('E0','E1'))}
    query_count=0
    for point in POINTS[1:]:
        paths=list((RAW/'virtual'/point).glob('*/result.json'));assert {p.parent.name for p in paths}==expected
        for path in paths:
            r=read(path);assert r['validity']=='VALID' and r['physics_steps']==0
            assert not r['teacher_used_for_control'] and not r['reference_retimed']
            assert sha(path.parent/'trajectory.npz')==r['arrays_sha256']
            query_count+=len(r['queries'])
    main_queries=query_count
    component=read(ROOT/'design/COMPONENT_CARD.json');valid=excluded=0
    for point in POINTS:
        for proposal in component['rows']:
            path=RAW/'component'/point/proposal['case']/'result.json';r=read(path)
            assert r['component_start']==proposal and not r['qualification_authority']
            if r['validity']=='PREQUERY_EXCLUDED':assert not r['queries'];excluded+=1
            else:
                assert r['validity']=='VALID' and r['physics_steps']==0
                assert not r['teacher_used_for_control'] and not r['reference_retimed']
                assert sha(path.parent/'trajectory.npz')==r['arrays_sha256'];valid+=1
                query_count+=len(r['queries'])
    assert valid+excluded==96 and query_count==len(virtual)
    assert main_queries<=12800 and query_count-main_queries<=1920
    videos=[]
    for point in POINTS[1:]:
        receipt=record(RAW/'delivery'/(point+'-VIDEO_RECEIPT.json'),'PASS_RECORDED_RGB_VIDEO_FULL_DECODE')
        assert len(receipt['chapters'])==8 and receipt['model_queries']==receipt['new_renders']==receipt['physics_steps']==0
        assert sha(Path(receipt['file']))==receipt['sha256']
        videos.append(dict(point=point,sha256=receipt['sha256'],frames=receipt['frames']))
    archives=[record(ROOT/'evidence'/('RESULT_ARCHIVE_'+p+'.json'),'PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS') for p in (*POINTS[1:],'component')]
    for name,n in (('OFFLINE2904.csv',2904),('VIRTUAL192.csv',192),('COMPONENT96.csv',96),('PAIRED_OFFLINE_CHANGES.csv',2904)):
        with (RAW/'delivery'/name).open() as stream:assert len(list(csv.DictReader(stream)))==n
        evidence.append(dict(path=str(RAW/'delivery'/name),sha256=sha(RAW/'delivery'/name)))
    result=dict(status='PASS_COMPLETE_PRIMARY_ARTIFACTS_AND_CUMULATIVE_BUDGET_RECONCILIATION',
        final_summary_archive='PENDING_SEPARATE_FIXED_REVISION_READBACK',optimizer_updates=training['optimizer_updates'],
        raw_training_updates=len(updates),raw_training_heads=len(trainheads),full_Trainer_checkpoints=checkpoints,
        offline_logical_rows=2904,exact_donor_offline_reuse=968,new_offline_queries=1936,
        virtual_logical_episodes=192,exact_donor_virtual_reuse=64,component_proposals=96,component_valid=valid,component_excluded=excluded,
        new_virtual_starts=len(starts),new_main_queries=main_queries,new_component_queries=query_count-main_queries,
        inference_attempts=len(attempts),inference_heads=len(heads),new_frame_pairs=len(frames),
        saved_state_teacher_label_calls=diagnosis['label_calls'],teacher_label_proposals=96,
        quality=verdict['interpretation'],virtual_research_candidates=verdict['reliable_virtual_research_candidates'],
        model_queries_by_audit=0,gradient_queries_by_audit=0,new_WBC_physics_hardware=0,
        old_secondary_status='SECONDARY_PARTIAL',videos=videos,archives=archives,evidence=evidence,
        selected_checkpoint='NONE',deployment_selected_checkpoint='NONE')
    (ROOT/'evidence/PRIMARY_COMPLETION_AUDIT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence','archives')},indent=2),flush=True)


if __name__=='__main__':main()
