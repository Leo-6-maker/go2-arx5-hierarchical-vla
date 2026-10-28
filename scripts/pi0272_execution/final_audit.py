"""Literal completed work and partial attempts; no model execution."""
import csv
import json
from collections import Counter
from pathlib import Path
from quality_worker import ROOT,RAW,read,sha,write,admitted


def main():
    admitted();card=read(ROOT/'design/EVALUATION_CARD.json');complete=read(ROOT/'evidence/QUERY_OUTPUTS_COMPLETE.json')
    assert complete['status']=='ALL_FIXED_PI0272_OUTPUTS_COMPLETE' and complete['evaluation_card_sha256']==sha(ROOT/'design/EVALUATION_CARD.json')
    registered={r['output']:r for r in card['rows']};outputs={r['path']:r for r in complete['rows']}
    assert len(registered)==len(outputs)==2148 and registered.keys()==outputs.keys()
    journals={}
    for name,cap in (('model_attempts',2560),('model_heads',7168)):
        records=[json.loads(s) for s in (ROOT/'evidence'/f'{name}.jsonl').read_text().splitlines() if s]
        assert len(records)<=cap and [r['ordinal'] for r in records]==list(range(1,len(records)+1));journals[name]=records
    attempts,heads=journals['model_attempts'],journals['model_heads'];counts=Counter(r['attempt'] for r in heads)
    assert set(counts)<=set(r['ordinal'] for r in attempts)
    assert set(r['request']['output'] for r in attempts)==set(outputs)
    technical=[]
    for a in attempts:
        p=Path(a['request']['output']);q=registered[str(p)];receipt=read(p.with_suffix('.receipt.json'));n=4 if q['readout']=='S4FIT' else 1
        assert a['request']==q and a['point']==q['point'] and a['input_sha256']==q['expected_condition_sha256']
        assert 0<=counts[a['ordinal']]<=n
        if receipt['attempt']==a['ordinal']:assert receipt['heads']==counts[a['ordinal']]==n
        else:
            assert receipt['attempt']>a['ordinal']
            technical.append(dict(attempt=a['ordinal'],heads=counts[a['ordinal']],saved_attempt=receipt['attempt'],output=str(p)))
    for p,r in outputs.items():
        receipt=read(Path(p).with_suffix('.receipt.json'))
        assert sha(p)==r['sha256']==receipt['output_sha256'] and sha(Path(p).with_suffix('.receipt.json'))==r['receipt_sha256']
    assert len(complete['repeatability'])==4 and all(r['exact_bytes'] and r['RNG_unchanged'] for r in complete['repeatability'])
    cpu=read(RAW/'delivery/CPU_REPLAY.json');assert cpu['status']=='PASS_PI0272_SAVED_READOUT_DECODE_AND_PHYSICAL_METRICS'
    assert {r['file']:r['sha256'] for r in cpu['predictions']}=={p:r['sha256'] for p,r in outputs.items()}
    assert cpu['model_queries']==cpu['gradient_queries']==cpu['physics_steps']==0
    for name,proof in cpu['tables'].items():
        path=RAW/'delivery'/name;assert sha(path)==proof['sha256']
        with path.open() as stream:assert len(list(csv.DictReader(stream)))==proof['rows']
    assert cpu['tables']['ALL_POINTS3504.csv']['rows']==3504
    assert len(read(RAW/'delivery/K8_BANKS192.json')['banks'])==192
    preserved=read(ROOT/'evidence/RESULT_ARCHIVE_predictions.json')
    assert preserved['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS'
    manifest_path=RAW/'archive/results-predictions/MEMBER_MANIFEST.json';manifest=read(manifest_path)
    assert manifest['tree_sha256']==preserved['tree_sha256'] and len(manifest['files'])==preserved['members']
    assert sha(manifest_path)==next(r['sha256'] for r in preserved['files'] if r['path']=='MEMBER_MANIFEST.json')
    kept={str(RAW/r['path']):r['sha256'] for r in manifest['files']}
    assert all(kept[p]==r['sha256'] and kept[str(Path(p).with_suffix('.receipt.json'))]==r['receipt_sha256'] for p,r in outputs.items())
    training=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json');trainers=[]
    for cp in training['complete_checkpoints']:
        path=ROOT/'evidence'/cp['arm']/f"checkpoint-{cp['step']}.full_readback.json";r=read(path)
        assert r['status']=='PASS_FULL_FILE_READBACK' and r['files']==cp['entries'];trainers.append(r)
    assert len(trainers)==4
    write(ROOT/'evidence/FINAL_ARTIFACT_AUDIT.json',dict(status='PASS_PI0272_ALL_FIXED_RESULTS_TRAINERS_AND_BUDGETS',
        effective_policies=['AGG4096/S4_REUSED','AGG4096/DIRECT0','S4FIT1024','DIRECT1024'],
        formal_updates=dict(S4FIT=1024,DIRECT=1024),raw_completed_updates=training['raw_completed_updates'],
        raw_training_heads=training['raw_training_heads'],new_quality_outputs=2148,quality_attempts=len(attempts),quality_heads=len(heads),
        technical_quality_attempts=technical,reused_AGG_outputs=1360,complete_trainer_readbacks=trainers,predictions_archive=preserved,
        new_teacher_labels=0,new_episodes=0,new_renders=0,new_WBC=0,new_physics=0,
        summary_archive_and_terminal_publication='STILL_REQUIRED',selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PASS_PI0272_ALL_FIXED_RESULTS_TRAINERS_AND_BUDGETS',len(attempts),len(heads),flush=True)


if __name__=='__main__':main()
