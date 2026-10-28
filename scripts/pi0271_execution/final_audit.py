"""Reconcile registered diagnosis outputs, literal consumption and preserved evidence."""
import argparse
import csv
from collections import Counter
from pathlib import Path
from support import ROOT, RAW, read, sha, write
from close_predecessor import journal


def counts(attempts,heads):
    head_counts=Counter(r['attempt'] for r in heads)
    assert set(head_counts)<=set(r['ordinal'] for r in attempts)
    assert all(0<=head_counts[r['ordinal']]<=4 for r in attempts)
    return head_counts


def main():
    from worker import admitted
    admitted()
    plan_path=ROOT/'design/QUERY_PLAN.json';plan=read(plan_path)
    complete=read(ROOT/'evidence/QUERY_OUTPUTS_COMPLETE.json')
    assert complete['status']=='ALL_FIXED_PI0271_OUTPUTS_COMPLETE' and complete['query_plan_sha256']==sha(plan_path)
    registered={r['request']['output']:r for r in plan['rows']}
    outputs={r['path']:r for r in complete['rows']}
    assert len(outputs)==len(registered)==len(plan['rows'])<=2400 and set(outputs)==set(registered)
    panels=Counter(r['panel'] for r in plan['rows'])
    assert panels['A1']==1184 and panels['A2']==1152 and panels['A3']<=64
    attempts=journal(ROOT/'evidence/model_attempts.jsonl',2560)
    heads=journal(ROOT/'evidence/model_heads.jsonl',10240);head_counts=counts(attempts,heads)
    assert set(r['request']['output'] for r in attempts)==set(outputs)
    technical=[]
    for attempt in attempts:
        path=Path(attempt['request']['output']);row=registered[str(path)];receipt=read(path.with_suffix('.receipt.json'))
        assert attempt['point']==row['point'] and attempt['request']==row['request']
        assert receipt['input_sha256']==attempt['input_sha256'] and receipt['noise_sha256']==attempt['noise_sha256']
        if receipt['attempt']!=attempt['ordinal']:
            assert receipt['attempt']>attempt['ordinal']
            technical.append(dict(attempt=attempt['ordinal'],heads=head_counts[attempt['ordinal']],
                saved_attempt=receipt['attempt'],state='NO_FINAL_OUTPUT_FOR_THIS_ATTEMPT',output=str(path)))
        else:assert receipt['heads']==head_counts[attempt['ordinal']]==4 and receipt['native_S4']
    for path,row in outputs.items():
        receipt=read(Path(path).with_suffix('.receipt.json'))
        assert sha(path)==row['sha256']==receipt['output_sha256']
        assert sha(Path(path).with_suffix('.receipt.json'))==row['receipt_sha256']
    labels=read(ROOT/'evidence/VISITED_LABELS_COMPLETE.json')
    assert labels['status']=='ALL_64_PROPOSALS_LABELLED_OR_MISSING_RETAINED' and len(labels['records'])==64
    teachers=journal(ROOT/'evidence/teacher_calls.jsonl',80)
    assert len(teachers)==labels['teacher_attempts']
    visited=read(ROOT/'design/VISITED_STATE_CARD_V2.json')['rows'];assert len(visited)==64
    visited_by_id={r['input_id']:r for r in visited};label_rows=[];retained=[]
    for row in labels['records']:
        assert sha(row['path'])==row['sha256'];value=read(row['path']);source=value['row']
        assert source==visited_by_id[source['input_id']] and value['DIAGNOSIS_ONLY'] and value['never_train_or_blind_TEST']
        assert value['model_queries']==value['render_calls']==value['physics_steps']==0
        if value.get('sample'):
            assert sha(value['sample'])==value['sample_sha256'];retained.append((value['sample'],value['sample_sha256']))
            call=teachers[value['teacher_call_ordinal']-1]
            assert call['input_id']==source['input_id'] and call['observation_sha256']==source['observation_sha256']
        label_rows.append(value);retained.append((row['path'],row['sha256']))
    assert len({r['row']['input_id'] for r in label_rows})==64
    assert all(r['input_id'] in visited_by_id for r in teachers)
    cpu=read(RAW/'delivery/CPU_REPLAY.json')
    assert cpu['status']=='PASS_ALL_NEW_S4_DECODE_METRICS_AND_FIXED_K8_REPLAY'
    assert {r['file']:r['sha256'] for r in cpu['predictions']}=={p:r['sha256'] for p,r in outputs.items()}
    assert cpu['model_queries']==cpu['teacher_calls']==cpu['episodes']==cpu['renders']==0
    expected={'SUPPORT_ALL_NOISES2336.csv':2336,'A1_SUPPORT1184.csv':1184,'PAIRED_SUPPORT1168.csv':1168,
        'A1_PAIRED592.csv':592,'K8_REGIONS960.csv':960,'PAIRED_K8_REGIONS480.csv':480,'VISITED128.csv':128,'PAIRED_VISITED64.csv':64}
    for name,proof in cpu['tables'].items():
        path=RAW/'delivery'/name;assert sha(path)==proof['sha256']
        with path.open() as stream:actual=len(list(csv.DictReader(stream)))
        assert actual==proof['rows'] and (name not in expected or actual==expected[name])
    assert set(expected)<=set(cpu['tables'])
    banks=read(RAW/'delivery/K8_BANKS192.json');assert len(banks['banks'])==192 and banks['K']==8 and banks['primary_scene_units']==24
    preserved={};archives=[]
    for part in ('support','visited'):
        receipt=read(ROOT/'evidence'/('RESULT_ARCHIVE_'+part+'.json'))
        assert receipt['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS'
        path=RAW/'archive'/('results-'+part)/'MEMBER_MANIFEST.json';manifest=read(path)
        assert manifest['tree_sha256']==receipt['tree_sha256'] and len(manifest['files'])==receipt['members']
        assert sha(path)==next(r['sha256'] for r in receipt['files'] if r['path']=='MEMBER_MANIFEST.json')
        preserved.update({str(RAW/r['path']):r['sha256'] for r in manifest['files']});archives.append(receipt)
    retained.extend((p,r['sha256']) for p,r in outputs.items())
    assert all(preserved[p]==digest for p,digest in retained)
    sources=[ROOT/'design/EXECUTION_FILESET.json',ROOT/'evidence/A0_DECISION.json',plan_path,
        ROOT/'evidence/QUERY_OUTPUTS_COMPLETE.json',ROOT/'evidence/VISITED_LABELS_COMPLETE.json',
        RAW/'delivery/CPU_REPLAY.json',RAW/'delivery/K8_BANKS192.json',Path(__file__)]
    write(ROOT/'evidence/FINAL_ARTIFACT_AUDIT.json',dict(status='PASS_COMPLETE_FIXED_DIAGNOSIS_ARTIFACTS_AND_BUDGETS',
        fixed_queries=dict(panels),unique_new_outputs=len(outputs),inference_attempts=len(attempts),S4_heads=len(heads),
        technical_inference_attempts=technical,teacher_attempts=len(teachers),teacher_proposals=64,
        label_outcomes=dict(Counter(r['status'] for r in label_rows)),A3_reused_AGG_predictions=sum(r['status']!='MISSING_AFTER_STOP' for r in visited),
        no_new_training_episode_render_or_physics_by_sealed_execution=True,
        source_files=[dict(path=str(p),sha256=sha(p)) for p in sources],archives=archives,
        summary_archive_and_terminal_publication='STILL_REQUIRED',selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))
    print('PASS_COMPLETE_FIXED_DIAGNOSIS_ARTIFACTS_AND_BUDGETS',len(attempts),len(heads),len(teachers),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:
        assert counts([dict(ordinal=1),dict(ordinal=2)],[dict(attempt=1)]*2+[dict(attempt=2)]*4)=={1:2,2:4}
        try:counts([dict(ordinal=1)],[dict(attempt=2)])
        except AssertionError:pass
        else:raise AssertionError('ORPHAN_HEAD_NOT_REJECTED')
        print('PASS_LITERAL_PARTIAL_HEAD_ACCOUNTING_NO_MODEL_QUERY')
    else:main()
