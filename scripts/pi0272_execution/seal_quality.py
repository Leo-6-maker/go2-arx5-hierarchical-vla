"""Seal only after both final audits and three exact effective-policy bindings."""
from pathlib import Path
from quality_worker import BASE,ROOT,read,sha,write


def main():
    audit=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json')
    assert audit['status']=='PASS_PI0272_BOTH_ACTUAL_1024_MATCHED_TRAINING' and len(audit['complete_checkpoints'])==4
    check=read(ROOT/'evidence/POSTTRAINING_CPU_CHECK.json')
    assert check['status']=='PASS_CPU_ADMISSION_BLOCK_REPLAY_INVALID_RETENTION_AND_ARITHMETIC'
    for r in check['source_files']:assert sha(r['path'])==r['sha256']
    paths=list((ROOT/'code').glob('*.py'))
    assert {(r['arm'],r['step']) for r in audit['complete_checkpoints']}=={(a,s) for a in ('S4FIT','DIRECT') for s in (512,1024)}
    # Model bytes are verified at the pinned upload revision; full Trainer readback remains a terminal requirement.
    for arm in ('S4FIT','DIRECT'):
        p=ROOT/'evidence'/arm/'checkpoint-1024.archive_receipt.json';receipt=read(p)
        assert receipt['status']=='UPLOADED' and (receipt['arm'],receipt['step'])==(arm,1024)
        expected=next(r for r in audit['complete_checkpoints'] if (r['arm'],r['step'])==(arm,1024))
        assert receipt['entries']==expected['entries'];paths.append(p)
    for point in ('S4FIT1024','DIRECT0','DIRECT1024'):
        p=ROOT/'model_bindings'/f'{point}.json';binding=read(p)
        assert binding['view_type']=='model-only' and binding['point']==point
        mode=ROOT/'design'/('S4FIT_READOUT_R1.json' if point=='S4FIT1024' else 'DIRECT_READOUT_R1.json')
        assert binding['readout_contract_sha256']==sha(mode)
        paths.extend((p,mode))
        if point!='DIRECT0':
            p=ROOT/'evidence'/point/'FROZEN_SCOPE.json'
            assert read(p)['status']=='PASS_FIXED_FINAL_MODEL_VIEW_PRECISION_FROZEN_SCOPE';paths.append(p)
    for campaign,name in (('PI0270','QUALITY_EXECUTION_FILESET.json'),('PI0271','EXECUTION_FILESET.json')):
        seal=BASE/campaign/'design'/name
        for r in read(seal)['files']:
            p=Path(r['path'])
            if p.suffix=='.py':assert sha(p)==r['sha256'];paths.append(p)
    paths.extend(ROOT/p for p in ('design/FITTING_CARD.json','design/EVALUATION_CARD.json',
        'contracts/ROT/source/dataset/statistics.json','evidence/MAIN_TRAINING_AUDIT.json',
        'evidence/PRECISION_CONTRACT_RESOLUTION.json','evidence/POSTTRAINING_CPU_CHECK.json'))
    files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paths))]
    write(ROOT/'design/QUALITY_EXECUTION_FILESET_R1.json',dict(schema='PI0272_PHASE_BOUND_QUALITY_R1',files=files,
        training_audit_sha256=sha(ROOT/'evidence/MAIN_TRAINING_AUDIT.json'),model_queries=0))
    print('SEALED_PI0272_QUALITY_R1',len(files),flush=True)


if __name__=='__main__':main()
