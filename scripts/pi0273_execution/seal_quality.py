"""Seal fixed evaluation only after both final training audits and model-only bindings."""
from pathlib import Path
from quality_worker import ROOT,BASE,read,sha,write

def main():
    audit=read(ROOT/'evidence/MAIN_TRAINING_AUDIT.json')
    assert audit['status']=='PASS_PI0273_BOTH_ACTUAL_5120_MATCHED_TRAINING'
    assert len(audit['complete_checkpoints'])==4
    check=read(ROOT/'evidence/QUALITY_CPU_FIXTURES.json')
    assert check['status']=='PASS_CONSTRUCTED_ADMISSION_INVALID_RETENTION_AND_PHYSICAL_SCORE'
    for r in check['source_files']:assert sha(r['path'])==r['sha256']
    native=read(ROOT/'evidence/NATIVE_CPU_FIXTURES.json');assert native['status']=='PASS_CONSTRUCTED_NATIVE_E0_E1_AND_GATE'
    for r in native['source_files']:assert sha(r['path'])==r['sha256']
    assert read(ROOT/'evidence/HISTORICAL_TABLE_REUSE.json')['status']=='PASS_CLOSED_AGG_PANELS_AND_S4FIT_LOCAL_CONTEXT_ONLY'
    paths=list((ROOT/'code').glob('*.py'))+list((ROOT/'design').glob('*.json'))
    for arm in ('BROAD','TARGET'):
        point=arm+'5120';p=ROOT/'model_bindings'/f'{point}.json';binding=read(p)
        assert binding['view_type']=='model-only' and binding['point']==point
        assert binding['readout_contract_sha256']==sha(ROOT/'design'/f'{arm}_READOUT_R1.json')
        paths.append(p)
        p=ROOT/'evidence'/point/'FROZEN_SCOPE.json'
        assert read(p)['status']=='PASS_FIXED_FINAL_MODEL_VIEW_PRECISION_FROZEN_SCOPE';paths.append(p)
        p=ROOT/'evidence'/arm/'checkpoint-5120.archive_receipt.json';receipt=read(p)
        expected=next(r for r in audit['complete_checkpoints'] if (r['arm'],r['step'])==(arm,5120))
        assert receipt['status']=='UPLOADED' and receipt['entries']==expected['entries'];paths.append(p)
    for campaign,name in (('PI0270','QUALITY_EXECUTION_FILESET.json'),('PI0271','EXECUTION_FILESET.json')):
        for r in read(BASE/campaign/'design'/name)['files']:
            p=Path(r['path'])
            if p.suffix=='.py':assert sha(p)==r['sha256'];paths.append(p)
    paths.extend(ROOT/p for p in ('contracts/ROT/source/dataset/statistics.json','evidence/MAIN_TRAINING_AUDIT.json',
        'evidence/PRECISION_CONTRACT_RESOLUTION.json','evidence/QUALITY_CPU_FIXTURES.json','evidence/NATIVE_CPU_FIXTURES.json',
        'evidence/QUALITY_MEMBERSHIP_READBACK.json','evidence/A0_SOURCE_AND_PLAN.json','evidence/HISTORICAL_TABLE_REUSE.json'))
    paths=sorted(set(p for p in paths if p.name!='QUALITY_EXECUTION_FILESET.json'))
    write(ROOT/'design/QUALITY_EXECUTION_FILESET.json',dict(status='FROZEN_PI0273_BEFORE_NEW_QUALITY',
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths],
        training_audit_sha256=sha(ROOT/'evidence/MAIN_TRAINING_AUDIT.json'),model_queries=0))
    print('SEALED_PI0273_QUALITY',len(paths),flush=True)

if __name__=='__main__':main()
