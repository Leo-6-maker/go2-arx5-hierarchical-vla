"""Bind admitted CPU evidence and actual imported mature trainer dependencies."""
import json
from pathlib import Path
from provision import ROOT,OLD,sha,read,write

def main():
    assert read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')['status']=='PASS_PI0269_HOPPER_ACTUAL_PROCESSOR_BOUNDARY'
    assert read(ROOT/'evidence/HOPPER_PHYSICAL_FIXTURES.json')['status']=='PASS_ROT_CONTRACT_PHYSICAL_AND_WEIGHTED_FIXTURES'
    assert read(ROOT/'evidence/SAVED_STATE_PREFIX_AUDIT.json')['status']=='PASS_BOUNDED_SAME_STATE_LABEL_DIAGNOSIS_NO_NEW_CORRECTNESS_DEFECT'
    paths=[]
    for row in read(OLD/'EXECUTION_FILESET.json')['files']:
        p=Path(row['path'])
        if not p.is_relative_to(OLD):
            assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
            paths.append(p)
    for arm in ('UNIFORM','PREFIX'):
        p=ROOT/'evidence'/('RUNTIME_BINDING_'+arm+'.json')
        assert read(p)['status']=='PASS_EXACT_FROZEN_HELPER_IMPORTS_AND_CPU_HOOK_INSTALL';paths.append(p)
    paths.extend(ROOT/'code'/name for name in ('train.py','temporal.py','retire_discarded_canary.py','checkpoint.py','training_audit.py'))
    paths.extend(ROOT/name for name in ('PAIRING_PLAN.json','DONOR_LOCAL_READBACK.json','STORAGE_ADMISSION.json','design/APPROVED_RUN_CARD.json',
        'contracts/ROT/source/dataset/statistics.json','evidence/HOPPER_CPU_ADMISSION.json','evidence/HOPPER_PHYSICAL_FIXTURES.json',
        'evidence/SAVED_STATE_PREFIX_AUDIT.json'))
    write(ROOT/'EXECUTION_FILESET.json',dict(schema='PI0269_QUALIFIED_TRAINING_FILESET_V1',
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paths))],
        source_fileset_sha256=sha(OLD/'EXECUTION_FILESET.json'),loss_change='Only fixed temporal physical reductions; native/full-speed untouched',
        no_new_model_query_by_seal=True))
    print('FROZEN_PI0269_ACTUAL_TRAINING_FILESET',len(set(paths)),flush=True)

if __name__=='__main__':main()
