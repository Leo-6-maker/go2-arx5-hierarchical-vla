"""Seal actual admitted new data, exact inherited code and current storage."""
from provision import ROOT, CONTROL, read, sha, write


def main():
    assert read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')['status']=='PASS_PI0270_ACTUAL_FROZEN_MAPPING_AND_CONTROL_INPUTS'
    assert read(ROOT/'evidence/RUNTIME_BINDING_AGG.json')['status']=='PASS_EXACT_FROZEN_HELPER_IMPORTS_AND_CPU_HOOK_INSTALL'
    assert read(CONTROL/'evidence/HOPPER_PHYSICAL_FIXTURES.json')['status']=='PASS_ROT_CONTRACT_PHYSICAL_AND_WEIGHTED_FIXTURES'
    paths=[]
    from pathlib import Path
    for row in read(CONTROL/'EXECUTION_FILESET.json')['files']:
        p=Path(row['path']);assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],p
        paths.append(p)
    for module in read(ROOT/'evidence/RUNTIME_BINDING_AGG.json')['modules']:paths.append(Path(module['path']))
    paths.extend(ROOT/'code'/name for name in ('train.py','cpu_admission.py','stage_data.py','provision.py',
        'runtime_binding_check.py','seal_training.py','checkpoint.py','training_audit.py','retire_discarded_canary.py'))
    paths.extend(ROOT/name for name in ('PAIRING_PLAN.json','RECOVERY_MAPPING.json','DONOR_LOCAL_READBACK.json','STORAGE_ADMISSION.json',
        'design/APPROVED_RUN_CARD.json','design/NEW_FIT192.json','contracts/ROT/source/dataset/statistics.json',
        'evidence/DATA_COVERAGE_ADMISSION.json','evidence/NEWFIT_ADMISSION.json','evidence/TRAINING_DATA_ARCHIVE.json',
        'evidence/TRAINING_DATA_LOCAL_READBACK.json','evidence/HOPPER_CPU_ADMISSION.json','evidence/RUNTIME_BINDING_AGG.json','evidence/DONOR_REUSE_BINDING.json'))
    manifest=ROOT/'archive/training-data/MEMBER_MANIFEST.json';paths.append(manifest)
    for row in read(manifest)['files']:
        p=ROOT/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
        paths.append(p)
    write(ROOT/'EXECUTION_FILESET.json',dict(schema='PI0270_FIXED_QUALIFIED_TRAINING_FILESET_V1',
          files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paths))],
          source_control_fileset_sha256=sha(CONTROL/'EXECUTION_FILESET.json'),
          data_change='Only fixed eligible learner-state RECOVERY rows; COMMON and full PREFIX recipe unchanged',
          no_model_or_optimizer_queries_by_seal=True))
    print('FROZEN_ACTUAL_PI0270_TRAINING_FILESET',len(set(paths)),flush=True)


if __name__=='__main__':main()
