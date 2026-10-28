"""Materialize the explicit R1 authority; retain V1 cards and CPU admission."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0272')
RELEASE='8a2ffbefe0c57536c0a7e340da0616c7987c74f1'
CANONICAL=5980876835


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    text=json.dumps(value,indent=2,sort_keys=True)+'\n';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)


def bind():
    source=ROOT/'authority/PRECISION_CONTRACT_RESOLUTION_R1.json';decision=read(source)
    canonical=read(ROOT/'authority/R1_CANONICAL.json')
    assert canonical['id']==CANONICAL and RELEASE in canonical['body']
    assert decision['status']=='RESOLVED_BY_EXPLICIT_SCIENTIFIC_AUTHORITY' and decision['launch_allowed']
    for name,digest in [('A2_PRECISION_CONTRACT_CONFLICT.json',decision['evidence']['conflict_reported_sha256']),
                        ('A2_PHASE_PRECISION_EVIDENCE.json',decision['evidence']['phase_reported_sha256'])]:
        assert sha(ROOT/'evidence'/name)==digest
    result=dict(decision,amendment_release=RELEASE,amendment_canonical=CANONICAL,
        canonical_body_sha256=hashlib.sha256(canonical['body'].encode()).hexdigest(),
        authorization_file_sha256=sha(source),runtime_binding_PASS_claimed=False,canary_PASS_claimed=False)
    write(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json',result)
    for arm in ('S4FIT','DIRECT'):
        original=ROOT/'design'/f'{arm}_READOUT.json';mode=read(original)
        mode.update(schema='PI0272_READOUT_CONTRACT_R1',amendment_release=RELEASE,amendment_canonical=CANONICAL,
            original_requested_mode_sha256=sha(original),query_dtype='EXPLICIT_PHASE_BOUND',
            query_dtype_by_phase=dict(training='torch.float32',quality='torch.bfloat16'),
            feature_dtype_by_phase=dict(training='torch.float32',quality='torch.bfloat16'),
            native_training_epsilon_dtype='torch.bfloat16',training_parameter_precision='EXACT_DONOR_MIXED_NAME_SHAPE_DTYPE_MAP',
            quality_parameter_precision='EXISTING_MODEL_ONLY_BFLOAT16_LOADING',
            generated_output_dtype='ACTUAL_NATIVE_READOUT_RECORDED_NOT_FORCED',
            phase_from_module_training_flag=False,cross_phase_output_equality_required=False,
            resolution_sha256=sha(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json'))
        write(ROOT/'design'/f'{arm}_READOUT_R1.json',mode)
    before=ast.parse((ROOT/'evidence/cpu-executed-source/train.py').read_text())
    after=ast.parse((ROOT/'code/train.py').read_text())
    dataset=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Pairs'),include_attributes=False)
    assert dataset(before)==dataset(after),'ACTUAL_CPU_DATASET_DEPENDENCY_CHANGED'
    cpu=read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')
    assert cpu['status']=='PASS_PI0272_BOTH_FULL_2048_ACTUAL_PROCESSOR_STREAMS'
    assert cpu['fitting_card_sha256']==sha(ROOT/'design/FITTING_CARD.json')
    for arm,digest in cpu['sample_audits'].items():assert sha(ROOT/'evidence/input-audit'/arm/'sample_audit.jsonl')==digest
    write(ROOT/'evidence/R1_CPU_ADMISSION_REUSE.json',dict(status='PASS_R1_REUSES_UNCHANGED_ACTUAL_4096_MICRO_CPU_INPUTS',
        original_receipt_sha256=sha(ROOT/'evidence/HOPPER_CPU_ADMISSION.json'),unchanged_dataset_AST=True,
        fitting_card_sha256=cpu['fitting_card_sha256'],evaluation_card_sha256=sha(ROOT/'design/EVALUATION_CARD.json'),
        sample_audits=cpu['sample_audits'],new_encodings=0,new_model_queries=0))
    print('R1_PRECISION_AUTHORITY_BOUND_EXISTING_CPU_INPUTS_REUSED')


def seal():
    assert read(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json')['amendment_release']==RELEASE
    assert read(ROOT/'evidence/CPU_FIXTURES_R1.json')['status']=='PASS_PI0272_CONSTRUCTED_DIRECT_AND_UNCHANGED_PHYSICAL_CPU_FIXTURES'
    paths=list((ROOT/'code').glob('*'))+list((ROOT/'authority').glob('*'))
    for arm in ('S4FIT','DIRECT'):
        p=ROOT/'evidence'/f'RUNTIME_BINDING_{arm}.json';binding=read(p)
        assert binding['status']=='PASS_PI0272_ACTUAL_FROZEN_HELPERS_AND_HOOKS'
        assert binding['readout_source_sha256']==sha(ROOT/'code/readout.py')
        paths.append(p);paths.extend(Path(m['path']) for m in binding['modules'])
    # Retain exact inherited code dependencies, without repeating archived data scans.
    inherited=read(ROOT.with_name('PI0270')/'EXECUTION_FILESET.json')
    for row in inherited['files']:
        p=Path(row['path'])
        if p.suffix=='.py':
            assert sha(p)==row['sha256'];paths.append(p)
    paths.extend(ROOT/p for p in ('DONOR_LOCAL_READBACK.json','DONOR_PARAMETER_SCOPE.json','STORAGE_ADMISSION.json',
        'contracts/ROT/source/dataset/statistics.json','design/FITTING_CARD.json','design/EVALUATION_CARD.json',
        'design/S4FIT_READOUT_R1.json','design/DIRECT_READOUT_R1.json','evidence/PRECISION_CONTRACT_RESOLUTION.json',
        'evidence/R1_CPU_ADMISSION_REUSE.json','evidence/CPU_FIXTURES_R1.json','evidence/HOPPER_CPU_ADMISSION.json'))
    paths=sorted({p for p in paths if p.is_file()})
    write(ROOT/'EXECUTION_FILESET.json',dict(schema='PI0272_R1_ACTUAL_TRAINING_FILESET',amendment_release=RELEASE,
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths],
        real_canaries_still_required=True,model_queries_by_seal=0))
    print('PI0272_R1_RUNTIME_SEALED',len(paths))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('bind','seal'));a=p.parse_args()
    bind() if a.action=='bind' else seal()
