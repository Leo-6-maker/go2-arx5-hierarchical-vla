"""Typed PV1 entry for established relay helpers; old scopes retain their guards."""
from pathlib import Path

from scripts.pi0292.durable import read
from scripts.pi0294.e1.scope import STATS
from scripts.pi0294.fd1 import sha
from .dataset import POPULATION,admit_new_support
from .decision import AUTHORITY
from .scope import EVAL_ROLE,FINAL_EVAL_ROLE,validate as validate_core
from .neural_scope import validate as validate_neural


def validate_runtime(root,profile,*,closeout=False,now=None):
    if now is not None:
        raise PermissionError('PV1_NATIVE_RUNTIME_CLOCK_ONLY_NO_INJECTED_DATE')
    root=Path(root)
    if profile=='hopper-neural':
        return validate_neural(root,closeout=closeout)[0]
    if profile!='4080-simulation':
        raise PermissionError('PV1_ONLY_BOUND_CORE_OR_HOPPER_POLICY_RUNTIME')
    raw_manifest=read(root/'RUN_MANIFEST.json')
    role=FINAL_EVAL_ROLE if raw_manifest.get('scope')==FINAL_EVAL_ROLE else EVAL_ROLE
    m,inputs=validate_core(root,role=role,closeout=closeout)
    stage=read(root/'PV1_STAGE_BINDING.json')
    plan_name='PV1_P5_LOCKBOX_PLAN.json' if role==FINAL_EVAL_ROLE else 'SOURCE_EVALUATION_PLAN.json'
    plan=read(root/'plans'/plan_name)
    expected_schema='PI0294_PV1_P5_PRESEALED_LOCKBOX80_V1' if role==FINAL_EVAL_ROLE else 'PI0294_PV1_PRESEALED_POLICY_EVALUATION_V1'
    if (stage['schema']!='PI0294_PV1_ADMITTED_SHARED_POLICY_STAGE_V1'
            or stage['authority_release']!=AUTHORITY or stage['population_plan_sha256']!=POPULATION
            or stage['Teacher_takeover'] is not False or stage['normalizer_refit'] is not False
            or stage['evaluation_data_for_training'] is not False
            or plan['schema']!=expected_schema
            or plan['population_plan_sha256']!=POPULATION
            or sha(root/'PV1_STAGE_BINDING.json')!=m['execution']['sealed_inputs']['PV1_STAGE_BINDING.json']):
        raise PermissionError('PV1_SOURCE_ONLY_FROZEN_EVALUATION_STAGE_AND_PLAN')
    bank=Path(inputs['new_dataset'])
    if (sha(bank/'DATASET_MANIFEST.json')!=stage['new_bank_manifest_sha256']
            or sha(bank/'statistics.json')!=STATS or sha(bank/'READY.json')!=stage['new_bank_ready_sha256']):
        raise PermissionError('PV1_DATA_GATE_BEFORE_ANY_AUTONOMOUS_EVALUATION')
    admit_new_support(read(bank/'DATASET_MANIFEST.json'),stage['route'])
    return m
