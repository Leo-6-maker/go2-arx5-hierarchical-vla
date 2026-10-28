"""Minimal policy-package entry; dry-run is the default and no robot transport exists."""
import argparse
import json
from pathlib import Path
import re


REQUIRED=('shared_weights','processor','statistics','runtime','precision','action_schema','dependency_lock','offline_inference_entry')


def validate_manifest(value):
    artifacts=value.get('artifacts',{})
    if (value.get('schema')!='PI0294_PV1_POLICY_MANIFEST_V1'
            or set(artifacts)!=set(REQUIRED) or value.get('shared_weights_for_tasks')!=['T1','T2']
            or value.get('default_mode')!='OFFLINE_DRY_RUN'
            or value.get('robot_or_WBC_connection_default') is not False
            or value.get('WBC_modified') is not False or value.get('real_robot_deployment')!='NONE'
            or any(not re.fullmatch('[0-9a-f]{64}',x.get('sha256','')) for x in artifacts.values())):
        raise PermissionError('PV1_OFFLINE_EXACT_HASHED_SHARED_POLICY_PACKAGE')
    return value


def dry_run(manifest):
    value=validate_manifest(manifest)
    return dict(status='PASS_OFFLINE_SCHEMA_DRY_RUN_NO_MODEL_LOAD',candidate=value['candidate'],
        tasks=['T1','T2'],models_loaded=0,generation_queries=0,robot_connections=0,WBC_connections=0)


def generate_once(manifest,observation,instruction,*,loader):
    validate_manifest(manifest)
    if instruction not in manifest.get('task_instructions',{}).values():
        raise PermissionError('PV1_OFFLINE_ONLY_PACKAGED_T1_T2_LANGUAGE')
    required={'body_rgb','wrist_rgb','eef10','timestamp_ns'}
    if set(observation)!=required:raise PermissionError('PV1_OFFLINE_EXACT_OBSERVATION_SCHEMA')
    instance=None
    try:
        instance=loader(manifest)
        action=instance.generate(observation,instruction)
        if not isinstance(action,list) or len(action)!=40 or any(not isinstance(row,list) or len(row)!=10 for row in action):
            raise ValueError('PV1_OFFLINE_GENERATION_MUST_RETURN_FINITE_H40_EEF10')
        if any(not isinstance(x,(int,float)) or isinstance(x,bool) or not -1e100<x<1e100 for row in action for x in row):
            raise ValueError('PV1_OFFLINE_GENERATION_MUST_RETURN_FINITE_H40_EEF10')
        return action
    finally:
        if instance is not None:instance.reset();instance.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();value=json.loads(Path(args.manifest).read_text())
    if not args.dry_run:raise PermissionError('PV1_OFFLINE_ENTRY_DEFAULTS_TO_DRY_RUN_NO_MODEL_OR_ROBOT')
    print(json.dumps(dry_run(value),sort_keys=True))


if __name__=='__main__':main()
