"""Reuse a complete diagnostic checkpoint after its post-save restore failed.

This never accepts a partial scientific endpoint or grants formal training. The
fresh successor still executes independent split2/resume4 and compares every
state to the original continuous4. Closed failures stay closed and unchanged.
"""
import json
from pathlib import Path
from . import target_gr00t_profile_v1 as profile
from .storage import digest


def identity(spec,header,current):
    p=profile.validate(spec);need=profile.need
    need(spec.get('execution_site')=='CORELAB_4080S'
        and spec.get('core_native_operation')=='FULL_ACTION_RESTORE_ONLY'
        and p['profile']==profile.PROFILE and p['mode']=='RESTORE4'
        and p['variant']=='FULL_ACTION_SIDE'and p['endpoint']==4 and spec['start_step']==0
        and all(p[k]==current[k]for k in('dataset_sha256','statistics_sha256','memory_policy'))
        and all(spec[k]==current[k]for k in('parent','bindings','GPU_uuid')),
        'COMPLETE_FOUR_STEP_RUNTIME_DIAGNOSTIC_ONLY_NOT_FORMAL_OR_OTHER_GENERATION')
    need(header.get('schema')=='WM02_TARGET_GR00T_FULL_STATE_DELTA_V1'
        and header.get('global_step')==4 and header.get('cursor')==128
        and header.get('generation_id')==spec['generation_id']
        and header.get('parent_manifest_sha256')==profile.PARENT and header.get('recipe')==p,
        'SEALED_COMPLETE_DIAGNOSTIC4_HEADER_AND_EXACT_GENERATION')


def continuous4(c):
    from .task_learning_training_admission_v1 import unchanged_training_math
    from .task_learning_probes_v1 import artifact
    need=profile.need;descriptor=c.get('sealed_continuous4_reference')
    need(isinstance(descriptor,dict)and set(descriptor)=={'path','sha256'},'EXPLICIT_SEALED_CONTINUOUS4_REFERENCE')
    config_path=Path(descriptor['path']);root=config_path.parent
    need(config_path==root/'ALLOCATION_CONFIG.json'and root.parent==Path(c['root']).parent
        and root.name.startswith('target-gr00t-core-native-full-')
        and digest(config_path)==descriptor['sha256'],'OWN_COLD_DIAGNOSTIC_REFERENCE_ROOT')
    old=json.loads(config_path.read_bytes());child=root/'continuous4'
    spec=json.loads((child/'EXECUTION_SPEC.json').read_bytes())
    lease=json.loads((root/'control/LEASE.json').read_bytes());worker=json.loads((root/'WORKER_EXIT.private.json').read_bytes())
    child_lease=json.loads((child/'control/LEASE.json').read_bytes());child_worker=json.loads((child/'WORKER_EXIT.private.json').read_bytes())
    result=json.loads((child/'RESULT.json').read_bytes());failure=result.get('failure')or''
    registry=json.loads(Path(c['registry']).read_bytes())
    need(old.get('original_Trainer_execution')is True and old.get('formal_training')is False
        and old.get('operation')=='FULL_ACTION_RESTORE_ONLY'and old['registry']==c['registry']
        and old['python']==c['python']
        and any(e.get('path')==str(root)and e.get('source_sha')==old['source_sha']for e in registry['entries'])
        and lease.get('status')==child_lease.get('status')=='CLOSED'
        and lease.get('worker_exit_code')==worker.get('exit_code')==child_lease.get('worker_exit_code')==child_worker.get('exit_code')==1
        and lease.get('config_sha256')==descriptor['sha256']
        and lease.get('result_sha256')==digest(root/'RESULT.json')
        and child_lease.get('spec_sha256')==digest(child/'EXECUTION_SPEC.json')
        and child_lease.get('result_sha256')==digest(child/'RESULT.json')
        and result.get('global_step')==result.get('GR00T_updates')==4
        and result.get('actual_progress',{}).get('cursor')==128
        and all(word in failure for word in('load_checkpoint','load_state_dict','OutOfMemoryError')),
        'PRESERVED_ACTUAL_POST_SAVE_OPTIMIZER_RESTORE_FAILURE_NOT_SCIENTIFIC_PROMOTION')
    checkpoint=child/'training/checkpoint-4';header=json.loads((checkpoint/'CHECKPOINT_COMPLETE.json').read_bytes())
    identity(spec,header,c)
    need(old['source_sha']==spec['source_sha']and spec['source_files']==old['source_files'],
        'EXACT_OLD_DIAGNOSTIC_SOURCE')
    unchanged_training_math(spec,c)
    for name,pin in header['files'].items():
        path=checkpoint/name
        need(path.resolve().is_relative_to(checkpoint)and digest(path)==pin,'COMPLETE_REFERENCE_CHECKPOINT_MEMBER_'+name)
    contract=json.loads((checkpoint/'PARAMETER_CONTRACT.json').read_bytes())
    numerical=json.loads((child/'WEIGHT_CACHE_EQUIVALENCE_ACTUAL.json').read_bytes())
    need(contract.get('variant')=='FULL_ACTION_SIDE'and contract.get('trainable_numel')==1304913028
        and len(contract['trainable'])==537 and numerical.get('gradient_tensors')==537
        and numerical.get('gradient_exact')is numerical.get('noise_and_RNG_exact')is True,
        'ACTUAL_COMPLETE_FULL_ACTION_AND_NUMERICAL_EQUIVALENCE_REFERENCE')
    after=child/'after/RESULT.json';records=json.loads(after.read_bytes())['records']
    need(len(records)==4 and all(r['model_global_step']==4 for r in records),
        'ACTUAL_UNRESTORED_FOUR_STEP_PROBE_REFERENCE')
    for row in records:need(digest(artifact(after.parent,row))==row['sha256'],'SEALED_REFERENCE_H40_PROBE')
    return dict(checkpoint=str(checkpoint),generation_id=header['generation_id'],
        after=dict(path=str(after),sha256=digest(after)),reference_root=str(root),
        checkpoint_sha256=digest(checkpoint/'CHECKPOINT_COMPLETE.json'),
        reused_optimizer_calls=4,qualification_weights_used_as_formal_parent=False)
