"""CPU-only binding of actual terminal state, never a new scientific window."""
import argparse
import json
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from .scope import validate,Ledger,STATS
from .remaining import panel
from .train import checked_resume


def verify_checkpoint(path,step,recipe):
    path=Path(path);complete=read(path/'CHECKPOINT_COMPLETE.json')
    if complete['global_step']!=step or not complete['full_resume'] or sha(path/'statistics.json')!=STATS:
        raise PermissionError('E1_REMAINDER_NATIVE_COMPLETE_STEP_STATS')
    required={'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','SAMPLER_STATE.json','statistics.json','config.json'}
    if not required.issubset(complete['files']) or not any(n.endswith('.safetensors') for n in complete['files']):
        raise PermissionError('E1_COMPLETE_NATIVE_MODEL_OPTIMIZER_RNG_FILES_REQUIRED')
    for name,digest in complete['files'].items():
        p=path/name
        if not p.resolve().is_relative_to(path.resolve()) or p.is_symlink() or sha(p)!=digest:raise ValueError('E1_REMAINDER_NATIVE_MEMBER_CHANGED:'+name)
    saved=read(path/'SAMPLER_STATE.json');trainer=read(path/'trainer_state.json')
    if saved['recipe']!=recipe or saved['sampler']['cursor']!=step*32 or trainer['global_step']!=step:
        raise PermissionError('E1_REMAINDER_EXACT_RECIPE_AND_CONSUMED_CURSOR')
    return sha(path/'CHECKPOINT_COMPLETE.json')


def build(root,core_state):
    root=Path(root);runtime=validate(root,'hopper-neural');ledger=Ledger(root,'hopper');ledger.guard();started=time.process_time()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('E1_REMAINDER_BINDING_CPU_ONLY')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});l=read(ledger.path)
    if l['GPU_reserved_s'] or len(l['PBS_submissions'])!=2 or any(x['status']!='NATIVE_TERMINAL' for x in l['PBS_submissions']):raise PermissionError('E1_PLAN_ONLY_AFTER_TWO_NATIVE_SETTLEMENTS')
    if not 1024<=l['optimizer_updates']<=2048 or l['optimizer_updates']+l['pending_updates']>2048:
        raise PermissionError('E1_REMAINDER_M1_COMPLETE_AND_TOTAL_UPDATE_CAP')
    core=read(core_state)
    if core['run_uuid']!=root.name or core['active_workers'] or core['physical_reservations'] or core['initial_history_hold']:
        raise PermissionError('E1_CORE_TERMINAL_HISTORY_AND_PHYSICS_REQUIRED')
    history=read(root/'evidence/ORACLE_HISTORY_ADJUDICATION.json')
    if history['status']!='C0_ALL_USED_CORRECTION_ANCHORS_EXACT_ORIGINAL_STATIC_HISTORY':raise PermissionError('E1_ORIGINAL_QUALIFIED_CORRECTION_HISTORY_REQUIRED')
    source_stop=bool(core['source_physics_stop']) or any(r['status']=='ACTUAL_SAFETY_FAIL' or r['live_native_failures'] or r['failure_stage']=='S1_INTEGRITY' for r in core['rows'])
    output=root/'plans/REMAINDER_FIXED_SCOPE.json'
    if output.exists():raise FileExistsError('PRESERVE_FIRST_FIXED_REMAINDER_BINDING')
    try:
        final=root/'plans/FINAL_CORRECTION_BINDING.private.json';binding=root/'plans/CORRECTION_BINDING.private.json'
        if not final.exists():
            if l['optimizer_updates']!=1024 or l['pending_updates']:raise PermissionError('E1_MISSING_CONSUMED_DATA_CHOICE_AFTER_M2_UPDATE_NO_GUESS')
            c0=read(root/'plans/C0_BASELINE_BINDING.private.json')
            if c0['status']!='PASS_ACTUAL_HOPPER_CORRECTION_PROCESSOR_LOADER' or c0['manifest_sha256']!=history['correction_manifest_sha256']:raise PermissionError('E1_VALID_EXISTING_C0_CONSUMPTION_REQUIRED')
            c0.update(final_dataset_choice='C0',choice_reason='PRIOR_ALLOCATION_ENDED_BEFORE_DATA_BINDING_ALREADY_QUALIFIED_C0_NO_OUTCOME_SELECTION',source_physics_stopped=source_stop)
            write(final,c0);final.chmod(0o400)
        data=read(final)
        if data['status']!='PASS_ACTUAL_HOPPER_CORRECTION_PROCESSOR_LOADER' or sha(Path(data['dataset'])/'DATASET_MANIFEST.json')!=data['manifest_sha256']:
            raise PermissionError('E1_EXACT_ALREADY_CONSUMED_DATA_NO_LATE_HOT_UPDATE')
        if binding.exists():
            if read(binding)!=data:raise PermissionError('E1_CORRECTION_DATA_CHOICE_DIVERGED')
        else:write(binding,data);binding.chmod(0o400)
        recipes=read(root/'plans/M2_RECIPE_RESOLVED.json');parent=root/'training/M1/checkpoint-1024'
        parent_digest=verify_checkpoint(parent,1024,read(root/'plans/M1_RECIPE_RESOLVED.json'))
        specs=[];previous_training_viable=True
        for base,branch,offset in [('M1-1024-CANONICAL','M1',0),('M2-BC','M2-BC',1024),('M2-CORR','M2-CORR',1536)]:
            remaining=panel(core['rows'],base,source_stopped=source_stop)
            spec=dict(base_label=base,label=base+'-REMAINDER-03',branch=branch,remaining_conditions=remaining['remaining'],
                prior_valid_conditions_preserved=remaining['complete'],new_engineering_recoveries=remaining['new_engineering_recoveries'])
            full=root/'training'/branch/('checkpoint-1024' if branch=='M1' else 'checkpoint-512')
            if branch=='M1':spec.update(training_action='COMPLETE',fixed_checkpoint_manifest_sha256=parent_digest)
            elif source_stop and branch=='M2-CORR':spec.update(training_action='BLOCKED',blocked_reason='SOURCE_SAFETY_STOP_CORRECTION_NOT_ADMITTED')
            elif (full/'CHECKPOINT_COMPLETE.json').exists() and l['optimizer_updates']>=offset+512:
                spec.update(training_action='COMPLETE',fixed_checkpoint_manifest_sha256=verify_checkpoint(full,512,recipes[branch]))
            elif l['pending_updates']:
                spec.update(training_action='BLOCKED',blocked_reason='UNKNOWN_OPTIMIZER_INTENT_NOT_REPEATED')
            elif (l['optimizer_updates']==offset or (l['optimizer_updates']<offset and previous_training_viable)) and not (root/'training'/branch/'session-start').exists():
                spec.update(training_action='NEW',training_checkpoint_relative=str(parent.relative_to(root)),training_checkpoint_manifest_sha256=parent_digest)
            elif offset<l['optimizer_updates']<offset+512:
                step=l['optimizer_updates']-offset;partial=root/'training'/branch/('checkpoint-'+str(step))
                try:
                    checked_resume(root,branch,partial,recipes[branch],l)
                    spec.update(training_action='RESUME',resume_step=step,training_checkpoint_relative=str(partial.relative_to(root)),training_checkpoint_manifest_sha256=sha(partial/'CHECKPOINT_COMPLETE.json'))
                except (OSError,ValueError,PermissionError,KeyError) as exc:spec.update(training_action='BLOCKED',blocked_reason='ACTUAL_PARTIAL_STATE_NOT_EXACTLY_RESUMABLE:'+type(exc).__name__)
            else:spec.update(training_action='BLOCKED',blocked_reason='NO_MATCHING_COMPLETE_NATIVE_STATE_OR_PRIOR_BRANCH_GAP')
            if branch=='M2-BC':previous_training_viable=spec['training_action'] in ['COMPLETE','NEW','RESUME']
            specs.append(spec)
        result=dict(schema='E1_FIXED_REMAINDER_SCOPE_V1',status='BOUND_ACTUAL_TERMINAL_REMAINDER_NOT_SUBMITTED',run_uuid=root.name,
            authority=runtime['authority']['protocol_ref'],source_sha=runtime['code']['vla_commit'],new_resource_window=False,
            optimizer_updates_at_binding=l['optimizer_updates'],pending_updates_at_binding=l['pending_updates'],
            fixed_M1_parent_manifest_sha256=parent_digest,source_physics_stopped=source_stop,
            core_state_sha256=sha(core_state),correction_manifest_sha256=data['manifest_sha256'],
            new_engineering_recoveries=sum(s['new_engineering_recoveries'] for s in specs),models=specs,
            valid_policy_outcomes_repeated=False,seed_or_recipe_changed=False,selected='NONE',deployment='NONE')
        write(output,result);output.chmod(0o400);return result
    finally:ledger.cpu(time.process_time()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--core-state',required=True);a=p.parse_args();print(json.dumps(build(a.root,a.core_state)),flush=True)
