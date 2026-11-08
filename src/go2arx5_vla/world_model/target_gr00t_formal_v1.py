"""Production endpoint slices after actual capacity, restore and FIT4 gates.

This is an adapter to the original single-A40 PBS owner, never a scheduler.
A slice can preserve a complete partial endpoint; it cannot call 2000 complete
or extend to 4000 merely because its allocation is ending.
"""
import json,time
from pathlib import Path
from . import target_gr00t_profile_v1 as profile
from .storage import digest


def validate_request(c,manifest):
    need=profile.need;train=[r for r in manifest['episodes']if r['split']=='TRAIN'];dev=[r for r in manifest['episodes']if r['split']=='DEV']
    from .target_gr00t_collection_v1 import COMMANDS
    need(c.get('allocation_mode')=='FORMAL_SLICE'and manifest.get('bundle_kind')=='FORMAL'
         and len(train)>=16 and len({r['parent']for r in train})>=4
         and {r['command']for r in train}==set(COMMANDS)
         and {r['command']for r in dev}==set(COMMANDS)
         and not ({r['parent']for r in train}&{r['parent']for r in dev})
         and manifest.get('TRAIN_only_statistics')is True and manifest.get('TEST_results_read')is False
         and manifest.get('WM_or_HOLD_or_diagnostic_relabelled')is False,
         'FORMAL_ACTUAL_DIVERSE_TARGET_TRAIN_AND_INDEPENDENT_DEV')
    endpoint=c.get('formal_target_endpoint');start=c.get('start_step');generation=c.get('science_generation_id')
    need(endpoint==2000 and type(start)is int and 0<=start<endpoint
         and isinstance(generation,str)and bool(generation),'FIRST_FORMAL_2000_FINITE_SCIENCE_IDENTITY')
    recipe=profile.recipe(c['dataset_sha256'],c['statistics_sha256'],mode='FORMAL',
        variant=c['variant'],endpoint=endpoint,microbatch=c.get('microbatch',1))
    q=c['qualification'];path=Path(q['path'])
    need(digest(path)==q['sha256'],'ACTUAL_FIT4_QUALIFICATION_RECEIPT')
    result=json.loads(path.read_bytes())
    need(result.get('status')=='FIT4_CHAIN_SEALED'and result.get('fit4_learnability_passed')is True
         and result.get('selected_variant')==c['variant']
         and result.get('actual_GR00T_updates',0)>=203,'ACTUAL_FIT4_LEARNABILITY_NOT_STEP_COUNT')
    qualified_config=json.loads((path.parent/'ALLOCATION_CONFIG.json').read_bytes())
    need(c['parent']['files']==qualified_config['parent']['files'],'SAME_QUALIFIED_PARENT_BYTES_FOR_FORMAL')
    restore=path.parent/'NEXT_UPDATE_RESTORE_ACTUAL.json';r=json.loads(restore.read_bytes())
    need(r.get('status')=='PASS_ACTUAL_UNINTERRUPTED_VERSUS_RESUMED_NEXT_UPDATE'
         and r.get('next_update_parameters_optimizer_scheduler_RNG_sampler_exact')is True,
         'ACTUAL_CAPACITY_NEXT_UPDATE_RESTORE_REQUIRED')
    need(c['parent_manifest_sha256']==profile.PARENT,'ORIGINAL_CORR_PARENT_FORMAL_FRESH_OPTIMIZER')
    if start:
        resume=c.get('resume')or{};cp=Path(resume.get('path',''))
        need(digest(cp/'CHECKPOINT_COMPLETE.json')==resume.get('sha256'),'EXACT_PRODUCTION_RESUME_HEADER')
        saved=json.loads((cp/'CHECKPOINT_COMPLETE.json').read_bytes())
        need(saved.get('schema')=='WM02_TARGET_GR00T_FULL_STATE_DELTA_V1'
             and saved.get('generation_id')==generation and saved.get('global_step')==start
             and saved.get('cursor')==32*start and saved.get('recipe',{}).get('mode')=='FORMAL'
             and saved['recipe']['dataset_sha256']==c['dataset_sha256']
             and saved['recipe']['statistics_sha256']==c['statistics_sha256']
             and saved['recipe']['variant']==c['variant'],'EXACT_SAME_FORMAL_GENERATION_NOT_FIT4_WEIGHTS')
    else:need(c.get('resume')is None,'NEW_FORMAL_NOT_DIAGNOSTIC_WEIGHT_CONTINUATION')
    return recipe


def execute(c,root,need,deadline,gpu,child,children,start,stop_requested):
    data=Path(c['bindings']['dataset']);manifest=json.loads((data/'DATASET_MANIFEST.json').read_bytes())
    recipe=validate_request(c,manifest)
    actual=child('formal-to-2000','FORMAL',c['variant'],2000,start_step=c['start_step'],
        resume=c.get('resume'),generation=c['science_generation_id'],microbatch=recipe['microbatch'])
    need(actual['exit_code']==0,'FORMAL_ENGINEERING_FAILURE_PRESERVED')
    return dict(status='FORMAL_ENDPOINT_SEALED'if actual['endpoint_complete']else'FORMAL_PARTIAL_COMPLETE_CHECKPOINT_SEALED',
        science_generation_id=c['science_generation_id'],target_endpoint=2000,global_step=actual['global_step'],
        endpoint_complete=actual['endpoint_complete'],actual_GR00T_updates=actual['GR00T_updates'],
        checkpoint=actual['checkpoint'],inference_package=actual.get('inference_package'),
        children=children,wall_s=time.monotonic()-start,
        native_physics=0,WM_updates=0,Actor_updates=0,baseline_changed=False,
        next=('EXPORT_AND_CORE_B0_B1_DEV_BEFORE_4000_DECISION'if actual['endpoint_complete']
              else'RECONCILE_PBS_TERMINAL_THEN_SAME_GENERATION_COMPLETE_STATE_RESUME'),
        formal4000_or8000_started=False,Core_DEV='NOT_RUN')
