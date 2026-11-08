"""Production endpoint slices after actual capacity, restore and FIT4 gates.

This is an adapter to the original single-A40 PBS owner, never a scheduler.
A slice can preserve a complete partial endpoint; it cannot call 2000 complete
or extend to 4000 merely because its allocation is ending.
"""
import json,math,time
from pathlib import Path
from . import target_gr00t_profile_v1 as profile
from .storage import digest

CORE_RESULT_WITNESS='WM02_TARGET_NATIVE_CLOSED_RESULT_WITNESS_V1'


def compact_core_result(row):
    """Read a closed original once on Core; retain its full hash, not its windows.

    The original large result stays at its producer. This read-only export
    does not qualify a model/task or create a lease. TRAIN diagnostics and zero
    integration failures remain labelled; the continuation consumer still
    requires the original DEV population and actual candidate execution.
    """
    spec,actual,worker,lease=(_pinned(row[k])for k in ('spec','result','worker','lease'))
    profile.need(lease.get('status')=='CLOSED'
        and lease.get('spec_sha256')==row['spec']['sha256']
        and lease.get('worker_exit_code')==worker.get('exit_code')
        and type(worker.get('exit_code'))is int
        and actual.get('source_sha')==spec.get('source_sha'),
        'CLOSED_ORIGINAL_CORE_RESULT_EXPORT_NOT_LIVE_OR_REWRITTEN')
    result={k:actual.get(k)for k in ('status','source_sha','phase','actual_target_integrations',
        'shutdown_clean','physics_stops')}
    reference=actual.get('reference_qualification')or{};trace=actual.get('trace_audit')or{}
    result.update(schema=CORE_RESULT_WITNESS,
        original_sha256={k:row[k]['sha256']for k in ('spec','result','worker','lease')},
        original_result_bytes=Path(row['result']['path']).stat().st_size,
        actual_data_role=spec.get('data_role'),actual_experiment_role=spec.get('experiment_role'),
        worker_exit_code=worker['exit_code'],failure=actual.get('failure'),
        task_condition_success=reference.get('task_condition_success'),
        task_score=reference.get('target_gr00t_task_score'),
        reference={k:reference.get(k)for k in ('schema','data_role','GR00T_queries',
            'GR00T_updates','expert_training_eligible','GR00T_native_interface_qualified',
            'full_task_success')},
        trace=dict(status=trace.get('status'),native_step_records=trace.get('native_step_records'),
            dropped=(trace.get('writer')or{}).get('dropped')),
        native_execution_authorized=False,model_qualification_granted=False)
    return result


def _pinned(descriptor):
    path=Path(descriptor['path'])
    profile.need(digest(path)==descriptor['sha256'],'PINNED_CONTINUATION_EVIDENCE')
    return json.loads(path.read_bytes())


def continuation_review(c,recipe):
    """Scientific decision from sealed probes/Core review, never site permission.

    Declare this rule before any formal training result: TRAIN H40 XYZ must
    improve over the previous500 probe; its other components and all fixed DEV
    components must not worsen. A separately pinned Core review must select
    continued learning rather than more data or execution diagnosis. Reusing
    this decision after an allocation boundary preserves the same generation.
    """
    need=profile.need;endpoint=recipe['endpoint'];previous={4000:2000,8000:4000}[endpoint]
    review=_pinned(c['continuation_review'])
    need(review.get('schema')=='WM02_TARGET_GR00T_CONTINUATION_REVIEW_V1'
        and review.get('generation_id')==c['science_generation_id']
        and review.get('from_endpoint')==previous and review.get('to_endpoint')==endpoint,
        'SAME_GENERATION_PREVIOUS_ENDPOINT_REVIEW')
    checkpoint=_pinned(review['checkpoint'])
    from .target_gr00t_training_v1 import semantic_recipe
    need(checkpoint.get('schema')=='WM02_TARGET_GR00T_FULL_STATE_DELTA_V1'
        and checkpoint.get('generation_id')==c['science_generation_id']
        and checkpoint.get('global_step')==previous and checkpoint.get('cursor')==32*previous
        and semantic_recipe(checkpoint['recipe'])==semantic_recipe(recipe),
        'SEALED_PREVIOUS_PRODUCTION_ENDPOINT')
    means={}
    for split in ('TRAIN','DEV'):
        pairs=[_pinned(review['probes'][split+'_'+when])['records']for when in ('before','after')]
        need(len(pairs[0])==len(pairs[1])==4,'FOUR_MATCHED_FIXED_CONTINUATION_PROBES')
        values=[{},{}]
        for index,rows in enumerate(pairs):
            step=previous-500 if index==0 else previous
            need(len({r['episode']for r in rows})==4,
                 'DISTINCT_FIXED_CONTINUATION_EPISODES')
            for row in rows:
                identity=row['model_identity']
                need(row['split']==split and row['model_global_step']==step
                    and identity['generation_id']==c['science_generation_id']
                    and identity['dataset_sha256']==c['dataset_sha256']
                    and identity['statistics_sha256']==c['statistics_sha256']
                    and identity['variant']==c['variant'] and identity['mode']=='FORMAL'
                    and identity['parent_manifest_sha256']==profile.PARENT
                    and (index==0 or identity.get('target_checkpoint_sha256')==review['checkpoint']['sha256']),
                    'CONTINUATION_PROBE_MODEL_DATA_AND_SPLIT')
                for horizon in ('H40','prefix8'):
                    for metric in ('XYZ_euclidean_m','SO3_rad','gripper_MAE'):
                        v=row['errors'][horizon][metric]
                        if isinstance(v,dict):v=v['mean']
                        need(type(v)in(int,float)and math.isfinite(v)and v>=0,
                             'FINITE_CONTINUATION_COMPONENTS')
                        values[index].setdefault(horizon+'/'+metric,[]).append(v)
        for before,after in zip(*pairs):
            need(all(before[k]==after[k]for k in ('episode','split','anchor','seed','noise','command')),
                 'SAME_CONTINUATION_INPUTS_NOISE_AND_COMMANDS')
        means[split]=[{k:math.fsum(x/len(v)for x in v)for k,v in side.items()}for side in values]
        before,after=means[split]
        need(all(after[k]<=before[k]for k in before)
            and (split!='TRAIN'or after['H40/XYZ_euclidean_m']<before['H40/XYZ_euclidean_m']),
            'TRAIN_IMPROVING_DEV_NOT_WORSE_NO_COMPONENT_TRADEOFF')
    core=_pinned(review['Core_DEV_review'])
    need(core.get('generation_id')==c['science_generation_id']
        and core.get('candidate_global_step')==previous
        and core.get('candidate_checkpoint_sha256')==review['checkpoint']['sha256']
        and core.get('dataset_sha256')==c['dataset_sha256']
        and core.get('statistics_sha256')==c['statistics_sha256']
        and type(core.get('planned_conditions'))is int and core['planned_conditions']>0
        and core.get('terminal_conditions')==core['planned_conditions']
        and core.get('all_worker_leases_closed')is True
        and core.get('TEST_opened')is False and core.get('WBC_changes')is False
        and core.get('decision')=='CONTINUE_SAME_DATA',
        'SEALED_CORE_DEV_DECISION_NOT_STEP_COUNT_OR_OPEN_LOOP_ALONE')
    package=_pinned(core['candidate_package'])
    need(package.get('generation_id')==c['science_generation_id']
        and package.get('global_step')==previous
        and package.get('checkpoint_sha256')==review['checkpoint']['sha256'],
        'SAME_REVIEWED_CORE_CANDIDATE_PACKAGE')
    witnesses=core['native_evidence'];names=set();arms=[];executed_candidate=False
    need(len(witnesses)==core['planned_conditions'],'EVERY_DECLARED_CORE_CONDITION_HAS_EVIDENCE')
    panel=_pinned(core['preregistration']);assignments=panel['assignments']
    need(panel.get('B1_global_step')==previous
        and panel.get('planned_native_conditions')==len(assignments)==len(witnesses),
        'ORIGINAL_PREDECLARED_CORE_CONDITION_POPULATION')
    expected={r['native_run']:r for r in assignments}
    need(len(expected)==len(assignments),'DISTINCT_PREDECLARED_CORE_CONDITIONS')
    for row in witnesses:
        spec,actual,worker,lease=(_pinned(row[k])for k in ('spec','result','worker','lease'))
        if actual.get('schema')==CORE_RESULT_WITNESS:
            original=actual.get('original_sha256')or{}
            need(all(original.get(k)==row[k]['sha256']for k in ('spec','worker','lease'))
                and isinstance(original.get('result'),str)and len(original['result'])==64
                and actual.get('actual_data_role')==spec.get('data_role')
                and actual.get('worker_exit_code')==worker.get('exit_code')
                and type(actual.get('original_result_bytes'))is int and actual['original_result_bytes']>0,
                'COMPACT_CORE_WITNESS_BOUND_TO_ORIGINAL_CLOSED_WORKER')
        policy=spec.get('target_gr00t_policy')or{};arm=policy.get('arm');arms.append(arm)
        need(spec.get('data_role')=='DEV'and arm in('B0','B1')
            and spec.get('phase')=='collect-target-gr00t-prefix-loop-v1'
            and isinstance(spec.get('registered_root'),str)and bool(spec['registered_root'])
            and spec['registered_root']not in names
            and actual.get('source_sha')==spec.get('source_sha')
            and type(actual.get('actual_target_integrations'))is int
            and actual['actual_target_integrations']>0 and actual.get('shutdown_clean')is True
            and actual.get('status')in('NATIVE_STAGE_PASSED','NATIVE_STAGE_FAILED_PRESERVED')
            and lease.get('status')=='CLOSED'and lease.get('spec_sha256')==row['spec']['sha256']
            and lease.get('worker_exit_code')==worker.get('exit_code')
            and type(worker.get('exit_code'))is int,
            'ACTUAL_DISTINCT_CLOSED_CORE_DEV_WORKER_NOT_PLAN_OR_TRAIN_DIAGNOSTIC')
        names.add(spec['registered_root'])
        condition=expected.get(Path(spec['registered_root']).name)
        need(condition is not None and condition['arm']==arm
            and condition['split']=='DEV'and condition['parent']==spec.get('data_parent')
            and condition['command']==spec.get('task_command')
            and condition['seeds']==spec.get('target_gr00t_query_seeds'),
            'EXACT_PREDECLARED_CORE_LAYOUT_COMMAND_ARM_AND_NOISE')
        if arm=='B1':
            need((policy.get('package')or{}).get('sha256')==core['candidate_package']['sha256'],
                 'EVERY_CORE_B1_USES_REVIEWED_CANDIDATE')
            executed_candidate|=actual['actual_target_integrations']>34303
    need(arms.count('B0')==arms.count('B1') and executed_candidate,
         'BOTH_CORE_ARMS_AND_ACTUAL_CANDIDATE_ACTION_REQUIRED')
    return dict(previous_endpoint=previous,next_endpoint=endpoint,means=means,
        Core_decision=core['decision'],site_admission='STILL_REQUIRED_BY_ORIGINAL_OWNER')


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
    need(type(endpoint)is int and endpoint in (2000,4000,8000)
         and type(start)is int and 0<=start<endpoint
         and isinstance(generation,str)and bool(generation),'FINITE_FORMAL_ENDPOINT_AND_SCIENCE_IDENTITY')
    recipe=profile.recipe(c['dataset_sha256'],c['statistics_sha256'],mode='FORMAL',
        variant=c['variant'],endpoint=endpoint,microbatch=c.get('microbatch',1))
    # The currently executable capacity/FIT4 chain measures microbatch1 only.
    # Larger execution batches need their own actual capacity/restore evidence.
    need(recipe['microbatch']==1,'SAME_ACTUALLY_QUALIFIED_MICROBATCH1_EXECUTION')
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
    if endpoint>2000:
        need(start>={4000:2000,8000:4000}[endpoint]and c.get('resume')is not None
             and isinstance(c.get('continuation_review'),dict),
             'CONTINUATION_REQUIRES_PREVIOUS_ENDPOINT_COMPLETE_STATE_AND_REVIEW')
        continuation_review(c,recipe)
    return recipe


def execute(c,root,need,deadline,gpu,child,children,start,stop_requested):
    data=Path(c['bindings']['dataset']);manifest=json.loads((data/'DATASET_MANIFEST.json').read_bytes())
    recipe=validate_request(c,manifest)
    endpoint=recipe['endpoint']
    actual=child('formal-to-'+str(endpoint),'FORMAL',c['variant'],endpoint,start_step=c['start_step'],
        resume=c.get('resume'),generation=c['science_generation_id'],microbatch=recipe['microbatch'])
    need(actual['exit_code']==0,'FORMAL_ENGINEERING_FAILURE_PRESERVED')
    return dict(status='FORMAL_ENDPOINT_SEALED'if actual['endpoint_complete']else'FORMAL_PARTIAL_COMPLETE_CHECKPOINT_SEALED',
        science_generation_id=c['science_generation_id'],target_endpoint=endpoint,global_step=actual['global_step'],
        endpoint_complete=actual['endpoint_complete'],actual_GR00T_updates=actual['GR00T_updates'],
        checkpoint=actual['checkpoint'],inference_package=actual.get('inference_package'),
        children=children,wall_s=time.monotonic()-start,
        native_physics=0,WM_updates=0,Actor_updates=0,baseline_changed=False,
        next=(('EXPORT_AND_CORE_B0_B1_DEV_BEFORE_4000_DECISION'if endpoint==2000 else
               'EXPORT_AND_CORE_DEV_BEFORE_8000_DECISION'if endpoint==4000 else
               'FREEZE_CANDIDATE_AND_DECLARE_FINAL_TEST_BEFORE_EXECUTION')if actual['endpoint_complete']
              else'RECONCILE_PBS_TERMINAL_THEN_SAME_GENERATION_COMPLETE_STATE_RESUME'),
        formal4000_or8000_started=endpoint>2000,Core_DEV='NOT_RUN')
