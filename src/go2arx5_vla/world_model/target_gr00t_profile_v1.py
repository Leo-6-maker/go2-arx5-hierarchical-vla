"""Independent target GR00T action-side learning identity, never FUSION/WM."""
import math,re

PROFILE = 'TARGET_GR00T_V1'
KIND = 'TARGET_GR00T_ACTION_SIDE_FT_V1'
PARENT = '94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38'
UPSTREAM = '51d4c89f72fda44cbf77285c6a8114b52676b8a1'
VARIANTS = ('FULL_ACTION_SIDE','INTERFACES_LAST6_DIT','INTERFACES_LAST2_DIT')
MODES = ('CAPACITY','FIT4','FORMAL')


def need(ok,reason):
    if not ok:raise PermissionError('TARGET_GR00T_PROFILE_' + reason)


def recipe(dataset_sha256, statistics_sha256, *, mode, variant, endpoint, microbatch=1,
           memory_policy='CUDA_RESIDENT'):
    need(all(type(x)is str and re.fullmatch('[0-9a-f]{64}',x)
             for x in (dataset_sha256,statistics_sha256)), 'EXACT_DATA_AND_PROCESSOR_PINS')
    need(mode in MODES and variant in VARIANTS and type(microbatch)is int
         and microbatch in (1,2,4,8), 'DECLARED_CAPACITY_VARIANT')
    need(type(endpoint)is int and endpoint in (
        (2,)if mode=='CAPACITY'else(200,500)if mode=='FIT4'else(2000,4000,8000)), 'FINITE_PREDECLARED_ENDPOINT')
    need(memory_policy in('CUDA_RESIDENT','CORE_LOSSLESS_CPU_ACTIVATIONS_2G_V1',
        'CORE_UNUSED_LOGITS1_CPU_ACTIVATIONS_2G_V2'),'DECLARED_MEMORY_EXECUTION')
    return dict(schema='WM02_TARGET_GR00T_ACTION_SIDE_RECIPE_V1',profile=PROFILE,kind=KIND,
        mode=mode,variant=variant,endpoint=endpoint,dataset_sha256=dataset_sha256,
        statistics_sha256=statistics_sha256,parent_manifest_sha256=PARENT,upstream_sha=UPSTREAM,
        trainable_scope=variant,visual_and_language_backbone_frozen=True,
        initial_optimizer='FRESH_FROM_ORIGINAL_CORR2048' if endpoint in (1,2,200,2000) else 'SAME_GENERATION_COMPLETE_STATE_REQUIRED',
        effective_batch=32,microbatch=microbatch,accumulation=32//microbatch,
        learning_rate=1e-4,weight_decay=1e-5,seed=20261108,
        scheduler='COSINE_FIXED_8000_WARMUP200'if mode=='FORMAL'else'INDEPENDENT_DIAGNOSTIC_COSINE500_WARMUP8',
        scheduler_horizon=8000 if mode=='FORMAL'else 500,warmup_updates=200 if mode=='FORMAL'else 8,
        save_every_updates=250,preserve_all_checkpoints=True,
        parameter_storage='PRESERVE_PINNED_PARENT_DTYPES_NO_GLOBAL_CAST',
        forward_precision='ORIGINAL_BF16_AUTOCAST',numeric_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',
        unused_category_storage='EXISTING_FROZEN_CATEGORY_TABLES_CPU',memory_policy=memory_policy,
        action_horizon=40,action_period_s=.05,physics_integrations=0,
        WM_updates=0,Actor_updates=0,external_adapter_updates=0,
        baseline_changed=False,WBC_changes=False,
        next_endpoint='REQUIRES_SEALED_TRAIN_DEV_AND_CORE_DECISION_NOT_AUTOMATIC_STEP_COUNT')


def validate(spec):
    p=spec.get('recipe')or{}
    expected=recipe(p.get('dataset_sha256'),p.get('statistics_sha256'),
        mode=p.get('mode'),variant=p.get('variant'),endpoint=p.get('endpoint'),microbatch=p.get('microbatch'),
        memory_policy=p.get('memory_policy','CUDA_RESIDENT'))
    start=spec.get('start_step')
    need(p==expected and spec.get('profile')==PROFILE and spec.get('kind')==KIND
         and spec.get('execution_site')in('VANDA_A40','CORELAB_4080S')
         and spec.get('physics')is False and spec.get('WBC_changes')is False
         and type(start)is int and 0<=start<p['endpoint']
         and spec.get('GR00T_updates')==p['endpoint']-start
         and spec.get('GR00T_updates',0)>0
         and spec.get('WM_updates')==spec.get('Actor_updates')==spec.get('TASK_ADAPTER_updates')==0
         and spec.get('parent_manifest_sha256')==PARENT
         and all(not spec.get(k)for k in ('fusion_binding','pilot1_binding','next1_binding','next2_binding','night1_binding')),
         'INDEPENDENT_NONZERO_INTERNAL_GR00T_NOT_ZERO_UPDATE_LEASE')
    if spec['execution_site']=='CORELAB_4080S':
        need(p['mode']in('CAPACITY','FIT4')and p['variant']=='INTERFACES_LAST2_DIT'
             and p['microbatch']==1,'CORE_FIRST_INTERNAL_LAST2_CAPACITY_AND_FIT4_ONLY')
    else:need(p['memory_policy']=='CUDA_RESIDENT','A40_ORIGINAL_RESIDENT_EXECUTION')
    if start:
        need(isinstance(spec.get('resume'),dict) and set(spec['resume'])=={'path','sha256'},
             'COMPLETE_SAME_GENERATION_RESUME')
    else:
        need(spec.get('resume')is None,'FRESH_OPTIMIZER_NO_OLD_SOURCE_RESUME')
        need(p['endpoint'] not in (500,4000,8000),'CONTINUATION_NOT_FRESH_RESTART')
    need(type(spec.get('deadline_epoch'))in(int,float)and math.isfinite(spec['deadline_epoch']),
         'FINITE_SITE_DEADLINE')
    return p


def whitelist(names, target_names, variant):
    """Exact internal GR00T parameter names; never an external adapter."""
    need(variant in VARIANTS,'KNOWN_ACTION_SCOPE')
    need(len(target_names)==14 and set(target_names)<=set(names),'SEVEN_ORIGINAL_TARGET_INTERFACES')
    blocks={n:int(m[1])for n in names if
            (m:=re.match(r'^action_head\.model\.transformer_blocks\.(\d+)\.',n))}
    need(set(blocks.values())==set(range(32)),'PINNED_32_DIT_BLOCK_ARCHITECTURE')
    if variant=='FULL_ACTION_SIDE':
        selected={n for n in names if n.startswith('action_head.')}
    else:
        count=6 if variant=='INTERFACES_LAST6_DIT'else 2
        selected=set(target_names)|{n for n,b in blocks.items()if b>=32-count}
    need(set(target_names)<=selected and selected and all(n.startswith('action_head.')for n in selected),
         'INTERNAL_ACTION_SIDE_AND_FROZEN_BACKBONE')
    return tuple(n for n in names if n in selected)
