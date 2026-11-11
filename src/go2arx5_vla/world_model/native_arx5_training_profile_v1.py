"""Official-parent native camera transfer: independent 500/25 training recipe.

No legacy FIT4, CORR2048 or FULL_ACTION_SIDE identity is admitted here. This
module defines software contracts; only actual runtime qualifications admit a
candidate update. Historical schedulers, readers and parameter gates are intact.
"""
from __future__ import annotations
import hashlib
import math

SCOPE = 'NATIVE_CAMERA_PUBLIC_ARX5_TRANSFER_V1'
PARENT_ID = '0e493187af087d156f6352d46ce14e049ff18e7c1cb1eabc8671357ab4b96e71'
REVISION = '2fc962b973bccdd5d8ce4f67cc63b264d6886495'
UPSTREAM = '51d4c89f72fda44cbf77285c6a8114b52676b8a1'
LR_SHA256 = '0d044913097f94796624a672d1aa3f5de17804fad80376e9dd05635776ebc31a'
STAGES = ('A_TARGET','B_SOURCE','B_TARGET')


def lr_multiplier(u):
    if type(u) is not int or not 0 <= u <= 500:
        raise ValueError('EXACT_500_UPDATE_SCHEDULE_POSITION')
    if u == 500:
        return 0.
    return (u+1)/25 if u < 25 else .5*(1+math.cos(math.pi*(u-25)/475))


def learning_rates():
    values = [1e-4*lr_multiplier(u) for u in range(500)]
    raw = (''.join(v.hex()+'\n' for v in values)).encode('ascii')
    if hashlib.sha256(raw).hexdigest() != LR_SHA256:
        raise ValueError('PRE_REGISTERED_500_USED_LR_SEQUENCE_CHANGED')
    return values


def target_visit_indices(manifest):
    """Five shuffled cycles of the existing equal-episode 3200 exposure list."""
    import numpy as np
    episodes = manifest['episodes']
    if len(episodes) != 100 or any(e['split'] != 'TRAIN' for e in episodes):
        raise ValueError('MATCHED_TRAIN100_EXPOSURE_REQUIRED')
    unique = [(e['id'],i) for e in episodes for i in e['supervised_anchors']]
    if len(unique) != len(set(unique)) or len(unique) != 1450:
        raise ValueError('EXACT_1450_UNIQUE_WINDOWS')
    lookup = {key:i for i,key in enumerate(unique)}
    visits = []
    for ep in episodes:
        if len(ep['optimization_anchors']) != 32:
            raise ValueError('ORIGINAL_32_VISITS_PER_EPISODE')
        visits += [lookup[ep['id'],a] for a in ep['optimization_anchors']]
    if len(visits) != 3200 or set(visits) != set(range(1450)):
        raise ValueError('ORIGINAL_COMPLETE_EXPOSURE_LIST')
    return np.asarray(visits, dtype=np.int64)[np.concatenate([
        np.random.Generator(np.random.PCG64(20261111+cycle)).permutation(3200) for cycle in range(5)])]


def source_visit_indices(windows):
    """Uniform epoch permutations AFTER the source task/episode window selection."""
    import numpy as np
    if type(windows) is not int or not 1 <= windows <= 3200:
        raise ValueError('ACTUAL_FINITE_UNIQUE_SOURCE_TRAIN_WINDOWS')
    count = math.ceil(16000/windows)
    return np.concatenate([np.random.Generator(np.random.PCG64(20261111+cycle)).permutation(windows)
                           for cycle in range(count)])[:16000].astype(np.int64)


def recipe(stage, dataset_sha256, statistics_sha256):
    if stage not in STAGES or any(len(x)!=64 or any(c not in '0123456789abcdef' for c in x)
                                 for x in (dataset_sha256,statistics_sha256)):
        raise ValueError('EXPLICIT_NEW_STAGE_AND_DATA_IDENTITIES')
    learning_rates()
    return dict(schema='NATIVE_ARX5_OFFICIAL_PARENT_STAGE_RECIPE_V1', scope=SCOPE,
        stage=stage, parent_files_identity=PARENT_ID, official_revision=REVISION, upstream=UPSTREAM,
        dataset_sha256=dataset_sha256, statistics_sha256=statistics_sha256,
        trainable_scope='INTERFACES_LAST2_DIT', target_category=10,
        category10_meaning='PROJECT_ADAPTATION_SLOT_NOT_OFFICIAL_ARX5_ID',
        initial_weights='B_SOURCE500_WHITELIST_ONLY' if stage=='B_TARGET' else 'OFFICIAL_RAW_PARENT',
        fresh_optimizer_scheduler_RNG_cursor=True, seed=20261111,
        RNG_namespace='B_SOURCE' if stage=='B_SOURCE' else 'MATCHED_TARGET',
        optimizer='AdamW',learning_rate=1e-4,weight_decay=1e-5,betas=[.9,.999],eps=1e-8,
        decay_grouping='ORIGINAL_TRAINER',max_grad_norm=1.,effective_batch=32,microbatch=1,accumulation=32,
        endpoint=500,sample_visits=16000,warmup_updates=25,used_lr_sha256=LR_SHA256,
        scheduler='COSINE_500_WARMUP25_FINAL_NEXT_LR_ZERO',
        checkpoints=[250,500],primary_endpoint=500,
        loss='ORIGINAL_MASKED_FM_ONLY',auxiliary_loss=False,
        forward_precision='ORIGINAL_BF16_AUTOCAST_FP32_TRAINABLE_PARAMETERS',
        numeric_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',
        CUDA_trainable_parameters_gradients_Adam_moments_activations=True,
        CPU_training_activation_offload=False,source_replay=False,contact_event_resampling=False,
        image_augmentation=False,original_DEV_TEST_used=False,WBC_changes=False,physics_dt_s=.002)


def parameter_contract(model):
    """Reuse validated target-category dispatch and the pure last2 selector."""
    from scripts.pi0294.generalization.minimal_adapter import ParameterContract,LEAVES,TargetCategoryLinear
    from .target_gr00t_profile_v1 import whitelist
    base = ParameterContract(model,'E1_ADAPTER_ONLY',10)
    names = whitelist(tuple(dict(model.named_parameters())),base.trainable_names,'INTERFACES_LAST2_DIT')
    for name,p in model.named_parameters():
        p.requires_grad_(name in names)
        if name in names and str(p.dtype) != 'torch.float32':
            raise ValueError('NEW_CORE_FP32_TRAINABLE_PARAMETER_QUALIFICATION_REQUIRED_'+name)
    for path in LEAVES:
        model.get_submodule(path).retain_frozen_tables_on_cpu()
    class Contract:
        trainable_names = names
        target = 10
        hooks = base.hooks
        def __init__(self):
            self.model = model
        def validate(self):
            if tuple(n for n,p in model.named_parameters() if p.requires_grad) != names:
                raise ValueError('NEW_OFFICIAL_PARENT_WHITELIST_CHANGED')
            if any(not isinstance(model.get_submodule(p),TargetCategoryLinear) or
                   model.get_submodule(p).target != 10 for p in LEAVES):
                raise ValueError('NEW_OFFICIAL_TARGET_CATEGORY_DISPATCH_CHANGED')
            if getattr(self,'_official_inference_consumed',False):
                raise ValueError('CONSUMED_ORDINARY_MODEL_CANNOT_TRAIN')
            if any(str(model.get_parameter(n).dtype) != 'torch.float32' for n in names):
                raise ValueError('CORE_TRAINABLE_PARAMETER_PRECISION_CHANGED')
        def parameters(self):
            self.validate()
            return [model.get_parameter(n) for n in names]
        def describe(self):
            self.validate()
            rows=[dict(name=n,shape=list(model.get_parameter(n).shape),numel=model.get_parameter(n).numel(),
                       dtype=str(model.get_parameter(n).dtype)) for n in names]
            return dict(schema='NATIVE_ARX5_OFFICIAL_PARENT_PARAMETER_CONTRACT_V1',parent_files_identity=PARENT_ID,
                trainable_scope='INTERFACES_LAST2_DIT',target_category=10,trainable=rows,
                actual_trainable_numel=sum(r['numel'] for r in rows),legacy_77847684_is_admission=False,
                frozen_visual_language_vlln_vl_attention_and_first30_DiT=True)
    return Contract()
