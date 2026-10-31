"""Matched maintenance arms on one admitted snapshot; not original E2.

Keep the existing native Trainer, loss, sampler, save and reload path. Only
the declared data version, finite endpoint and actual DiT whitelist differ.
"""
import re

KIND='V7_BOOK_M32_FINITE_LEARNING'
SCHEMA='BOOK_M32_MATCHED_DATA_LIMITED_ACTION_V1'
INITIAL='b248513ae76f0bb1ec29978a7c7242eabb5155063d8e732f61e182035a676b8c'


def plan(dataset_sha256,statistics_sha256,population_sha256,*,blocks,steps):
    from .book_fit import recovery_plan,full_plan
    if blocks not in(0,1,2)or type(blocks)is not int or type(steps)is not int or steps not in(10,512):
        raise ValueError('ONLY_DECLARED_A_B_AND_TEN_UPDATE_CAPACITY_OR_FIXED512')
    if blocks==0 and steps!=512:raise ValueError('CAPACITY_PROBE_IS_FOR_NEW_LIMITED_ACTION_ARM')
    for pin in(dataset_sha256,statistics_sha256,population_sha256):
        if type(pin)is not str or re.fullmatch('[0-9a-f]{64}',pin)is None:raise ValueError('EXACT_NATIVE_ADMITTED_DATA_PINS')
    p=recovery_plan();original=full_plan()
    p.update(schema=SCHEMA,dataset_manifest_sha256=dataset_sha256,statistics_sha256=statistics_sha256,
        TRAIN_population_sha256=population_sha256,initial_delta_checkpoint_manifest_sha256=INITIAL,
        parent_restore_dataset_manifest_sha256=original['dataset_manifest_sha256'],
        parent_restore_statistics_sha256=original['statistics_sha256'],
        max_steps=steps,TRAIN_visits=steps*32,TRAIN_episodes=21,TRAIN_windows=12533,
        pure_full_teacher_episodes=15,mixed_recovery_episodes=6,limited_DiT_blocks=blocks,
        trainable='EXACT_TARGET_INTERFACE_PLUS_DECLARED_LAST_DIT_BLOCKS',shared_DiT_frozen=blocks==0,
        capacity_qualification_only=steps==10,comparison_endpoint=512,
        optimization_population='DECLARED_TRAIN15_PLUS_SIX_ACTUAL_EXPERT_SUFFIXES_12533_WINDOWS',
        sampling='EXISTING_PCG64_CYCLIC_ALL_EXPERT_H40_WINDOWS_AB_IDENTICAL',
        recovery_fraction_per_complete_cycle=3518/12533,
        diagnostic_windows=[dict(id='BOOK_SOURCE_SEED000',anchor=a,split='TRAIN')for a in(0,48,80,160,240,320,400,600)]
            +[dict(id='BOOK_M32_RECOVERY000_EXPERT_SUFFIX',anchor=a,split='TRAIN')for a in(0,8,16,32,48,80,160,592)],
        warmup_updates=16,scheduler_horizon_steps=512,seed=2026103105,
        optimizer_initialization='FRESH_ADAMW_FROM_CLOSED_RECOVERY512_SAME_FOR_A_AND_B',
        parent_data_or_original_E1_E2_changed=False,gradient_checkpointing=False,
        purpose='One matched improved-data comparison; ten-update limited-action memory qualification is not a final candidate or original E2.')
    return p


def is_plan(p):
    if p.get('schema')=='BOOK_M32_FIXED_DATA_LAST2_CONTINUOUS4096_V1':
        try:return p==curve_plan(p['dataset_manifest_sha256'],p['statistics_sha256'],p['TRAIN_population_sha256'])
        except(KeyError,TypeError,ValueError):return False
    if p.get('schema')!=SCHEMA:return False
    try:return p==plan(p['dataset_manifest_sha256'],p['statistics_sha256'],p['TRAIN_population_sha256'],
                       blocks=p['limited_DiT_blocks'],steps=p['max_steps'])
    except(KeyError,TypeError,ValueError):return False


def curve_plan(dataset_sha256,statistics_sha256,population_sha256):
    """One new continuous schedule, not continuation of exhausted short cosine."""
    p=plan(dataset_sha256,statistics_sha256,population_sha256,blocks=2,steps=512)
    p.update(schema='BOOK_M32_FIXED_DATA_LAST2_CONTINUOUS4096_V1',max_steps=4096,
        TRAIN_visits=131072,scheduler_horizon_steps=4096,warmup_updates=128,
        checkpoint_steps=[1024,2048,4096],comparison_endpoint=4096,
        generation_queries=64,native_diagnostic_loss_forwards=64,
        optimizer_initialization='FRESH_ADAMW_FROM_SAME_CLOSED_RECOVERY512_CONTINUOUS4096_NOT_B512_RESUME',
        sufficient_training_or_convergence_claim=False,
        purpose='Fixed TRAIN/data/statistics/last2 DiT/uniform sampler learning curve. Fresh closed initialb248 and one4096 cosine; full1024/2048/4096 states plus matched TRAIN diagnostics. Schedule differs from short512; not an update-count-only causal effect.')
    return p


def block_parameter_names(names,count):
    """Actual pinned N1.7 has 32 blocks; indices are numeric, never lexical."""
    if type(count)is not int or count not in(1,2):raise ValueError('LAST_ONE_OR_TWO_ONLY')
    matches={n:int(m[1])for n in names if(m:=re.match(r'^action_head\.model\.transformer_blocks\.(\d+)\.',n))}
    if set(matches.values())!=set(range(32)):raise ValueError('ACTUAL_PINNED_32_DIT_BLOCKS_REQUIRED')
    selected=tuple(n for n in names if n in matches and matches[n]>=32-count)
    if not selected:raise ValueError('ACTUAL_LAST_BLOCK_PARAMETERS_REQUIRED')
    return selected


class LimitedContract:
    def __init__(self,base,count):
        base.validate()
        if base.arm!='E1_ADAPTER_ONLY'or base.target!=10 or len(base.trainable_names)!=14:
            raise PermissionError('CLOSED_TARGET_INTERFACE_PARENT_BEFORE_LIMITED_ACTION')
        self.base_description=base.describe();self.model=base.model;self.target=base.target
        self.target_names=base.trainable_names;self.block_count=count
        params=dict(self.model.named_parameters());self.block_names=block_parameter_names(tuple(params),count)
        wanted=set(self.target_names)|set(self.block_names)
        for n,p in params.items():p.requires_grad_(n in wanted)
        self.trainable_names=tuple(n for n in params if n in wanted)
        self.base=base;self.arm='M32_INTERFACE_PLUS_LAST_'+str(count)+'_DIT_BLOCKS'
        self.validate()

    def validate(self):
        if (self.model.action_head.state_dropout_prob!=0
                or tuple(n for n,p in self.model.named_parameters()if p.requires_grad)!=self.trainable_names
                or self.block_names!=block_parameter_names(tuple(dict(self.model.named_parameters())),self.block_count)):
            raise ValueError('LIMITED_ACTION_EXACT_WHITELIST_AND_FROZEN_REST')
        for row in self.base_description['trainable']:
            p=self.model.get_parameter(row['name'])
            leaf=self.model.get_submodule(row['name'].rsplit('.',1)[0])
            if list(p.shape)!=row['shape']or leaf.target!=10:raise ValueError('ORIGINAL_TARGET_SLICE_CHANGED')
        self.base.enforce_modes()

    def parameters(self):
        self.validate();p=dict(self.model.named_parameters());return [p[n]for n in self.trainable_names]

    def describe(self):
        self.validate();p=dict(self.model.named_parameters())
        return dict(schema='M32_LIMITED_ACTION_PARAMETER_CONTRACT_V1',arm=self.arm,target_category=10,
            upstream_commit=self.base_description['upstream_commit'],module_mode=self.base_description['module_mode'],
            actual_DiT_block_indices=list(range(32-self.block_count,32)),
            trainable=[dict(name=n,shape=list(p[n].shape),numel=p[n].numel())for n in self.trainable_names],
            trainable_numel=sum(p[n].numel()for n in self.trainable_names),
            original_target_interface_parameters=14,all_other_parameters_and_buffers_frozen=True,
            real_model_qualification=False,native_adopted=False)


def promote_contract(contract,p):
    if not is_plan(p):raise PermissionError('EXACT_M32_PLAN_BEFORE_PARAMETER_CHANGE')
    return LimitedContract(contract,p['limited_DiT_blocks'])if p['limited_DiT_blocks']else contract
