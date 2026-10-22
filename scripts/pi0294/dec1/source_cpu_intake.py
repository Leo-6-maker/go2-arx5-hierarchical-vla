"""Pure cold source CPU intake; no IO, lease writer, runner or native GO.

Outer scope must verify actual standard entry/host/registry/leases, exact source
capsule and CI, fresh reservation readback and finite process/asset admission.
These functions validate supplied immutable bytes; they do not observe a host.
"""
from copy import deepcopy
import hashlib
import json
import math
import re
from . import source_commissioning_plan as commissioning
from .authority import AUTHORITY,PARENT_AUTHORITY,HARD_CLOSE

ROLE='DEC1_SOURCE_CPU_COMPONENT_COMMISSIONING_V1'
STAGE_SCHEMA='DEC1_SOURCE_CPU_COMPONENT_STAGE_V1'
BUDGET_SCHEMA='DEC1_SOURCE_CPU_COMPONENT_BUDGET_V1'
PLAN_OWNER_RELEASE='80854c8d3914ffc77d55259e07257b904795ca0d'
PLAN_SHA256='65f9d193239105ca3ff78916bb468856d002938981c1d75cb65bdedb4c6307ad'
PIN_REPORT_SHA256='d653b7045163546120fcd739eec22db58f753dda0b4d00051e93d67df93db938'
ORIGINAL_13={
    "scripts/pi0294/scene.py": "29570d70e6d75519eae4fed02aad0a582efe6119deeea048b30bc99e2cf05b11",
    "scripts/pi0294/scene_cards.py": "08530a62ff3f34509f3efb6459f20d5e0c600cdeb8dc31b01e7a766269a0dd91",
    "scripts/pi0294/teacher.py": "994b741654075f9835b5d9bd4f8db70b7476664343b3c204594330dff41711e7",
    "scripts/pi0294/score.py": "143dab0826a7490331a0740bc9e9f322f2aebe5d8a810960bec21ba169e1c956",
    "scripts/pi0294/s1_preflight.py": "9ada71cf25a4347d6cd12232e42d32c7901e552ee839de091b0cc725d69205ec",
    "scripts/pi0292/physics_relay.py": "ac41798731b3db045a7f0d0aba3a95f421ac0fb7302a4858b07567e85fdd779e",
    "scripts/pi0292/live_inputs.py": "bcd1670b184b3f9b4ec13e24bad9bde0c16273b9728ac486afa0a98aaf67cdef",
    "scripts/pi0292/eval_contract.py": "8d18bef9216672d3b7d9d8f4d90a7592c524ea257947187c4f2b8c97dd2a6075",
    "scripts/pi0292/train_adapter.py": "1e0b5abca796d4023d6e935be4c43c20deb6f4de873359be742e13e189401ed0",
    "scripts/pi0291_arx5_data/config.py": "debc6d9d0358bc6dcf5a1107c5723bb6eb8ff4b396fc35f7901fb67f771f593c",
    "scripts/pi0291_arx5_data/engine.py": "623dc5ebf59fd9c470152ee3d60bb15b11ea554ef4b41bf911e62c20dec46c30",
    "scripts/pi0294/numerics.py": "0d29a075cb7521a43521355b68088cb848b797fd5e24fbcf9c60ab5a90c68449",
    "scripts/pi0294/local_backbone.py": "0c8dfdbc5b2c10db5105b5716f553e06594b757c6ec1354643e23d2da30ba6c1"
}
GLUE_2={
    "scripts/pi0294/episode.py": {
        "original_sha256": "ec8205fdb01cc238425e3bcfca5539711640f8aa2a6d7883c0c2ad4bc7ae8728",
        "current_sha256": "e299e84d72697143758f31c427be02bdadf5572c8a670e47a43aa144dbd30262",
        "explicit_glue_publication": "cbc0225482cb7992d8dec78357f7ad13c5adf2d5"
    },
    "scripts/pi0292/relay_protocol.py": {
        "original_sha256": "2ecd3ceb94a9f63316c02e6ccc58ce03fa37a2df7cd33291a6ff5de17f8070f0",
        "current_sha256": "39b19c67678ca177a8c88309adcadb2a0995a1ea1117399f047bfba3a3523aa1",
        "explicit_glue_publication": "641acd1be36d4525846830a208e0badf06e3d988"
    }
}
PATHS=dict(stage='PV1_STAGE_BINDING.json',
    plan='plans/DEC1_SOURCE_CPU_COMPONENT_PLAN.json',
    population='plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json',
    adoption='plans/DEC1_NATIVE_ADOPTION_AND_BUDGET.json',
    budget='plans/DEC1_SOURCE_CPU_BUDGET.json')
DIMS=('CoreLab_CPU_s','local_CPU_s','episode_starts','physics_ticks','constructors',
    'CoreLab_output_bytes','local_output_bytes','transfer_bytes')
CAPS=dict(zip(DIMS,(96*3600,8*3600,136,5000000,12000,64*1024**3,8*1024**3,64*1024**3)))
VAP1_AUTHORITY='43f7fe6f86c683f2e38e5c8802eb8cde208e575c'
VAP1_ADOPTION_SHA256='a26aeb5076e075791e1dfea4e5da0ccaae05ccbb7fa578a1583dea1c4764ed89'
VAP1_PLAN='plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json'
COMPONENTS={'main','counter_qualification','registration','control','static_reconstruction','audit_closeout'}


def _require(ok,reason):
    if not ok:raise PermissionError(reason)


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def _object(raw,name):
    _require(type(raw) is bytes,'SOURCE_CPU_EXACT_BYTES:'+name)
    def pairs(values):
        result={}
        for key,value in values:
            _require(key not in result,'SOURCE_CPU_DUPLICATE_JSON_KEY:'+name)
            result[key]=value
        return result
    try:value=json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))
    except (ValueError,TypeError) as error:raise PermissionError('SOURCE_CPU_JSON:'+name) from error
    _require(type(value) is dict,'SOURCE_CPU_OBJECT:'+name)
    return value


def _number(value,name,integer=False):
    _require(type(value) is int if integer else type(value) in (int,float),
        'SOURCE_CPU_NUMERIC_TYPE:'+name)
    _require(math.isfinite(value) and value>=0,'SOURCE_CPU_FINITE_NONNEGATIVE:'+name)
    return value


def _mapping(value,name):
    _require(type(value) is dict and set(value)==set(DIMS),'SOURCE_CPU_COMPLETE_DIMENSIONS:'+name)
    for key,item in value.items():_number(item,name+'.'+key,not key.endswith('_CPU_s'))
    return value


def validate_pins(source_files,code,stage):
    """Historical13 plus exact authorized2; never silently replace old15 pins."""
    expected=dict(ORIGINAL_13,**{name:row['current_sha256'] for name,row in GLUE_2.items()})
    _require(len(ORIGINAL_13)==13 and len(GLUE_2)==2 and len(expected)==15,
        'SOURCE_CPU_EXACT_13_PLUS_2')
    _require(type(source_files) is dict and set(source_files)==set(expected),
        'SOURCE_CPU_EXACT_FIFTEEN_SOURCE_BYTES')
    _require(all(type(raw) is bytes and _sha(raw)==expected[name] for name,raw in source_files.items()),
        'SOURCE_CPU_FROZEN_NUMERIC_OR_GLUE_BYTES_CHANGED')
    historical=dict(ORIGINAL_13,**{name:row['original_sha256'] for name,row in GLUE_2.items()})
    _require(code.get('frozen_original_13_files')==ORIGINAL_13
        and code.get('authorized_interface_glue_files')==GLUE_2
        and code.get('frozen_numerical_files')==historical
        and stage.get('original_pin_report_sha256')==PIN_REPORT_SHA256,
        'SOURCE_CPU_EXACT_OLD_AND_NEW_DELTA_PROVENANCE')
    return deepcopy(expected)


def effective_caps(dec1,amendment_bytes=None):
    """Use only the already adopted exact VAP1 delta; no new allowance."""
    caps=deepcopy(CAPS)
    if dec1.get('VAP1_authority_release') is None:
        _require(amendment_bytes is None,'SOURCE_CPU_UNADOPTED_VAP1_DELTA')
        return caps
    _require(dec1['VAP1_authority_release']==VAP1_AUTHORITY
        and type(amendment_bytes) is bytes and _sha(amendment_bytes)==VAP1_ADOPTION_SHA256,
        'SOURCE_CPU_EXACT_ALREADY_ADOPTED_VAP1_BYTES')
    a=_object(amendment_bytes,'VAP1 adopted delta')
    delta=dict(local_CPU_s=[28800,57600],CoreLab_CPU_s=[345600,576000],
        GPU_site_output_bytes=[68719476736,171798691840],
        CoreLab_output_bytes=[68719476736,137438953472],transfer_bytes=[68719476736,137438953472])
    _require(a.get('schema')=='VAP1_ADOPTION_AND_DELTA_BUDGET_V1'
        and a.get('authority_release')==VAP1_AUTHORITY
        and a.get('original_T0_epoch')==dec1['original_T0_epoch']
        and a.get('original_hard_close_epoch')==dec1['hard_close_epoch']==HARD_CLOSE
        and a.get('old_clocks_and_counters_reset') is False
        and a.get('cumulative_subcap_delta')==delta,
        'SOURCE_CPU_EXISTING_VAP1_SCOPE_CLOCK_AND_EXACT_DELTA')
    _require(dec1.get('caps',{}).get('CoreLab_CPU_s')==576000
        and dec1['caps'].get('local_CPU_s')==57600
        and dec1.get('VAP1_cumulative_io_caps_bytes',{}).get('CoreLab')==137438953472
        and dec1['VAP1_cumulative_io_caps_bytes'].get('transfer')==137438953472,
        'SOURCE_CPU_NATIVE_VAP1_COUNTER_CAPS_NOT_OLD_COPY')
    caps.update(CoreLab_CPU_s=576000,local_CPU_s=57600,
        CoreLab_output_bytes=137438953472,transfer_bytes=137438953472)
    return caps


def validate_budget(budget,parent,dec1,adoption,*,now_epoch,caps=None):
    """Nonadditive snapshot math; fresh native truth/leases stay outer-scope IO."""
    _require(budget.get('schema')==BUDGET_SCHEMA and budget.get('native_complete') is True,
        'SOURCE_CPU_NATIVE_COMPLETE_BUDGET_REQUIRED')
    _require(budget.get('unknown_effects')==[] and parent.get('unknown_effects')==[]
        and dec1.get('unknown_effects')==[],'SOURCE_CPU_UNKNOWN_EFFECTS_NO_NEW_START')
    _require(type(budget.get('request_id')) is str and bool(budget['request_id'])
        and budget.get('campaign_run_uuid')==adoption['campaign_run_uuid'],
        'SOURCE_CPU_ONCE_REQUEST_AND_CAMPAIGN')
    _require(adoption['original_T0_epoch']<=_number(budget['observed_epoch'],'observed_epoch')
        <=now_epoch<_number(budget['valid_until_epoch'],'valid_until_epoch')
        <=adoption['original_T0_epoch']+160*3600<=HARD_CLOSE,
        'SOURCE_CPU_ORIGINAL_CLOCK_AND_FRESH_BOUNDED_BUDGET')
    components=budget.get('components')
    _require(type(components) is dict and set(components)==COMPONENTS,
        'SOURCE_CPU_ALL_WORKER_STATIC_CONTROL_AND_CLOSEOUT_COMPONENTS')
    for name,row in components.items():_mapping(row,'components.'+name)
    main=components['main'];counter=components['counter_qualification']
    _require(main['episode_starts']==4 and main['physics_ticks']==128000
        and main['constructors']==324 and main['CoreLab_CPU_s']>0,
        'SOURCE_CPU_ORIGINAL_FOUR_MAIN_COMPONENTS')
    _require(counter['episode_starts']==counter['physics_ticks']==0 and counter['constructors']==2
        and counter['CoreLab_CPU_s']>0,'SOURCE_CPU_ZERO_TIME_TWO_COUNTER_CONSTRUCTORS')
    _require(all(components[n]['episode_starts']==0 for n in COMPONENTS-{'main'}),
        'SOURCE_CPU_NO_EXTRA_COMMISSIONING_EPISODES')
    _require(all(components[n]['CoreLab_CPU_s']>0 for n in ('registration','control','audit_closeout')),
        'SOURCE_CPU_REGISTRATION_CONTROL_AND_CLOSEOUT_CPU_REQUIRED')
    requested={key:sum(row[key] for row in components.values()) for key in DIMS}
    _require(_mapping(budget.get('requested'),'requested')==requested,
        'SOURCE_CPU_COMPLETE_COMPONENT_SUM')
    used=_mapping(budget.get('DEC1_used'),'DEC1_used')
    reserved=_mapping(budget.get('DEC1_reserved'),'DEC1_reserved')
    for key in ('CPU_s','episode_starts','physics_ticks','constructors'):
        _number(dec1[key],'DEC1.'+key,key!='CPU_s')
    _require(used['CoreLab_CPU_s']==dec1['CPU_s']
        and used['episode_starts']==dec1['episode_starts']
        and used['physics_ticks']==dec1['physics_ticks']
        and used['constructors']==dec1['constructors'],
        'SOURCE_CPU_ACTUAL_CORE_DEC1_COUNTERS_NOT_RESET')
    # Output/transfer and local DEC1 projections have no complete legacy Core
    # schema. Owner must supply a sealed actual readback/projection receipt;
    # this module validates its binding only, never fabricates that evidence.
    _require(type(budget.get('native_projection_receipt_sha256')) is str
        and re.fullmatch('[0-9a-f]{64}',budget['native_projection_receipt_sha256']) is not None,
        'SOURCE_CPU_NATIVE_FULL_PROJECTION_RECEIPT_REQUIRED')
    caps=CAPS if caps is None else caps
    _require(all(used[k]+reserved[k]+requested[k]<=caps[k] for k in DIMS),
        'SOURCE_CPU_DEC1_NONADDITIVE_CAP')
    _require(_number(budget['source_commissioning_used'],'source_used',True)
        +_number(budget['source_commissioning_reserved'],'source_reserved',True)+4<=4
        and budget.get('target_component_reserved')==4
        and type(budget.get('target_component_reserved')) is int,
        'SOURCE_CPU_FOUR_SOURCE_FOUR_TARGET_WITHIN_ORIGINAL_EIGHT')
    floor=parent['final_reserve_inside_total']
    _require(floor['episode_starts']==100 and floor['live_ticks']==floor['shadow_ticks']==1600000,
        'SOURCE_CPU_ORIGINAL_FINAL_RESERVE_UNCHANGED')
    children=parent['children'];_require(type(children) is list,'SOURCE_CPU_PARENT_CHILDREN')
    _require(sum(_number(c.get('CPU_reserved_s',0),'child.CPU_reserved_s') for c in children)
        <=_number(parent['CPU_reserved_s']['CoreLab'],'parent.CoreLab_CPU_reserved_s'),
        'SOURCE_CPU_ALL_PARENT_CHILD_CPU_RESERVATIONS_PRESERVED')
    pending={k:sum(_number(c.get(k+'_reserved',0),'child.'+k,True) for c in children)
        for k in ('episode_starts','physics_ticks','engine_starts')}
    physics=parent['physics_ticks'];_require(type(physics) is dict
        and set(physics)=={'live','shadow','reconstruction'},'SOURCE_CPU_PARENT_TICK_DOMAINS')
    parent_used=dict(CoreLab_CPU_s=parent['CPU_s']['CoreLab'],local_CPU_s=parent['CPU_s']['local_PC'],
        episode_starts=parent['episode_starts'],physics_ticks=sum(_number(v,'parent.tick',True) for v in physics.values()),
        constructors=parent['engine_starts'])
    parent_pending=dict(CoreLab_CPU_s=parent['CPU_reserved_s']['CoreLab'],
        local_CPU_s=parent['CPU_reserved_s']['local_PC'],episode_starts=pending['episode_starts'],
        physics_ticks=pending['physics_ticks'],constructors=pending['engine_starts'])
    parent_caps=dict(CoreLab_CPU_s=320*3600,local_CPU_s=48*3600,episode_starts=900,
        physics_ticks=40000000-3200000,constructors=100000)
    for key in parent_used:
        _require(_number(parent_used[key],'parent.used.'+key,not key.endswith('_CPU_s'))
            +_number(parent_pending[key],'parent.pending.'+key,not key.endswith('_CPU_s'))
            +requested[key]<=parent_caps[key],'SOURCE_CPU_PARENT_OR_PROTECTED_FINAL_CAP:'+key)
        _require(used[key]+reserved[key]<=parent_used[key]+parent_pending[key],
            'SOURCE_CPU_DEC1_CLASSIFICATION_MUST_BE_INSIDE_PARENT:'+key)
    remaining=_mapping(budget.get('parent_remaining_after_spent_reserved_and_final'),'parent_remaining')
    site=_mapping(budget.get('site_remaining'),'site_remaining')
    for key in parent_used:
        _require(remaining[key]==parent_caps[key]-parent_used[key]-parent_pending[key],
            'SOURCE_CPU_PARENT_REMAINING_EXACT:'+key)
    for key,upper in dict(CoreLab_output_bytes=300*1024**3,local_output_bytes=20*1024**3,
            transfer_bytes=300*1024**3).items():
        _require(remaining[key]<=upper,'SOURCE_CPU_ORIGINAL_OUTPUT_OR_TRANSFER_CAP:'+key)
    _require(all(requested[k]<=min(remaining[k],site[k]) for k in DIMS),
        'SOURCE_CPU_PARENT_FINAL_AND_SITE_REMAINING')
    _require(budget.get('site_permission')=='EXISTING_REGISTERED_CORELAB_4080_SIMULATION'
        and type(budget.get('concurrent_CoreLab_cores')) is int
        and 1<=budget['concurrent_CoreLab_cores']<=4,
        'SOURCE_CPU_EXISTING_SITE_AND_ORIGINAL_CORE_LIMIT')
    _require(all(type(budget.get(k)) is int and budget[k]==0 for k in
        ('GR00T_queries','GPU_submissions','training_updates')),
        'SOURCE_CPU_ZERO_MODEL_GPU_AND_UPDATES')
    return requested


def validate(manifest,*,stage_bytes,plan_bytes,population_bytes,adoption_bytes,
             parent_bytes,dec1_bytes,budget_bytes,source_files,now_epoch,amendment_bytes=None):
    """Validate supplied intake ONLY. Never a native dispatch/lease assertion."""
    _number(now_epoch,'now_epoch')
    stage=_object(stage_bytes,'stage');plan=_object(plan_bytes,'plan')
    adoption=_object(adoption_bytes,'adoption');parent=_object(parent_bytes,'parent')
    dec1=_object(dec1_bytes,'DEC1');budget=_object(budget_bytes,'budget')
    _require(type(manifest) is dict and manifest.get('scope')==ROLE
        and manifest.get('schema')=='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1',
        'SOURCE_CPU_EXACT_NEW_ROLE_NOT_OLD_DEV')
    caps=effective_caps(dec1,amendment_bytes)
    amendments=[AUTHORITY]+([VAP1_AUTHORITY] if amendment_bytes is not None else [])
    _require(manifest.get('authority',{}).get('protocol_ref')==PARENT_AUTHORITY
        and manifest['authority'].get('amendments')==amendments,
        'SOURCE_CPU_EXPLICIT_NONADDITIVE_DEC1_AUTHORITY')
    _require(stage.get('schema')==STAGE_SCHEMA and stage.get('role')==ROLE
        and stage.get('authority_release')==AUTHORITY
        and stage.get('component_plan_owner_release')==PLAN_OWNER_RELEASE,
        'SOURCE_CPU_EXACT_STAGE_AND_FIXED_PLAN_OWNER_RELEASE')
    _require(_sha(plan_bytes)==PLAN_SHA256,'SOURCE_CPU_ONCE_PRESEALED_PLAN_BYTES')
    commissioning.validate(plan,population_bytes,owner_release=PLAN_OWNER_RELEASE)
    code=manifest['code'];runtime=code['vla_commit']
    _require(type(runtime) is str and re.fullmatch('[0-9a-f]{40}',runtime) is not None
        and stage.get('runtime_source_release')==runtime,
        'SOURCE_CPU_RUNTIME_SEAL_IDENTITY_SEPARATE_FROM_PLAN_OWNER')
    _require(type(stage.get('runtime_CI_receipt_sha256')) is str
        and re.fullmatch('[0-9a-f]{64}',stage['runtime_CI_receipt_sha256']) is not None,
        'SOURCE_CPU_EXACT_RUNTIME_CI_RECEIPT_BINDING')
    _require(all(stage.get(k) is v for k,v in dict(source_only=True,new_training_data=False,
        student_neural_ACK_qualification=False,target_task_qualification=False,
        DEV_selection=False,LOCKBOX=False).items()),'SOURCE_CPU_COMPONENT_ONLY_NOT_POLICY_RELEASE')
    _require(adoption.get('schema')=='DEC1_NATIVE_ADOPTION_AND_BUDGET_V1'
        and adoption.get('status')=='DEC1_AUTHORITY_ADOPTED_CONTROL_ONLY'
        and adoption.get('authority_release')==AUTHORITY
        and adoption.get('original_authority_release')==PARENT_AUTHORITY
        and adoption.get('original_hard_close_epoch')==HARD_CLOSE
        and adoption.get('clock_reset') is False and adoption.get('additive_budget') is False,
        'SOURCE_CPU_ACTUAL_ADOPTION_NO_NEW_CLOCK_OR_BUDGET')
    campaign=adoption['campaign_run_uuid'];t0=adoption['original_T0_epoch']
    _require(_number(t0,'original_T0_epoch')+168*3600==HARD_CLOSE,
        'SOURCE_CPU_ORIGINAL_SEVEN_DAY_WINDOW')
    _require(stage.get('campaign_run_uuid')==campaign and parent.get('run_uuid')==campaign
        and dec1.get('campaign_run_uuid')==campaign
        and dec1.get('authority_release')==AUTHORITY
        and parent.get('T0_epoch')==t0 and dec1.get('original_T0_epoch')==t0
        and parent.get('hard_closeout_epoch')==dec1.get('hard_close_epoch')==HARD_CLOSE,
        'SOURCE_CPU_SAME_ORIGINAL_CAMPAIGN_AND_CLOCK')
    raw=dict(stage=stage_bytes,plan=plan_bytes,population=population_bytes,
        adoption=adoption_bytes,budget=budget_bytes)
    sealed=manifest.get('execution',{}).get('sealed_inputs',{})
    _require(amendment_bytes is None or sealed.get(VAP1_PLAN)==_sha(amendment_bytes),
        'SOURCE_CPU_SEALED_EXISTING_VAP1_DELTA')
    _require(all(sealed.get(PATHS[k])==_sha(v) for k,v in raw.items())
        and sealed.get('plans/PUBLICATION_AND_CI.private.json')==stage['runtime_CI_receipt_sha256']
        and sealed.get('plans/SOURCE_ORIGINAL_PIN_IDENTITY.json')==PIN_REPORT_SHA256
        and sealed.get('plans/SOURCE_CPU_NATIVE_RESOURCE_PROJECTION.json')==budget.get('native_projection_receipt_sha256'),
        'SOURCE_CPU_ALL_INTAKE_BYTES_SEALED_IN_MANIFEST')
    _require(budget.get('parent_snapshot_sha256')==_sha(parent_bytes)
        and budget.get('DEC1_snapshot_sha256')==_sha(dec1_bytes)
        and budget.get('adoption_sha256')==_sha(adoption_bytes),
        'SOURCE_CPU_ACTUAL_SNAPSHOT_BYTES_AND_ADOPTION_BOUND')
    _require(manifest.get('data',{}).get('manifest_sha256')==PLAN_SHA256,
        'SOURCE_CPU_DATA_IS_COMPONENT_PLAN_NOT_DEV_OR_READY')
    pins=validate_pins(source_files,code,stage)
    requested=validate_budget(budget,parent,dec1,adoption,now_epoch=now_epoch,caps=caps)
    limits=manifest['limits']
    expected_limits=dict(CPU_s=requested['CoreLab_CPU_s'],local_CPU_upper_s=requested['local_CPU_s'],
        physics_ticks=requested['physics_ticks'],engine_starts=requested['constructors'],
        commissioning_episodes=4,policy_episodes=4,training_updates=0,
        output_bytes=requested['CoreLab_output_bytes'])
    _require(all(type(limits.get(k)) in (int,float) and type(limits.get(k)) is not bool
        and limits[k]==v for k,v in expected_limits.items()),
        'SOURCE_CPU_MANIFEST_LIMITS_MATCH_COMPLETE_REQUEST')
    _require(manifest['execution'].get('profile')=='4080-simulation',
        'SOURCE_CPU_ORIGINAL_STANDARD_PROFILE')
    _require(manifest['limits']['PV1_T0_epoch']==t0
        and manifest['limits']['PV1_hard_close_epoch']==HARD_CLOSE
        and now_epoch<manifest['limits']['operation_deadline_epoch']<=budget['valid_until_epoch'],
        'SOURCE_CPU_MANIFEST_ORIGINAL_BOUNDED_DEADLINE')
    return dict(status='PASS_PURE_SOURCE_CPU_INTAKE_NOT_NATIVE_ADMISSION',role=ROLE,
        component_plan_owner_release=PLAN_OWNER_RELEASE,runtime_source_release=runtime,
        component_plan_sha256=PLAN_SHA256,requested=deepcopy(requested),frozen_pins=pins,
        native_dispatch_admitted=False,actual_lease_or_site_observed=False,
        GR00T_queries=0,GPU_submissions=0,training_updates=0)
