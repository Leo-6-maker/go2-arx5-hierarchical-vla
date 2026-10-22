"""Owner-tested pure synthetic metadata, no native admission/evidence.

Independent process/network rejection is installed before project imports.
The single real-pin test reads source text only; it never imports those files.
"""
import sys
_AUDIT={'active':True,'blocked':[]}
def _deny(event,args):
    if _AUDIT['active'] and (event.startswith(('subprocess.','socket.'))
            or event in ('os.system','os.fork','os.forkpty','os.posix_spawn','os.spawn','pty.spawn')):
        _AUDIT['blocked'].append(event);raise PermissionError('SOURCE_CPU_FIXTURE_EXTERNAL_EFFECT:'+event)
sys.addaudithook(_deny)
from copy import deepcopy
import hashlib,json
from pathlib import Path
import pytest
from scripts.pi0294.dec1 import source_cpu_intake as unit
from scripts.pi0294.dec1.test_source_commissioning_plan import population
assert not _AUDIT['blocked'],'IMPORT_ATTEMPTED_EXTERNAL_EFFECT'
_AUDIT['active']=False

@pytest.fixture(autouse=True)
def no_external_effects():
    n=len(_AUDIT['blocked']);_AUDIT['active']=True
    try:yield
    finally:
        _AUDIT['active']=False
        assert len(_AUDIT['blocked'])==n


def encode(value):return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def packet(population,monkeypatch):
    value,make_population=population;population_bytes=make_population(value)
    plan=unit.commissioning.build(population_bytes,owner_release=unit.PLAN_OWNER_RELEASE)
    plan_bytes=encode(plan);monkeypatch.setattr(unit,'PLAN_SHA256',sha(plan_bytes))
    paths=set(unit.ORIGINAL_13)|set(unit.GLUE_2)
    source={p:('CPU_FIXTURE_NOT_NATIVE:'+p).encode() for p in paths}
    old13={p:sha(source[p]) for p in unit.ORIGINAL_13}
    glue=deepcopy(unit.GLUE_2)
    for p in glue:glue[p]['current_sha256']=sha(source[p])
    monkeypatch.setattr(unit,'ORIGINAL_13',old13);monkeypatch.setattr(unit,'GLUE_2',glue)
    historical=dict(old13,**{p:r['original_sha256'] for p,r in glue.items()})
    t0=unit.HARD_CLOSE-168*3600;campaign='CPU_FIXTURE_ORIGINAL_CAMPAIGN';runtime='f'*40
    parent=dict(run_uuid=campaign,T0_epoch=t0,hard_closeout_epoch=unit.HARD_CLOSE,
        CPU_s=dict(CoreLab=1000,local_PC=2000),CPU_reserved_s=dict(CoreLab=100,local_PC=0),
        children=[dict(CPU_reserved_s=100,episode_starts_reserved=0,physics_ticks_reserved=0,engine_starts_reserved=0)],
        episode_starts=10,physics_ticks=dict(live=1000,shadow=1000,reconstruction=0),engine_starts=20,
        final_reserve_inside_total=dict(episode_starts=100,live_ticks=1600000,shadow_ticks=1600000),
        unknown_effects=[],GPU_inflight_observation=dict(job_id='ACTIVE_A_FIXTURE',state='R'))
    dec1=dict(authority_release=unit.AUTHORITY,campaign_run_uuid=campaign,original_T0_epoch=t0,
        hard_close_epoch=unit.HARD_CLOSE,CPU_s=540,episode_starts=0,physics_ticks=0,constructors=0,unknown_effects=[])
    adoption=dict(schema='DEC1_NATIVE_ADOPTION_AND_BUDGET_V1',status='DEC1_AUTHORITY_ADOPTED_CONTROL_ONLY',
        authority_release=unit.AUTHORITY,original_authority_release=unit.PARENT_AUTHORITY,campaign_run_uuid=campaign,
        original_T0_epoch=t0,original_hard_close_epoch=unit.HARD_CLOSE,clock_reset=False,additive_budget=False)
    def dims(**values):return dict.fromkeys(unit.DIMS,0)|values
    components=dict(main=dims(CoreLab_CPU_s=14400,episode_starts=4,physics_ticks=128000,
            constructors=324,CoreLab_output_bytes=4*768*1024**2),
        counter_qualification=dims(CoreLab_CPU_s=100,constructors=2),
        registration=dims(CoreLab_CPU_s=180),control=dims(CoreLab_CPU_s=200,local_CPU_s=300),
        static_reconstruction=dims(),audit_closeout=dims(CoreLab_CPU_s=500))
    requested={k:sum(c[k] for c in components.values()) for k in unit.DIMS}
    remaining=dims(CoreLab_CPU_s=320*3600-1100,local_CPU_s=48*3600-2000,
        episode_starts=900-10,physics_ticks=40000000-3200000-2000,constructors=100000-20,
        CoreLab_output_bytes=250*1024**3,local_output_bytes=10*1024**3,transfer_bytes=200*1024**3)
    budget=dict(schema=unit.BUDGET_SCHEMA,native_complete=True,request_id='CPU_FIXTURE_NOT_NATIVE_REQUEST',
        campaign_run_uuid=campaign,unknown_effects=[],observed_epoch=t0+100,valid_until_epoch=t0+10*3600,
        components=components,requested=requested,DEC1_used=dims(CoreLab_CPU_s=540,local_CPU_s=500),
        DEC1_reserved=dims(),native_projection_receipt_sha256='a'*64,
        source_commissioning_used=0,source_commissioning_reserved=0,target_component_reserved=4,
        parent_remaining_after_spent_reserved_and_final=remaining,site_remaining=deepcopy(remaining),
        site_permission='EXISTING_REGISTERED_CORELAB_4080_SIMULATION',concurrent_CoreLab_cores=2,
        GR00T_queries=0,GPU_submissions=0,training_updates=0)
    stage=dict(schema=unit.STAGE_SCHEMA,role=unit.ROLE,authority_release=unit.AUTHORITY,
        campaign_run_uuid=campaign,component_plan_owner_release=unit.PLAN_OWNER_RELEASE,
        runtime_source_release=runtime,runtime_CI_receipt_sha256='b'*64,
        original_pin_report_sha256=unit.PIN_REPORT_SHA256,source_only=True,new_training_data=False,
        student_neural_ACK_qualification=False,target_task_qualification=False,DEV_selection=False,LOCKBOX=False)
    manifest=dict(schema='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1',scope=unit.ROLE,
        authority=dict(protocol_ref=unit.PARENT_AUTHORITY,amendments=[unit.AUTHORITY]),
        code=dict(vla_commit=runtime,frozen_original_13_files=deepcopy(old13),
            authorized_interface_glue_files=deepcopy(glue),frozen_numerical_files=historical),
        data=dict(manifest_sha256=sha(plan_bytes)),execution=dict(profile='4080-simulation',sealed_inputs={}),
        limits=dict(PV1_T0_epoch=t0,PV1_hard_close_epoch=unit.HARD_CLOSE,operation_deadline_epoch=t0+9*3600,
            CPU_s=requested['CoreLab_CPU_s'],local_CPU_upper_s=requested['local_CPU_s'],
            physics_ticks=requested['physics_ticks'],engine_starts=requested['constructors'],
            commissioning_episodes=4,policy_episodes=4,training_updates=0,output_bytes=requested['CoreLab_output_bytes']))
    return dict(manifest=manifest,stage=stage,plan=plan,population_bytes=population_bytes,
        adoption=adoption,parent=parent,dec1=dec1,budget=budget,source_files=source,now_epoch=t0+101)


def call(p):
    """Seal synthetic supplied bytes, never perform IO or manufacture native receipt."""
    raw={n:encode(p[n]) for n in ('stage','plan','adoption','parent','dec1')}
    p['budget'].update(parent_snapshot_sha256=sha(raw['parent']),DEC1_snapshot_sha256=sha(raw['dec1']),
        adoption_sha256=sha(raw['adoption']))
    raw['budget']=encode(p['budget']);raw['population']=p['population_bytes']
    p['manifest']['execution']['sealed_inputs']={unit.PATHS[k]:sha(raw[k]) for k in unit.PATHS}
    p['manifest']['execution']['sealed_inputs'].update({
        'plans/PUBLICATION_AND_CI.private.json':p['stage']['runtime_CI_receipt_sha256'],
        'plans/SOURCE_ORIGINAL_PIN_IDENTITY.json':unit.PIN_REPORT_SHA256,
        'plans/SOURCE_CPU_NATIVE_RESOURCE_PROJECTION.json':p['budget'].get('native_projection_receipt_sha256')})
    return unit.validate(p['manifest'],stage_bytes=raw['stage'],plan_bytes=raw['plan'],
        population_bytes=raw['population'],adoption_bytes=raw['adoption'],parent_bytes=raw['parent'],
        dec1_bytes=raw['dec1'],budget_bytes=raw['budget'],source_files=p['source_files'],now_epoch=p['now_epoch'])


def test_complete_four_teacher_pure_intake_allows_active_A_without_native_GO(packet):
    result=call(packet)
    assert result['status']=='PASS_PURE_SOURCE_CPU_INTAKE_NOT_NATIVE_ADMISSION'
    assert result['native_dispatch_admitted'] is False and result['actual_lease_or_site_observed'] is False
    assert result['GR00T_queries']==result['GPU_submissions']==result['training_updates']==0
    assert packet['parent']['GPU_inflight_observation']['state']=='R'


def test_plan_owner_is_original808_runtime_is_independent_future_commit(packet):
    assert packet['manifest']['code']['vla_commit']!=unit.PLAN_OWNER_RELEASE
    assert call(packet)['component_plan_owner_release']==unit.PLAN_OWNER_RELEASE
    packet['stage']['component_plan_owner_release']=packet['manifest']['code']['vla_commit']
    with pytest.raises(PermissionError,match='FIXED_PLAN_OWNER'):call(packet)


@pytest.mark.parametrize('part',['registration','control','static_reconstruction','audit_closeout'])
def test_missing_even_zero_static_or_closing_component_is_rejected(packet,part):
    del packet['budget']['components'][part]
    with pytest.raises(PermissionError,match='ALL_WORKER_STATIC_CONTROL'):call(packet)


@pytest.mark.parametrize('change',['not_complete','no_projection','missing_parent_remaining','missing_site_remaining',
    'component_sum','old_scope','model_query','bool_zero','target_slots','extra_source_episode',
    'original_pin_relabel','numeric_byte','glue_byte','glue_release','clock','expired','unknown',
    'lost_CPU','parent_final','site_CPU','DEC1_CPU','global_CPU','limits','population','plan'])
def test_scope_budget_and_frozen_identity_mutations_fail_closed(packet,change):
    p=packet;b=p['budget'];m=p['manifest']
    if change=='not_complete':b['native_complete']=False
    elif change=='no_projection':b.pop('native_projection_receipt_sha256')
    elif change=='missing_parent_remaining':b.pop('parent_remaining_after_spent_reserved_and_final')
    elif change=='missing_site_remaining':b.pop('site_remaining')
    elif change=='component_sum':b['requested']['CoreLab_CPU_s']-=1
    elif change=='old_scope':m['scope']='PI0294_PV1_CORE_POLICY_EVALUATION_V1'
    elif change=='model_query':b['GR00T_queries']=1
    elif change=='bool_zero':b['GPU_submissions']=False
    elif change=='target_slots':b['target_component_reserved']=0
    elif change=='extra_source_episode':b['source_commissioning_used']=1
    elif change=='original_pin_relabel':m['code']['frozen_numerical_files']['scripts/pi0294/episode.py']=unit.GLUE_2['scripts/pi0294/episode.py']['current_sha256']
    elif change=='numeric_byte':p['source_files'][next(iter(unit.ORIGINAL_13))]+=b'changed'
    elif change=='glue_byte':p['source_files']['scripts/pi0294/episode.py']+=b'changed'
    elif change=='glue_release':m['code']['authorized_interface_glue_files']['scripts/pi0294/episode.py']['explicit_glue_publication']='e'*40
    elif change=='clock':p['adoption']['original_T0_epoch']+=1
    elif change=='expired':p['now_epoch']=b['valid_until_epoch']
    elif change=='unknown':p['dec1']['unknown_effects']=['UNKNOWN_EFFECT']
    elif change=='lost_CPU':b['DEC1_used']['CoreLab_CPU_s']=0
    elif change=='parent_final':p['parent']['final_reserve_inside_total']['episode_starts']=0
    elif change=='site_CPU':b['site_remaining']['CoreLab_CPU_s']=1
    elif change=='DEC1_CPU':b['DEC1_reserved']['CoreLab_CPU_s']=96*3600
    elif change=='global_CPU':p['parent']['CPU_reserved_s']['CoreLab']=320*3600
    elif change=='limits':m['limits']['physics_ticks']=0
    elif change=='population':p['population_bytes']+=b' '
    elif change=='plan':p['plan']['source_component_upper']=8
    with pytest.raises(PermissionError):call(p)


def test_snapshot_bytes_and_manifest_seal_cannot_be_relabelled(packet):
    call(packet);p=packet
    raw={n:encode(p[n]) for n in ('stage','plan','adoption','parent','dec1','budget')}
    raw['parent']+=b' '
    with pytest.raises(PermissionError,match='SNAPSHOT_BYTES'):
        unit.validate(p['manifest'],stage_bytes=raw['stage'],plan_bytes=raw['plan'],population_bytes=p['population_bytes'],
            adoption_bytes=raw['adoption'],parent_bytes=raw['parent'],dec1_bytes=raw['dec1'],budget_bytes=raw['budget'],
            source_files=p['source_files'],now_epoch=p['now_epoch'])
    raw['parent']=encode(p['parent']);raw['stage']+=b' '
    with pytest.raises(PermissionError,match='BYTES_SEALED'):
        unit.validate(p['manifest'],stage_bytes=raw['stage'],plan_bytes=raw['plan'],population_bytes=p['population_bytes'],
            adoption_bytes=raw['adoption'],parent_bytes=raw['parent'],dec1_bytes=raw['dec1'],budget_bytes=raw['budget'],
            source_files=p['source_files'],now_epoch=p['now_epoch'])


def test_exact_published_thirteen_and_two_glue_static_bytes_only():
    root=Path(__file__).resolve().parents[3]
    paths=set(unit.ORIGINAL_13)|set(unit.GLUE_2)
    source={p:(root/p).read_bytes() for p in paths}
    original=dict(unit.ORIGINAL_13,**{p:r['original_sha256'] for p,r in unit.GLUE_2.items()})
    code=dict(frozen_original_13_files=unit.ORIGINAL_13,authorized_interface_glue_files=unit.GLUE_2,
        frozen_numerical_files=original)
    result=unit.validate_pins(source,code,dict(original_pin_report_sha256=unit.PIN_REPORT_SHA256))
    assert len(result)==15 and result['scripts/pi0294/episode.py']!=original['scripts/pi0294/episode.py']
