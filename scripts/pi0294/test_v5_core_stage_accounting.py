"""CPU conservation/identity regressions. Fixtures are not native receipts."""
from copy import deepcopy
import uuid
import pytest
from scripts.pi0294.test_v5_closed_training_handoff import fixture
from scripts.pi0294.pv1 import mix_accounting as a, mix_core_accounting as c
from scripts.pi0294.pv1 import mix_stage_handoff as h, successor_cd as s
from scripts.pi0294.pv1.successor_core_context import SCHEMA, EVAL_ROLE


def closed():
    g, l, capture = fixture()
    pool = s.whole_package()
    l['CPU_reserved_s'].update(CoreLab=pool['Core_CPU_s'], local_PC=pool['local_CPU_s'])
    l['CPU_s'].update(CoreLab=0., local_PC=0.)
    l['network_transfer_accounted_bytes'] = 0
    g['unknown_effects'] = []; g['pending_updates'] = []
    l = h.project_training_to_LOCAL(local_bytes=a.raw(l), global_bytes=a.raw(g),
        terminal_capture=capture, arm='C50')['proposed_LOCAL']
    # Explicit CPU fixture for the additional physical Core GPU-process CPU.
    # The production proposal also rechecks the full actual cap/matrix/final
    # before the existing onceIO may adopt it; no fixture is native adoption.
    from scripts.pi0294.pv1.mix_core_cpu_projection import FIELD,PER_PANEL
    marker=dict(schema='V5_CD_EXISTING_CORE_GPU_CPU_PROJECTION_V1',task=s.TASK,
        scope_grant_sha256=s.GRANT_SHA,per_panel_Core_CPU_upper_s=PER_PANEL,
        additional_Core_CPU_reserved_s=2*PER_PANEL,panels=['EVAL:C50:N294101','EVAL:D25:N294101'],
        caps_changed={},new_scientific_work=False,original_GPU_site_CPU_retained=True,
        before_GLOBAL_sha256=a.sha(a.raw(g)),before_LOCAL_sha256=a.sha(a.raw(l)),observed_epoch=1150.)
    for value in (g,l):
        value[FIELD]=deepcopy(marker);value[a.PACKAGE][FIELD]=deepcopy(marker)
        value[a.PACKAGE]['remaining_resources']['Core_CPU_s']+=2*PER_PANEL
    l['CPU_reserved_s']['CoreLab']+=2*PER_PANEL
    return g, l, capture


def paid():
    g, l, capture = closed()
    p = c.prepay_control(g, l, terminal_capture=capture, arm='C50', now_epoch=1200.)
    return p['proposed_global'], p['proposed_LOCAL'], capture


def record(g, capture):
    proof = h.closed_endpoint(global_bytes=a.raw(g), arm='C50', **capture)
    return dict(schema='V5_CD_CORE_NATIVE_GPU_ALLOCATION_V1', campaign_run_uuid='20261018T182956Z-localpc-pv1',
        native_platform='CORELAB_RTX4080', platform_evaluation_version=c.VERSION, owner_user='liuyu-nfs',
        authority_release=s.TASK_SOURCE, scope_grant_sha256=s.GRANT_SHA,
        stage='V5_MATCHED_DEV24', noise_bank=294101, remaining_conditions=24,
        checkpoint_step=1024, statistics_sha256=a.STATS, parent_checkpoint_manifest_sha256=a.PARENT,
        owner_uid=2009, hostname='corelab-4080s', physical_GPU_count=1,
        physical_GPU_kind='RTX_4080_SUPER', walltime_s=28800, accounting_pad_s=60,
        reserved_GPU_s=28860, CPU_upper_s=80000, declared_process_CPU_limit_s=79760,
        active_cpu_limit=4, minimum_GPU_memory_bytes=30*1024**3,
        requested_memory_bytes=48*1024**3, optimizer_update_limit=0,
        reserved_optimizer_updates=0, PBS_submissions_consumed=0, query_limit=2097,
        charges_DEC1_subcaps=False, new_robot_physics=0, legacy_GPU_seconds_field_is_projection=True,
        additional_output_upper_bytes=c.remaining_stage()['Hopper_storage_bytes'],
        evaluation_candidate='C50', source_sha='c'*40, request_id=str(uuid.uuid4()),
        submission_nonce=str(uuid.uuid4()), request_sha256='f'*64,
        run_manifest_sha256='e'*64, checkpoint_manifest_sha256=proof['checkpoint_manifest_sha256'],
        training_handoff_sha256=a.sha(a.raw(proof)), planned_panel_sha256='a'*64,
        cold_input_audit_sha256='b'*64, T_CD=1000., deadline_epoch=g[s.FIELD]['deadline_epoch'],
        created_epoch=1201., child_run_uuid='v5-c50-fixture-eval', native_handle=None,
        GPU_uuid='GPU-11111111-1111-1111-1111-111111111111', status=c.STATES[0],
        reserved_neural_queries=2097)


def physical(g):
    return dict(schema=SCHEMA, scope=EVAL_ROLE, candidate='C50', run_uuid='v5-c50-fixture-eval',
        code=dict(vla_commit='c'*40), authority=dict(protocol_ref=s.TASK_SOURCE, scope_grant_sha256=s.GRANT_SHA),
        execution=dict(profile='4080-simulation'), limits=dict(CPU_s=93900, physics_ticks=832000,
        policy_episodes=26, engine_starts=2108, cpus=[0, 1, 2], operation_deadline_epoch=g[s.FIELD]['deadline_epoch']))


def registered():
    g, l, capture = paid(); r = record(g, capture); m = physical(g)
    p = c.register(g, l, r, a.raw(m), terminal_capture=capture, now_epoch=1202.)
    return p['proposed_global'], p['proposed_LOCAL'], r, m


def started(r):
    boot = '22222222-2222-4222-8222-222222222222'
    return dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_STARTED_V1', hostname='corelab-4080s', uid=2009,
        GPU_uuid=r['GPU_uuid'], request_id=r['request_id'], request_sha256=r['request_sha256'],
        submission_nonce=r['submission_nonce'], run_manifest_sha256=r['run_manifest_sha256'],
        source_sha=r['source_sha'], process_state='RUNNING', native_model_permission=False,
        physics_authorized=False, started_epoch=1203., observed_epoch=1204.,
        boot_id=boot, pid=2000, proc_start=12345, native_handle=c.process_handle(boot, 2000, 12345))


def test_control_consumes_exact_prepaid_dimensions_once_and_preserves_all_old_history():
    g, l, capture = closed(); old_g, old_l = deepcopy(g), deepcopy(l)
    p = c.prepay_control(g, l, terminal_capture=capture, arm='C50', now_epoch=1200.)
    v, w = p['proposed_global'], p['proposed_LOCAL']
    assert g == old_g and l == old_l and w['old_history_fixture'] == l['old_history_fixture']
    assert v['used'] == g['used'] and w['PBS_submissions'] == l['PBS_submissions']
    assert v['CPU_reserved_s'] == g['CPU_reserved_s']-7200 and v['CPU_s'] == g['CPU_s']+7200
    assert w['network_transfer_accounted_bytes'] == 1024**3
    assert w['Core_native_GPU_current_storage_bytes'] == 128*1024**2
    assert v[a.PACKAGE] == w[a.PACKAGE]
    for n, amount in c.CONTROL.items():
        assert v[a.PACKAGE]['remaining_resources'][n] == g[a.PACKAGE]['remaining_resources'][n]-amount
    with pytest.raises(PermissionError): c.prepay_control(v, w, terminal_capture=capture, arm='C50', now_epoch=1201.)


def test_stage_moves_existing_pool_without_reserving_GPU_CPU_or_PBS_twice():
    g, l, capture = paid(); r = record(g, capture); m = physical(g)
    p = c.register(g, l, r, a.raw(m), terminal_capture=capture, now_epoch=1202.)
    v, w = p['proposed_global'], p['proposed_LOCAL']; stage = a.stage_vector('EVAL')
    assert v['reserved'] == g['reserved'] and v['CPU_reserved_s'] == g['CPU_reserved_s']
    assert v['used'] == g['used'] and v['PBS_submissions'] == g['PBS_submissions']
    assert w['CPU_reserved_s'] == l['CPU_reserved_s'] and w['GPU_reserved_s'] == l['GPU_reserved_s']
    for n, amount in c.remaining_stage().items():
        assert v[a.PACKAGE]['remaining_resources'][n] + amount == g[a.PACKAGE]['remaining_resources'][n]
    assert v[a.PACKAGE]['completed_stages'] == ['TRAIN:C50:1024']
    assert v[c.FIELD] == w[c.FIELD] and v[c.FIELD][r['request_id']]['native_handle'] is None
    cpu = w['V5_CD_Core_CPU_substages'][c.key('C50')]
    assert cpu['run_manifest_sha256'] == a.sha(a.raw(m)) and cpu['CPU_s'] == stage['Core_CPU_s']
    assert w['Core_native_GPU_reserved']['constructors'] == stage['engine_starts']
    assert not p['native_process_or_model_admission']
    with pytest.raises(PermissionError): c.register(v, w, r, a.raw(m), terminal_capture=capture, now_epoch=1203.)


@pytest.mark.parametrize('field,value', [('evaluation_candidate','B_COVERAGE'),('checkpoint_step',4096),
    ('statistics_sha256','bad'),('parent_checkpoint_manifest_sha256','bad'),('stage','P4_DEV'),
    ('shared_GPU_grant_sha256','a'*64),('PBS_JOBID','99.stdct-mgmt-02'),('queue_dispatch_mode','old'),
    ('platform_training_version',a.VERSION),('reserved_optimizer_updates',1),('query_limit',1937),
    ('created_epoch',True),('reserved_GPU_s',True),('physical_GPU_kind','NVIDIA_A40')])
def test_no_legacy_identity_shared_exception_or_changed_recipe(field,value):
    g, _, capture = paid(); r = record(g, capture); r[field] = value
    with pytest.raises(PermissionError): c.validate_row(r)


def test_actual_closed_training_and_exact_LOCAL_projection_cannot_be_replaced_by_PASS_flag():
    g, l, capture = closed(); capture['OS_exit_bytes'] = a.raw(dict(returncode=1))
    with pytest.raises((PermissionError, KeyError)):
        c.prepay_control(g, l, terminal_capture=capture, arm='C50', now_epoch=1200.)
    g, l, capture = closed(); l['V5_CD_terminal_training_mirrors'] = {}
    with pytest.raises(PermissionError): c.prepay_control(g, l, terminal_capture=capture, arm='C50', now_epoch=1200.)


def test_first_update_or_partial_endpoint_cannot_start_Eval_or_D25():
    g, l, capture = closed(); g['PBS_submissions'][0]['committed_optimizer_updates'] = 1023
    with pytest.raises(PermissionError): c.prepay_control(g, l, terminal_capture=capture, arm='C50', now_epoch=1200.)
    g, l, capture = closed()
    with pytest.raises(PermissionError): c.prepay_control(g, l, terminal_capture=capture, arm='D25', now_epoch=1200.)


def test_fresh_real_process_binding_not_guessed_PID_or_prepared_handle():
    g, l, r, _ = registered(); native = started(r)
    p = c.bind(g, l, request_id=r['request_id'], process=native, now_epoch=1205.)
    v, w = p['proposed_global'], p['proposed_LOCAL']
    assert v[c.FIELD][r['request_id']]['native_handle'] == native['native_handle']
    assert v['reserved'] == g['reserved'] and v['used'] == g['used'] and v[c.FIELD] == w[c.FIELD]
    with pytest.raises(PermissionError): c.bind(v, w, request_id=r['request_id'], process=native, now_epoch=1206.)
    native['native_handle'] = '99.stdct-mgmt-02'
    with pytest.raises(PermissionError): c.bind(g, l, request_id=r['request_id'], process=native, now_epoch=1205.)
    native = started(r)
    with pytest.raises(PermissionError): c.bind(g, l, request_id=r['request_id'], process=native, now_epoch=1400.)


def test_pre_effect_query_CPU_debits_conserve_current_and_future_and_never_repeat():
    g, l, r, _ = registered(); native = started(r)
    g = c.bind(g, l, request_id=r['request_id'], process=native, now_epoch=1205.)['proposed_global']
    args=dict(request_id=r['request_id'], native_handle=native['native_handle'])
    with pytest.raises(PermissionError): c.charge(g, **args, kind='generation_queries', count=1, effect_id='query/0')
    v = c.charge(g, **args, kind='model_loads', count=1, effect_id='model/0')
    v = c.charge(v, **args, kind='generation_queries', count=1, effect_id='query/0')
    v = c.charge(v, **args, kind='CPU_s', count=12.5, effect_id='CPU/0')
    assert v['generation_queries'] == g['generation_queries']+1
    assert v['reserved']['neural_queries'] == g['reserved']['neural_queries']-1
    assert v['CPU_reserved_s'] == g['CPU_reserved_s']-12.5 and v['CPU_s'] == g['CPU_s']+12.5
    assert v[a.PACKAGE] == g[a.PACKAGE] and v['optimizer_updates'] == g['optimizer_updates']
    with pytest.raises(PermissionError): c.charge(v, **args, kind='generation_queries', count=1, effect_id='query/0')
    v['unknown_effects'].append(dict(kind='UNKNOWN_PRESERVED'))
    with pytest.raises(PermissionError): c.charge(v, **args, kind='model_loads', count=1, effect_id='model/1')


@pytest.mark.parametrize('dimension', ['H200_s','neural_queries','Hopper_CPU_s','Hopper_storage_bytes'])
def test_ordinary_global_validator_retains_all_current_plus_future_holds(dimension):
    g, _, _, _ = registered()
    if dimension in g['reserved']: g['reserved'][dimension] -= 1
    elif dimension == 'Hopper_CPU_s': g['CPU_reserved_s'] -= 1
    else: g['storage_reserved_bytes'] -= 1
    with pytest.raises(PermissionError): a.validate_global(g)
