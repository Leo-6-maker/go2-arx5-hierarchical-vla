"""Captured old failure; all new retry identities/physical/PBS are CPU fixtures.

No test creates a production root, calls SSH, imports a model, or grants native
qualification. The original current Core row is marked terminal only inside a
clearly labelled future fixture; its real closed state is not asserted here.
"""
import base64
import gzip
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.pi0294.generalization import minimal_execution as e, minimal_retry_execution as retry
from scripts.pi0294.generalization import minimal_retry_history as history, minimal_retry_paid_io as paid
from scripts.pi0294.generalization import minimal_terminal as terminal
from scripts.pi0294.test_minimal_retry_custody import evidence as captured_evidence
from scripts.pi0294.test_minimal_terminal import captured_CPU_terminal


@pytest.fixture
def evidence(request):
    if not os.environ.get('ARX5_RETRY_ACTUAL_CAPTURE_DIRECTORY'):
        pytest.skip('explicit private captured original closed failure required')
    return request.getfixturevalue('captured_evidence')


@pytest.fixture
def prepared(evidence):
    root = Path(os.environ['ARX5_RETRY_ACTUAL_CAPTURE_DIRECTORY'])
    def saved(pattern):
        v = json.loads(sorted(root.glob(pattern))[-1].read_bytes())
        return json.loads(v['stdout']) if 'stdout' in v else v
    inv = saved('FIRST512_RETRY_BOTH_OLD_ROOTS_UNIQUE_INODE_READONLY_*private.json')
    old_closed = saved('FIRST512_NATIVE_COLD01_OLD_COLD_LEASE_INDEPENDENT_READONLY_*private.json')
    # Later real D25 terminal bytes fill the previously unbound Core record
    # ONLY in this hybrid CPU fixture. The earlier holds remain conservatively
    # retained; it is not a new canonical snapshot or native settlement.
    g = json.loads(evidence['global_bytes'])
    later = json.loads((root / 'D25_RETRY02_NETWORK_TERMINAL_INDEPENDENT_DUAL_READBACK.private.json').read_bytes())
    later_g = json.loads(gzip.decompress(base64.b64decode(later['global']['GLOBAL_gzip_base64'])))
    for key, row in g[e.original.CORE_ROWS].items():
        if row['status'] != 'NATIVE_TERMINAL':
            g[e.original.CORE_ROWS][key] = deepcopy(later_g[e.original.CORE_ROWS][key])
    ev = dict(evidence, global_bytes=e.raw(g))
    h = history.capture(**ev); old = h['original_row']
    r = json.loads(base64.b64decode(h['original_marker']['request_base64']))
    r.update(retry.fixed_request_fields(), request_id='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
        submission_nonce='bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb', source_sha='c' * 40,
        run_manifest_sha256='d' * 64, closed_predecessor_request_id=old['request_id'],
        closed_predecessor_row_sha256=h['original_row_sha256'],
        closed_predecessor_marker_sha256=h['original_marker_sha256'], retry_history_sha256=e.sha(e.raw(h)))
    custody = dict(schema=retry.guard.SCHEMA, request_id=r['request_id'], submission_nonce=r['submission_nonce'],
        run_uuid=retry.RUN, old_metadata_ceiling=e.CONTROL['Hopper_storage_bytes'],
        paid_output_ceiling=e.native_resources()['Hopper_storage_bytes'],
        engineering_metadata_allowance=retry.CONTROL['Hopper_storage_bytes'], old_inventory_sha256='f' * 64,
        old_roots=[dict(run_uuid='arx5-minimal-raw-parent-first512-01', manifest_sha256='e' * 64,
                       closed_lease_sha256=old_closed['lease_sha256']),
                   dict(run_uuid='arx5-minimal-raw-parent-first512-02', manifest_sha256=old['run_manifest_sha256'],
                       closed_lease_sha256=evidence['closed_readback']['lease_sha256'])])
    r['physical_custody_sha256'] = e.sha(e.raw(custody))
    proof = dict(status='ACTUAL_RETRY_PHYSICAL_BOUNDS_ONLY_NO_PERMISSION', old_inventory_sha256='f' * 64,
        old_metadata_ceiling=e.CONTROL['Hopper_storage_bytes'], paid_output_ceiling=e.native_resources()['Hopper_storage_bytes'],
        old_allocated_bytes=inv['unique_allocated_bytes'], new_allocated_bytes=8192,
        next_output_upper_bytes=retry.guard.CONTROL_GROWTH_RESERVE,
        new_root='/cpu-fixture/children/' + retry.RUN,
        old_roots=['/cpu-fixture/children/' + x['run_uuid'] for x in custody['old_roots']],
        ledger_changed=False, native_permission=False)
    now = inv['observed_epoch'] + 1
    return dict(**ev, request_bytes=e.raw(r), old_cold_close_readback=old_closed,
        root_inventory=inv, physical_custody=custody, physical_proof=proof,
        physical_observed_epoch=now, now_epoch=now)


def propose(prepared):
    return retry.register(**prepared)['proposed_global']


def bound(prepared):
    g = propose(prepared); r = json.loads(prepared['request_bytes'])
    return retry.bind(g, r['request_id'], '987654.stdct-mgmt-02', now_epoch=prepared['now_epoch'] + 1)


def terminal_fixture(g, *, success=False):
    """Reuse original five-session fixture with the NEW row explicitly chosen."""
    _, row = retry._selected(g)
    # Existing helper chooses old e.rows[0]; only its fixture input is replaced.
    fixture = deepcopy(g); fixture.pop(retry.FIELD)
    # Build terminal using its original old row, then bind every real identity
    # to the explicit new row; this is synthetic test data, never native data.
    args = captured_CPU_terminal(fixture, success=success)
    for kind in ('native', 'effects', 'audit'):
        args[kind].update({k: row[k] for k in ('request_id', 'request_sha256', 'job_id', 'source_sha')})
    args['native'].update(submission_nonce=row['submission_nonce'], prior_readonly_preparation_CPU_upper_s=0,
        conservative_CPU_s=1000, new_native_conservative_CPU_s=1000,
        finished_epoch=row['submitted_epoch'] + 5, observed_epoch=row['submitted_epoch'] + 6)
    args['audit']['effect_summary_sha256'] = e.sha(e.raw(args['effects']))
    args['now_epoch'] = row['submitted_epoch'] + 7
    args['physical_proof'] = deepcopy(g[retry.FIELD]['physical_preflight'])
    args['physical_proof']['new_allocated_bytes'] = 1000000
    args['physical_observed_epoch'] = row['submitted_epoch'] + 6
    return args


def test_exact_fixed_plan_keeps_all_science_and_zero_new_output_reservation():
    old = e.fixed_request_fields(); new = retry.fixed_request_fields()
    changes = {k for k in old if old[k] != new[k]}
    assert changes == {'schema', 'platform_training_version', 'profile',
                       'prior_readonly_preparation_CPU_upper_s', 'additional_output_upper_bytes'}
    assert new['optimizer_update_limit'] == 1028 and new['query_limit'] == 147
    assert new['session_plan'] == e.native_sessions() and new['model_load_limit'] == 11
    assert new['scheduler_horizon'] == 1024 and new['formal_terminal_steps'] == 512
    assert retry.native_resources()['Hopper_CPU_s'] == 165600
    assert retry.native_resources()['Hopper_storage_bytes'] == 0
    assert new['paid_output_allowance_bytes'] == 144294967296


def test_two_distinct_rows_keep_complete_original_failed_history(prepared):
    old = json.loads(prepared['global_bytes']); before = deepcopy(prepared)
    result = retry.register(**prepared); g = result['proposed_global']; rows = retry.rows(g)
    assert len(rows) == 2 and rows[0] == e.rows(old)[0]
    assert g[e.FIELD] == old[e.FIELD] and g[e.package.FIELD] == old[e.package.FIELD]
    assert g['PBS_submissions'][:-1] == old['PBS_submissions']
    assert g[e.original.PACKAGE] == old[e.original.PACKAGE]
    assert prepared == before and result['native_admission'] is False
    assert result['original_global_reader_hooks_required'] is True
    assert retry.selected(g, rows[1]['request_id']) == rows[1]
    assert retry.selected(g, rows[0]['request_id']) == rows[0]
    assert rows[0]['conservative_CPU_s'] == 1153 and rows[0]['conservative_H200_s'] == 257
    assert g['unknown_effects'] == g['pending_updates'] == []


def test_control_paid_once_full_compute_held_and_old_gross_never_refunded(prepared):
    old = json.loads(prepared['global_bytes']); g = propose(prepared)
    assert g['CPU_s'] == old['CPU_s'] + 3600
    assert g['CPU_reserved_s'] == old['CPU_reserved_s'] + 165600 - 3600
    assert g['reserved']['H200_s'] == old['reserved']['H200_s'] + 59520 - 3600
    assert g['reserved']['optimizer_updates'] == old['reserved']['optimizer_updates'] + 1028
    assert g['reserved']['neural_queries'] == old['reserved']['neural_queries'] + 147
    assert g['new_storage_current_bytes'] == old['new_storage_current_bytes'] + 64 * 1024**2
    assert g['new_storage_peak_bytes'] >= old['new_storage_peak_bytes']
    assert g['storage_reserved_bytes'] == old['storage_reserved_bytes'] - 64 * 1024**2
    assert g['used']['PBS_submissions'] == old['used']['PBS_submissions'] + 1
    assert g['used']['H200_s'] == old['used']['H200_s']
    # The candidate alone omitted these costs: the complete proposal adds them.
    candidate = paid.candidate(prepared['global_bytes'], prepared['local_bytes'],
        **{k: prepared[k] for k in ('native_readback', 'closed_readback', 'old_cold_close_readback', 'root_inventory', 'now_epoch')},
        new_request_id=g[retry.FIELD]['request_id'], new_nonce=g['PBS_submissions'][-1]['submission_nonce'],
        new_root=prepared['physical_proof']['new_root'])
    dims = g[retry.FIELD]['complete_future_dimensions']
    for key, extra in [('Hopper_CPU_s', 3600), ('local_CPU_s', 600), ('transfer_bytes', 64 * 1024**2),
                       ('Core_storage_bytes', 64 * 1024**2)]:
        assert dims[key]['projected'] == candidate['full_dimensions'][key]['projected'] + extra
    assert all(v['shortfall'] == 0 for v in dims.values())


def test_registration_and_native_binding_are_single_use(prepared):
    g = propose(prepared); request_id = g[retry.FIELD]['request_id']
    with pytest.raises(PermissionError, match='ONCE_NO_EXISTING'):
        retry.register(**dict(prepared, global_bytes=e.raw(g)))
    b = retry.bind(g, request_id, '987654.stdct-mgmt-02', now_epoch=prepared['now_epoch'] + 1)
    assert retry.selected(b, request_id)['status'] == e.STATES[1]
    with pytest.raises(PermissionError): retry.bind(b, request_id, '987654.stdct-mgmt-02', now_epoch=prepared['now_epoch'] + 2)
    with pytest.raises(PermissionError): retry.bind(g, e.rows(g)[0]['request_id'], '987654.stdct-mgmt-02', now_epoch=prepared['now_epoch'] + 1)
    with pytest.raises(PermissionError): retry.bind(g, request_id, '1501631.stdct-mgmt-02', now_epoch=prepared['now_epoch'] + 1)


def test_new_cpu_and_site_limits_use_no_second_prior900(prepared):
    r = json.loads(prepared['request_bytes'])
    actual = retry.process_CPU_limits(r)
    assert actual == dict(parent_RLIMIT_CPU_s=164760, isolated_child_CPU_upper_s=600,
        terminal_observer_CPU_upper_s=240, prior_readonly_preparation_CPU_upper_s=0,
        whole_reserved_CPU_upper_s=165600, new_CPU_budget=0)
    assert retry.project_envelope(r)['CPU_hours'] == 595.2
    assert retry.project_envelope(r)['actual_process_CPU_upper_s'] == 165600
    profile = dict(select='1:ncpus=36:mem=240gb:ngpus=1:gpu_model=A40',
                   ncpus=36, mem='240gb', ngpus=1, walltime='16:30:00')
    assert retry.select_profile(profile, r) == (36, 240 * 1024**3)
    with pytest.raises(PermissionError): retry.select_profile(dict(profile, walltime='17:00:00'), r)


@pytest.mark.parametrize('key', ['unknown_effects', 'pending_updates'])
def test_bind_does_not_ignore_new_global_latches(prepared, key):
    g = propose(prepared); g[key] = [{'preserve': True}]; before = deepcopy(g)
    with pytest.raises(PermissionError, match='LATCHES'):
        retry.bind(g, g[retry.FIELD]['request_id'], '987654.stdct-mgmt-02', now_epoch=prepared['now_epoch'] + 1)
    assert g == before


def test_unchanged_original_native_reader_still_rejects_unhooked_version(prepared):
    from scripts.pi0294.pv1.neural_ledger import check
    g = propose(prepared)
    # Explicitly document that this adapter is NOT a native deployment or a
    # global permission bypass. The original reader requires cold dispatch.
    with pytest.raises(PermissionError): check(g, allow_unknown=True)


@pytest.mark.parametrize('change', ['unknown', 'pending', 'old_request', 'old_nonce', 'science', 'clock',
    'old_source', 'old_manifest', 'old_history', 'missing_guard', 'stale_physical', 'unsafe_growth',
    'wrong_inodes', 'old_bytes', 'foreign_path', 'wrong_lease', 'negative_physical'])
def test_no_unsafe_replacement_can_be_proposed(prepared, change):
    p = deepcopy(prepared); g = json.loads(p['global_bytes']); r = json.loads(p['request_bytes'])
    old = e.rows(g)[0]
    if change == 'unknown': g['unknown_effects'] = [{'must': 'remain'}]
    elif change == 'pending': g['pending_updates'] = [{'must': 'remain'}]
    elif change == 'old_request': r['request_id'] = old['request_id']
    elif change == 'old_nonce': r['submission_nonce'] = old['submission_nonce']
    elif change == 'science': r['query_limit'] += 1
    elif change == 'clock': r['stop_new_training_epoch'] += 1
    elif change == 'old_source': r['source_sha'] = old['source_sha']
    elif change == 'old_manifest': r['run_manifest_sha256'] = old['run_manifest_sha256']
    elif change == 'old_history': r['retry_history_sha256'] = '0' * 64
    elif change == 'missing_guard': r.pop('physical_guard_policy')
    elif change == 'stale_physical': p['physical_observed_epoch'] -= 121
    elif change == 'unsafe_growth': p['physical_proof']['next_output_upper_bytes'] = 0
    elif change == 'wrong_inodes': p['physical_proof']['old_inventory_sha256'] = '0' * 64
    elif change == 'old_bytes': p['physical_proof']['old_allocated_bytes'] -= 1
    elif change == 'foreign_path': p['physical_proof']['old_roots'][0] = '/different/parent'
    elif change == 'wrong_lease':
        p['physical_custody']['old_roots'][1]['closed_lease_sha256'] = '0' * 64
        r['physical_custody_sha256'] = e.sha(e.raw(p['physical_custody']))
    else: p['physical_proof']['new_allocated_bytes'] = -1
    p.update(global_bytes=e.raw(g), request_bytes=e.raw(r)); before = deepcopy(p)
    with pytest.raises(PermissionError): retry.register(**p)
    assert p == before


@pytest.mark.parametrize('change', ['old_row', 'old_marker', 'original_package', 'new_row', 'request_bytes',
    'double_row', 'no_index', 'refund', 'paid_output', 'native_hold', 'guard'])
def test_changed_or_forged_two_row_index_is_rejected(prepared, change):
    g = propose(prepared); row = g['PBS_submissions'][-1]; mark = g[retry.FIELD]
    if change == 'old_row': g['PBS_submissions'][-2]['conservative_CPU_s'] -= 1
    elif change == 'old_marker': g[e.FIELD]['terminal']['native']['conservative_CPU_s'] -= 1
    elif change == 'original_package': g[e.package.FIELD]['registered_requests'].append(row['request_id'])
    elif change == 'new_row': row['cold_input_audit_sha256'] = '0' * 64
    elif change == 'request_bytes': mark['request_sha256'] = '0' * 64
    elif change == 'double_row': g['PBS_submissions'].append(deepcopy(row))
    elif change == 'no_index': g.pop(retry.FIELD)
    elif change == 'refund': g['new_storage_current_bytes'] -= 144294967296
    elif change == 'paid_output': row['paid_output_allowance_bytes'] += 1
    elif change == 'native_hold': mark['native_hold']['optimizer_updates'] = 1024
    else: mark['physical_custody']['paid_output_ceiling'] += 1
    with pytest.raises(PermissionError): retry.validate(g)


def test_local_control_projection_is_explicit_conservative_and_idempotent(prepared):
    g = propose(prepared); local = json.loads(prepared['local_bytes']); before = deepcopy(local)
    original = e.budget_LOCAL(g, local); actual = retry.budget_LOCAL(g, local)
    assert local == before and actual['CPU_s']['local_PC'] == original['CPU_s']['local_PC'] + 600
    assert actual['network_transfer_accounted_bytes'] == original['network_transfer_accounted_bytes'] + 64 * 1024**2
    assert actual.get('Core_native_GPU_current_storage_bytes') == original.get('Core_native_GPU_current_storage_bytes')
    assert g[retry.FIELD]['remaining_resources']['Core_storage_bytes'] == g[e.package.FIELD]['remaining_resources']['Core_storage_bytes']
    assert retry.budget_LOCAL(g, actual) == actual
    assert actual['ARX5_minimal_first512_retry_control_projection']['canonical_LOCAL_write_proven'] is False


def test_prior900_exemption_requires_exact_closed_predecessor_and_new_request(prepared):
    g = bound(prepared); _, row = retry._selected(g)
    assert retry.terminal_CPU_prior(g, row) == 0
    with pytest.raises(PermissionError): retry.terminal_CPU_prior(g, e.rows(g)[0])
    wrong = deepcopy(row); wrong['request_id'] = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc'
    with pytest.raises(PermissionError): retry.terminal_CPU_prior(g, wrong)
    bad = deepcopy(g); bad[e.FIELD]['terminal']['native']['prior_readonly_preparation_CPU_upper_s'] = 0
    with pytest.raises(PermissionError): retry.terminal_CPU_prior(bad, row)


@pytest.mark.parametrize('success', [True, False])
def test_new_terminal_accounts_actual_cpu_once_and_preserves_old_paid_output(prepared, success):
    g = bound(prepared); _, row = retry._selected(g); before = deepcopy(g)
    args = terminal_fixture(g, success=success); out = retry.settle(g, row['request_id'], **args)
    assert g == before and out[e.FIELD] == g[e.FIELD] and out[e.package.FIELD] == g[e.package.FIELD]
    assert out['CPU_s'] == g['CPU_s'] + 1000
    assert out['used']['H200_s'] == g['used']['H200_s'] + 3600
    assert out['new_storage_current_bytes'] == g['new_storage_current_bytes']
    assert out['new_storage_peak_bytes'] == g['new_storage_peak_bytes']
    assert retry.selected(out, row['request_id'])['scientific_endpoint_complete'] is success
    assert retry.forecast_remaining(out) == out[retry.FIELD]['remaining_resources']
    assert out['PBS_submissions'][:-1] == g['PBS_submissions'][:-1]
    with pytest.raises(PermissionError): retry.settle(out, row['request_id'], **args)


@pytest.mark.parametrize('change', ['unknown', 'pending', 'double900', 'wrong_prior', 'wrong_job', 'notF',
    'missing_restore', 'too_many_updates', 'fake_audit', 'wrong_cursor', 'new1024', 'unsafe_native_time',
    'physical_missing_margin', 'physical_overflow', 'physical_old_changed', 'physical_stale', 'physical_before_F'])
def test_retry_terminal_keeps_original_science_and_unknown_fences(prepared, change):
    g = bound(prepared); _, row = retry._selected(g); args = terminal_fixture(g, success=True)
    n, x, a = args['native'], args['effects'], args['audit']
    if change == 'unknown': x['unknown_query_intents'] = [{'preserve': True}]
    elif change == 'pending': g['pending_updates'] = [{'preserve': True}]
    elif change == 'double900': n['conservative_CPU_s'] += 900
    elif change == 'wrong_prior': n['prior_readonly_preparation_CPU_upper_s'] = 900
    elif change == 'wrong_job': n['job_id'] = '1501631.stdct-mgmt-02'
    elif change == 'notF': n['job_state'] = 'R'
    elif change == 'missing_restore': a['two_full_optimizer_scheduler_RNG_sampler_restore_pairs_exact'] = False
    elif change == 'too_many_updates': x['optimizer_updates_committed'] += 1
    elif change == 'fake_audit': a['status'] = 'CPU_PASS'
    elif change == 'wrong_cursor': next(iter(a['endpoints'].values()))['cursor'] = 32768
    elif change == 'new1024': next(iter(a['endpoints'].values()))['global_step'] = 1024
    elif change == 'unsafe_native_time':
        n['finished_epoch'] = row['deadline_epoch'] + 1; n['observed_epoch'] = n['finished_epoch'] + 1
        args['now_epoch'] = n['observed_epoch'] + 1
    elif change == 'physical_missing_margin': args['physical_proof']['next_output_upper_bytes'] = 0
    elif change == 'physical_overflow': args['physical_proof']['new_allocated_bytes'] = row['paid_output_allowance_bytes']
    elif change == 'physical_old_changed': args['physical_proof']['old_inventory_sha256'] = '0' * 64
    elif change == 'physical_stale': args['now_epoch'] += 121
    else: args['physical_observed_epoch'] = n['finished_epoch'] - 1
    a['effect_summary_sha256'] = e.sha(e.raw(x)); before = deepcopy(g)
    with pytest.raises(PermissionError): retry.settle(g, row['request_id'], **args)
    assert g == before
