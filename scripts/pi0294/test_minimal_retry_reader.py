"""Original reader integration; captured history plus labelled future fixtures.

No check/forecast/ledger validator is mocked. Request/PBS/physical proofs in
``prepared`` are future CPU fixtures, not registered native facts. The later
D25 failure regression reads its captured complete original ledger bytes and
must remain rejected for missing full Eval escrow, without modifying them.
"""
import base64
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.pi0294.generalization import minimal_execution as original
from scripts.pi0294.generalization import minimal_retry_execution as retry
from scripts.pi0294.generalization import minimal_package as package, minimal_scope as scope
from scripts.pi0294.pv1 import mix_vanda_first, mix_core_recovery
from scripts.pi0294.pv1.neural_ledger import check
from scripts.pi0294.dec1 import a40_training_context as context, a40_training_runtime as runtime
from scripts.pi0294.test_minimal_retry_execution import (
    captured_evidence, evidence, prepared, propose, bound, terminal_fixture)


def test_original_reader_accepts_two_explicit_rows_without_rewriting_old_history(prepared):
    g = propose(prepared); before = deepcopy(g)
    assert check(g) is g
    assert package.validate(g) == g[package.FIELD]
    assert original.rows(g) == [g[retry.FIELD]['history']['original_row']]
    assert len(retry.rows(g)) == 2
    assert retry.selected(g, g[retry.FIELD]['request_id'])['platform_training_version'] == retry.VERSION
    assert g == before


@pytest.mark.parametrize('success', [False, True])
def test_original_reader_accepts_bound_and_original_science_terminal_with_postF_proof(prepared, success):
    g = bound(prepared); assert check(g) is g
    row = retry.selected(g, g[retry.FIELD]['request_id'])
    value = retry.settle(g, row['request_id'], **terminal_fixture(g, success=success))
    before = deepcopy(value)
    assert check(value) is value
    proof = scope.first512_forecast(original.raw(value), prepared['local_bytes'])
    assert proof['additional'] == retry.forecast_remaining(value)
    assert proof['additional'] == value[retry.FIELD]['remaining_resources']
    assert value[retry.FIELD]['status'] == ('COMPLETE' if success else 'INCOMPLETE')
    assert value[original.FIELD] == g[original.FIELD]
    assert value[package.FIELD] == g[package.FIELD]
    assert value == before


def test_old_global_reader_and_forecast_remain_exact_without_retry_marker(prepared):
    g = json.loads(prepared['global_bytes']); before = deepcopy(g)
    assert check(g) is g
    assert package.validate(g) == g[package.FIELD]
    proof = scope.first512_forecast(prepared['global_bytes'], prepared['local_bytes'])
    assert proof['additional'] == original.forecast_remaining(g)
    projected = mix_vanda_first.budget_LOCAL(g, json.loads(prepared['local_bytes']))
    assert 'ARX5_minimal_first512_retry_control_projection' not in projected
    assert g == before


def test_complete_forecast_counts_retry_compute_and_control_without_old_output_refund(prepared):
    before = scope.first512_forecast(prepared['global_bytes'], prepared['local_bytes'])
    g = propose(prepared)
    after = scope.first512_forecast(original.raw(g), prepared['local_bytes'])
    assert len(after['dimensions']) == 14
    assert after['additional'] == retry.forecast_remaining(g)
    assert after['additional'] != original.forecast_remaining(g)
    expected = dict(H200_s=59520-3600, Hopper_CPU_s=165600,
        optimizer_updates=1028, neural_queries=147)
    for key, prior in before['dimensions'].items():
        assert after['dimensions'][key]['projected'] - prior['projected'] == expected.get(key, 0), key
    assert after['cap_changes_required'] == {}
    assert after['native_admission'] is False and after['LOCAL_is_historical'] is True
    assert g[retry.FIELD]['history']['original_paid_output_bytes'] == 144294967296
    assert g['new_storage_current_bytes'] >= json.loads(prepared['global_bytes'])['new_storage_current_bytes']


def test_original_LOCAL_projection_accounts_each_control_once_keeps_canonical_input_unchanged(prepared):
    g = propose(prepared); local = json.loads(prepared['local_bytes']); saved = deepcopy(local)
    recovered = mix_core_recovery.budget_LOCAL(g, local)
    old = original.budget_LOCAL(g, recovered)
    projected = mix_vanda_first.budget_LOCAL(g, local)
    assert projected == retry.budget_LOCAL(g, recovered)
    assert projected['CPU_s']['local_PC'] - old['CPU_s']['local_PC'] == 600
    assert projected['network_transfer_accounted_bytes'] - old['network_transfer_accounted_bytes'] == 64*1024**2
    assert retry.budget_LOCAL(g, projected) == projected
    assert projected['ARX5_minimal_first512_retry_control_projection']['canonical_LOCAL_write_proven'] is False
    assert local == saved


@pytest.mark.parametrize('mutation', ['missing_index', 'missing_history', 'old_row_relabelled',
    'retry_relabelled_old', 'retry_unknown_version', 'missing_native_platform', 'old_row_removed',
    'duplicate_retry', 'history_digest', 'paid_output_refund', 'CPU_hold', 'GPU_hold',
    'query_hold', 'update_hold', 'original_package', 'physical_preflight', 'UNKNOWN'])
def test_original_GLOBAL_reader_rejects_incomplete_two_row_context(prepared, mutation):
    g = propose(prepared); mark = g[retry.FIELD]; row = g['PBS_submissions'][-1]
    if mutation == 'missing_index': del g[retry.FIELD]
    elif mutation == 'missing_history': del mark['history']
    elif mutation == 'old_row_relabelled': original.rows(g)[0]['platform_training_version'] = retry.VERSION
    elif mutation == 'retry_relabelled_old': row['platform_training_version'] = original.VERSION
    elif mutation == 'retry_unknown_version': row['platform_training_version'] = retry.VERSION + '_OTHER'
    elif mutation == 'missing_native_platform': del row['native_platform']
    elif mutation == 'old_row_removed':
        g['PBS_submissions'].remove(original.rows(g)[0]); g['used']['PBS_submissions'] -= 1
    elif mutation == 'duplicate_retry':
        g['PBS_submissions'].append(deepcopy(row)); g['used']['PBS_submissions'] += 1
    elif mutation == 'history_digest': mark['history']['original_row_sha256'] = '0'*64
    elif mutation == 'paid_output_refund': g['new_storage_current_bytes'] -= 144294967296
    elif mutation == 'CPU_hold': g['CPU_reserved_s'] -= 165600
    elif mutation == 'GPU_hold': g['reserved']['H200_s'] -= 59520
    elif mutation in ('query_hold', 'update_hold'):
        key = 'neural_queries' if mutation == 'query_hold' else 'optimizer_updates'
        # Captured history may legitimately hold extra escrow. A negative
        # must cross the combined required floor, not merely subtract147.
        g['reserved'][key] = (retry.forecast_remaining(g)[key]
            + g[original.original.PACKAGE]['remaining_resources'][key] - 1)
    elif mutation == 'original_package': g[package.FIELD]['registered_requests'].append(row['request_id'])
    elif mutation == 'physical_preflight': mark['physical_preflight']['new_allocated_bytes'] += 1
    elif mutation == 'UNKNOWN': g['unknown_effects'] = [{'unchanged': True}]
    before = deepcopy(g)
    with pytest.raises((PermissionError, KeyError)): check(g)
    assert g == before


@pytest.mark.parametrize('key', ['unknown_effects', 'pending_updates'])
def test_full_future_reader_keeps_effect_fences_even_if_row_shape_is_valid(prepared, key):
    g = propose(prepared); g[key] = [{'unchanged': True}]; before = deepcopy(g)
    with pytest.raises(PermissionError): scope.first512_forecast(original.raw(g), prepared['local_bytes'])
    assert g == before


def test_actual_later_D25_failure_still_requires_full_original_panel_hold(evidence):
    root = Path(os.environ['ARX5_RETRY_ACTUAL_CAPTURE_DIRECTORY'])
    captured = json.loads((root/'D25_RETRY02_NETWORK_TERMINAL_INDEPENDENT_DUAL_READBACK.private.json').read_bytes())
    gb = gzip.decompress(base64.b64decode(captured['global']['GLOBAL_gzip_base64']))
    lb = gzip.decompress(base64.b64decode(captured['local']['LOCAL_gzip_base64']))
    assert original.sha(gb) == captured['global']['GLOBAL_sha256']
    assert original.sha(lb) == captured['local']['LOCAL_sha256']
    g = json.loads(gb); before = deepcopy(g)
    assert check(g) is g  # Authentic closed history, but not permission to retry.
    with pytest.raises(PermissionError, match='EXACT_ORIGINAL_C_D_COMPUTE_HOLDS_RETAINED'):
        scope.first512_forecast(gb, lb)
    assert g == before and retry.FIELD not in g


@pytest.mark.parametrize('adapter', [original, retry])
def test_typed_profile_selects_exact_adapter_and_preserves_CPU_prior(adapter):
    assert context.minimal_training_adapter(adapter.VERSION, adapter.PROFILE) is adapter
    r = adapter.fixed_request_fields()
    assert r['prior_readonly_preparation_CPU_upper_s'] == (0 if adapter is retry else 900)
    assert r['CPU_upper_s'] == 165600


@pytest.mark.parametrize('version,profile', [(retry.VERSION, original.PROFILE),
    (original.VERSION, retry.PROFILE), (retry.VERSION, None),
    (original.VERSION, None), ('UNKNOWN', retry.PROFILE), (None, original.PROFILE)])
def test_typed_profile_cannot_alias_another_scope(version, profile):
    with pytest.raises(PermissionError, match='MINIMAL'): context.minimal_training_adapter(version, profile)


@pytest.mark.parametrize('capacity', [(8, 64), (36, 240)])
def test_original_PBS_reader_accepts_explicit_retry_capacity_with_full_request(prepared, monkeypatch, capacity):
    r = json.loads(prepared['request_bytes']); cpu, mem = capacity
    job_id, host = '987654.stdct-mgmt-02', 'GN-A40-001'
    limits = dict(select=f'1:ncpus={cpu}:mem={mem}gb:ngpus=1:gpu_model=A40',
        ncpus=cpu, mem=f'{mem}gb', ngpus=1, walltime='16:30:00')
    job = dict(job_state='R', Job_Owner='e1192548@login', Job_Name=r['job_name'],
        project=r['project'], Rerunable=False, run_count=1,
        Variable_List=dict(PV1_REQUEST_ID=r['request_id'], PV1_SUBMISSION_NONCE=r['submission_nonce']),
        exec_host=host+'/0', Resource_List=limits)
    monkeypatch.setenv('PBS_JOBID', job_id); monkeypatch.setenv('USER', 'e1192548')
    monkeypatch.setattr(runtime.socket, 'gethostname', lambda:host)
    def read_only_job(argv, **kw):
        assert argv == ['/opt/pbs/bin/qstat', '-f', '-F', 'json', job_id]
        assert kw == dict(capture_output=True, text=True, timeout=25, check=True)
        return SimpleNamespace(stdout=json.dumps(dict(Jobs={job_id:job})))
    # Only the unavailable site process/output is a fixture. Original own-job,
    # request and capacity validators all run; no permission file is produced.
    monkeypatch.setattr(runtime.subprocess, 'run', read_only_job)
    assert runtime.native_job(r) == (job_id, host, job)
    assert retry.process_CPU_limits(r)['prior_readonly_preparation_CPU_upper_s'] == 0
    bad = deepcopy(r); bad['prior_readonly_preparation_CPU_upper_s'] = 900
    with pytest.raises(PermissionError): runtime.native_job(bad)
    job['Resource_List']['walltime'] = '08:00:00'
    with pytest.raises(PermissionError): runtime.native_job(r)
