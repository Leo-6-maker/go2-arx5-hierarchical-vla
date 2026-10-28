"""Captured historical ledgers, pure proposals; no native training evidence."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import pytest
from scripts.pi0294.pv1 import mix_vanda_first as f, mix_accounting as a
from scripts.pi0294.pv1 import successor_cd as s, mix_core_accounting as core


@pytest.fixture
def captured():
    base = Path(os.environ.get('PI0294_PRIVATE_WORKSPACE', '/home/liuyu/ISSAC-GR00T'))
    p = base/'work/private/v5-execution-20261025/C50_PREMODEL_REPAIR_ACTUAL_FULL_BUDGET_CPU_PREPARED.private.json'
    if not p.exists():
        pytest.skip('private actual closed failure ledgers unavailable')
    v = json.loads(p.read_bytes())
    authority = Path(__file__).resolve().parents[2]/'reports/PI0294/design/parallel-science-20261022/V5_D25_VANDA_FIRST_DIRECT_AUTHORIZATION_20261026.json'
    marker = v['proposal']['marker']
    return dict(global_bytes=base64.b64decode(v['GLOBAL_base64']),
        local_bytes=base64.b64decode(v['LOCAL_base64']),
        authority_bytes=authority.read_bytes(),
        closure_capture_bytes=a.raw(v['closure_capture']),
        metadata_capture_bytes=a.raw(v['metadata_capture']),
        replacement_request_id=marker['replacement_request_id'],
        replacement_submission_nonce=marker['replacement_submission_nonce'],
        now_epoch=time.time())


def test_original_order_still_rejects_early_D_without_authority(captured):
    g = json.loads(captured['global_bytes'])
    assert s.execution_order(g) == s.ORDER
    with pytest.raises(PermissionError):
        a.prepay_training_control(g, arm='D25', now_epoch=time.time())


def test_actual_closed_boundary_changes_only_execution_order_and_existing_repair(captured):
    original = deepcopy(captured)
    out = f.propose(**captured)
    g, l = json.loads(captured['global_bytes']), json.loads(captured['local_bytes'])
    v = out['proposed_global']
    assert captured == original
    assert out['actual_LOCAL_changed'] is False
    assert s.execution_order(v) == f.ORDER
    assert v[s.FIELD] == g[s.FIELD] == l[s.FIELD]
    for name in ('used', 'CPU_s', 'new_storage_current_bytes', 'new_storage_peak_bytes',
                 'PBS_submissions', a.CORE_ROWS, 'optimizer_updates', 'generation_queries'):
        assert v[name] == g[name]
    assert out['full_remaining_proof']['shortfalls'] == {}
    assert v[a.PACKAGE]['completed_stages'] == ['TRAIN:C50:1024']
    assert v[a.PACKAGE]['endpoints'] == g[a.PACKAGE]['endpoints']
    assert v[a.PACKAGE]['remaining_resources']['optimizer_updates'] == 1024
    assert v[a.PACKAGE]['remaining_resources']['H200_s'] == 2*28860+22560
    assert core.before_stage('C50', v) == list(f.ORDER[:2])
    assert core.before_stage('D25', v) == list(f.ORDER[:3])


def test_prepaid_D_and_deferred_LOCAL_projection_conserve_transfer_and_CPU(captured):
    v = f.propose(**captured)['proposed_global']
    l = json.loads(captured['local_bytes'])
    old = f.budget_LOCAL(v, l)
    new = a.prepay_training_control(v, arm='D25', now_epoch=time.time())
    projected = f.budget_LOCAL(new, l)
    assert projected['network_transfer_accounted_bytes'] == old['network_transfer_accounted_bytes']+256*1024**2
    assert projected['CPU_s']['local_PC'] == old['CPU_s']['local_PC']+15
    assert projected['CPU_reserved_s']['local_PC'] == old['CPU_reserved_s']['local_PC']-15
    assert projected['CPU_s']['Hopper'] == old['CPU_s']['Hopper']+1800
    assert not s.recompute(new, l, Core_storage_upper_bytes=f.CORE_STORAGE,
                           local_storage_upper_bytes=f.LOCAL_STORAGE)['shortfalls']
    with pytest.raises(PermissionError):
        a.prepay_training_control(new, arm='D25', now_epoch=time.time())
    with pytest.raises(PermissionError):
        core.paired(new, projected)


@pytest.mark.parametrize('case', range(7))
def test_scope_or_snapshot_substitution_and_repeat_rejected(captured, case):
    p = deepcopy(captured)
    if case == 0:
        p['authority_bytes'] += b' '
    elif case == 1:
        p['local_bytes'] += b' '
    elif case == 2:
        p['now_epoch'] = json.loads(p['global_bytes'])[s.FIELD]['stop_new_training_epoch']
    elif case == 3:
        g = json.loads(p['global_bytes']);g[a.PACKAGE]['endpoints']['C50']['checkpoint_manifest_sha256'] = '0'*64;p['global_bytes'] = a.raw(g)
    elif case == 4:
        g = f.propose(**p)['proposed_global'];p['global_bytes'] = a.raw(g)
    elif case == 5:
        g = json.loads(p['global_bytes']);g[a.CORE_ROWS][next(iter(g[a.CORE_ROWS]))]['query_attempts'] = 1;p['global_bytes'] = a.raw(g)
    else:
        g = json.loads(p['global_bytes']);g['unknown_effects'] = ['unresolved'];p['global_bytes'] = a.raw(g)
    with pytest.raises((PermissionError, KeyError, ValueError)):
        f.propose(**p)


def test_invalid_order_and_forged_snapshot_marker_rejected(captured):
    g = f.propose(**captured)['proposed_global']
    for change in ({'physical_order': list(s.ORDER)}, {'confirmed_LOCAL_sha256': '0'*64},
                   {'no_Core_admission_from_snapshot': False}, {'authority_sha256': '1'*64}):
        bad = deepcopy(g);bad[f.FIELD].update(change)
        with pytest.raises(PermissionError):
            a.validate_global(bad)


def test_Core_sync_cannot_invent_unfinished_D25_endpoint(captured):
    g = f.propose(**captured)['proposed_global']
    with pytest.raises((PermissionError, KeyError, TypeError)):
        f.reconcile_LOCAL(a.raw(g), captured['local_bytes'], D25_terminal_capture={})
