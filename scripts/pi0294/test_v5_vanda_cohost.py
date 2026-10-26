"""Original captured endpoints/budgets and isolated resource-fence regressions."""
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import pytest
from scripts.pi0294.pv1 import mix_vanda_cohost as c, mix_accounting as a
from scripts.pi0294.pv1 import mix_vanda_first as f, successor_cd as s


@pytest.fixture
def captured():
    workspace = Path(os.environ.get('PI0294_PRIVATE_WORKSPACE', '/home/liuyu/ISSAC-GR00T'))
    base = workspace/'work/private/v5-execution-20261025'
    paths = sorted(base.glob('D25_VANDA_FIRST_REAL_CLOSED_HANDOFF_READONLY_*'))
    if not paths:
        pytest.skip('private full actual terminal bytes unavailable')
    d = json.loads(paths[-1].read_bytes())['value']['closed_capture']
    cc = json.loads(sorted(base.glob('C50_REAL_CLOSED_HANDOFF_READONLY_*'))[-1].read_bytes())['value']['closed_capture']
    prior = json.loads((base/'C50_PREMODEL_REPAIR_ACTUAL_FULL_BUDGET_CPU_PREPARED.private.json').read_bytes())
    auth = Path(__file__).resolve().parents[2]/'reports/PI0294/design/parallel-science-20261022/V5_CD_VANDA_COLOCATED_EVAL_DIRECT_AUTHORIZATION_20261026.json'
    return dict(global_bytes=base64.b64decode(d['global_base64']),
                historical_LOCAL_bytes=base64.b64decode(prior['LOCAL_base64']),
                authority_bytes=auth.read_bytes(), captures={'C50': cc, 'D25': d}, now_epoch=time.time())


def test_original_resource_partition_retains_raw_and_fits_gross():
    p = c.storage_contract()
    assert p['physical_per_attempt_bytes'] == 768*1024**2
    assert p['attempts'] == 26 and p['query_calls'] == 2080
    assert p['total_bytes'] == 41607495680 < p['original_gross_upper_bytes'] == 42641095480
    assert p['model_payload_copies'] == 0
    assert p['raw_artifacts_deleted_or_truncated'] is False


def test_captured_proposal_retains_every_historical_cost_and_all_future_work(captured):
    old = deepcopy(captured)
    before = json.loads(captured['global_bytes'])
    out = c.propose(**captured); after = out['proposed_global']
    assert captured == old
    assert out['actual_LOCAL_changed'] is out['native_admission'] is out['native_IO_performed'] is False
    for k in ('used', 'CPU_s', 'PBS_submissions', 'new_storage_current_bytes', 'new_storage_peak_bytes',
              'storage_reserved_bytes', a.CORE_ROWS, s.FIELD, f.FIELD):
        assert after[k] == before[k]
    remaining = before[a.PACKAGE]['remaining_resources']
    planned = after[a.PACKAGE]['remaining_resources']
    assert {k:planned[k]-v for k,v in remaining.items() if planned[k] != v} == {
        'PBS_submissions': 3, 'Hopper_CPU_s': 189000}
    assert out['full_remaining_proof']['shortfalls'] == {}
    assert out['full_remaining_proof']['dimensions']['PBS_submissions']['projected'] == 34
    assert out['full_remaining_proof']['dimensions']['Hopper_CPU_s']['projected'] < 213*3600
    assert c.validated(after)['closed_endpoints']['D25']['global_sha256'] == a.sha(captured['global_bytes'])
    assert c.validated(after)['closed_endpoints']['C50']['global_sha256'] == a.sha(captured['global_bytes'])


@pytest.mark.parametrize('kind', ['authority','LOCAL','repeat','unknown','active','missing_D','wrong_arm','late','false_success'])
def test_no_implicit_authority_stale_endpoint_or_replay(captured, kind):
    p = deepcopy(captured)
    if kind == 'authority': p['authority_bytes'] += b' '
    elif kind == 'LOCAL': p['historical_LOCAL_bytes'] += b' '
    elif kind == 'repeat': p['global_bytes'] = a.raw(c.propose(**p)['proposed_global'])
    elif kind in ('unknown','active'):
        g = json.loads(p['global_bytes'])
        if kind == 'unknown': g['unknown_effects'] = ['unresolved']
        else: g['PBS_submissions'][-1]['status'] = 'SUBMITTED_PENDING_NATIVE'
        p['global_bytes'] = a.raw(g)
    elif kind == 'missing_D': p['captures'].pop('D25')
    elif kind == 'wrong_arm': p['captures']['D25'] = p['captures']['C50']
    elif kind == 'late': p['now_epoch'] = json.loads(p['global_bytes'])[s.FIELD]['deadline_epoch']-28860
    else:
        d = p['captures']['D25']; raw = d['terminal_capture_base64']['OS_exit_bytes']
        v = json.loads(base64.b64decode(raw)); v['returncode'] = 1
        d['terminal_capture_base64']['OS_exit_bytes'] = base64.b64encode(a.raw(v)).decode()
    with pytest.raises((PermissionError, KeyError, ValueError)):
        c.propose(**p)


def test_storage_guard_counts_hardlinks_once_without_changing_inputs(tmp_path):
    p = tmp_path/'one'; p.write_bytes(b'a'*1000); old = p.stat()
    os.link(p, tmp_path/'two')
    measured = c.enforce_output(tmp_path, 8192)
    assert measured['logical_bytes'] == 1000 and measured['unique_inodes'] == 1
    assert p.stat().st_mtime_ns == old.st_mtime_ns and p.read_bytes() == b'a'*1000


def test_output_overflow_stops_and_preserves_full_raw(tmp_path):
    p = tmp_path/'raw'; p.write_bytes(b'actual-raw'*1024)
    before = p.read_bytes()
    with pytest.raises(PermissionError, match='PRESERVE_ALL'):
        c.enforce_output(tmp_path, 4096)
    assert p.read_bytes() == before


def test_symlink_and_mountlike_path_escape_rejected(tmp_path):
    (tmp_path/'escape').symlink_to('/tmp', target_is_directory=True)
    with pytest.raises(PermissionError): c.tree_bytes(tmp_path)
    with pytest.raises(PermissionError): c.tree_bytes(tmp_path, exclude=['/etc'])
