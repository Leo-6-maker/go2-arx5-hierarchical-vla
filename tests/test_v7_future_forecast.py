"""The old E2 forecast must distinguish project caps from real site capacity."""
from copy import deepcopy
import base64
import gzip
import json
from pathlib import Path

import pytest

from scripts.pi0294.generalization import minimal_adoption as a, minimal_core_accounting as c
from scripts.pi0294.test_minimal_core_runtime import fixture
from scripts.pi0294.test_minimal_core_v7_successor import v7_fixture


def test_policy_requires_complete_validated_row_not_a_flag():
    assert not a.v7_project_policy({'execution_policy_sha256': c.V7_POLICY})
    _, _, _, _, legacy, _ = fixture()
    assert not a.v7_project_policy({c.FIELD: {'legacy': legacy}})
    _, r = v7_fixture()
    row = deepcopy(legacy)
    row.update(r, request=deepcopy(r), request_sha256=c.digest(r), child_run_uuid=r['run_uuid'])
    assert a.v7_project_policy({c.FIELD: {'new': row}})
    row['optimizer_updates'] = 1
    with pytest.raises(PermissionError):
        a.v7_project_policy({c.FIELD: {'new': row}})


def test_exact_current_E2_forecast_retains_deficit_without_new_GO():
    p = Path(__file__).resolve().parents[1].parent / 'private/minimal-adaptation-20261027/MAINTENANCE30_CONTROL_IO'
    if not (p / 'E2_HELD_NATIVE_AND_ORIGINAL_LOCAL.private.json').exists():
        pytest.skip('Private actual E2 captured LOCAL not provided')
    gb = gzip.decompress(base64.b64decode(json.loads((p / 'E1_FIXED512_CONTROLIO/V7_PANEL08/COMPLETE24_FINAL_INDEPENDENT_READ.private.json').read_bytes())['GLOBAL_gzip_base64']))
    lb = gzip.decompress(base64.b64decode(json.loads((p / 'E2_HELD_NATIVE_AND_ORIGINAL_LOCAL.private.json').read_bytes())['LOCAL_gzip_base64']))
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    g = json.loads(gb)
    before = deepcopy(g)
    v = first512_forecast(gb, lb)
    assert v['V7_project_aggregate_caps_retired']
    assert v['cap_changes_required'] == {}
    assert set(v['retired_project_cap_changes']) == {'Hopper_storage_bytes'}
    d = v['dimensions']['Hopper_storage_bytes']
    assert d['projected'] == 1542239209720
    assert d['shortfall'] == 40074397944 and d['final'] == 40 * 1024 ** 3
    assert d['matrix'] == 9442971836 and not d['project_cap_is_approval_gate']
    assert v['dimensions']['optimizer_updates']['projected'] == 15936
    assert v['native_admission'] is False and before == g
