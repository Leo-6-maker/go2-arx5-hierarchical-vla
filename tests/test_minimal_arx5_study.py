from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.pi0294.generalization import minimal_study as s


def fixture_manifest():
    rows = []
    for family in ('F0', 'F1', 'F2'):
        for geometry in range(3):
            for task in ('T1', 'T2'):
                group = f'PI0294/TRAIN/{family}/{geometry:03d}'
                rid = f'PI0294-TRAIN-{family}-{geometry:03d}-{task}'
                row = dict(id=rid, group=group, task=task, family=family, windows=601,
                           split='TRAIN', origin='new', first_attempt=True)
                row.update({k: hashlib.sha256(f'{rid}:{k}'.encode()).hexdigest() for k in
                            ('sha256', 'raw_FILE_MANIFEST_sha256', 'samples_sha256', 'physics_sha256', 'source_sha')})
                rows.append(row)
    return dict(schema='PI0294_FROZEN_MULTITASK_NATIVE_V1', frozen=True, fixture=False,
                official_source_sha=s.UPSTREAM, statistics_sha256=s.R0, RAW_DATA_READY=True, episodes=rows)


def panel(m):
    data = s.raw(m)
    return s.paired_panel(data, expected_manifest_sha256=hashlib.sha256(data).hexdigest())


def test_six_geometry_twelve_parent_selection_is_order_independent_and_balanced():
    m = fixture_manifest()
    a = panel(m)
    m['episodes'].reverse()
    b = panel(m)
    assert a['parents'] == b['parents']
    assert len(a['parents']) == 12 and a['overlapping_windows'] == 7212
    assert [r['family'] for r in a['parents'][::2]] == ['F0', 'F1', 'F2', 'F0', 'F1', 'F2']
    assert a['native_bank_qualified'] is False


def test_all_epoch_windows_consumed_once_and_resume_is_same_cursor_suffix():
    p = panel(fixture_manifest())
    rows = list(s.paired_cursors(p))
    assert len(rows) == 32768
    assert len({(r['parent'], r['anchor']) for r in rows[:7212]}) == 7212
    assert rows[:16384] == list(s.paired_cursors(p, stop=16384))
    assert rows[16384:] == list(s.paired_cursors(p))[16384:]
    summary = s.stream_summary(p)
    assert sum(summary['pilot_exposure'].values()) == 512 * 32
    assert sum(summary['full_exposure'].values()) == 1024 * 32
    assert summary['consumed_by_native_training'] == 0


@pytest.mark.parametrize('field,value', [('fixture', True), ('frozen', False), ('statistics_sha256', 'changed')])
def test_wrong_input_manifest_rejected(field, value):
    m = fixture_manifest(); m[field] = value
    with pytest.raises(ValueError, match='ORIGINAL_FROZEN'):
        panel(m)


def test_wrong_byte_identity_rejected():
    with pytest.raises(ValueError, match='EXACT_SAVED'):
        s.paired_panel(s.raw(fixture_manifest()), expected_manifest_sha256='0' * 64)


def test_dev_payload_or_geometry_cannot_leak_to_selected_training_pool():
    m = fixture_manifest()
    bad = deepcopy(m['episodes'][0]); bad['id'] = 'DEV-OTHER-ID'; bad['split'] = 'DEV'
    m['episodes'].append(bad)
    with pytest.raises(ValueError, match='NO_DEV_LINEAGE'):
        panel(m)


def test_incomplete_task_pairs_do_not_count_as_extra_geometries():
    m = fixture_manifest()
    m['episodes'] = [r for r in m['episodes'] if r['task'] != 'T2']
    with pytest.raises(ValueError, match='THREE_FAMILIES'):
        panel(m)


def test_actual_frozen_twelve_parent_metadata_reproduces_published_cursor_hashes():
    root = Path(__file__).resolve().parents[1]
    captured = json.loads((root / 'reports/PI0294/design/parallel-science-20261022/'
        'ARX5_MINIMAL_ADAPTATION_TWELVE_PARENT_AND_CURSOR_METADATA_FROZEN_20261027.json').read_text())
    p = captured['panel']
    assert s.digest(p) == 'd603324fee6cc69f5b488b47cfaf13dd4bc1feb8d8dc6411a1a739822cfff74d'
    assert {r['family'] for r in p['parents']} == set(s.FAMILIES)
    assert s.stream_summary(p) == captured['paired_cursor']
    assert captured['new_training_updates'] == 0 and p['native_bank_qualified'] is False
