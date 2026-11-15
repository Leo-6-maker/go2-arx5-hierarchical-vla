import copy
import hashlib
import json
from collections import Counter
import pytest

from go2arx5_vla.sla_adapter.mixed_visits_v4 import Catalog, MixedVisits, visit_record


def catalogs():
    rows = [dict(split='TRAIN', task=f'task{t}', episode=f'ep{e}', phase=p, anchor=a)
            for t in range(2) for e in range(3)
            for p, a in (('approach', 0), ('grasp', 10), ('release', 20))]
    return {d: Catalog(d, digit * 64, rows) for d, digit in (('T', 'a'), ('S_cmd', 'b'))}


def test_m75_exact_quota_and_partition_independent_metadata():
    s = MixedVisits('M75_3K', catalogs(), seed=20261113)
    counts = Counter()
    for _ in range(1000):
        rows = s.next_batch()
        assert len(rows) == 32
        assert Counter(v.domain for v in rows) == {'T': 24, 'S_cmd': 8}
        assert tuple(v for i in range(0, 32, 8) for v in rows[i:i+8]) == rows
        counts.update(v.domain for v in rows)
    assert counts == {'T': 24000, 'S_cmd': 8000}
    assert s.state_dict()['domain_visits'] == counts


def test_four_updates_equal_independent_two_plus_two_and_new_state_load():
    c = catalogs()
    continuous = MixedVisits('M75_3K', c, seed=17)
    expected = [continuous.next_batch() for _ in range(4)]
    first = MixedVisits('M75_3K', c, seed=17)
    observed = [first.next_batch() for _ in range(2)]
    state = json.loads(json.dumps(first.state_dict()))
    fresh = MixedVisits('M75_3K', c, seed=17)
    fresh.restore(state)
    observed.extend(fresh.next_batch() for _ in range(2))
    assert observed == expected
    assert fresh.state_dict() == continuous.state_dict()


def bank_hashes(arm, c):
    s = MixedVisits(arm, c, seed=42)
    bank = {'T': hashlib.sha256(), 'S_cmd': hashlib.sha256()}
    for _ in range(3000):
        rows = s.next_batch()
        # Per-domain ordinal recovers the bank independently of batch shuffle.
        for row in sorted(rows, key=lambda r: (r.domain, r.within_domain_visit)):
            bank[row.domain].update(json.dumps(visit_record(row), sort_keys=True).encode())
    return {d: h.hexdigest() for d, h in bank.items()}, s.state_dict()


def test_m75_and_seq_share_samples_and_random_tags_with_different_order():
    c = catalogs()
    mixed, m = bank_hashes('M75_3K', c)
    sequential, s = bank_hashes('SEQ_3K', c)
    assert mixed == sequential
    assert m['domain_visits'] == s['domain_visits'] == {'T': 72000, 'S_cmd': 24000}


def test_restore_rejects_other_dataset_seed_and_exposure_cursor():
    c = catalogs()
    s = MixedVisits('M75_3K', c, seed=9)
    s.next_batch()
    for key, value in (('seed', 10), ('cursor', 2), ('catalogs', {})):
        wrong = copy.deepcopy(s.state_dict());wrong[key] = value
        with pytest.raises(ValueError):s.restore(wrong)


def test_source_aux_and_nontrain_cannot_become_command_catalogs():
    with pytest.raises(ValueError):Catalog('S_aux', 'a'*64, [])
    bad = [dict(split='DEV', task='t', episode='e', phase='p', anchor=0)]
    with pytest.raises(ValueError):Catalog('S_cmd', 'b'*64, bad)
    bad[0]['split'] = 'TRAIN'
    with pytest.raises(ValueError):Catalog('S_cmd', 'b'*64, bad * 2)


def test_source_only_requires_no_target_catalog_and_tail_preserves_prefix():
    c = catalogs()
    with pytest.raises(ValueError):MixedVisits('S_ONLY_3K', c, seed=1)
    only = MixedVisits('S_ONLY_3K', {'S_cmd': c['S_cmd']}, seed=1)
    assert {v.domain for v in only.next_batch()} == {'S_cmd'}
    a = MixedVisits('M75_3K', c, seed=2, updates=3000)
    b = MixedVisits('M75_3K', c, seed=2, updates=4000)
    assert a.batch(0) == b.batch(0) and a.batch(2999) == b.batch(2999)
