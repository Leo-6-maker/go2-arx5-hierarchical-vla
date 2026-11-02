"""The proposed pair changes two slots, including overlapping pools."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.pi0294.multitask import book_phase_boundary as pb
from scripts.pi0294.multitask.book_m32_export import COUNTS
from scripts.pi0294.multitask.book_phase_sampling import digest


def fixture():
    ids = [*[f'BOOK_SOURCE_SEED{i:03}' for i in range(15)],
           'BOOK_RECOVERY_TRAIN000_EXPERT_SUFFIX',
           *[f'BOOK_M32_RECOVERY{i:03}_EXPERT_SUFFIX' for i in range(5)]]
    entries = dict(zip(ids, COUNTS))
    entries.update({f'BOOK_LATE_RECOVERY_TRAIN{i:03}_EXPERT_SUFFIX': n
                    for i,n in enumerate((513,529,497,521,481,497,361))})
    index = dict(schema=pb.SCHEMA, dataset_manifest_sha256='1'*64,
                 statistics_sha256='2'*64, TRAIN_population_sha256='3'*64, episodes=[])
    rows = []
    for name,n in sorted(entries.items()):
        late = name.startswith('BOOK_LATE_RECOVERY_')
        phases = (['preinsert']*100 if late else ['approach']*50+['align']*50)
        phases += ['close']*20+['verify_grasp']*20+['lift']*50+['preinsert']*70
        phases += ['insert']*40+['release']*20+['withdraw']*20
        phases = (phases+['hold']*n)[:n]
        source = list(range(601-n,601))
        index['episodes'].append(dict(id=name, windows=n, expert_suffix='EXPERT_SUFFIX' in name,
            new_late_takeover=late, source_observation_index=source, stages=phases,
            prefix8_closed=[s not in ('release','withdraw','hold') for s in phases],
            source_planning_indices=list(range(0,640,8))))
        rows.extend((dict(id=name, split='TRAIN', task='BOOK'),a) for a in range(n))
    pins = dict(index_sha256=digest(index), dataset_sha256='1'*64,
                statistics_sha256='2'*64, population_sha256='3'*64)
    return rows,index,pins


def test_pair_keeps_specialty_streams_identical_and_audits_overlap():
    rows,index,pins = fixture()
    groups = pb.pools(rows,index,**pins)
    before = np.random.get_state()
    control, cs = pb.sequence(rows,index,boundary=False,seed=2026103105,return_streams=True,**pins)
    boundary, bs = pb.sequence(rows,index,boundary=True,seed=2026103105,return_streams=True,**pins)
    assert len(control)==len(boundary)==131072
    assert cs['uniform'].shape==(4096,16) and bs['uniform'].shape==(4096,14)
    assert bs[pb.LATE].shape==(4096,2) and pb.LATE not in cs
    for name in cs:
        if name!='uniform':
            np.testing.assert_array_equal(cs[name],bs[name])
    late = {i for values in groups[pb.LATE].values() for i in values}
    assert len(late)==112
    assert sum(int(i) in late for i in boundary)>8192  # Overlap is counted.
    assert {rows[int(i)][0]['id'] for i in bs[pb.LATE].flat}==set(groups[pb.LATE])
    counts = {name:sum(rows[int(i)][0]['id']==name for i in bs[pb.LATE].flat)
              for name in groups[pb.LATE]}
    assert max(counts.values())-min(counts.values())<=1
    assert len(set(bs[pb.LATE].flat))==112
    assert set(control)==set(boundary)==set(range(15932))
    np.testing.assert_array_equal(before[1],np.random.get_state()[1])
    np.testing.assert_array_equal(boundary[32768:32785],
        pb.sequence(rows,index,boundary=True,seed=2026103105,**pins)[32768:32785])


def test_metadata_does_not_change_rows_and_hold_is_not_carry():
    rows,index,pins = fixture();original = deepcopy(rows)
    groups = pb.pools(rows,index,**pins)
    metadata = {e['id']:e for e in index['episodes']}
    for pool in ('carry_boundary','carry_other'):
        assert all(metadata[rows[i][0]['id']]['stages'][rows[i][1]] in ('lift','preinsert','insert')
                   for values in groups[pool].values() for i in values)
    assert not any(name.startswith('BOOK_LATE_RECOVERY_') for name in groups['early_recovery'])
    assert rows==original


def test_sealed_identity_split_and_unique_window_join():
    rows,index,pins = fixture()
    bad = deepcopy(index);bad['episodes'][0]['stages'][0]='hold'
    with pytest.raises(PermissionError):
        pb.pools(rows,bad,**pins)
    badrows = deepcopy(rows);badrows[0][0]['split']='DEV'
    with pytest.raises(PermissionError):
        pb.pools(badrows,index,**pins)
    with pytest.raises(ValueError):
        pb.pools([rows[0],*rows[:-1]],index,**pins)
    with pytest.raises(ValueError):
        pb.sequence(rows,index,boundary=True,seed=1,updates=8192,**pins)
