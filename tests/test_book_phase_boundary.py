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


def recipes(rows,index,pins,steps=4096):
    entries=[dict(id=e['id'],windows=e['windows']) for e in index['episodes']]
    return [pb.plan(variant,pins['dataset_sha256'],pins['statistics_sha256'],
        pins['population_sha256'],pins['index_sha256'],entries,steps=steps)
        for variant in ('CONTROL','BOUNDARY')]


def test_recipe_pair_freezes_all_science_except_two_sampling_slots():
    from scripts.pi0294.multitask.book_bc_dagger import plan as data_plan
    from scripts.pi0294.multitask.book_m32_learning import is_plan
    rows,index,pins=fixture();control,boundary=recipes(rows,index,pins)
    assert is_plan(control) and is_plan(boundary)
    assert {k for k in control if control[k]!=boundary[k]}=={'pb_variant','phase_slots_per_batch'}
    base=data_plan('D2',pins['dataset_sha256'],pins['statistics_sha256'],pins['population_sha256'],control['TRAIN_entries'])
    allowed={'schema','sampling','purpose','pb_variant','phase_index_schema',
             'phase_index_sha256','phase_slots_per_batch','phase_metadata_student_input'}
    assert all(control[k]==v for k,v in base.items() if k not in allowed)
    for k,value in [('warmup_updates',200),('learning_rate',3e-5),('loss','CHANGED'),('effective_batch_size',64)]:
        bad=deepcopy(boundary);bad[k]=value
        assert not is_plan(bad)


def test_actual_fit_sampler_qualification_prefix_resume_and_population():
    from scripts.pi0294.multitask.book_fit import FitSampler,optimization_rows
    rows,index,pins=fixture();_,p=recipes(rows,index,pins)
    _,probe=recipes(rows,index,pins,steps=10)
    assert optimization_rows(rows,p)==rows
    sampler=FitSampler(rows,fit_recipe=p,phase_index=index)
    actual=list(sampler)
    expected=pb.sequence(rows,index,boundary=True,seed=p['seed'],**pins)
    np.testing.assert_array_equal([i for _,i in actual],expected)
    assert list(FitSampler(rows,fit_recipe=probe,phase_index=index))==actual[:320]
    cursor=65536
    resumed=FitSampler(rows,start=cursor,stop=cursor+32,fit_recipe=p,phase_index=index)
    assert list(resumed)==actual[cursor:cursor+32]
    assert resumed.state_dict(cursor)['pb_variant']=='BOUNDARY'
    assert resumed.state_dict(cursor)['phase_index_sha256']==pins['index_sha256']
    assert sampler.index_at(cursor)==expected[cursor]
    with pytest.raises(PermissionError):
        FitSampler(rows,fit_recipe=p)
    with pytest.raises(PermissionError):
        optimization_rows(rows[:-1],p)


def test_checkpoint_requires_exact_index_and_intermediate_restore_name():
    from scripts.pi0294.multitask.book_checkpoint import validate_delta_header,qualification_file_name
    rows,index,pins=fixture();p,_=recipes(rows,index,pins)
    members={name:'a'*64 for name in ('target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth',
        'trainer_state.json','statistics.json','SAMPLER_STATE.json','FROZEN_PARENT_TENSORS.json',
        'ADAPTER_PARAMETER_CONTRACT.json','TRAIN_DEV_SEAL.json','TRAIN_POPULATION.json')}
    members['TRAIN_PHASE_INDEX.json']=pins['index_sha256']
    h=dict(schema='BOOK_M32_PARENT_BOUND_TRAINING_DELTA_V1',recipe=p,global_step=1024,cursor=32768,
        standalone_model=False,full_resume_with_required_parent=True,
        required_parent_checkpoint_manifest_sha256=p['checkpoint_manifest_sha256'],files=members)
    assert validate_delta_header(h,p['checkpoint_manifest_sha256'])==h
    name='ACTUAL_DELTA_CHECKPOINT_RESTORE_1024.json'
    assert qualification_file_name(dict(delta_native_qualification_file=name),h)==name
    with pytest.raises(PermissionError):
        qualification_file_name({},h)
    bad=deepcopy(h);bad['files']['TRAIN_PHASE_INDEX.json']='b'*64
    with pytest.raises(PermissionError):
        validate_delta_header(bad,p['checkpoint_manifest_sha256'])
