from copy import deepcopy
import json
import numpy as np
import pytest
from scripts.pi0294.multitask.book_phase_sampling import INDEX_SCHEMA,SLOTS,digest,pools,sequence
from scripts.pi0294.multitask.book_m32_learning import phase_plan,curve_plan,is_plan
from scripts.pi0294.multitask.book_fit import FitSampler
from scripts.pi0294.multitask.book_m32_export import COUNTS


def fixture():
    names=[*[f'BOOK_SOURCE_SEED{i:03}'for i in range(15)],'BOOK_RECOVERY_TRAIN000_EXPERT_SUFFIX',
           *[f'BOOK_M32_RECOVERY{i:03}_EXPERT_SUFFIX'for i in range(5)]]
    index=dict(schema=INDEX_SCHEMA,dataset_manifest_sha256='1'*64,statistics_sha256='2'*64,TRAIN_population_sha256='3'*64,episodes=[])
    rows=[]
    for e,(name,n)in enumerate(zip(names,COUNTS)):
        start=0 if e<15 else 5  # Not local anchor modulo8: regression fixture.
        stages=['approach']*16+['align']*16+['close']*16+['verify_grasp']*8+['lift']*32+['preinsert']*160+['release']*24+['withdraw']*64
        stages+=['hold']*(n-len(stages))
        index['episodes'].append(dict(id=name,windows=n,source_observation_index=list(range(start,start+n)),
            stages=stages,prefix8_closed=[48<=a<240 for a in range(n)],source_planning_indices=list(range(0,n+5,8)),expert_suffix=e>=15))
        rows.extend((dict(id=name,split='TRAIN'),a)for a in range(n))
    p=phase_plan('1'*64,'2'*64,'3'*64,digest(index))
    return rows,index,p


def test_only_sampling_changes_from_matched_control():
    rows,index,p=fixture();u=curve_plan('1'*64,'2'*64,'3'*64)
    changes={k for k in set(p)|set(u)if p.get(k)!=u.get(k)}
    assert changes=={'schema','sampling','purpose','phase_index_schema','phase_index_sha256','phase_slots_per_batch','phase_metadata_student_input'}
    assert is_plan(p)and sum(dict(SLOTS).values())==32
    for key,value in [('seed',0),('max_steps',8192),('initial_delta_checkpoint_manifest_sha256','4'*64),('learning_rate',.002),('phase_metadata_student_input',True)]:
        bad=deepcopy(p);bad[key]=value;assert not is_plan(bad)


def test_full_coverage_episode_balance_deterministic_resume_and_index_at():
    rows,index,p=fixture();sampler=FitSampler(rows,fit_recipe=p,phase_index=index)
    original=list(sampler);assert len(original)==131072 and len({i for c,i in original})==12533
    for cursor in(1,7,31,32,32768,65537,131070):
        resumed=FitSampler(rows,start=cursor,fit_recipe=p,phase_index=index)
        assert next(iter(resumed))==original[cursor] and resumed.index_at(cursor)==original[cursor][1]
        assert resumed.state_dict(cursor)['phase_index_sha256']==p['phase_index_sha256']
    assert list(FitSampler(rows,start=131060,fit_recipe=p,phase_index=index))==original[131060:]
    assert rows==fixture()[0] # Metadata/labels are never added to dataset rows.


def test_real_source_boundaries_closed_prefix_and_final_open_hold():
    rows,index,p=fixture();group=pools(rows,index,p)
    for key in('carry_boundary','carry_other'):
        for name,indices in group[key].items():
            e=next(e for e in index['episodes']if e['id']==name)
            for i in indices:
                r,a=rows[i];assert e['stages'][a]in('lift','preinsert','insert')and e['prefix8_closed'][a]
                assert (e['source_observation_index'][a]in e['source_planning_indices'])==(key=='carry_boundary')
    assert any(rows[i][1]%8!=0 for name,v in group['carry_boundary'].items()if 'SUFFIX'in name for i in v)
    for k,g in group.items():
        if k=='uniform':continue
        assert all(next(e for e in index['episodes']if e['id']==name)['stages'][rows[i][1]]!='hold'for name,v in g.items()for i in v)


def test_no_index_substitution_dev_or_unknown_phase():
    rows,index,p=fixture()
    with pytest.raises(PermissionError):FitSampler(rows,fit_recipe=p)
    bad=deepcopy(index);bad['episodes'][0]['stages'][0]='privileged_failure'
    with pytest.raises(PermissionError):FitSampler(rows,fit_recipe=p,phase_index=bad)
    badrows=deepcopy(rows);badrows[0][0]['split']='DEV'
    with pytest.raises(PermissionError):FitSampler(badrows,fit_recipe=p,phase_index=index)
    probe=phase_plan('1'*64,'2'*64,'3'*64,digest(index),steps=10)
    assert is_plan(probe)and probe['TRAIN_visits']==320 and probe['scheduler_horizon_steps']==4096
    assert probe['capacity_qualification_only']and probe['warmup_updates']==128


def test_native_dataset_uses_row_identities_without_loading_a_batch():
    rows,index,p=fixture()
    class AuditedDataset:
        def __init__(self):self.rows=rows
        def __len__(self):return len(self.rows)
        def __getitem__(self,index):raise AssertionError('SAMPLER_MUST_NOT_READ_MODEL_BATCH')
    actual=FitSampler(AuditedDataset(),fit_recipe=p,phase_index=index)
    expected=FitSampler(rows,fit_recipe=p,phase_index=index)
    assert list(actual)==list(expected)
