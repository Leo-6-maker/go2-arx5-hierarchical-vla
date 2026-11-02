from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scripts.pi0294.multitask import shared_task_sampling as shared


def dataset():
    from scripts.pi0294.multitask.book_m32_export import COUNTS
    counts={'T1':[601]*10,'T2':[601]*10,
        'BOOK':[601]*21+list(COUNTS[15:])+[513,529,497,521,481,497,361],
        'DOOR':[601]*5}
    rows=[]
    for task,values in counts.items():
        for i,windows in enumerate(values):
            row=dict(id=f'{task}_{i:03}',task=task,split='TRAIN',windows=windows,sha256=f'{i:064x}')
            rows.extend((row,a)for a in range(windows))
    assert len(rows)==34563
    return SimpleNamespace(rows=rows,manifest_sha='a'*64)


@pytest.fixture(scope='module')
def full():
    d=dataset();s=shared.TaskBalancedSampler(d);return d,s


def test_actual_population_grain_and_each_full_batch_has_all_tasks(full):
    d,s=full;report=shared.exposure(d.rows,s.values)
    assert report['every_batch_exact_task8']and report['visits']==262144
    assert report['unique_windows']==34563 and report['training_admitted']is False
    assert all(report['tasks'][t]['visits']==65536 for t in shared.TASKS)
    assert sum(r['full_episode']for t in report['tasks'].values()for r in t['episodes'])==46
    assert sum(not r['full_episode']for t in report['tasks'].values()for r in t['episodes'])==13
    for t in report['tasks'].values():
        visits=[e['visits']for e in t['episodes']]
        assert max(visits)-min(visits)<=1
        assert all(e['maximum_window_visits']-e['minimum_window_visits']<=1 for e in t['episodes'])


def test_native10_prefix_and_committed_cursor_resume_keep_all_future_draws(full):
    d,s=full;short=shared.TaskBalancedSampler(d,stop=320)
    assert list(short)==list(s)[:320]and short.sequence_sha256==s.sequence_sha256
    cursor=32777;resumed=shared.TaskBalancedSampler(d,start=cursor)
    assert list(resumed)==list(s)[cursor:]
    assert resumed.state_dict(cursor)==s.state_dict(cursor)
    with pytest.raises(IndexError):resumed.index_at(cursor-1)
    with pytest.raises(ValueError):resumed.state_dict(cursor-1)
    with pytest.raises(IndexError):s.index_at(shared.HORIZON)


def test_sampling_does_not_consume_model_numpy_rng(full):
    d,s=full;before=np.random.get_state();shared.sequence(d.rows,visits=320);after=np.random.get_state()
    assert before[0]==after[0]and np.array_equal(before[1],after[1])and before[2:]==after[2:]


def test_metadata_reorder_preserves_logical_draws_but_changes_bound_row_identity(full):
    d,s=full;reverse=SimpleNamespace(rows=list(reversed(d.rows)),manifest_sha=d.manifest_sha)
    other=shared.TaskBalancedSampler(reverse,stop=320)
    assert other.row_identity_sha256!=s.row_identity_sha256
    assert other.state_dict(0)!=s.state_dict(0)
    for a,b in zip(s.values[:320],other.values[:320]):
        r,i=d.rows[a];q,j=reverse.rows[b];assert(r['id'],i)==(q['id'],j)


@pytest.mark.parametrize('fault',['DEV','task','duplicate','missing','inconsistent','out_of_range'])
def test_invalid_or_incomplete_windows_are_never_silently_dropped(fault):
    d=dataset()
    if fault in('DEV','task','inconsistent'):
        row=deepcopy(d.rows[0][0]);d.rows[0]=(row,0)
        if fault=='DEV':row['split']='DEV'
        elif fault=='task':row['task']='OTHER'
        else:row['windows']=600
    elif fault=='duplicate':d.rows.append(d.rows[0])
    elif fault=='missing':d.rows.pop(0)
    else:d.rows[0]=(d.rows[0][0],-1)
    with pytest.raises(PermissionError):shared.TaskBalancedSampler(d)


def test_new_sampler_cannot_expand_the_finite_horizon_or_negative_range():
    d=dataset()
    for options in(dict(stop=shared.HORIZON+1),dict(start=-1),dict(start=True),dict(seed=True)):
        with pytest.raises(ValueError):shared.TaskBalancedSampler(d,**options)


def test_raw_recipe_keeps_native10_prefix_and_8192_schedule_separate_from_book():
    pins=['1'*64,'2'*64,'3'*64,'4'*64]
    full=shared.recipe(*pins);short=shared.recipe(*pins,steps=10)
    assert shared.is_recipe(full)and shared.is_recipe(short)
    for key in ['initialization','raw_parent_files_identity','effective_batch_size','seed','warmup_updates',
        'scheduler_horizon_steps','learning_rate','weight_decay','sampling','task_slots_per_batch32']:
        assert full[key]==short[key]
    assert full['warmup_updates']==256 and full['checkpoint_steps']==[2048,4096,6144,8192]
    assert full['generation_queries']==80 and short['generation_queries']==32
    assert full['full_expert_episodes']==46 and full['expert_suffix_entries']==13
    for key,value in [('initialization','BOOK_B248'),('effective_batch_size',64),('warmup_updates',128),('state_history_length',3)]:
        bad=dict(full);bad[key]=value;assert not shared.is_recipe(bad)
