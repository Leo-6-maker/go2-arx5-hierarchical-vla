from copy import deepcopy
import pytest
from scripts.pi0294.multitask.book_bc_dagger import plan,is_plan,validate_population,DATA_SCHEMA,POPULATION_SCHEMA
from scripts.pi0294.multitask.book_fit import optimization_rows,FitSampler
from scripts.pi0294.multitask.book_m32_learning import is_plan as is_m32
from scripts.pi0294.multitask.book_m32_export import COUNTS


def entries():
    names=[*[f'BOOK_SOURCE_SEED{i:03}'for i in range(15)],'BOOK_RECOVERY_TRAIN000_EXPERT_SUFFIX',*[f'BOOK_M32_RECOVERY{i:03}_EXPERT_SUFFIX'for i in range(5)]]
    return [dict(id=n,windows=w)for n,w in zip(names,COUNTS)]


def recipe(arm='D0',extra=None):return plan(arm,'1'*64,'2'*64,'3'*64,entries()+([]if extra is None else[extra]))


def test_empty_expansion_does_not_start_nominal_dagger_or_full_demo_training():
    assert is_m32(recipe())and is_plan(recipe())
    for arm in('D1','D2'):
        with pytest.raises(PermissionError,match='NO_NEW'):recipe(arm)
    for extra in(dict(id='BOOK_DEV002',windows=601),dict(id='BOOK_SOURCE_SEED000',windows=601)):
        with pytest.raises(ValueError):recipe('D1',extra)
    with pytest.raises(ValueError):recipe('D2',dict(id='BOOK_GEOMETRY_TRAIN001',windows=601))


def test_matched_arms_only_change_real_population_keep_parent_stats_batch_and_schedule():
    from scripts.pi0294.multitask.source_export import source_identity
    from scripts.pi0294.multitask.book_geometry_cards import geometry_card
    exported=source_identity(geometry_card(1),dict(native_LIBERO_goal_final=True))
    ps=[recipe(),recipe('D1',dict(id=exported['episode_id'],windows=601)),
        recipe('D2',dict(id='BOOK_LATE_RECOVERY_TRAIN000_EXPERT_SUFFIX',windows=401))]
    keys=('initial_delta_checkpoint_manifest_sha256','statistics_sha256','learning_rate','weight_decay','max_steps','warmup_updates','scheduler_horizon_steps','seed','effective_batch_size','limited_DiT_blocks','diagnostic_windows','sampling')
    assert all(all(p[k]==ps[0][k]for k in keys)for p in ps)
    for p in ps:
        assert is_m32(p)and p['max_steps']==4096 and p['checkpoint_steps']==[1024,2048,4096]
        changed=deepcopy(p);changed['learning_rate']=1e-3;assert not is_m32(changed)


def test_train_only_real_population_and_sampler_resume_coverage():
    p=recipe();rows=[(dict(id=e['id'],split='TRAIN',task='BOOK'),i)for e in p['TRAIN_entries']for i in range(e['windows'])]
    assert len(optimization_rows(rows,p))==12533
    bad=deepcopy(rows);bad[0][0]['split']='DEV'
    with pytest.raises(PermissionError):optimization_rows(bad,p)
    class D:
        def __len__(self):return len(rows)
    a=list(FitSampler(D(),fit_recipe=p,start=31,stop=35))
    assert a==list(FitSampler(D(),fit_recipe=p,start=0,stop=35))[31:]
    manifest=dict(schema=DATA_SCHEMA,arm='D0',training_admitted=True,statistics_sha256='2'*64,TRAIN_windows=12533,
        episodes=[dict(e,split='TRAIN',task='BOOK',sha256='a'*64,raw_root='saved/'+e['id'])for e in p['TRAIN_entries']])
    population=dict(schema=POPULATION_SCHEMA,arm='D0',statistics_sha256='2'*64,DEV_TEST_confirm_admitted=False,
        episodes=[dict(e,sha256='a'*64,raw_root='saved/'+e['id'])for e in p['TRAIN_entries']])
    assert validate_population(manifest,population,p)
    for key,value in [('training_admitted',False),('statistics_sha256','4'*64),('arm','D2')]:
        bad=deepcopy(manifest);bad[key]=value
        with pytest.raises(PermissionError):validate_population(bad,population,p)
    bad=deepcopy(population);bad['episodes'][0]['sha256']='b'*64
    with pytest.raises(PermissionError,match='PROVENANCE'):validate_population(manifest,bad,p)
