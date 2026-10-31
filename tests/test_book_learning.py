import pytest
from scripts.pi0294.multitask.book_learning import recipe,BookSampler


def test_new_book_recipe_is_distinct_and_preserves_controlled_comparison():
    r=recipe()
    assert r['parent']=='QUALIFIED_E1_512_NOT_RAW_OR_E2'
    assert r['original_E1_E2_recipes_data_R0_unchanged'] and not r['DEV_for_optimization']
    assert r['max_steps']==128 and r['per_device_train_batch_size']*r['gradient_accumulation_steps']==32
    assert r['shared_DiT_frozen'] and r['checkpoint_steps']==[128]


def test_finite_training_sampler_exact_absolute_cursor_and_resume():
    data=list(range(3005));full=list(BookSampler(data));suffix=list(BookSampler(data,start=64))
    assert full[64:]==suffix and [c for c,_ in full]==list(range(4096))
    assert set(i for _,i in full[:3005])==set(range(3005))
    sampler=BookSampler(data,start=64)
    for c,i in suffix[::127]:assert sampler.index_at(c)==i
    with pytest.raises(IndexError):sampler.index_at(4096)
