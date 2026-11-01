from copy import deepcopy
import pytest
from scripts.pi0294.multitask.book_gpu_capacity import GIB, REQUEST, entry_requirement


def test_all_existing_training_and_dev_keep_original_24gib_entry():
    assert entry_requirement({}) == 24*GIB
    assert entry_requirement(dict(kind='V7_BOOK_M32_FINITE_LEARNING',recipe=dict(max_steps=4096))) == 24*GIB


def test_finite_native_capacity_test_does_not_admit_model_or_silently_reduce_training():
    m=dict(kind='V7_BOOK_M32_FINITE_LEARNING',GPU_capacity_qualification=deepcopy(REQUEST),
        recipe=dict(max_steps=10,capacity_qualification_only=True,limited_DiT_blocks=2,TRAIN_visits=320))
    assert entry_requirement(m)==13*GIB
    for key,value in [('max_steps',4096),('limited_DiT_blocks',0),('TRAIN_visits',10),('capacity_qualification_only',False)]:
        bad=deepcopy(m);bad['recipe'][key]=value
        with pytest.raises(PermissionError):entry_requirement(bad)
    bad=deepcopy(m);bad['GPU_capacity_qualification']['microbatch']=2
    with pytest.raises(PermissionError):entry_requirement(bad)
