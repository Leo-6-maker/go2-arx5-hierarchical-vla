"""Real CPU DataLoader regression; tiny data is not PV1 native qualification."""
from types import SimpleNamespace
from unittest.mock import patch
import pickle,random
import pytest
from . import trainer as c

class TinyDataset:
    def __len__(self):return 200
    def __getitem__(self,index):
        import torch
        return dict(index=index,value=torch.tensor([index/8,index/16],dtype=torch.float32))
    def make_sampler(self,dataset,*,start,stop,seed):
        assert seed==20261019
        return range(start,stop)

class TinyTrainer:
    """Only the upstream constructor is a CPU double, not the loader."""
    pass

def loader(workers):
    import torch
    with patch.object(c,'e1_trainer',return_value=TinyTrainer):cls=c.make_trainer_class()
    obj=cls();obj.args=SimpleNamespace(data_seed=20261019,max_steps=4,dataloader_num_workers=workers,dataloader_pin_memory=False)
    obj.state=SimpleNamespace(global_step=2);obj._train_batch_size=2;obj.train_dataset=TinyDataset()
    obj.data_collator=torch.utils.data.default_collate
    return obj,obj.get_train_dataloader()

def test_worker2_is_really_started_and_has_identical_ordered_batches_and_resume_cursor():
    torch=pytest.importorskip('torch');import numpy as np
    torch.manual_seed(98765);before=torch.get_rng_state().clone()
    python_before=pickle.dumps(random.getstate());numpy_before=pickle.dumps(np.random.get_state())
    baseline,b0=loader(0);parallel,b2=loader(2)
    assert b0.loader.num_workers==0 and b2.loader.num_workers==2
    assert b2.loader.multiprocessing_context.get_start_method()=='spawn' and b2.loader.prefetch_factor==2
    rows0=list(b0);rows2=list(b2)
    assert len(rows0)==len(rows2)==32
    assert all(torch.equal(a['index'],b['index']) and torch.equal(a['value'],b['value']) for a,b in zip(rows0,rows2))
    assert torch.equal(torch.get_rng_state(),before)
    assert pickle.dumps(random.getstate())==python_before and pickle.dumps(np.random.get_state())==numpy_before
    assert baseline.consumed==parallel.consumed==64
    assert baseline.start_cursor==parallel.start_cursor==2*32
    assert b2.batches==32 and b2.wait_s>=0
    # A resumed getter reconstructs the original64 cursor, regardless of
    # how far the earlier iterator prefetched or yielded without an optimizer.
    resumed=parallel.get_train_dataloader();first=next(iter(resumed))
    assert first['index'].tolist()==[64,65] and parallel.consumed==64

def test_legacy_arguments_leave_zero_workers_and_do_not_add_prefetch_to_zero():
    assert c.loader_options(SimpleNamespace())==dict(num_workers=0,pin_memory=True,drop_last=True,persistent_workers=False)
    v=c.loader_options(SimpleNamespace(dataloader_num_workers=4,dataloader_pin_memory=True))
    assert v['num_workers']==4 and v['prefetch_factor']==2 and v['worker_init_fn'] is c._worker_threads

@pytest.mark.parametrize('bad',[True,False,-1,1,3,8,36,2.,'2',None])
def test_unbounded_or_ambiguous_worker_settings_reject_before_constructing_loader(bad):
    with pytest.raises(PermissionError):c.loader_options(SimpleNamespace(dataloader_num_workers=bad))

@pytest.mark.parametrize('bad',[1,0,'true',None])
def test_pin_flag_is_explicit(bad):
    with pytest.raises(PermissionError):c.loader_options(SimpleNamespace(dataloader_num_workers=2,dataloader_pin_memory=bad))
