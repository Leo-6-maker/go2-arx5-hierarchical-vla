"""CPU regression of the actual direct-get_action hook bypass, no native claim."""
from types import SimpleNamespace
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from torch import nn
from scripts.pi0294.generalization import minimal_worker as worker
from scripts.pi0294.generalization import minimal_inference_export as export


class Head(nn.Module):
    def __init__(self):
        super().__init__();self.shared=nn.Dropout(.7)

    def get_action(self):
        # Pinned upstream: process shared visual/language features BEFORE the
        # initial randn and call head.get_action directly, not head.__call__.
        self.shared(torch.ones(96))
        return torch.randn(1,40,132)


class Graph(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Identity();self.action_head=Head()
        self.action_head.register_forward_pre_hook(lambda module,args:module.eval())

    def get_action(self):
        self.backbone(torch.zeros(1));return self.action_head.get_action()


def test_old_probe_train_restore_changes_noise_despite_same_seed():
    model=Graph().eval();torch.manual_seed(19);before=model.get_action()
    model.train(True)  # Exact original probes.py finally behavior.
    torch.manual_seed(19);after=model.get_action()
    assert model.action_head.training  # Direct method bypassed the eval hook.
    assert not torch.equal(before,after)


@pytest.mark.parametrize('wrapped',[False,True])
def test_worker_inference_draw_enforces_complete_eval_before_seed(tmp_path,monkeypatch,wrapped):
    graph=Graph().train(True);instance=SimpleNamespace(model=graph,parameter_contract=object()if wrapped else None)
    w=worker.Worker.__new__(worker.Worker);w.root=tmp_path;w.evidence=tmp_path/'evidence';w.instance=instance;w.ledger=object();w.guard=lambda:None
    def draw(root,sid,name,inst,payload,seed,target,ledger):
        assert all(not m.training for m in inst.model.modules())
        torch.manual_seed(seed);arr=inst.model.get_action().numpy()
        with target.open('xb')as f:np.savez(f,normalized=arr,physical_H40=arr[0,:,:10],noise_seed=np.asarray(seed,dtype=np.uint64),noise_fingerprint_json=np.asarray('same_cpu_fixture'),observation_fingerprint=np.asarray('same_input'))
    monkeypatch.setattr(worker.original_model,'fixed_draw',draw)
    reference=w.draw('S','reference',{},19);graph.train(True)
    reloaded=w.draw('S','reloaded',{},19)
    assert export.compare_saved_draws(reference,reloaded)['new_queries']==0
    assert reference.stat().st_mode&0o777==0o400
    with pytest.raises(PermissionError):w.draw('S','reference',{},19)
