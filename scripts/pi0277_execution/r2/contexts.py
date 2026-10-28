"""Make the original mode context reusable across retained-graph recomputation."""
def reusable(factory):
    def create(*args,**kwargs):
        class Context:
            def __init__(self):self.stack=[]
            def __enter__(self):
                current=factory(*args,**kwargs);self.stack.append(current)
                return current.__enter__()
            def __exit__(self,*error):return self.stack.pop().__exit__(*error)
        return Context()
    return create

def check():
    from contextlib import contextmanager
    from types import SimpleNamespace
    import torch
    from torch.utils.checkpoint import checkpoint
    mode=SimpleNamespace(training=True)
    @contextmanager
    def original():
        previous=mode.training;mode.training=False
        try:yield
        finally:mode.training=previous
    def rollout(factory,probe):
        weight=torch.tensor(.37,dtype=torch.float64,requires_grad=True);x=torch.tensor(.61,dtype=torch.float64);states=[]
        def step(x,w):assert mode.training is False;return torch.tanh(w*x)
        for _ in range(4):
            x=x+.25*checkpoint(step,x,weight,use_reentrant=False,preserve_rng_state=True,context_fn=lambda:(factory(),factory()));states.append(x)
        assert mode.training is True
        loss=(x+.23).square()
        if probe:
            norms=torch.autograd.grad(loss,states,retain_graph=True);assert all(g.isfinite().all() and g.abs().sum()>0 for g in norms)
        loss.backward();assert mode.training is True
        return x.detach(),weight.grad
    reference=rollout(original,False)
    try:rollout(original,True)
    except AttributeError:pass
    else:raise AssertionError('Original one-use context did not reproduce the defect')
    repaired=rollout(reusable(original),True)
    assert all(torch.equal(a,b) for a,b in zip(reference,repaired))
    return dict(status='PASS_RETAINED_GRAPH_CONTEXT_AND_EXACT_GRADIENT',new_model_queries=0,optimizer_updates=0,mode_restored=True)

if __name__=='__main__':
    import json;print(json.dumps(check()))
