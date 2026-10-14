"""Original native training plus one read-only pre-update full-resume receipt."""
import argparse
import hashlib
import pickle
import random
from pathlib import Path

from scripts.pi0292.durable import read,write
from . import native_train,neural


def install_observer(root,checkpoint):
    original=native_train.make_trainer_class
    def factory():
        from transformers import TrainerCallback
        import numpy as np
        import torch
        from safetensors import safe_open
        Native=original()
        class Observed(Native):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                trainer=self
                class ResumeObserver(TrainerCallback):
                    seen=False
                    def on_step_begin(self,arguments,state,control,**kw):
                        if self.seen:return
                        self.seen=True
                        if state.global_step!=512 or trainer.consumed!=16384:
                            raise ValueError('R3_ORIGINAL_STEP_AND_CONSUMED_CURSOR_NOT_RESTORED')
                        tensors=trainer.model.state_dict();checked=0
                        for file in sorted(checkpoint.glob('*.safetensors')):
                            with safe_open(str(file),framework='pt',device='cpu') as archive:
                                for name in archive.keys():
                                    expected=archive.get_tensor(name);actual=tensors[name].detach().cpu()
                                    if expected.dtype!=actual.dtype or not torch.equal(expected,actual):
                                        raise ValueError('R3_MODEL_RESUME_TENSOR_CHANGED:'+name)
                                    checked+=1
                        if checked!=len(tensors):raise ValueError('R3_MODEL_RESUME_KEY_COVERAGE')
                        expected=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
                        optimizer_equal=native_train.tree_identity(expected)==native_train.tree_identity(trainer.optimizer.state_dict())
                        del expected
                        expected_scheduler=torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=False)
                        scheduler_equal=native_train.tree_identity(expected_scheduler)==native_train.tree_identity(trainer.lr_scheduler.state_dict())
                        saved=torch.load(checkpoint/'rng_state.pth',map_location='cpu',weights_only=False)
                        digest=lambda x:hashlib.sha256(pickle.dumps(x)).hexdigest()
                        cuda=saved['cuda'] if isinstance(saved['cuda'],list) else [saved['cuda']]
                        rng_equal=(digest(random.getstate())==digest(saved['python'])
                            and digest(np.random.get_state())==digest(saved['numpy'])
                            and torch.equal(torch.get_rng_state(),saved['cpu'])
                            and len(cuda)==len(torch.cuda.get_rng_state_all())
                            and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())))
                        if not(optimizer_equal and scheduler_equal and rng_equal):
                            raise ValueError('R3_ACTUAL_OPTIMIZER_SCHEDULER_OR_RNG_RESTORE_MISMATCH')
                        write(root/'evidence/FULLSTATE_RESTORE.json',dict(
                            status='PASS_ACTUAL_PRE_UPDATE_MODEL_ADAM_SCHEDULER_RNG_SAMPLER',
                            run_uuid=root.name,global_step=512,consumed_samples=16384,
                            model_state_tensors_compared=checked,optimizer_equal=optimizer_equal,
                            scheduler_equal=scheduler_equal,rng_equal=rng_equal,
                            sampler=trainer.sampler.state_dict(trainer.consumed),
                            updates_before_receipt=0,original_native_training_unchanged=True,
                            audit_consumed_no_RNG=True,selected='NONE',deployment='NONE'))
                self.add_callback(ResumeObserver())
        return Observed
    native_train.make_trainer_class=factory


def main():
    p=argparse.ArgumentParser()
    for name in ('root','request','bindings','resume'):p.add_argument('--'+name,required=True)
    a=p.parse_args();a.mode='main';a.microbatch=2;a.workers=0;a.max_steps=4096;a.stop_at=2048
    root=Path(a.root);request=read(a.request)
    if request['mode']!='main' or request['resume']!=a.resume or request['stop_at']!=2048:
        raise PermissionError('R3_EXACT_FULLSTATE_OFFLINE_REQUEST_REQUIRED')
    install_observer(root,Path(a.resume))
    neural.run(a)


if __name__=='__main__':main()
