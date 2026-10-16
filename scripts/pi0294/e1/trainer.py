"""SMALL8 loader and checkpoint instrumentation around the unchanged native loss."""
import hashlib
import json
from pathlib import Path
import shutil

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from .dataset import SmallSampler


def learning_rate_multiplier(step,total=1024,warmup=32):
    import math
    if step<warmup:return step/warmup
    return .1+.9*(1+math.cos(math.pi*min(1.,(step-warmup)/(total-warmup))))/2


def check_trace(directory,dataset,start,stop):
    actual={}
    for path in Path(directory).glob('worker-*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line);cursor=row['cursor']
            if start<=cursor<stop:
                if cursor in actual:raise ValueError('DUPLICATE_SMALL8_CONSUMED_CURSOR')
                actual[cursor]=row
    sampler=SmallSampler(dataset,start=start,stop=stop)
    for cursor,index in sampler:
        if actual.get(cursor)!=dict(cursor=cursor,index=index,**dataset.identity(index)):
            raise ValueError('ACTUAL_SMALL8_CONSUMED_IDENTITY_MISMATCH')
    return dict(status='PASS_ACTUAL_SMALL8_CONSUMED_IDENTITIES',count=stop-start,
        sha256=hashlib.sha256(json.dumps([actual[c] for c in range(start,stop)],sort_keys=True).encode()).hexdigest())


def make_trainer_class():
    import torch
    from gr00t.experiment.trainer import Gr00tTrainer
    from scripts.pi0294.native_train import make_trainer_class as original,MeasuredLoader

    class SmallTrainer(original()):
        def get_train_dataloader(self):
            self.args.ignore_data_skip=True;self.start_cursor=self.consumed=self.state.global_step*32
            self.sampler=SmallSampler(self.train_dataset,start=self.consumed,stop=self.args.max_steps*32)
            self.measured_loader=MeasuredLoader(torch.utils.data.DataLoader(self.train_dataset,
                batch_size=self._train_batch_size,sampler=self.sampler,collate_fn=self.data_collator,
                num_workers=0,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(20261016)))
            return self.measured_loader

        def train(self,resume_from_checkpoint=None,**kwargs):
            if resume_from_checkpoint:
                checkpoint=Path(resume_from_checkpoint);saved=read(checkpoint/'SAMPLER_STATE.json');state=read(checkpoint/'trainer_state.json')
                expected=SmallSampler(self.train_dataset,stop=self.args.max_steps*32).state_dict(state['global_step']*32)
                if saved['sampler']!=expected or saved['recipe']!=self.recipe:raise ValueError('EXACT_E1_RESUME_SAMPLER_RECIPE')
            return Gr00tTrainer.train(self,resume_from_checkpoint=resume_from_checkpoint,**kwargs)

        def create_scheduler(self,num_training_steps,optimizer=None):
            if self.lr_scheduler is None:
                self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer,
                    lambda step:learning_rate_multiplier(step,self.args.max_steps,self.recipe['warmup_updates']))
                self._created_lr_scheduler=True
            return self.lr_scheduler

        def _save_checkpoint(self,model,trial):
            if self.consumed!=self.state.global_step*32:raise ValueError('E1_OPTIMIZER_SAMPLE_CURSOR_MISMATCH')
            # Only the terminal parent needs optimizer state; 256 is inference-only.
            self.args.save_only_model=self.state.global_step==256
            Gr00tTrainer._save_checkpoint(self,model,trial)
            self.args.save_only_model=False
            checkpoint=Path(self.args.output_dir)/('checkpoint-'+str(self.state.global_step))
            if sha(self.train_dataset.root/'statistics.json')!=self.train_dataset.statistics_sha:raise ValueError('E1_FROZEN_STATS_CHANGED')
            self.train_dataset.processor.save_pretrained(checkpoint/'processor')
            shutil.copyfile(self.train_dataset.root/'statistics.json',checkpoint/'statistics.json')
            trace=check_trace(self.train_dataset.trace_root,self.train_dataset,self.start_cursor,self.consumed)
            write(checkpoint/'SAMPLER_STATE.json',dict(sampler=self.sampler.state_dict(self.consumed),recipe=self.recipe,
                manifest_sha256=self.train_dataset.manifest_sha,statistics_sha256=self.train_dataset.statistics_sha,worker_trace=trace))
            files={str(p.relative_to(checkpoint)):sha(p) for p in sorted(checkpoint.rglob('*')) if p.is_file()}
            required=['config.json','statistics.json','trainer_state.json','SAMPLER_STATE.json']
            if self.state.global_step==1024:required+=['optimizer.pt','scheduler.pt','rng_state.pth']
            if any(n not in files for n in required) or not any(n.endswith('.safetensors') for n in files):
                raise ValueError('E1_ACTUAL_CHECKPOINT_MEMBERS_INCOMPLETE')
            write(checkpoint/'CHECKPOINT_COMPLETE.json',dict(global_step=self.state.global_step,files=files,
                full_resume=self.state.global_step==1024,exact_resume_from_M2048=False))
    return SmallTrainer
