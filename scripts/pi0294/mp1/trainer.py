"""Reuse the measured official E1 Trainer with MP1's explicit data seed."""
from scripts.pi0294.native_train import MeasuredLoader
from scripts.pi0294.e1.trainer import make_trainer_class as e1_trainer


def make_trainer_class():
    import torch

    class MP1Trainer(e1_trainer()):
        def get_train_dataloader(self):
            self.args.ignore_data_skip=True;self.start_cursor=self.consumed=self.state.global_step*32
            self.sampler=self.train_dataset.make_sampler(self.train_dataset,start=self.consumed,stop=self.args.max_steps*32,seed=self.args.data_seed)
            self.measured_loader=MeasuredLoader(torch.utils.data.DataLoader(self.train_dataset,
                batch_size=self._train_batch_size,sampler=self.sampler,collate_fn=self.data_collator,
                num_workers=0,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(self.args.data_seed)))
            return self.measured_loader

    return MP1Trainer
