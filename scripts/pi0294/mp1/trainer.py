"""Reuse the measured official E1 Trainer with MP1's explicit data seed."""
from scripts.pi0294.native_train import MeasuredLoader
from scripts.pi0294.e1.trainer import make_trainer_class as e1_trainer


def _worker_threads(worker_id):
    """Bound each spawned CPU reader; preprocessing stays deterministic."""
    import torch
    torch.set_num_threads(1)


def loader_options(arguments):
    """Respect an explicitly qualified caller's worker count; legacy stays0."""
    workers=getattr(arguments,'dataloader_num_workers',0)
    if type(workers) is not int or workers not in (0,2,4):
        raise PermissionError('MP1_EXPLICIT_BOUNDED_CPU_LOADER_WORKERS')
    options=dict(num_workers=workers,pin_memory=getattr(arguments,'dataloader_pin_memory',True),
        drop_last=True,persistent_workers=False)
    if type(options['pin_memory']) is not bool:
        raise PermissionError('MP1_EXPLICIT_PIN_MEMORY_FLAG')
    if workers:options.update(multiprocessing_context='spawn',prefetch_factor=2,worker_init_fn=_worker_threads)
    return options


def make_trainer_class():
    import torch

    class MP1Trainer(e1_trainer()):
        def get_train_dataloader(self):
            self.args.ignore_data_skip=True;self.start_cursor=self.consumed=self.state.global_step*32
            self.sampler=self.train_dataset.make_sampler(self.train_dataset,start=self.consumed,stop=self.args.max_steps*32,seed=self.args.data_seed)
            self.measured_loader=MeasuredLoader(torch.utils.data.DataLoader(self.train_dataset,
                batch_size=self._train_batch_size,sampler=self.sampler,collate_fn=self.data_collator,
                **loader_options(self.args),generator=torch.Generator().manual_seed(self.args.data_seed)))
            return self.measured_loader

    return MP1Trainer
