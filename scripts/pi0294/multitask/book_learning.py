"""First finite book-only learning run using the existing native GR00T Trainer.

This is an E1-warmstarted new-task experiment, not either controlled E1/E2 arm.
The native owner registers and seals its inputs before calling this module.
"""
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time

from scripts.pi0292.train_adapter import CyclicSampler

PARENT = '383ad9b8fc3b81efffde132d4f06de92fb96fbdf9ec0b6c92c097e0f58eb2a8d'
DATA = '47163b091f60c6d30e49b50699177727f8909e2100533fa4bc4555b8431c010d'
STATS = '060ff05d4c34205593e234d43a5a76bd5c2305171289489d51fb78b68122e096'
SEAL = '3d5c4a9ad1a94b6c68f38644e1b9a6dc67f7735806667851049ac250b9cae1cf'


def recipe():
    return dict(schema='BOOK5_E1_WARMSTART_TARGET_ONLY128_V1',
        parent_checkpoint_manifest_sha256=PARENT, parent='QUALIFIED_E1_512_NOT_RAW_OR_E2',
        dataset_manifest_sha256=DATA, statistics_sha256=STATS, TRAIN_DEV_seal_sha256=SEAL,
        TRAIN_episodes=5, TRAIN_windows=3005, DEV_conditions=10, DEV_for_optimization=False,
        trainable='SEVEN_TARGET_CATEGORY10_ENCODER_DECODER_SLICES', shared_DiT_frozen=True,
        backbone_frozen=True, target_category=10, state_dropout_prob=0.,
        max_steps=128, scheduler_horizon_steps=128, warmup_updates=8,
        per_device_train_batch_size=1, gradient_accumulation_steps=32, effective_batch_size=32,
        learning_rate=1e-4, weight_decay=1e-5, adam_beta1=.9, adam_beta2=.999,
        adam_epsilon=1e-8, max_grad_norm=1., optimizer='adamw_torch',
        bf16=True, tf32=True, seed=2026103101, dataloader_num_workers=0,
        sampling='NUMPY_PCG64_SEED_PLUS_CYCLE_PERMUTATION_V1',
        loss='UNCHANGED_OFFICIAL_MASKED_FLOW_MATCHING', H40=40, prefix=8,
        checkpoint_steps=[128], retain_all_checkpoints=True,
        reason='First independent book learning/DEV on shared GPU0; 1.303B shared-action AdamW is not assumed to fit alongside the existing GPU user.',
        original_E1_E2_recipes_data_R0_unchanged=True)


class BookSampler(CyclicSampler):
    def __init__(self, dataset, *, start=0, stop=4096):
        self.dataset = dataset
        super().__init__(len(dataset), seed=recipe()['seed'], start=start, stop=stop, positioned=True)

    def index_at(self, cursor):
        import numpy as np
        if not self.start <= cursor < self.stop:
            raise IndexError('BOOK_TRAIN_CURSOR_OUTSIDE_FINITE_RUN')
        cycle, offset = divmod(cursor, self.size)
        return int(np.random.Generator(np.random.PCG64(self.seed+cycle)).permutation(self.size)[offset])


def make_trainer_class(contract, guard):
    import torch
    from gr00t.experiment.trainer import Gr00tTrainer
    from scripts.pi0292.durable import read, write
    from scripts.pi0294.native_train import make_trainer_class as native, MeasuredLoader
    from scripts.pi0294.e1.trainer import learning_rate_multiplier, check_trace
    from scripts.pi0294.native_dataset import sha

    class BookTrainer(native()):
        def get_train_dataloader(self):
            self.start_cursor = self.consumed = self.state.global_step*32
            self.sampler = BookSampler(self.train_dataset, start=self.consumed)
            self.measured_loader = MeasuredLoader(torch.utils.data.DataLoader(self.train_dataset,
                batch_size=1, sampler=self.sampler, collate_fn=self.data_collator,
                num_workers=0, pin_memory=True, drop_last=True,
                generator=torch.Generator().manual_seed(self.recipe['seed'])))
            return self.measured_loader

        def create_optimizer(self):
            contract.validate()
            result = super().create_optimizer()
            actual = [p for group in self.optimizer.param_groups for p in group['params']]
            if {id(p) for p in actual} != {id(p) for p in contract.parameters()}:
                raise ValueError('BOOK_OPTIMIZER_EXACT_DECLARED_CATEGORY_SLICES')
            return result

        def create_scheduler(self, num_training_steps, optimizer=None):
            if self.lr_scheduler is None:
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer,
                    lambda step: learning_rate_multiplier(step, 128, 8))
                self._created_lr_scheduler = True
            return self.lr_scheduler

        def training_step(self, model, inputs, num_items_in_batch=None):
            guard();contract.validate()
            return super().training_step(model, inputs, num_items_in_batch)

        def _save_checkpoint(self, model, trial):
            guard()
            if self.consumed != self.state.global_step*32:
                raise ValueError('BOOK_CHECKPOINT_COMMITTED_CURSOR')
            Gr00tTrainer._save_checkpoint(self, model, trial)
            cp = Path(self.args.output_dir)/('checkpoint-'+str(self.state.global_step))
            self.train_dataset.processor.save_pretrained(cp/'processor')
            for name in ('statistics.json', 'TRAIN_DEV_SEAL.json'):
                shutil.copyfile(self.train_dataset.root/name, cp/name)
            write(cp/'ADAPTER_PARAMETER_CONTRACT.json', contract.describe())
            write(cp/'SAMPLER_STATE.json', dict(recipe=self.recipe,
                sampler=self.sampler.state_dict(self.consumed), manifest_sha256=self.train_dataset.manifest_sha,
                statistics_sha256=self.train_dataset.statistics_sha,
                worker_trace=check_trace(self.train_dataset.trace_root, self.train_dataset, self.start_cursor, self.consumed)))
            members = {str(p.relative_to(cp)):sha(p) for p in sorted(cp.rglob('*')) if p.is_file()}
            required = {'config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json',
                        'SAMPLER_STATE.json','ADAPTER_PARAMETER_CONTRACT.json','statistics.json','TRAIN_DEV_SEAL.json'}
            if not required <= set(members) or not any(n.endswith('.safetensors') for n in members):
                raise ValueError('BOOK_FULL_RESUME_CHECKPOINT_MEMBERS_REQUIRED')
            write(cp/'CHECKPOINT_COMPLETE.json', dict(schema='BOOK5_FULL_WRAPPED_TRAINING_CHECKPOINT_V1',
                global_step=self.state.global_step, cursor=self.consumed, full_resume=True,
                parent_checkpoint_manifest_sha256=PARENT, recipe=self.recipe, files=members,
                official_unwrapped_inference_export=False, actual_reload_check='PENDING'))
    return BookTrainer


def run(root, manifest_sha256):
    from scripts.pi0292.durable import read, write, OwnerLock
    from scripts.pi0294.native_dataset import NativeDataset, ProcessedCollator, sha, official_import
    from scripts.pi0294.e1.smoke import node_local, put
    from scripts.pi0294.generalization.minimal_adapter import ParameterContract, frozen_fingerprint
    from scripts.pi0294.generalization.minimal_model import loader_kwargs, validate_loading
    from scripts.pi0294.local_backbone import local_model_name
    from scripts.pi0294.numerics import prepare_environment, configure_torch, MODE
    from scripts.pi0294.neural import COSMOS_REVISION
    root = node_local(root); m = read(root/'RUN_MANIFEST.json')
    if (sha(root/'RUN_MANIFEST.json') != manifest_sha256 or m['recipe'] != recipe()
            or m['kind'] != 'V7_BOOK5_FINITE_LEARNING' or m['owner_uid'] != os.getuid()
            or os.uname().nodename != 'corelab-4080s' or not time.time() < m['deadline_epoch']):
        raise PermissionError('BOOK_SEALED_FINITE_NATIVE_MANIFEST')
    lease = read(root/'control/LEASE.json')
    if lease['status'] != 'ACTIVE' or lease['manifest_sha256'] != manifest_sha256:
        raise PermissionError('BOOK_EXISTING_OWNER_ACTIVE_LEASE')
    registry = read(root/'control/REGISTERED_PATHS.json')
    if not any(r.get('path') == str(root) and r.get('role') == m['role'] for r in registry['entries']):
        raise PermissionError('BOOK_EXISTING_PATH_REGISTRY')
    for name, digest in m['sealed_inputs'].items():
        if sha(root/name) != digest:
            raise PermissionError('BOOK_SEALED_INPUT_CHANGED:'+name)
    bindings = read(root/'NEURAL_BINDINGS.private.json')
    for name in ('source','python','cosmos','dataset','cosmos_hub'):
        node_local(bindings[name])
    checkpoint = node_local(m['checkpoint'])
    if sha(checkpoint/'CHECKPOINT_COMPLETE.json') != PARENT:
        raise PermissionError('BOOK_EXACT_DECLARED_E1_WARMSTART')
    parent = read(checkpoint/'CHECKPOINT_COMPLETE.json')
    for name, digest in parent['files'].items():
        if sha(checkpoint/name) != digest:
            raise PermissionError('BOOK_PARENT_CHANGED_AT_NEW_TASK_VERSION_BOUNDARY:'+name)
    data = NativeDataset(bindings['dataset'], bindings['source'], bindings['cosmos'],
                         for_optimization=True, trace_root=root/'evidence/worker_trace')
    if (data.manifest_sha != DATA or data.statistics_sha != STATS or len(data) != 3005
            or sha(data.root/'TRAIN_DEV_SEAL.json') != SEAL):
        raise PermissionError('BOOK_ONLY_FIVE_FROZEN_QUALIFIED_TRAIN_EPISODES')
    data.make_sampler = BookSampler
    lock = OwnerLock(root.parent/'global-native-GPU-locks'/(m['GPU_uuid']+'.lock'))
    started = time.monotonic(); CPU = time.process_time(); loads = steps = 0; failure = None
    model = trainer = contract = None
    def guard():
        if time.time() >= m['deadline_epoch'] or (root/'STOP.json').exists():
            raise PermissionError('BOOK_FINITE_DEADLINE_OR_STOP')
        fs = os.statvfs(root)
        if fs.f_bavail*fs.f_frsize < m['protected_remaining_bytes']+m['free_margin_bytes']:
            raise PermissionError('BOOK_RETAIN_ORIGINAL_PROTECTED_WORK_AND_REAL_FREE_MARGIN')
    def stop(signum, frame):
        raise TimeoutError('BOOK_FINITE_NATIVE_WALL_OR_OWN_STOP')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGALRM, stop)
    signal.alarm(max(1, int(m['deadline_epoch']-time.time())))
    try:
        guard(); prepare_environment(MODE); official_import(bindings['source'])
        import torch
        from transformers import AutoModel, TrainingArguments, TrainerCallback, set_seed
        from scripts.pi0294.native_train import tensor_identity
        configure_torch(torch, MODE); torch.set_num_threads(4); set_seed(recipe()['seed'])
        if torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] < 24*1024**3:
            raise PermissionError('BOOK_REAL_SHARED_GPU_FREE24GIB_REQUIRED')
        put(root/'MODEL_LOAD_INTENT.json', dict(parent=PARENT, observed_epoch=time.time()));loads += 1
        import gr00t.model.gr00t_n1d7.gr00t_n1d7
        model, info = AutoModel.from_pretrained(checkpoint, **loader_kwargs(local_model_name(root,bindings), bindings['cosmos_hub'],COSMOS_REVISION))
        validate_loading(model, info); model.to('cuda')
        contract = ParameterContract(model,'E1_ADAPTER_ONLY',10)
        if contract.describe()['trainable_numel'] != 10180740:
            raise ValueError('BOOK_EXPECTED_SEVEN_TARGET_SLICES')
        before = {n:tensor_identity(p) for n,p in model.named_parameters() if p.requires_grad}
        frozen = frozen_fingerprint(model);data.initialize()
        write(root/'MODEL_LOADED.json', dict(parent=PARENT, contract=contract.describe(), observed_epoch=time.time(),free_GPU_bytes=torch.cuda.mem_get_info()[0]))
        arguments = TrainingArguments(output_dir=str(root/'training'),max_steps=128,per_device_train_batch_size=1,
            gradient_accumulation_steps=32,learning_rate=1e-4,weight_decay=1e-5,adam_beta1=.9,adam_beta2=.999,
            adam_epsilon=1e-8,max_grad_norm=1.,bf16=True,tf32=True,optim='adamw_torch',
            dataloader_num_workers=0,dataloader_pin_memory=True,remove_unused_columns=False,
            seed=recipe()['seed'],data_seed=recipe()['seed'],eval_strategy='no',logging_strategy='steps',
            logging_steps=1,save_strategy='no',save_only_model=False,save_total_limit=None,
            gradient_checkpointing=False,report_to=[],disable_tqdm=True,ignore_data_skip=True,load_best_model_at_end=False)
        holder = {}
        class Audit(TrainerCallback):
            def on_step_begin(self,args,state,control,**kwargs):
                guard();write(root/'evidence/UPDATE_PENDING.json',dict(next_step=state.global_step+1,epoch=time.time()))
            def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
                for name,p in model.named_parameters():
                    if p.grad is not None and (not p.requires_grad or not torch.isfinite(p.grad).all()):
                        raise ValueError('BOOK_FINITE_DECLARED_GRADIENTS_ONLY:'+name)
            def on_step_end(self,args,state,control,**kwargs):
                torch.cuda.synchronize();t=holder['trainer']
                if t.consumed != state.global_step*32:
                    raise ValueError('BOOK_ACTUAL_COMMITTED_SAMPLE_CURSOR')
                record=dict(step=state.global_step,cursor=t.consumed,native_loss=t.last_native_loss,
                    learning_rate=t.lr_scheduler.get_last_lr(),epoch=time.time())
                with(root/'evidence/OPTIMIZER_STEPS.jsonl').open('a')as f:f.write(json.dumps(record)+'\n');f.flush()
                write(root/'evidence/UPDATE_COMMITTED.json',record)
                write(root/'evidence/PROGRESS.json',record)
                if state.global_step==128:control.should_save=True;control.should_training_stop=True
                return control
        trainer = make_trainer_class(contract,guard)(model=model,args=arguments,train_dataset=data,
            data_collator=ProcessedCollator(data.processor),callbacks=[Audit()],evidence_dir=root/'evidence',recipe=recipe())
        holder['trainer']=trainer
        trainer.train();steps=trainer.state.global_step
        if steps != 128 or trainer.consumed != 4096 or frozen_fingerprint(model) != frozen:
            raise ValueError('BOOK_FIXED128_ENDPOINT_OR_FROZEN_PARAMETERS_CHANGED')
        after={n:tensor_identity(p)for n,p in model.named_parameters()if p.requires_grad}
        changed=[n for n in before if before[n]!=after[n]]
        if not changed:raise ValueError('BOOK_OPTIMIZER_DID_NOT_CHANGE_PARAMETERS')
        cp=root/'training/checkpoint-128';complete=read(cp/'CHECKPOINT_COMPLETE.json')
        for name,digest in complete['files'].items():
            if sha(cp/name)!=digest:raise ValueError('BOOK_CHECKPOINT_READBACK_CHANGED:'+name)
        optimizer=trainer.optimizer.state_dict();scheduler=trainer.lr_scheduler.state_dict()
        restored=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
        if len(restored['state'])!=len(optimizer['state']) or not restored['state']:
            raise ValueError('BOOK_SAVED_OPTIMIZER_STATE_REQUIRED')
        # Actual strict reload of every wrapped model member into the qualified
        # architecture; no second live model, extra checkpoint or inference export.
        from transformers.modeling_utils import load_sharded_checkpoint
        load_sharded_checkpoint(model,str(cp),strict=True,prefer_safe=True)
        if {n:tensor_identity(p)for n,p in model.named_parameters()if p.requires_grad}!=after or frozen_fingerprint(model)!=frozen:
            raise ValueError('BOOK_ACTUAL_FULL_WRAPPED_MODEL_RELOAD_DIFFERS')
        trainer.optimizer.load_state_dict(restored);trainer.lr_scheduler.load_state_dict(torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False))
        if trainer.lr_scheduler.state_dict()!=scheduler:raise ValueError('BOOK_SCHEDULER_RESTORE_DIFFERS')
        trainer._load_rng_state(str(cp))
        write(root/'ACTUAL_CHECKPOINT_RELOAD.json',dict(checkpoint_manifest_sha256=sha(cp/'CHECKPOINT_COMPLETE.json'),
            exact_model_reload=True,optimizer_state_entries=len(restored['state']),scheduler_reload=True,RNG_reload=True,
            sampler_cursor=read(cp/'SAMPLER_STATE.json')['sampler']['cursor'],changed_trainable_parameters=changed,
            frozen_parameters_and_buffers_unchanged=True,DEV_rollouts=0,observed_epoch=time.time()))
    except BaseException as error:
        failure=type(error).__name__+': '+str(error)
        raise
    finally:
        signal.alarm(0)
        if trainer is not None:steps=trainer.state.global_step
        write(root/'RESULT.json',dict(status='COMPLETE_FIXED128_RELOADABLE'if failure is None else'FAILED',error=failure,
            model_load_attempts=loads,optimizer_updates=steps,CPU_s=time.process_time()-CPU,wall_s=time.monotonic()-started,
            DEV_rollouts=0,physics_integrations=0,completed_epoch=time.time()))
        trainer=model=contract=None;gc.collect()
        lock.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args();run(args.root,args.manifest_sha256)
