"""Cold C50/D25 contracts around mature PV1 data selection and Trainer.

CPU preparation only: no launcher, admission, owner or new budget is defined.
Legacy validators/4096 runs are untouched and reject these distinct identities.
"""
from copy import deepcopy
from functools import partial
import hashlib
from types import SimpleNamespace
import numpy as np

from scripts.pi0294.scene_cards import digest
from .recipe import primary_recipe
from scripts.pi0294.e1.scope import STATS
from .sampler import MatchedSampler, lane_at as legacy_lane
from scripts.pi0294.e1.trainer import learning_rate_multiplier

SCHEMA='V5_MATCHED_MIX_1024_V1'
ARMS=('C50','D25')
TOTAL=1024
VISITS=TOTAL*32


def recipe(original,arm):
    if arm not in ARMS:raise PermissionError('V5_ONLY_C50_D25')
    # Derive frozen numerical fields from the existing validated recipe. The
    # resulting identity is new and cannot be passed to validate_primary.
    r=primary_recipe(original,'COVERAGE_FIRST','B_COVERAGE')
    r.update(schema=SCHEMA,route='V5_MIX1024',arm=arm,stage='V5_MATCHED_MIX',
        max_steps=TOTAL,scheduler_horizon_steps=TOTAL,sample_visits=VISITS,
        checkpoints=[TOTAL],development_candidate_steps=[TOTAL],
        sampling='V5_PAIRED_EXPERT_LANE_REPLACEMENT_V1',new_expert_fraction=.5 if arm=='C50' else .25,
        entire_branch_new_fraction=.375 if arm=='C50' else .1875,
        C1_contract='ORIGINAL_ALGORITHM_IDENTICAL_AT_EACH_CURSOR_NO_FORCED_FORK_RATIO',
        initialization='LEGACY_MATCHED',statistics_sha256=STATS,requires_valid_native_scope=True)
    return r


def validate(r,original):
    if r!=recipe(original,r.get('arm')):raise PermissionError('V5_EXACT_COLD_RECIPE')
    return r


def lane_at(cursor,arm):
    if type(cursor) is not int or not 0<=cursor<VISITS or arm not in ARMS:raise PermissionError('V5_EXACT_CURSOR_ARM')
    task,lane=legacy_lane(cursor,'COVERAGE_FIRST','B_COVERAGE')
    if arm=='D25' and lane.startswith('new_') and cursor%32<24:lane='original_'+task
    return task,lane


class MixSampler(MatchedSampler):
    def __init__(self,dataset,*,arm,start=0,stop=VISITS,seed=20261019,positioned=True):
        if arm not in ARMS or seed!=20261019 or type(start)is not int or type(stop)is not int or not 0<=start<=stop<=VISITS:
            raise PermissionError('V5_MATCHED_SEED_AND_EXACT_RESUME_RANGE')
        # Reuse the legacy bank/parent/window integrity checks only. This
        # support-view is never an execution recipe or old B run identity.
        support=SimpleNamespace(rows=dataset.rows,PV1_route='COVERAGE_FIRST',PV1_arm='B_COVERAGE',manifest_sha=dataset.manifest_sha)
        super().__init__(support,start=start,stop=stop,seed=seed,positioned=positioned)
        self.dataset=dataset;self.route='V5_MIX1024';self.arm=arm

    def selection_at(self,cursor):
        if type(cursor)is not int or not self.start<=cursor<self.stop:raise IndexError('V5_CURSOR_OUTSIDE_SESSION')
        task,lane=lane_at(cursor,self.arm)
        # Exact legacy absolute-cursor/lane substream, retaining shared picks.
        lane_seed=int.from_bytes(hashlib.sha256(lane.encode()).digest()[:4],'big')
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed,cursor,lane_seed])))
        parents=sorted(self.bank[lane]);parent=parents[int(rng.integers(len(parents)))];windows=self.bank[lane][parent]
        anchor,index=windows[int(rng.integers(len(windows)))]
        return dict(task=task,lane=lane,parent=parent,anchor=anchor,index=index)

    def state_dict(self,consumed):
        if type(consumed)is not int or not self.start<=consumed<=self.stop or consumed%32:
            raise ValueError('V5_ONLY_COMMITTED_UPDATE_CURSOR')
        return dict(algorithm='V5_PAIRED_EXPERT_LANE_REPLACEMENT_V1',route=self.route,arm=self.arm,
            seed=self.seed,cursor=consumed,stop=self.stop,manifest_sha256=self.dataset.manifest_sha,
            support_sha256=self.support_sha256,metadata_not_model_input=True)


def bind_dataset_reader(reader,arm):
    """Bind an already-qualified PV1 reader; does not initialize or load it."""
    if arm not in ARMS or reader.PV1_route!='COVERAGE_FIRST' or reader.PV1_arm!='B_COVERAGE':
        raise PermissionError('V5_REUSES_ONLY_FULL_FROZEN_COVERAGE_READER')
    class View:
        def __init__(self):
            self.rows=reader.rows;self.manifest_sha=digest(dict(schema=SCHEMA,base_reader_manifest=reader.manifest_sha))
            self.make_sampler=partial(MixSampler,arm=arm)
        def __len__(self):return len(reader)
        def __getitem__(self,index):return reader[index]
        def __getattr__(self,name):return getattr(reader,name)
    return View()


def lr_schedule(r):
    if r.get('schema')!=SCHEMA or r.get('max_steps')!=TOTAL or r.get('scheduler_horizon_steps')!=TOTAL or r.get('warmup_updates')!=64:
        raise PermissionError('V5_FIXED_1024_LR_HORIZON')
    return [r['learning_rate']*learning_rate_multiplier(i,TOTAL,64) for i in range(TOTAL+1)]


def trainer_arguments(r,original):
    validate(r,original)
    return dict(max_steps=TOTAL,per_device_train_batch_size=2,gradient_accumulation_steps=16,
        learning_rate=r['learning_rate'],weight_decay=r['weight_decay'],adam_beta1=r['adam_beta1'],
        adam_beta2=r['adam_beta2'],adam_epsilon=r['adam_epsilon'],max_grad_norm=r['max_grad_norm'],
        bf16=r['bf16'],tf32=r['tf32'],optim=r['optimizer'],dataloader_num_workers=0,
        dataloader_pin_memory=True,remove_unused_columns=False,seed=r['seed'],data_seed=r['seed'],
        eval_strategy='no',logging_strategy='steps',logging_steps=1,save_strategy='no',save_only_model=False,
        gradient_checkpointing=False,report_to=[],disable_tqdm=True,ignore_data_skip=True,load_best_model_at_end=False)


def step_boundary(step,cursor,*,pending,unknown):
    if type(step)is not int or not 0<=step<=TOTAL or type(cursor)is not int or cursor!=32*step or pending or unknown:
        raise PermissionError('V5_KNOWN_COMMITTED_UPDATE_ONLY')
    return dict(should_training_stop=step==TOTAL,should_save=step==TOTAL,full_resume_required=True)


def resume_state(saved,complete,trainer_state,r,sampler,member_hashes):
    """Validate actual full-state metadata; caller must stream/hash members."""
    step=complete.get('global_step');files=complete.get('files',{})
    required={'config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','SAMPLER_STATE.json','statistics.json'}
    if (r.get('schema')!=SCHEMA or r.get('arm')!=sampler.arm or r.get('max_steps')!=TOTAL
        or r.get('seed')!=sampler.seed or r.get('statistics_sha256')!=STATS
        or type(step)is not int or not 0<step<TOTAL or complete.get('full_resume')is not True or not required<=set(files)
        or not any(n.endswith('.safetensors') for n in files) or not any(n.startswith('processor/') for n in files)
        or files!=member_hashes or trainer_state.get('global_step')!=step or saved.get('recipe')!=r
        or saved.get('sampler')!=sampler.state_dict(step*32)
        or saved.get('manifest_sha256')!=sampler.dataset.manifest_sha
        or saved.get('statistics_sha256')!=STATS or files.get('statistics.json')!=STATS):
        raise PermissionError('V5_EXACT_FULL_RESUME_NO_OPTIMIZER_OR_CURSOR_RESET')
    return step



def make_trainer_class():
    """Mature loss/loader/scheduler; full-state save also at interrupted step256.

    This factory is not a launcher. An existing authorized owner must supply
    native admission, update intents/commits, parent checks and resource guards.
    """
    from pathlib import Path
    import os
    import shutil
    from gr00t.experiment.trainer import Gr00tTrainer
    from scripts.pi0294.mp1.trainer import make_trainer_class as mature
    from scripts.pi0294.e1.trainer import check_trace
    from scripts.pi0292.durable import write,read
    from scripts.pi0294.native_dataset import sha
    class MixTrainer(mature()):
        def __init__(self,*args,**kwargs):
            r=kwargs.get('recipe',{})
            if r.get('schema')!=SCHEMA or r.get('arm')not in ARMS or kwargs['args'].max_steps!=TOTAL:
                raise PermissionError('V5_TYPED_TRAINER_RECIPE_NOT_LEGACY')
            super().__init__(*args,**kwargs)
        def _save_checkpoint(self,model,trial):
            if self.consumed!=self.state.global_step*32 or not 0<self.state.global_step<=TOTAL:
                raise PermissionError('V5_SAVE_ONLY_KNOWN_CONSUMED_CURSOR')
            self.args.save_only_model=False
            Gr00tTrainer._save_checkpoint(self,model,trial)
            checkpoint=Path(self.args.output_dir)/('checkpoint-'+str(self.state.global_step))
            if sha(self.train_dataset.root/'statistics.json')!=STATS:raise PermissionError('V5_R0_CHANGED')
            self.train_dataset.processor.save_pretrained(checkpoint/'processor')
            shutil.copyfile(self.train_dataset.root/'statistics.json',checkpoint/'statistics.json')
            trace=check_trace(self.train_dataset.trace_root,self.train_dataset,self.start_cursor,self.consumed)
            write(checkpoint/'SAMPLER_STATE.json',dict(sampler=self.sampler.state_dict(self.consumed),recipe=self.recipe,
                manifest_sha256=self.train_dataset.manifest_sha,statistics_sha256=STATS,worker_trace=trace))
            files={str(p.relative_to(checkpoint)):sha(p) for p in sorted(checkpoint.rglob('*')) if p.is_file()}
            required={'config.json','statistics.json','trainer_state.json','SAMPLER_STATE.json','optimizer.pt','scheduler.pt','rng_state.pth'}
            if not required<=set(files) or not any(n.endswith('.safetensors') for n in files):raise PermissionError('V5_FULL_SAVE_INCOMPLETE')
            write(checkpoint/'CHECKPOINT_COMPLETE.json',dict(global_step=self.state.global_step,files=files,full_resume=True,
                exact_resume_from_M2048=False,origin_job_id=os.environ.get('PBS_JOBID'),
                origin_source_sha=read(Path(self.args.output_dir).parents[1]/'RUN_MANIFEST.json')['code']['vla_commit']))
    return MixTrainer
