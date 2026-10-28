"""Explicit GEN1 identity around the unchanged C50 numerical training recipe.

This module is cold integration. No native request, GPU or model release is
created here; those remain the existing owner/runtime's responsibility.
"""
from copy import deepcopy
from . import amd_bank as bank,amd_pilot_scope as scope
from scripts.pi0294.pv1 import mix1024 as numerical
from scripts.pi0294.pv1.recipe import PARENT,commit_update as mature_commit
from scripts.pi0294.scene_cards import digest

SCHEMA='GEN1_INDEPENDENT_CORR2048_LAYOUT_V0_1024_RECIPE_V1'


def recipe(original):
    r=numerical.recipe(original,'C50')
    r.update(schema=SCHEMA,route='GEN1_LAYOUT_V0',arm=bank.IDENTITY,
        stage='GEN1_INDEPENDENT_TRAIN',sampling='GEN1_FROZEN_C50_NUMERICAL_LANE_PARENT_WINDOW_V1',
        initial_optimizer='FRESH_INDEPENDENT_FROM_COMMON_CORR2048',
        initial_parent_checkpoint_manifest_sha256=PARENT)
    return r


def validate(value,original):
    scope.need(value==recipe(original),'GEN_EXACT_RECIPE_NO_C_D_ALIAS_OR_NEW_NUMERICS')
    return value


def trainer_arguments(value,original):
    validate(value,original)
    # Only the argument factory sees this transient numerical reference. The
    # actual manifest, request, sampler, update journal and save use GEN1.
    return numerical.trainer_arguments(numerical.recipe(original,'C50'),original)


def begin_update(value,*,local_step,request_id,recipe_value):
    scope.need(recipe_value.get('schema')==SCHEMA and recipe_value.get('arm')==bank.IDENTITY
        and recipe_value.get('max_steps')==1024 and recipe_value.get('sample_visits')==32768
        and recipe_value.get('initial_parent_checkpoint_manifest_sha256')==PARENT
        and type(local_step)is int and 0<=local_step<1024 and bool(request_id)
        and value['pending_updates']==value['unknown_effects']==[]
        and value['optimizer_updates']==local_step,'GEN_ONE_EXACT_KNOWN_NATIVE_UPDATE_INTENT')
    out=deepcopy(value);runs=out['training_runs'];expected=dict(recipe_sha256=digest(recipe_value),last_native_step=local_step)
    if not runs and local_step==0:runs[bank.IDENTITY]=expected
    scope.need(runs=={bank.IDENTITY:expected},'GEN_NO_OLD_ARM_OR_OPTIMIZER_RESET')
    out['pending_updates']=[dict(arm=bank.IDENTITY,run_key=bank.IDENTITY,
        local_step_before=local_step,request_id=request_id)]
    return out


def commit_update(value,*,local_step_after,request_id):
    scope.need(type(local_step_after)is int and 1<=local_step_after<=1024
        and set(value['training_runs'])=={bank.IDENTITY},'GEN_ONLY_INDEPENDENT_1024_RANGE')
    return mature_commit(value,arm=bank.IDENTITY,local_step_after=local_step_after,request_id=request_id)


def resume_state(saved,complete,trainer_state,recipe_value,sampler,member_hashes):
    step=complete.get('global_step');files=complete.get('files',{})
    required={'config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','SAMPLER_STATE.json','statistics.json'}
    scope.need(recipe_value.get('schema')==SCHEMA and recipe_value.get('arm')==bank.IDENTITY
        and recipe_value.get('max_steps')==1024 and type(step)is int and 0<step<1024
        and complete.get('full_resume')is True and required<=set(files)
        and any(n.endswith('.safetensors')for n in files)and any(n.startswith('processor/')for n in files)
        and files==member_hashes and trainer_state.get('global_step')==step
        and saved.get('recipe')==recipe_value and saved.get('sampler')==sampler.state_dict(step*32)
        and saved.get('manifest_sha256')==sampler.dataset.manifest_sha
        and saved.get('statistics_sha256')==files.get('statistics.json')==scope.a.STATS,
        'GEN_EXACT_FULL_OPTIMIZER_SCHEDULER_RNG_SAMPLER_RESTORE')
    return step


def make_trainer_class():
    """Same mature Trainer/loss/scheduler, with complete GEN1 state saves."""
    from pathlib import Path
    import os,shutil
    from gr00t.experiment.trainer import Gr00tTrainer
    from scripts.pi0294.mp1.trainer import make_trainer_class as mature
    from scripts.pi0294.e1.trainer import check_trace
    from scripts.pi0292.durable import read,write
    from scripts.pi0294.native_dataset import sha
    class Trainer(mature()):
        def __init__(self,*args,**kwargs):
            r=kwargs.get('recipe',{})
            scope.need(r.get('schema')==SCHEMA and r.get('arm')==bank.IDENTITY
                and kwargs['args'].max_steps==1024,'GEN_TYPED_TRAINER_ONLY')
            super().__init__(*args,**kwargs)
        def _save_checkpoint(self,model,trial):
            scope.need(self.consumed==self.state.global_step*32 and 0<self.state.global_step<=1024,
                'GEN_SAVE_KNOWN_CONSUMED_CURSOR')
            self.args.save_only_model=False;Gr00tTrainer._save_checkpoint(self,model,trial)
            checkpoint=Path(self.args.output_dir)/('checkpoint-'+str(self.state.global_step))
            scope.need(sha(self.train_dataset.root/'statistics.json')==scope.a.STATS,'GEN_R0_UNCHANGED')
            self.train_dataset.processor.save_pretrained(checkpoint/'processor')
            shutil.copyfile(self.train_dataset.root/'statistics.json',checkpoint/'statistics.json')
            trace=check_trace(self.train_dataset.trace_root,self.train_dataset,self.start_cursor,self.consumed)
            write(checkpoint/'SAMPLER_STATE.json',dict(sampler=self.sampler.state_dict(self.consumed),recipe=self.recipe,
                manifest_sha256=self.train_dataset.manifest_sha,statistics_sha256=scope.a.STATS,worker_trace=trace))
            files={str(p.relative_to(checkpoint)):sha(p)for p in sorted(checkpoint.rglob('*'))if p.is_file()}
            required={'config.json','statistics.json','trainer_state.json','SAMPLER_STATE.json','optimizer.pt','scheduler.pt','rng_state.pth'}
            scope.need(required<=set(files)and any(n.endswith('.safetensors')for n in files),'GEN_COMPLETE_SAVE_MEMBERS')
            write(checkpoint/'CHECKPOINT_COMPLETE.json',dict(global_step=self.state.global_step,files=files,full_resume=True,
                exact_resume_from_M2048=False,origin_job_id=os.environ.get('PBS_JOBID'),training_identity=bank.IDENTITY,
                origin_source_sha=read(Path(self.args.output_dir).parents[1]/'RUN_MANIFEST.json')['code']['vla_commit']))
    return Trainer
