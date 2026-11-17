"""One B1-parent FM500 versus full-generated-H40 loss500 comparison."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .runtime_v3 import DeviceResidentEngine
from .runtime_v1 import read_pin
from .checkpoint_v1 import sha
from .close_temporal_v1 import CloseEngine
from .time_shift7_training_v1 import ShiftDataset,physical
from .config_v1 import SEED
from . import full_h40_v6

PLAN='ae3d3c0dceade437b0104a150cd0402fef650052'
RECIPE='SLA_B1_PARENT2000_FRESH_ADAM_FULL_GENERATED_H40_V6'
B1='b9ba974c374b3c3a09d7742004fc799337d5e7f1b9c9c7d290af066766e4706c'


class TargetEngine(DeviceResidentEngine):
    visits_for_dataset=CloseEngine.visits_for_dataset
    learning_rate_values_for_spec=CloseEngine.learning_rate_values_for_spec

    def dataset_for_spec(self,spec):
        from openpi_client.image_tools import resize_with_pad
        audit=read_pin(spec['target_audit'])
        if(spec['operation']!=RECIPE or spec['plan_commit']!=PLAN or spec['microbatch']!=4
           or spec['training_matmul_precision']!='highest' or spec['planned_candidate_updates']!=500
           or spec['new_optimizer_updates_budget']!=500 or spec['new_training_visits_budget']!=16000
           or spec['B1_parent_package']['sha256']!=B1 or spec['inference_route']!='cached_actual'
           or spec['full_H40_lambda']!={'T-FM':0.,'T-H40':.1}.get(spec['arm'])
           or spec['aux_samples_per_update']!=4 or spec['aux_sampling_steps']!=10
           or spec['aux_microbatch']!=1 or spec['dense_manifest_sha256']!='4bcc5ffcd8920d874599a54cca73792c50a49ea6013d65ca7826817a9f80d7de'
           or not audit['passes']or audit['manifest_sha256']!=spec['dense_manifest_sha256']):
            raise ValueError('EXPLICIT_MATCHED_V6_B1_FULL_GENERATED_H40_ONLY')
        return ShiftDataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad,
            max_cache_bytes=spec['frame_cache_bytes'],time_audit=spec['time_codec_audit'],shift_audit=spec['shift_layout_audit'])

    def __init__(self,spec,output):
        super().__init__(spec,output)
        import jax,jax.numpy as jnp,flax.nnx as nnx
        from .provider_v1 import active_rows
        from .mobile_contract_v52 import contract
        from . import batch_v2
        from openpi.models import pi0
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields
        from scripts.sla.close_goal_g1_v1 import make_sampler as deployed_sampler
        package=Path(spec['B1_parent_package']['path']);parent=read_pin(dict(path=str(package/'MANIFEST.private.json'),sha256=B1));contract(parent)
        if parent['bindings']['endpoint_lambda']!=0. or parent['bindings']['frozen_fingerprint']!=self.initial_frozen:
            raise ValueError('REAL_B1_WITH_UNCHANGED_FOUNDATION_REQUIRED')
        rows,structure=active_rows(dict(arrays=parent['adapter_arrays']),self.active)
        if len(rows)!=28 or sum(np.prod(r['shape'])for r in rows)!=52153376:raise ValueError('ORIGINAL_28_ACTIVE_ARRAYS_REQUIRED')
        arrays=[]
        for row in rows:
            path=package/row['file']
            if sha(path)!=row['sha256']:raise ValueError('FROZEN_B1_ADAPTER_BYTES_CHANGED')
            arrays.append(jax.device_put(np.load(path,allow_pickle=False)))
        self.active,=jax.tree_util.tree_unflatten(structure,arrays);self.opt_state=self.tx.init(self.active)
        lr=hashlib.sha256(''.join(x.hex()+'\n'for x in self.learning_rate_values).encode()).hexdigest()
        if lr!=spec['learning_rates_sha256']:raise ValueError('ORIGINAL_500_ENTRY_LR_REQUIRED')
        self.bindings.update(recipe=RECIPE,goal_plan_sha=PLAN,B1_parent_package=spec['B1_parent_package'],
            parent_cumulative_updates=2000,parent_cumulative_visits=64000,new_planned_updates=500,new_planned_visits=16000,
            successor_cumulative_updates=2500,successor_cumulative_visits=80000,full_H40_lambda=spec['full_H40_lambda'],
            full_H40_margin=.20,full_H40_scale=.25,full_sampling_steps=10,aux_samples_per_update=4,aux_microbatch=1,
            aux_slot_rule='4*(COMMITTED_STEP%8)+arange4',aux_noise_rule='SEED+7000000_FOLD_ABSOLUTE_CURSOR',
            aux_forward_precision='ORIGINAL_CACHED_ACTUAL_DEFAULT',FM_training_precision='highest',aux_gradient_rematerialization=True,
            clean_proxy=False,codec_version='SLA_REFERENCE_CONTINUED_GRIP_TIME_SHIFT7_V1',
            time_codec_audit=spec['time_codec_audit'],shift_layout_audit=spec['shift_layout_audit'],
            derived_statistics_sha256=self.dataset.manifest['statistics']['sha256'],dense_manifest_sha256=self.dataset.manifest_sha,
            fresh_Adam=True,state_token_dropout=.25,inference_route='cached_actual',BASE_head=False,spatial_loss=False,
            reference_error_input=False,teacher_memory_input=False,parent_host_cast_v7=True)
        fields,_=extracted_fields(batch_v2,pi0);graph=self.graph
        def fm(active,frozen,obs,actions,noise,tau):
            with jax.default_matmul_precision('highest'):
                t=tau[:,None,None];field=fields['train_field'](nnx.merge(graph,active,frozen),obs,t*noise+(1-t)*actions,tau)
                error=(field-(noise-actions))**2
                return jnp.mean(error),(jnp.mean(error[...,:7]),jnp.mean(error[...,7:]))
        self.grad=jax.jit(jax.value_and_grad(fm,has_aux=True))
        sampler=full_h40_v6.make_sampler(graph,fields,rematerialize=True,precision=None)
        self.sample=sampler
        low=jnp.asarray(self.dataset.statistics['actions'].q01,jnp.float32);high=jnp.asarray(self.dataset.statistics['actions'].q99,jnp.float32)
        def aux(active,frozen,obs,noise):return full_h40_v6.loss(sampler(active,frozen,obs,noise),low,high,xp=jnp)
        self.aux_evaluate=jax.jit(aux);self.aux_gradient=jax.jit(jax.value_and_grad(aux,has_aux=True))
        original_sampler=deployed_sampler(graph,fields,cached=True,precision=None)
        self.infer=lambda a,f,o,n:original_sampler(a,f,o,n)[0]
        self.add_aux=jax.jit(lambda acc,g:jax.tree.map(lambda a,b:a+spec['full_H40_lambda']*b/4,acc,g),donate_argnums=(0,))
        self.gradient_norm=jax.jit(lambda g:__import__('optax').global_norm(g))
        self.aux_noise=jax.jit(lambda cursor:jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(SEED+7000000),cursor),(1,40,32)))
        self.add_fm_metrics=jax.jit(lambda m,v:m+jnp.stack(v)/8,donate_argnums=(0,))

    def _aux_four(self,obs,first_cursor,accumulator,*,reference_check=False,diagnostic=False):
        import jax,jax.numpy as jnp
        records=[]
        for i in range(4):
            item=jax.tree.map(lambda x:x[i:i+1],obs);noise=self.aux_noise(jnp.uint32(first_cursor+i))
            key='diagnostic_full_generation'if diagnostic else'training_full_generation'
            self.counts[key+'_attempts']=self.counts.get(key+'_attempts',0)+1
            if self.spec['full_H40_lambda']:
                self.counts['full_trajectory_backward_attempts']=self.counts.get('full_trajectory_backward_attempts',0)+1
                (value,detail),grad=self.aux_gradient(self.active,self.frozen,item,noise)
                accumulator=self.add_aux(accumulator,grad)
            else:value,detail=self.aux_evaluate(self.active,self.frozen,item,noise)
            jax.block_until_ready((value,detail,accumulator))
            values=np.asarray(jnp.r_[value,jnp.stack(detail)])
            if not np.isfinite(values).all():raise ValueError('FINITE_FULL_TEN_STEP_OBJECTIVE_REQUIRED')
            self.counts[key+'s']=self.counts.get(key+'s',0)+1
            row=dict(absolute_cursor=first_cursor+i,loss=float(values[0]),head=float(values[1]),tail=float(values[2]),interior=float(values[3]))
            if reference_check:
                self.counts['reference_comparison_generations']=self.counts.get('reference_comparison_generations',0)+1
                original=self.infer(self.active,self.frozen,item,noise);static=self.sample(self.active,self.frozen,item,noise);jax.block_until_ready((original,static))
                # static is a second, separately charged qualification forward;
                # backprop's earlier output is not claimed as this prediction.
                self.counts['static_comparison_generations']=self.counts.get('static_comparison_generations',0)+1
                x,y=np.asarray(original),np.asarray(static);row['sampler_max_abs_difference']=float(np.max(abs(x-y)));row['sampler_exact_equal']=bool(np.array_equal(x,y))
                if not np.array_equal(x,y):raise ValueError('STATIC_SCAN_DIFFERS_FROM_ORIGINAL_CACHED_SAMPLER')
                row['noise_sha256']=hashlib.sha256(np.asarray(noise).tobytes()).hexdigest()
            records.append(row)
        return accumulator,records

    def diagnose(self):
        import jax,jax.numpy as jnp
        from .batch_v2 import host_batch,device_batch
        before=list(jax.tree.leaves((self.active,self.opt_state,self.rng)))
        host=host_batch(self.dataset,self.tokenizer,self.visit_map,0,4);obs,actions=device_batch(host)
        noise,tau=self.draw_bank(self.rng,np.arange(32,dtype=np.uint32))
        self.counts['diagnostic_forward_backward_attempts']=self.counts.get('diagnostic_forward_backward_attempts',0)+1
        self.counts['diagnostic_windows']=self.counts.get('diagnostic_windows',0)+4
        (fm,parts),gradient=self.grad(self.active,self.frozen,obs,actions,noise[:4],tau[:4])
        gradient,records=self._aux_four(obs,0,gradient,reference_check=True,diagnostic=True)
        norm=float(self.gradient_norm(gradient));jax.block_until_ready(gradient)
        if not np.isfinite(norm)or norm<=0:raise ValueError('REAL_FINITE_NONZERO_FULL_GENERATED_GRADIENT_REQUIRED')
        if self.step or self.cursor or self.counts['optimizer_updates'] or any(a is not b for a,b in zip(before,jax.tree.leaves((self.active,self.opt_state,self.rng)))):
            raise ValueError('ZERO_UPDATE_DIAGNOSIS_MUTATED_TRAINING_STATE')
        return dict(FM32_micro4=float(fm),gradient_norm=norm,aux=records,optimizer_updates=0,optimization_visits=0,
            diagnostic_windows=4,full_trajectory_gradients=4 if self.spec['full_H40_lambda'] else 0,comparison_generations=8,total_offline_generations=12,
            backward_complete=True,training_precision='highest',aux_forward='cached_actual',GPU_memory=jax.devices()[0].memory_stats())

    def update(self):
        import jax,jax.numpy as jnp
        from .batch_v2 import prepared_batches,device_batch
        started=time.perf_counter();accumulator=jax.tree.map(jnp.zeros_like,self.active);metrics=jnp.zeros(3,jnp.float32);logs=[];aux_obs=None
        noise,tau=self.draw_bank(self.rng,np.arange(self.cursor,self.cursor+32,dtype=np.uint32));first=full_h40_v6.aux_slots(self.step)[0]
        def count(k):self.counts[k]=self.counts.get(k,0)+1
        batches=prepared_batches(self.dataset,self.tokenizer,self.visit_map,self.cursor,4,executor=self.loader,on_read=count)
        try:
            for host in batches:
                if time.time()>=self.spec['deadline_epoch']-60:raise TimeoutError('FINITE_V6_BEFORE_UPDATE_COMMIT')
                obs,actions=device_batch(host);offset=int(host['cursors'][0])-self.cursor
                count('forward_backward_attempts');self.counts['attempted_training_visits']+=4
                (value,parts),gradient=self.grad(self.active,self.frozen,obs,actions,noise[offset:offset+4],tau[offset:offset+4])
                accumulator=self.accumulate(accumulator,gradient);metrics=self.add_fm_metrics(metrics,(value,*parts))
                if offset==first:aux_obs=obs
                logs.append(dict(indices=host['indices'],first_absolute_cursor=int(host['cursors'][0]),state_token_dropped=host['state_token_dropped']))
        finally:batches.close()
        if aux_obs is None:raise ValueError('FOUR_FIXED_IN_BATCH_AUX_ANCHORS_REQUIRED')
        accumulator,aux=self._aux_four(aux_obs,self.cursor+first,accumulator)
        norm=float(self.gradient_norm(accumulator));value=np.asarray(metrics)
        if not np.isfinite(value).all()or not np.isfinite(norm)or norm<=0:raise ValueError('FINITE_REAL_EFFECTIVE32_PLUS_AUX_GRADIENT_REQUIRED')
        self.counts['completed_forward_backward']+=8;self.counts['completed_training_visits']+=32;count('optimizer_update_attempts')
        self.active,self.opt_state,_=self.apply(self.active,self.opt_state,accumulator);jax.block_until_ready((self.active,self.opt_state))
        self.step+=1;self.cursor+=32;count('optimizer_updates')
        row=dict(step=self.step,cursor=self.cursor,mean_FM32_loss=float(value[0]),active7=float(value[1]),padding25=float(value[2]),
            full_H40_loss=float(np.mean([x['loss']for x in aux])),aux=aux,micro=logs,grad_norm_before_clip=norm,
            used_lr=self.learning_rate_values[self.step-1],wall_s=time.perf_counter()-started,GPU_memory=jax.devices()[0].memory_stats())
        with(self.output/'updates.private.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
        return row
