"""Fresh-Adam successors of the real SHIFT7 function, with one loss variable."""
import hashlib
from pathlib import Path
import time
import numpy as np
from .runtime_v3 import DeviceResidentEngine
from .close_temporal_v1 import CloseEngine
from .time_shift7_training_v1 import ShiftDataset, physical
from .runtime_v1 import read_pin
from .checkpoint_v1 import sha
from .gripper_time_shift7_v1 import VERSION,SHIFT,SLOTS
from .gripper_latent_v2 import VERSION as STATS_CODEC
from .mobile_endpoint_v52 import endpoint_loss,clean_estimate

RECIPE='SLA_MOBILE_SHIFT7_PARENT1500_FRESH_ADAM_ENDPOINT500_V52'
PLAN='90d39196bdc345b977ee54ed64e750cf3561aa72'
PARENT='ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb'


class SuccessorEngine(DeviceResidentEngine):
    visits_for_dataset=CloseEngine.visits_for_dataset
    learning_rate_values_for_spec=CloseEngine.learning_rate_values_for_spec

    def dataset_for_spec(self,spec):
        from openpi_client.image_tools import resize_with_pad
        audit=read_pin(spec['target_audit'])
        if (spec['operation']!=RECIPE or spec['plan_commit']!=PLAN
                or spec['microbatch']!=4 or spec['training_matmul_precision']!='highest'
                or spec['planned_candidate_updates']!=500 or spec['new_optimizer_updates_budget']!=500
                or spec['new_training_visits_budget']!=16000 or spec['planned_target_visits']!=16000
                or spec['endpoint_lambda']!={'B1':0.,'B2':.1}.get(spec['arm'])or spec['inference_route']!='cached_actual'
                or spec['dense_manifest_sha256']!='4bcc5ffcd8920d874599a54cca73792c50a49ea6013d65ca7826817a9f80d7de'
                or spec['SHIFT7_parent_package']['sha256']!=PARENT
                or not audit['passes']or audit['manifest_sha256']!=spec['dense_manifest_sha256']):
            raise ValueError('FIXED_MATCHED_SHIFT7_SUCCESSOR500_V52_REQUIRED')
        return ShiftDataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad,
            max_cache_bytes=spec['frame_cache_bytes'],time_audit=spec['time_codec_audit'],shift_audit=spec['shift_layout_audit'])

    def __init__(self,spec,output):
        super().__init__(spec,output)
        import jax
        import jax.numpy as jnp
        import flax.nnx as nnx
        from openpi.models import pi0
        from . import batch_v2
        from .provider_v1 import active_rows
        from .provider_time_shift7_v1 import contract
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields
        from scripts.sla.close_goal_g1_v1 import make_sampler
        package=Path(spec['SHIFT7_parent_package']['path']);parent=read_pin(dict(path=str(package/'MANIFEST.private.json'),sha256=PARENT));contract(parent)
        if parent['bindings']['frozen_fingerprint']!=self.initial_frozen:raise ValueError('SAME_FROZEN_DENSE_PARENT')
        rows,structure=active_rows(dict(arrays=parent['adapter_arrays']),self.active)
        if len(rows)!=28 or sum(np.prod(r['shape'])for r in rows)!=52153376:raise ValueError('UNCHANGED_28_ACTIVE_ARRAYS')
        values=[]
        for row in rows:
            p=package/row['file']
            if sha(p)!=row['sha256']:raise ValueError('REAL_SHIFT7_PARENT_ARRAY_BYTES_CHANGED')
            values.append(jax.device_put(np.load(p,allow_pickle=False)))
        self.active,=jax.tree_util.tree_unflatten(structure,values)
        self.opt_state=self.tx.init(self.active)
        lr_sha=hashlib.sha256(''.join(v.hex()+'\n'for v in self.learning_rate_values).encode('ascii')).hexdigest()
        if lr_sha!=spec['learning_rates_sha256']:raise ValueError('MATCHED_50_WARMUP_LR_BANK')
        self.bindings.update(recipe=RECIPE,successor_parent_package=spec['SHIFT7_parent_package'],
            parent_cumulative_updates=1500,parent_cumulative_visits=48000,new_planned_updates=500,new_planned_visits=16000,
            successor_cumulative_updates=2000,successor_cumulative_visits=64000,
            codec_version=VERSION,statistics_codec_version=STATS_CODEC,parameter_slot_shift=SHIFT,
            time_parameter_slots=[[int(i)for i in g]for g in SLOTS],time_codec_audit=spec['time_codec_audit'],shift_layout_audit=spec['shift_layout_audit'],
            derived_statistics_sha256=self.dataset.manifest['statistics']['sha256'],dense_manifest_sha256=self.dataset.manifest_sha,
            endpoint_lambda=spec['endpoint_lambda'],endpoint_margin_m_s2=.20,endpoint_scale_m_s2=.25,endpoint_tau_max=.5,
            clean_proxy='X_TAU_MINUS_TAU_FIELD',objective='FM32_PLUS_PHYSICAL_XYZ_ENDPOINT_HINGE',fresh_Adam=True,
            BASE_head=False,spatial_loss=False,reference_error_input=False,teacher_memory_input=False,state_token_dropout=.25,
            inference_route='cached_actual',goal_plan_sha=PLAN,parent_host_cast_v7=True)
        fields,_=extracted_fields(batch_v2,pi0);graph=self.graph
        low=jnp.asarray(self.dataset.statistics['actions'].q01,jnp.float32);high=jnp.asarray(self.dataset.statistics['actions'].q99,jnp.float32);coefficient=spec['endpoint_lambda']
        def objective(active,frozen,obs,actions,noise,tau):
            with jax.default_matmul_precision('highest'):
                t=tau[:,None,None];x=t*noise+(1-t)*actions;field=fields['train_field'](nnx.merge(graph,active,frozen),obs,x,tau)
                error=(field-(noise-actions))**2;fm=jnp.mean(error);edge=endpoint_loss(clean_estimate(x,tau,field),tau,low,high,xp=jnp)
                return fm+coefficient*edge,(fm,jnp.mean(error[...,:7]),jnp.mean(error[...,7:]),edge,jnp.mean((tau<=.5).astype(jnp.float32)))
        self.grad=jax.jit(jax.value_and_grad(objective,has_aux=True))
        sample=make_sampler(graph,fields,cached=True,precision=None)
        self.infer=lambda a,f,o,n:sample(a,f,o,n)[0]
        self.add_successor_metrics=jax.jit(lambda acc,vals:acc+jnp.stack(vals)/self.accumulation_steps,donate_argnums=(0,))
        self.successor_summary=jax.jit(lambda metrics,grad:jnp.r_[metrics,__import__('optax').global_norm(grad)])

    def update(self):
        import jax
        import jax.numpy as jnp
        from .batch_v2 import prepared_batches,device_batch
        start=time.perf_counter();accumulator=jax.tree.map(jnp.zeros_like,self.active);metrics=jnp.zeros((6,),jnp.float32);logs=[]
        self.counts['rng_bank_generation_attempts']=self.counts.get('rng_bank_generation_attempts',0)+1
        noise,tau=self.draw_bank(self.rng,np.arange(self.cursor,self.cursor+32,dtype=np.uint32))
        def counted(key):self.counts[key]=self.counts.get(key,0)+1
        batches=prepared_batches(self.dataset,self.tokenizer,self.visit_map,self.cursor,self.microbatch,prefetch=True,executor=self.loader,on_read=counted)
        try:
            for host in batches:
                if time.time()>=self.spec['deadline_epoch']:raise TimeoutError('FINITE_V52_BEFORE_OPTIMIZER_COMMIT')
                obs,actions=device_batch(host);offset=int(host['cursors'][0])-self.cursor
                self.counts['forward_backward_attempts']+=1;self.counts['attempted_training_visits']+=self.microbatch
                (loss,aux),gradient=self.grad(self.active,self.frozen,obs,actions,noise[offset:offset+self.microbatch],tau[offset:offset+self.microbatch])
                accumulator=self.accumulate(accumulator,gradient);metrics=self.add_successor_metrics(metrics,(loss,*aux))
                logs.append(dict(indices=host['indices'],state_token_dropped=host['state_token_dropped'],first_absolute_cursor=int(host['cursors'][0])))
        finally:batches.close()
        summary=np.asarray(self.successor_summary(metrics,accumulator))
        if not np.isfinite(summary).all()or summary[-1]<=0:raise ValueError('FINITE_NONZERO_REAL_SUCCESSOR_GRADIENT_REQUIRED')
        self.counts['completed_forward_backward']+=self.accumulation_steps;self.counts['completed_training_visits']+=32;counted('optimizer_update_attempts')
        self.active,self.opt_state,_=self.apply(self.active,self.opt_state,accumulator);jax.block_until_ready((self.active,self.opt_state))
        self.step+=1;self.cursor+=32;self.counts['optimizer_updates']+=1;counted('rng_banks_generated')
        row=dict(step=self.step,cursor=self.cursor,objective=float(summary[0]),mean_FM32_loss=float(summary[1]),active7=float(summary[2]),padding25=float(summary[3]),physical_endpoint_loss=float(summary[4]),eligible_tau_fraction=float(summary[5]),grad_norm_before_clip=float(summary[6]),used_lr=self.learning_rate_values[self.step-1],micro=logs,wall_s=time.perf_counter()-start,GPU_memory=jax.devices()[0].memory_stats())
        with(self.output/'updates.private.jsonl').open('a')as f:f.write(__import__('json').dumps(row)+'\n');f.flush()
        return row

    def generate(self,index,noise_id):
        import jax
        from .model_v1 import observation
        item=self.dataset[index];obs,_=observation(item,self.tokenizer);noise=jax.random.normal(jax.random.PRNGKey(2026111200+noise_id),(1,40,32))
        self.counts['ordinary_RGB_generation_attempts']+=1;value=self.infer(self.active,self.frozen,obs,noise);jax.block_until_ready(value);value=np.asarray(value)
        state,*_=self.dataset.raw(index);ep,_=self.dataset.rows[index];trajectory,_=physical(value,state,self.dataset.frames[ep['id']]['T_policy_from_mount'],self.dataset.statistics,self.dataset.reference(index));self.counts['ordinary_RGB_generations']+=1
        return value,trajectory
