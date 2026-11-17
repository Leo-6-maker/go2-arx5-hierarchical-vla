"""Source-only native dual-arm H50 FM32, with frozen official dense parent."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,inspect,json,os,time
from pathlib import Path
import numpy as np
from .runtime_v1 import Engine,read_pin
from .checkpoint_v1 import sha
from .config_v1 import SEED
from .model_v1 import load_parent,observation
from .source_native_v6 import SourceDataset,SCHEMA
from .source_batch_v6 import SourceTokenizer,draw_bank,prepared_batches,host_batch

PLAN='ae3d3c0dceade437b0104a150cd0402fef650052'
RECIPE='SLA_SOURCE_NATIVE_DUAL_ARM_H50_FM1000_V6'
PI0_SHA='24b32d7c6ed5e409459afa8ca5af96a4dae09c2dcdf48e828907e411f4f12342'
CONFIG_SHA='e15d71586f1a561b35c5651227607613fb16d2df411f3708b1c9d36b02544285'


class SourceEngine:
    checkpoint=Engine.checkpoint
    resume=Engine.resume
    residency=Engine.residency
    verify_frozen=Engine.verify_frozen

    def __init__(self,spec,output):
        import jax,jax.numpy as jnp,flax.nnx as nnx,optax
        from openpi.models import pi0,pi0_config
        from openpi.training.optimizer import AdamW
        from openpi_client.image_tools import resize_with_pad
        from . import batch_v2
        from .batch_v2 import accumulation_steps
        from .mfree_training_v1 import learning_rates
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields
        from scripts.sla.close_goal_g1_v1 import make_sampler
        if(spec['plan_commit']!=PLAN or spec['operation']!=RECIPE or spec['microbatch']not in(1,2,4)
           or spec['planned_candidate_updates']!=1000 or spec['new_training_visits_budget']!=32000
           or spec['training_matmul_precision']!='highest' or spec['physics'] or spec['new_physical_requests_budget']):
            raise ValueError('EXPLICIT_SOURCE_NATIVE_FIXED1000_RECIPE_REQUIRED')
        if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise ValueError('ONE_ACTUAL_GPU_REQUIRED')
        if sha(inspect.getfile(pi0))!=PI0_SHA or sha(inspect.getfile(pi0_config))!=CONFIG_SHA:
            raise ValueError('OFFICIAL18896_NATIVE_MODEL_BYTES_REQUIRED')
        rule=read_pin(spec['initial_function_rule'])
        if os.environ.get('XLA_FLAGS')!=rule['XLA_FLAGS']:raise ValueError('REGISTERED_XLA_FLAGS_REQUIRED')
        asset=read_pin(spec['parent_manifest'])
        if(asset['repo']!='haofuly/SLA_base' or not asset['all_files_verified']
           or asset['revision']!='5bbdf3012ab643554aa2eb83ddeae9ace3dce476'
           or Path(spec['parent_params']).resolve()!=(Path(spec['parent_manifest']['path']).parent/'model/params').resolve()
           or sha(spec['tokenizer']['path'])!=spec['tokenizer']['sha256']):raise ValueError('VERIFIED_OFFICIAL_BASE_AND_TOKENIZER_REQUIRED')
        self.spec=spec;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        jax.config.update('jax_compilation_cache_dir',spec['compilation_cache_dir'])
        self.dataset=SourceDataset(spec['dataset'],spec['source_manifest_sha256'],split='TRAIN',resize_with_pad=resize_with_pad)
        self.dev=SourceDataset(spec['dataset'],spec['source_manifest_sha256'],split='SOURCE_DEV',resize_with_pad=resize_with_pad)
        self.tokenizer=SourceTokenizer(spec['tokenizer']['path']);m=self.dataset.manifest
        path=Path(spec['dataset'])/m['visit_map']['file']
        if sha(path)!=m['visit_map']['sha256']:raise ValueError('SEALED_SOURCE32000_VISITS_REQUIRED')
        self.visit_map=np.load(path,allow_pickle=False)
        if self.visit_map.shape!=(32000,)or self.visit_map.min()<0 or self.visit_map.max()>=len(self.dataset):
            raise ValueError('TYPED_SOURCE32000_INDICES_REQUIRED')
        self.learning_rate_values=learning_rates();lr_sha=hashlib.sha256(''.join(x.hex()+'\n'for x in self.learning_rate_values).encode()).hexdigest()
        if lr_sha!=spec['learning_rates_sha256']:raise ValueError('PINNED_MFREE1000_LR_REQUIRED')
        self.step=self.cursor=0;self.microbatch=spec['microbatch'];self.accumulation_steps=accumulation_steps(self.microbatch)
        self.rng=jax.device_put(jax.random.PRNGKey(SEED));self.counts=dict(actual_parent_loads=0,optimizer_updates=0,
            forward_backward_attempts=0,completed_forward_backward=0,attempted_training_visits=0,completed_training_visits=0,
            ordinary_RGB_generation_attempts=0,ordinary_RGB_generations=0,physical_requests=0)
        self.graph,self.active,self.frozen,self.inventory=load_parent(spec['parent_params'],host_cast=True,action_horizon=50)
        self.counts['actual_parent_loads']=1;self.initial_frozen=self.inventory['frozen_fingerprint']
        (self.output/'PARAMETERS.private.json').write_text(json.dumps(self.inventory,indent=2))
        self.bindings=dict(recipe=RECIPE,plan_commit=PLAN,parent_manifest=spec['parent_manifest'],tokenizer=spec['tokenizer']['sha256'],
            source_manifest_sha256=spec['source_manifest_sha256'],statistics_sha256=m['statistics']['sha256'],
            visit_map_sha256=m['visit_map']['sha256'],lr_sha256=lr_sha,seed=SEED,frozen_fingerprint=self.initial_frozen,
            schema=SCHEMA,horizon=50,active_dimensions=14,padding_dimensions=18,full_task_instruction=True,three_current_RGB=True,
            state_token_dimensions=14,state_token_dropout=.25,optimizer=dict(b1=.9,b2=.95,eps=1e-8,weight_decay=1e-5,clip_norm=1.),
            source_sha=spec['source_sha'],source_seal=spec['source_seal'],fresh_Adam=True,SHIFT7=False,source_dispatch='UNKNOWN',
            official_pi0_sha256=PI0_SHA,official_config_sha256=CONFIG_SHA,XLA_FLAGS=rule['XLA_FLAGS'],microbatch=self.microbatch)
        rate=jnp.asarray(self.learning_rate_values,jnp.float32)
        self.tx=AdamW(b1=.9,b2=.95,eps=1e-8,weight_decay=1e-5,clip_gradient_norm=1.).create(lambda n:rate[jnp.minimum(n,999)])
        self.opt_state=self.tx.init(self.active);graph=self.graph;fields,_=extracted_fields(batch_v2,pi0)
        def objective(active,frozen,obs,actions,noise,tau):
            with jax.default_matmul_precision('highest'):
                t=tau[:,None,None];field=fields['train_field'](nnx.merge(graph,active,frozen),obs,t*noise+(1-t)*actions,tau)
                error=(field-(noise-actions))**2
                return jnp.mean(error),(jnp.mean(error[...,:14]),jnp.mean(error[...,14:]))
        self.grad=jax.jit(jax.value_and_grad(objective,has_aux=True));self.draw_bank=jax.jit(draw_bank)
        divisor=self.accumulation_steps
        self.accumulate=jax.jit(lambda a,g:jax.tree.map(lambda x,y:x+y/divisor,a,g),donate_argnums=(0,))
        self.add_metrics=jax.jit(lambda a,v:a+jnp.stack(v)/divisor,donate_argnums=(0,))
        self.summary=jax.jit(lambda v,g:jnp.r_[v,optax.global_norm(g)])
        def apply(active,state,grad):
            updates,state=self.tx.update(grad,state,active)
            return optax.apply_updates(active,updates),state
        self.apply=jax.jit(apply,donate_argnums=(0,1))
        sampler=make_sampler(graph,fields,cached=True,precision=None)
        self.infer=lambda a,f,o,n:sampler(a,f,o,n)[0]
        self.loader=ThreadPoolExecutor(max_workers=1)

    def close(self):self.loader.shutdown(wait=True,cancel_futures=True)

    def diagnose(self):
        import jax,jax.numpy as jnp
        from .batch_v2 import device_batch
        host=host_batch(self.dataset,self.tokenizer,self.visit_map,0,self.microbatch);obs,actions=device_batch(host)
        noise,tau=self.draw_bank(self.rng,np.arange(32,dtype=np.uint32))
        self.counts['diagnostic_backward_attempts']=1;self.counts['diagnostic_windows']=self.microbatch
        (value,parts),grad=self.grad(self.active,self.frozen,obs,actions,noise[:self.microbatch],tau[:self.microbatch])
        summary=np.asarray(self.summary(jnp.r_[value,jnp.stack(parts)],grad))
        if not np.isfinite(summary).all()or summary[-1]<=0:raise ValueError('SOURCE_REAL_FINITE_NONZERO_GRADIENT_REQUIRED')
        self.counts['diagnostic_backward_completed']=1
        return dict(backward_complete=True,windows=self.microbatch,loss=summary.tolist(),optimizer_updates=0,optimization_visits=0,
            GPU_memory=jax.devices()[0].memory_stats(),scope='S_NATIVE_H50_NO_TARGET_COMPILER')

    def update(self):
        import jax,jax.numpy as jnp
        from .batch_v2 import device_batch
        start=time.perf_counter();acc=jax.tree.map(jnp.zeros_like,self.active);metrics=jnp.zeros(3,jnp.float32);logs=[]
        noise,tau=self.draw_bank(self.rng,np.arange(self.cursor,self.cursor+32,dtype=np.uint32))
        def count(k):self.counts[k]=self.counts.get(k,0)+1
        batches=prepared_batches(self.dataset,self.tokenizer,self.visit_map,self.cursor,self.microbatch,executor=self.loader,on_read=count)
        try:
            for host in batches:
                if time.time()>=self.spec['deadline_epoch']-180:raise TimeoutError('SOURCE_DEADLINE_BEFORE_UPDATE_COMMIT')
                obs,actions=device_batch(host);offset=int(host['cursors'][0])-self.cursor
                count('forward_backward_attempts');self.counts['attempted_training_visits']+=self.microbatch
                (value,parts),grad=self.grad(self.active,self.frozen,obs,actions,noise[offset:offset+self.microbatch],tau[offset:offset+self.microbatch])
                acc=self.accumulate(acc,grad);metrics=self.add_metrics(metrics,(value,*parts))
                logs.append(dict(indices=host['indices'],first_absolute_cursor=int(host['cursors'][0]),state_token_dropped=host['state_token_dropped'],host_wait_s=host['host_wait_s']))
        finally:batches.close()
        result=np.asarray(self.summary(metrics,acc))
        if not np.isfinite(result).all():raise ValueError('SOURCE_NONFINITE_LOSS_OR_GRADIENT_NO_UPDATE')
        self.counts['completed_forward_backward']+=self.accumulation_steps;self.counts['completed_training_visits']+=32;count('optimizer_update_attempts')
        self.active,self.opt_state=self.apply(self.active,self.opt_state,acc);jax.block_until_ready((self.active,self.opt_state))
        self.step+=1;self.cursor+=32;count('optimizer_updates')
        row=dict(step=self.step,cursor=self.cursor,FM32=float(result[0]),active14=float(result[1]),padding18=float(result[2]),
            grad_norm_before_clip=float(result[3]),used_lr=self.learning_rate_values[self.step-1],micro=logs,wall_s=time.perf_counter()-start,
            GPU_memory=jax.devices()[0].memory_stats())
        with(self.output/'updates.private.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
        return row

    def generate(self,index,noise_id,*,dataset=None):
        import jax
        d=self.dataset if dataset is None else dataset
        obs,_=observation(d[index],self.tokenizer)
        noise=jax.random.normal(jax.random.PRNGKey(noise_id),(1,50,32));self.counts['ordinary_RGB_generation_attempts']+=1
        result=self.infer(self.active,self.frozen,obs,noise);jax.block_until_ready(result);normalized=np.asarray(result)
        if normalized.shape!=(1,50,32)or not np.isfinite(normalized).all():raise ValueError('SOURCE_FINITE_FULL_H50_REQUIRED')
        decoded=d.statistics['actions'].unnormalize(normalized[0,:,:14]);self.counts['ordinary_RGB_generations']+=1
        return normalized,decoded
