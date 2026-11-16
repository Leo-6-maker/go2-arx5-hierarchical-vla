"""One predeclared physical-width/time objective, with unchanged FM32 and PEFT."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .runtime_v3 import DeviceResidentEngine
from .mfree_training_v1 import MobileDataset, learning_rates
from .runtime_v1 import read_pin, probe_indices
from .mixed_visits_v4 import Catalog, MixedVisits
from .config_v1 import SEED
from .checkpoint_v1 import sha

RECIPE='SLA_CLOSE_PARENT1000_NEW500_MATCHED_V1'


def physical_gripper_terms(clean, target, tau, low, high, *, xp, sigmoid):
    """Robot-time deltas, not diffusion-time derivatives; no output clipping."""
    span=high-low+1e-6
    width=lambda x: .002+.040*sigmoid((x[...,6]+1)/2*span+low)
    a,b=width(clean),width(target)
    point=xp.mean(((a-b)/.040)**2,axis=-1)
    delta=xp.mean(((xp.diff(a,axis=-1)-xp.diff(b,axis=-1))/.002)**2,axis=-1)
    eligible=(tau<=.5).astype(point.dtype)
    # Zero high-tau samples, then average over the effective batch. This makes
    # micro4/8 accumulation independent of eligible counts in each partition.
    count=max(int(tau.shape[0]),1)
    return xp.sum(point*eligible)/count,xp.sum(delta*eligible)/count


class CloseEngine(DeviceResidentEngine):
    def dataset_for_spec(self,spec):
        from openpi_client.image_tools import resize_with_pad
        audit=read_pin(spec['target_audit'])
        if (spec['operation']!=RECIPE or spec['planned_candidate_updates']!=500
                or spec['new_optimizer_updates_budget']!=500 or spec['new_training_visits_budget']!=16000
                or spec['planned_target_visits']!=16000 or spec['microbatch']not in (4,8)
                or spec['training_matmul_precision']!='highest'
                or spec['gripper_lambda']not in (0.,1.)
                or audit['schema']!='MFREE_TRAIN_FP32_DEPLOYMENT_AUDIT_V1' or not audit['passes']
                or audit['manifest_sha256']!=spec['dense_manifest_sha256']):
            raise ValueError('EXACT_MATCHED_CLOSE500_C20_FROZEN_STATS_RECIPE')
        return MobileDataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad,
                             max_cache_bytes=spec['frame_cache_bytes'])

    def visits_for_dataset(self,dataset):
        catalog=Catalog('T',dataset.manifest_sha,dataset.catalog_rows())
        planner=MixedVisits('T100_3K',{'T':catalog},seed=SEED)
        self.selected_indices=probe_indices(dataset)
        return np.fromiter((v.row_index for step in range(500)for v in planner.batch(step)),dtype='<i8',count=16000)

    def learning_rate_values_for_spec(self,spec):return learning_rates()[:500]

    def __init__(self,spec,output):
        super().__init__(spec,output)
        import jax
        import jax.numpy as jnp
        import flax.nnx as nnx
        from openpi.models import pi0
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields
        from scripts.sla.close_goal_g1_v1 import make_sampler
        from . import batch_v2
        from .provider_v1 import active_rows
        package=Path(spec['C20_parent_package']['path'])
        if sha(package/'MANIFEST.private.json')!=spec['C20_parent_package']['sha256']:
            raise ValueError('UNCHANGED_C20_1000_PARENT_PACKAGE')
        parent=json.loads((package/'MANIFEST.private.json').read_bytes())
        if parent['step']!=1000 or parent['cursor']!=32000 or parent['bindings']['frozen_fingerprint']!=self.initial_frozen:
            raise ValueError('REAL_PARENT1000_AND_FROZEN_DENSE_IDENTITY')
        rows,structure=active_rows(dict(arrays=parent['adapter_arrays']),self.active);values=[]
        for row in rows:
            path=package/row['file']
            if sha(path)!=row['sha256']:raise ValueError('PARENT_TRAINED_ADAPTER_BYTES')
            values.append(jax.device_put(np.load(path,allow_pickle=False)))
        self.active,=jax.tree_util.tree_unflatten(structure,values)
        # A common NEW Adam is initialized on the unchanged trained function.
        self.opt_state=self.tx.init(self.active)
        self.bindings.update(recipe=RECIPE,parent_C20_package=spec['C20_parent_package'],parent_step=1000,
            parent_visits=32000,new_planned_updates=500,new_planned_visits=16000,
            codec_version=parent['bindings']['codec_version'],parent_host_cast_v7=True,
            derived_statistics_sha256=self.dataset.manifest['statistics']['sha256'],
            dense_manifest_sha256=self.dataset.manifest_sha,dataset_id=self.dataset.manifest['dataset_id'],
            boundary_version='MOBILE_ACCEPTED_BOUNDARY_050_V1',BASE_head=False,spatial_loss=False,
            reference_error_input=False,teacher_memory_input=False,state_token_dropout=.25,
            gripper_lambda=spec['gripper_lambda'],aux_tau_max=.5,
            auxiliary='MEAN_NORMALIZED_PHYSICAL_WIDTH_AND_ROBOT_TIME_DELTA_ERRORS',
            inference_route=spec['inference_route'],goal_plan_sha=spec['goal_plan_sha'])
        fields,_=extracted_fields(batch_v2,pi0);graph=self.graph
        low=jnp.asarray(self.dataset.statistics['actions'].q01[6],jnp.float32)
        high=jnp.asarray(self.dataset.statistics['actions'].q99[6],jnp.float32)
        coefficient=float(spec['gripper_lambda'])

        def objective(active,frozen,obs,actions,noise,tau):
            with jax.default_matmul_precision('highest'):
                expanded=tau[:,None,None];x=expanded*noise+(1-expanded)*actions
                field=fields['train_field'](nnx.merge(graph,active,frozen),obs,x,tau)
                error=(field-(noise-actions))**2
                fm=jnp.mean(error);active7=jnp.mean(error[...,:7]);padding=jnp.mean(error[...,7:])
                point,delta=physical_gripper_terms(x-expanded*field,actions,tau,low,high,
                                                   xp=jnp,sigmoid=jax.nn.sigmoid)
                return fm+coefficient*(point+delta),(fm,active7,padding,point,delta)

        self.grad=jax.jit(jax.value_and_grad(objective,has_aux=True))
        route=spec['inference_route']
        sample=make_sampler(graph,fields,cached=route.startswith('cached'),
                            precision='highest'if route.endswith('highest')else None)
        self.infer=lambda active,frozen,obs,noise:sample(active,frozen,obs,noise)[0]
        self.add_close_metrics=jax.jit(lambda acc,vals:acc+jnp.stack(vals)/self.accumulation_steps,donate_argnums=(0,))
        self.close_summary=jax.jit(lambda metrics,grad:jnp.r_[metrics,__import__('optax').global_norm(grad)])

    def update(self):
        import jax
        import jax.numpy as jnp
        from .batch_v2 import prepared_batches,device_batch
        start=time.perf_counter();accumulator=jax.tree.map(jnp.zeros_like,self.active)
        metrics=jnp.zeros((6,),jnp.float32);logs=[]
        self.counts['rng_bank_generation_attempts']=self.counts.get('rng_bank_generation_attempts',0)+1
        noise,tau=self.draw_bank(self.rng,np.arange(self.cursor,self.cursor+32,dtype=np.uint32))
        batches=prepared_batches(self.dataset,self.tokenizer,self.visit_map,self.cursor,self.microbatch,
                                 prefetch=True,executor=self.loader)
        try:
            for host in batches:
                if time.time()>=self.spec['deadline_epoch']:raise TimeoutError('FINITE_CLOSE500_BEFORE_COMMIT')
                obs,actions=device_batch(host);offset=int(host['cursors'][0])-self.cursor
                self.counts['forward_backward_attempts']+=1;self.counts['attempted_training_visits']+=self.microbatch
                (loss,aux),gradient=self.grad(self.active,self.frozen,obs,actions,
                    noise[offset:offset+self.microbatch],tau[offset:offset+self.microbatch])
                accumulator=self.accumulate(accumulator,gradient)
                metrics=self.add_close_metrics(metrics,(loss,*aux))
                logs.append(dict(indices=host['indices'],state_token_dropped=host['state_token_dropped'],
                                 first_absolute_cursor=int(host['cursors'][0])))
        finally:batches.close()
        summary=np.asarray(self.close_summary(metrics,accumulator))
        if not np.isfinite(summary).all()or summary[-1]<=0:raise ValueError('REAL_FINITE_NONZERO_CLOSE_GRADIENT_BEFORE_UPDATE')
        self.counts['completed_forward_backward']+=self.accumulation_steps
        self.counts['completed_training_visits']+=32
        self.counts['optimizer_update_attempts']=self.counts.get('optimizer_update_attempts',0)+1
        self.active,self.opt_state,_=self.apply(self.active,self.opt_state,accumulator)
        jax.block_until_ready((self.active,self.opt_state));self.step+=1;self.cursor+=32;self.counts['optimizer_updates']+=1
        row=dict(step=self.step,cursor=self.cursor,objective=float(summary[0]),mean_FM32_loss=float(summary[1]),
            active7=float(summary[2]),padding25=float(summary[3]),physical_width_loss=float(summary[4]),
            physical_width_delta_loss=float(summary[5]),grad_norm_before_clip=float(summary[6]),
            used_lr=self.learning_rate_values[self.step-1],micro=logs,wall_s=time.perf_counter()-start,
            GPU_memory=jax.devices()[0].memory_stats())
        with(self.output/'updates.private.jsonl').open('a')as handle:handle.write(json.dumps(row)+'\n');handle.flush()
        return row
