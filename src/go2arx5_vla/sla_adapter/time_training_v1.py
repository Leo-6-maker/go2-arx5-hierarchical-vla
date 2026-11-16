"""One allowed representation fallback; ordinary FM, original C20 statistics."""
from pathlib import Path
import numpy as np
from .mfree_training_v1 import MobileDataset
from .close_temporal_v1 import CloseEngine
from .runtime_v1 import read_pin
from .checkpoint_v1 import sha
from .gripper_time_v1 import VERSION,BLOCKS,decode
from .gripper_latent_v2 import VERSION as ORIGINAL_CODEC
from .codec_v1 import Codec,pad32

RECIPE='SLA_REFERENCE_CONTINUED_TIME_PARENT1000_NEW500_V1'


def physical(normalized,state,frame,statistics,reference):
    values=np.asarray(normalized)
    if values.shape!=(1,40,32)or not np.isfinite(values).all():raise ValueError('FINITE_H40_32_TIME_CONTAINER')
    active7=statistics['actions'].unnormalize(values[0,:,:7]);active7[:,6]=0.
    result=Codec(frame).decode(state,pad32(active7))
    width,detail=decode(values[0,:,6],reference,statistics['actions'].q01[6],statistics['actions'].q99[6])
    result[:,9]=(width-.002)/.040
    return result,detail


class TimeDataset(MobileDataset):
    def __init__(self,*args,time_audit,**kwargs):
        super().__init__(*args,**kwargs)
        audit=read_pin(time_audit)
        if(audit['schema']!='SLA_REFERENCE_CONTINUED_TIME_CODEC_C20_LABEL_AUDIT_V1'or not audit['passes']
                or audit['codec_version']!=VERSION or audit['dataset_manifest_sha256']!=self.manifest_sha
                or audit['statistics_sha256']!=self.manifest['statistics']['sha256']or audit['windows']!=len(self)):
            raise ValueError('ACTUAL_ORIGINAL_C20_TIME_CODEC_RECONSTRUCTION_REQUIRED')
        p=audit['target_bank'];q=audit['reference_contexts']
        if sha(p['path'])!=p['sha256']or sha(q['path'])!=q['sha256']:raise ValueError('IMMUTABLE_TIME_SIDECAR_BYTES')
        with np.load(p['path'],allow_pickle=False)as z:self.time_targets=z['normalized_gripper_parameters'].copy()
        self.contexts=read_pin(q)['contexts']
        if self.time_targets.shape!=(len(self),40)or self.time_targets.dtype!=np.float32 or not np.isfinite(self.time_targets).all():
            raise ValueError('FINITE_ALL_C20_PARAMETER_LABELS')
        for i,(ep,anchor)in enumerate(self.rows):
            c=self.contexts[i]
            if(c['index'],c['episode'],c['anchor'])!=(i,ep['id'],anchor):raise ValueError('EXACT_C20_CONTEXT_ROW_ALIGNMENT')

    def __getitem__(self,index):
        item=super().__getitem__(index)
        item['actions'][:,6]=self.time_targets[index]
        return item

    def reference(self,index):
        c=self.contexts[index]
        return {k:c[k]for k in ('width_m','source_native_s','query_native_s')}


class TimeEngine(CloseEngine):
    def dataset_for_spec(self,spec):
        from openpi_client.image_tools import resize_with_pad
        audit=read_pin(spec['target_audit'])
        if(spec['operation']!=RECIPE or spec['gripper_lambda']!=0. or spec['microbatch']!=4
                or spec['planned_candidate_updates']!=500 or spec['new_optimizer_updates_budget']!=500
                or spec['new_training_visits_budget']!=16000 or spec['training_matmul_precision']!='highest'
                or not audit['passes']or audit['manifest_sha256']!=spec['dense_manifest_sha256']):
            raise ValueError('ONE_REPRESENTATION_ONLY_FALLBACK_EXACT500')
        return TimeDataset(spec['dataset'],spec['dense_manifest_sha256'],resize_with_pad=resize_with_pad,
                           max_cache_bytes=spec['frame_cache_bytes'],time_audit=spec['time_codec_audit'])

    def __init__(self,spec,output):
        super().__init__(spec,output)
        import jax
        import jax.numpy as jnp
        import flax.nnx as nnx
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields
        from . import batch_v2
        from openpi.models import pi0
        self.bindings.update(recipe=RECIPE,codec_version=VERSION,statistics_codec_version=ORIGINAL_CODEC,
            time_codec_audit=spec['time_codec_audit'],time_parameter_slots=[list(pair)for pair in BLOCKS],
            auxiliary='DISABLED_ORDINARY_FM_ON_TIME_PARAMETER_LABELS',previous_reference_used_by_decoder=True,
            teacher_phase_input=False,extra_history_tokens=False,raw_width_postprocessing=False)
        fields,_=extracted_fields(batch_v2,pi0);graph=self.graph
        def objective(active,frozen,obs,actions,noise,tau):
            with jax.default_matmul_precision('highest'):
                expanded=tau[:,None,None];x=expanded*noise+(1-expanded)*actions
                field=fields['train_field'](nnx.merge(graph,active,frozen),obs,x,tau)
                error=(field-(noise-actions))**2;fm=jnp.mean(error)
                # Shared metric slots for disabled physical auxiliaries are zero.
                return fm,(fm,jnp.mean(error[...,:7]),jnp.mean(error[...,7:]),jnp.asarray(0.),jnp.asarray(0.))
        self.grad=jax.jit(jax.value_and_grad(objective,has_aux=True))

    def generate(self,index,noise_id):
        import jax
        from .model_v1 import observation
        item=self.dataset[index];obs,_=observation(item,self.tokenizer)
        noise=jax.random.normal(jax.random.PRNGKey(2026111200+noise_id),(1,40,32))
        self.counts['ordinary_RGB_generation_attempts']+=1
        value=self.infer(self.active,self.frozen,obs,noise);jax.block_until_ready(value);value=np.asarray(value)
        state,*_=self.dataset.raw(index);ep,_=self.dataset.rows[index]
        trajectory,_=physical(value,state,self.dataset.frames[ep['id']]['T_policy_from_mount'],
                              self.dataset.statistics,self.dataset.reference(index))
        self.counts['ordinary_RGB_generations']+=1
        return value,trajectory
