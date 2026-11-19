"""Explicit dual-arm H50 input path; the target H40 guard stays unchanged."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import time
import numpy as np
from .model_v1 import Tokenizer
from .batch_v2 import drop_state, accumulation_steps
from .source_native_v6 import CAMERAS


class SourceTokenizer(Tokenizer):
    def __call__(self,prompt,normalized_state32,*,drop_state=False):
        state=np.asarray(normalized_state32)
        if state.shape!=(32,) or np.any(state[14:]!=0):
            raise ValueError('NATIVE_TWO_ARM14_STATE_PLUS18_ZERO_CONTAINER_REQUIRED')
        return self.delegate.tokenize(prompt,None if drop_state else state[:14],use_discrete_state_format=True)


def host_batch(dataset,tokenizer,visit_map,start,microbatch,*,on_read=None,dropper=drop_state):
    accumulation_steps(microbatch);began=time.perf_counter();rows=[];tokens=[];masks=[];drops=[]
    indices=[int(i)for i in visit_map[start:start+microbatch]]
    if len(indices)!=microbatch:raise ValueError('COMPLETE_SOURCE_MICROBATCH_REQUIRED')
    for offset,index in enumerate(indices):
        if on_read:on_read('host_window_read_attempts')
        item=dataset[index]
        if on_read:on_read('host_window_reads')
        drop=dropper(start+offset);token,mask=tokenizer(item['prompt'],item['state'],drop_state=drop)
        rows.append(item);tokens.append(token);masks.append(mask);drops.append(drop)
    obs=dict(image={k:np.stack([r['image'][k]for r in rows])for k in CAMERAS},
        image_mask={k:np.asarray([r['image_mask'][k]for r in rows],bool)for k in CAMERAS},
        image_padding_mask={k:np.stack([r['image_padding_mask'][k]for r in rows])for k in CAMERAS},
        state=np.stack([r['state']for r in rows]),tokenized_prompt=np.stack(tokens),tokenized_prompt_mask=np.stack(masks))
    actions=np.stack([r['actions']for r in rows])
    if obs['state'].shape!=(microbatch,32)or actions.shape!=(microbatch,50,32)or np.any(actions[...,14:]!=0):
        raise ValueError('SOURCE_H50_ACTIVE14_PADDING18_REQUIRED')
    if any(obs['image'][k].shape!=(microbatch,224,224,3)for k in CAMERAS):raise ValueError('THREE_CURRENT_RGB224_REQUIRED')
    return dict(observation=obs,actions=actions,indices=indices,cursors=np.arange(start,start+microbatch,dtype=np.uint32),
        state_token_dropped=drops,prepare_wall_s=time.perf_counter()-began)


def prepared_batches(dataset,tokenizer,visit_map,cursor,microbatch,*,executor=None,on_read=None):
    if cursor%32 or cursor+32>len(visit_map):raise ValueError('SOURCE_COMPLETE_EFFECTIVE32_BOUNDARY_REQUIRED')
    starts=[cursor+i*microbatch for i in range(accumulation_steps(microbatch))]
    with(ThreadPoolExecutor(max_workers=1)if executor is None else nullcontext(executor))as pool:
        future=pool.submit(host_batch,dataset,tokenizer,visit_map,starts[0],microbatch,on_read=on_read)
        for i in range(len(starts)):
            began=time.perf_counter();batch=future.result();wait=time.perf_counter()-began
            if i+1<len(starts):future=pool.submit(host_batch,dataset,tokenizer,visit_map,starts[i+1],microbatch,on_read=on_read)
            yield dict(batch,host_wait_s=wait)


def draws(rng,cursors):
    import jax,jax.numpy as jnp
    keys=jax.vmap(lambda i:jax.random.fold_in(rng,i))(jnp.asarray(cursors,jnp.uint32))
    parts=jax.vmap(lambda k:jax.random.split(k,3))(keys)
    noise=jax.vmap(lambda k:jax.random.normal(k,(1,50,32)))(parts[:,1])[:,0]
    tau=jax.vmap(lambda k:jax.random.beta(k,1.5,1,(1,)))(parts[:,2])[:,0]
    return noise,tau*.999+.001


def draw_bank(rng,cursors):
    if cursors.shape!=(32,):raise ValueError('SOURCE_FIXED32_ABSOLUTE_CURSOR_BANK_REQUIRED')
    return draws(rng,cursors)
