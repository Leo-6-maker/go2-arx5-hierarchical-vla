"""Full generated H40 objective and a differentiable ten-step sampler.

No physical output processing. The existing deployment sampler stays intact.
"""

def acceleration_samples(xyz, *, xp):
    velocity=xp.diff(xyz,axis=-2)/.05
    zero=xp.zeros_like(velocity[...,:1,:])
    delta=xp.diff(xp.concatenate((zero,velocity,zero),axis=-2),axis=-2)/.05
    square=xp.sum(delta*delta,axis=-1)
    # Match the original Euclidean norm, with its finite zero subgradient.
    return xp.where(square>0,xp.sqrt(xp.where(square>0,square,xp.ones_like(square))),xp.zeros_like(square))


def loss(normalized,q01,q99,*,xp):
    if normalized.shape[-2:]!=(40,32):raise ValueError('COMPLETE_NORMALIZED_H40_REQUIRED')
    xyz=(normalized[...,:3]+1)*.5*(q99[:3]-q01[:3]+1e-6)+q01[:3]
    acceleration=acceleration_samples(xyz,xp=xp)
    penalty=xp.maximum((acceleration-.20)/.25,0)**2
    return xp.mean(xp.max(penalty,axis=-1)),(xp.mean(acceleration[...,0]),xp.mean(acceleration[...,-1]),xp.mean(xp.max(acceleration[...,1:-1],axis=-1)))


def aux_slots(step):
    if type(step)is not int or step<0:raise ValueError('COMMITTED_NONNEGATIVE_UPDATE_REQUIRED')
    first=4*(step%8)
    return tuple(range(first,first+4))


def make_sampler(graph,fields,*,rematerialize=True,precision=None):
    """Same cached vector field and Euler recurrence as cached_actual.

    The deployed while_loop is not reverse-mode differentiable. A static
    ten-step scan preserves the recurrence and permits gradients through all
    ten evaluations. Numerical equality needs actual-model qualification.
    """
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    def sample(active,frozen,observation,noise):
        with jax.default_matmul_precision(precision):
            model=nnx.merge(graph,active,frozen)
            prefix=fields['prefix_field'](model,observation)
            def step(carry,_):
                value,tau=carry
                field=fields['cache_field'](model,*prefix,value,tau)
                return (value+(-1.0/10)*field,tau+(-1.0/10)),None
            body=jax.checkpoint(step,prevent_cse=False)if rematerialize else step
            (value,_),_=jax.lax.scan(body,(noise,jnp.asarray(1.,jnp.float32)),xs=None,length=10)
            return value
    return jax.jit(sample)
