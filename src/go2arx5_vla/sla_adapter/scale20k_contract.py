"""Immutable data, recipe, and optimizer routing for the SCALE20K arms."""
from collections import defaultdict
import hashlib
import math

import numpy as np


PLAN = '926226914a0c08ee1ebb143e0edea1434728107b'
RECIPE = 'SLA_SCALE20K_DOMAIN_ISOLATED_V1'
PARENT_REVISION = '5bbdf3012ab643554aa2eb83ddeae9ace3dce476'
DOMAINS = ('target', 'robodojo', 'libero')
HORIZONS = dict(zip(DOMAINS, (40, 50, 10), strict=True))
BATCH = {
    'SCALE-MIX20K': dict(zip(DOMAINS, (32, 48, 48), strict=True)),
    'SCALE-TGT5K': dict(zip(DOMAINS, (32, 0, 0), strict=True)),
}
CHECKPOINTS = {'SCALE-MIX20K': (500, 2000, 5000, 10000, 20000), 'SCALE-TGT5K': (500, 2000, 5000)}
DOMAIN_WEIGHT = dict(zip(DOMAINS, (1., 1.5, 1.5), strict=True))
BANK_SEEDS = dict(zip(DOMAINS, (2026111901, 2026111902, 2026111903), strict=True))
PRIVATE_ROOTS = frozenset(('action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out'))
LORA_NAMES = frozenset(('lora_a', 'lora_b'))


def is_lora_parameter(name):
    # Attention layers use separate fields. The upstream MLP stores both
    # factors alongside the original einsum under compound field names.
    return name in LORA_NAMES or name.endswith(('_lora_a', '_lora_b'))


def learning_rates():
    peak, final, warmup, total = 2e-5, 2e-6, 500, 20000
    return np.asarray([peak*k/warmup if k <= warmup else
        final + .5*(peak-final)*(1+math.cos(math.pi*(k-warmup)/(total-warmup)))
        for k in range(1,total+1)],dtype='<f8')


def bank_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def cursors(arm, step):
    if type(step) is not int or not 0 <= step <= CHECKPOINTS[arm][-1]:
        raise ValueError('SCALE20K_COMPLETE_GLOBAL_UPDATE_REQUIRED')
    return {domain:step*size for domain,size in BATCH[arm].items()}


def visits(episodes, domain):
    """Task-before-episode sampling over unique, explicitly admitted windows."""
    groups = defaultdict(list)
    names = set()
    global_index = 0
    for episode in episodes:
        if episode['episode'] in names or episode['split'] != 'TRAIN' or not episode['admitted']:
            raise ValueError('SCALE20K_REQUIRES_UNIQUE_ADMITTED_TRAIN_EPISODES')
        names.add(episode['episode'])
        task, count = episode['task'], episode['window_count']
        if not isinstance(task,str) or not task.strip() or type(count) is not int or count <= 0:
            raise ValueError('SCALE20K_EPISODE_REQUIRES_TASK_AND_VALID_WINDOW_COUNT')
        groups[task].append((global_index,count))
        global_index += count
    if not groups:
        raise ValueError('SCALE20K_TRAIN_DOMAIN_MUST_HAVE_REAL_EPISODES')

    tasks = sorted(groups)
    rng = np.random.Generator(np.random.PCG64(BANK_SEEDS[domain]))
    total = 20000*BATCH['SCALE-MIX20K'][domain]
    assignment = rng.permutation(len(tasks))
    order = np.tile(assignment,math.ceil(total/len(tasks)))[:total]
    output = np.empty(total,dtype='<i8')
    for index,task in enumerate(tasks):
        locations = np.flatnonzero(order == index)
        ranges = np.asarray(groups[task],dtype=np.int64)
        choices = ranges[rng.integers(len(ranges),size=len(locations))]
        output[locations] = choices[:,0]+rng.integers(choices[:,1])
    return output


def parameter_partition(paths):
    """Classify the actual PaliGemma tree from upstream's two-expert naming."""
    groups = dict(frozen=[], vlm_lora=[], action_expert=[], domain_template=[])
    seen = set()
    for path in paths:
        path = tuple(path)
        if not path or path in seen:
            raise ValueError('SCALE20K_PARAMETER_PATHS_MUST_BE_NONEMPTY_AND_UNIQUE')
        seen.add(path)
        group = 'frozen'
        if path[0] in PRIVATE_ROOTS:
            group = 'domain_template'
        elif path[:2] == ('PaliGemma','llm'):
            if any(part.endswith('_1') for part in path[2:]):
                group = 'action_expert'
            elif is_lora_parameter(path[-1]):
                group = 'vlm_lora'
        groups[group].append(path)
    if (not all(groups.values()) or {p[0] for p in groups['domain_template']} != PRIVATE_ROOTS
            or not any(not is_lora_parameter(p[-1]) for p in groups['action_expert'])):
        raise ValueError('SCALE20K_REQUIRES_FROZEN_VISION_VLM_LORA_DENSE_EXPERT_AND_PRIVATE_IO')
    return {group:tuple(sorted(members)) for group,members in groups.items()}


def scale_graph_config(domain):
    from openpi.models.pi0_config import Pi0Config

    return Pi0Config(pi05=True,dtype='bfloat16',action_dim=32,action_horizon=HORIZONS[domain],
                     paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m',max_token_len=200)


def make_adamw(*, beta1, beta2, eps, weight_decay, clip_norm):
    """Use a single clip for the joint update and no inactive-domain invocation."""
    import jax
    import optax

    transform = optax.chain(optax.scale_by_adam(b1=beta1,b2=beta2,eps=eps),
                            optax.add_decayed_weights(weight_decay))

    @jax.jit
    def apply_partition(parameters,state,gradient,lr,scale):
        clipped = jax.tree.map(lambda leaf:leaf*scale,gradient)
        direction,next_state = transform.update(clipped,state,parameters)
        updates = jax.tree.map(lambda leaf:-lr*leaf,direction)
        return optax.apply_updates(parameters,updates),next_state

    def initialize(parameters):
        return {key:transform.init(value) for key,value in parameters.items()}

    def apply(parameters, states, gradient, lr):
        keys = set(gradient)
        if keys not in ({'shared','target'},{'shared',*DOMAINS}) or set(parameters) != {'shared',*DOMAINS} or set(states) != set(parameters):
            raise ValueError('SCALE20K_WRONG_ACTIVE_DOMAINS_OR_OPTIMIZER_PARTITION')
        norm = float(optax.global_norm(gradient))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('SCALE20K_NONFINITE_OR_ZERO_OPTIMIZER_GRADIENT')
        scale = min(1.,clip_norm/norm)
        next_parameters, next_states = dict(parameters),dict(states)
        for key,value in gradient.items():
            next_parameters[key],next_states[key] = apply_partition(parameters[key],states[key],value,lr,scale)
        return next_parameters,next_states,norm

    return initialize,apply
