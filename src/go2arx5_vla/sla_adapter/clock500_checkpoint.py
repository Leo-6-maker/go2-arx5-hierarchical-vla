"""Durable branch checkpoints that cannot be mistaken for the formal ADAPT lineage."""
from pathlib import Path

from . import adapt5k_checkpoint, scale20k_checkpoint
from .clock500 import ARMS, ENDPOINT, PARENT, PLAN
from .runtime_v1 import read_pin


SCHEMA = 'SLA_CLOCK500_BRANCH_COMPLETE_STATE_V1'


def boundary(binding, phase):
    adapt = binding['original_adapt_contract']
    if (binding['plan']!=PLAN or binding['arm'] not in ARMS or binding['parent_phase']!=PARENT
            or not PARENT<phase<=ENDPOINT or (adapt['stage_start'],adapt['stage_stop'])!=(PARENT,5000)
            or not all(scale20k_checkpoint.is_digest(binding[key]['sha256'])
                       for key in ('parent_state','D2','comparison'))):
        raise ValueError('CLOCK500_COMPLETE_BRANCH_REQUIRES_PINNED_PARENT_D2_AND_ARM')
    return dict(schema=SCHEMA,binding=binding,phase_step=phase,parent_phase=PARENT,
        branch_updates=phase-PARENT,branch_training_H40=4*(phase-PARENT),
        progress=adapt5k_checkpoint.identity(adapt,phase),formal_ADAPT_parent_eligible=False,physics=False)


def save(path, *, phase, binding, parameters, optimizer, rng):
    import jax

    stack = scale20k_checkpoint.training_tree(parameters,optimizer,rng)
    if len(jax.tree.leaves(stack))!=140:
        raise ValueError('CLOCK500_REQUIRES_ALL_140_MODEL_ADAM_RNG_ARRAYS')
    return scale20k_checkpoint.seal_arrays(path,stack,boundary(binding,phase))


def read_complete(checkpoint, binding):
    root = Path(checkpoint['path']).resolve(strict=True)
    receipt = read_pin(dict(path=str(root/'COMPLETE.json'),sha256=checkpoint['sha256']))
    expected = boundary(binding,receipt['phase_step'])
    if {k:v for k,v in receipt.items() if k!='arrays'}!=expected or len(receipt['arrays'])!=140:
        raise ValueError('CLOCK500_BRANCH_STATE_IDENTITY_OR_140_ARRAYS_CHANGED')
    return root,receipt


def restore(checkpoint, binding, *, parameters, optimizer, rng):
    root,marker = read_complete(checkpoint,binding)
    template = scale20k_checkpoint.training_tree(parameters,optimizer,rng)
    return scale20k_checkpoint.restore_arrays(root,marker,template),marker['phase_step']
