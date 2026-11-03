"""Full deterministic WM optimizer/bootstrap/RNG recovery."""
import random
from pathlib import Path
import numpy as np
import torch


def save(path, model, optimizers, bootstrap, *, updates, replay_identity, source_sha, recipe):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve existing model endpoint')
    state = dict(schema='WM01_FULL_CHECKPOINT_V1', model=model.state_dict(),
        optimizers=[o.state_dict() for o in optimizers], bootstrap=[r.bit_generator.state for r in bootstrap],
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        numpy_rng=np.random.get_state(), python_rng=random.getstate(), updates=updates,
        replay_identity=replay_identity, source_sha=source_sha, recipe=recipe,
        replay_cursor=updates*recipe['effective_batch'], actor_updates=0)
    with path.open('xb') as stream:
        torch.save(state, stream)
        stream.flush()
        __import__('os').fsync(stream.fileno())


def restore(path, model, optimizers, bootstrap, *, replay_identity, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state['schema'] != 'WM01_FULL_CHECKPOINT_V1' or state['replay_identity'] != replay_identity:
        raise ValueError('Checkpoint/replay identity mismatch')
    model.load_state_dict(state['model'])
    for optimizer, value in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(value)
    for rng, value in zip(bootstrap, state['bootstrap']):
        rng.bit_generator.state = value
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    return state
