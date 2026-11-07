"""Small independent TASK_ADAPTER primitives; frozen GR00T is untouched.

Inputs contain pixels, measured EEF state and the declared task instruction.
Object truth, Teacher phase, run IDs, scores and future states are not inputs.
The candidate exports to NumPy so native execution need not import Torch.
"""
import numpy as np
import copy
import os
from pathlib import Path

SCHEMA = 'WM02_STABLE_TASK_ADAPTER_V1'
COMMANDS = ('HOLD_OPEN', 'APPROACH_RED_4MM_OPEN')
SEEDS = (110801,110802,110803)
STEP_M = .004
INPUT_DIM, HIDDEN_DIM = 116,32


def need(ok, reason):
    if not ok:
        raise ValueError('STABLE_TASK_ADAPTER_' + reason)


def features(*, body_rgb, wrist_rgb, actual_target_eef10, task_command):
    """Explicit field selection prevents metadata or privileged-state leakage."""
    state = np.asarray(actual_target_eef10,dtype=float)
    need(state.shape == (10,) and np.isfinite(state).all() and 0 <= state[9] <= 1,
         'ACTUAL_FINITE_EEF10')
    need(task_command in COMMANDS, 'DECLARED_TASK_INSTRUCTION')
    values = []
    for image in (body_rgb,wrist_rgb):
        image = np.asarray(image)
        need(image.shape == (256,256,3) and image.dtype == np.uint8, 'ACTUAL_256_RGB_BYTES')
        rgb = image.astype(np.float32)
        thumbnail = rgb.reshape(4,64,4,64,3).mean(axis=(1,3)).reshape(-1)/255.
        red = (rgb[:,:,0] > 80) & (rgb[:,:,0] > 1.4*rgb[:,:,1]) & (rgb[:,:,0] > 1.4*rgb[:,:,2])
        ys,xs = np.nonzero(red)
        moments = [float(xs.mean()/255.*2-1),float(ys.mean()/255.*2-1),float(red.mean()),1.] if len(xs) else [0.,0.,0.,0.]
        values.extend([thumbnail,np.asarray(moments)])
    values.extend([state,np.asarray([float(task_command == name) for name in COMMANDS])])
    result = np.concatenate(values).astype(np.float32)
    need(result.shape == (INPUT_DIM,) and np.isfinite(result).all(), 'EXACT_DEPLOYABLE_FEATURES')
    return result


def normalizer(train_features):
    x = np.asarray(train_features,dtype=np.float32)
    need(x.ndim == 2 and x.shape[1] == INPUT_DIM and len(x) >= 2
         and np.isfinite(x).all(), 'TRAIN_ONLY_FINITE_FEATURE_MATRIX')
    return x.mean(axis=0),np.maximum(x.std(axis=0),np.float32(1e-3))


def make_model(mean, scale):
    import torch
    from torch import nn
    mean,scale = (np.asarray(v,dtype=np.float32) for v in (mean,scale))
    need(mean.shape == scale.shape == (INPUT_DIM,) and np.isfinite(mean).all()
         and np.isfinite(scale).all() and (scale > 0).all(), 'OWN_TRAIN_NORMALIZER')
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('mean',torch.from_numpy(mean.copy()))
            self.register_buffer('scale',torch.from_numpy(scale.copy()))
            self.hidden = nn.Linear(INPUT_DIM,HIDDEN_DIM)
            self.output = nn.Linear(HIDDEN_DIM,3)

        def forward(self,x):
            raw = self.output(torch.tanh(self.hidden((x-self.mean)/self.scale)))
            # This is the candidate's explicit bounded output parameterization,
            # never clipping or rescaling a generated GR00T action.
            return STEP_M*raw/torch.sqrt(1.+raw.square().sum(dim=-1,keepdim=True))
    return Adapter()


def export_arrays(model):
    return dict(schema=np.array(SCHEMA),
        **{k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()})


def numpy_predict(arrays, features):
    need(str(arrays['schema']) == SCHEMA, 'EXACT_CANDIDATE_EXPORT')
    x = np.asarray(features,dtype=np.float32)
    need(x.shape[-1:] == (INPUT_DIM,) and np.isfinite(x).all(), 'FINITE_EXACT_STUDENT_INPUT')
    hidden = np.tanh(((x-arrays['mean'])/arrays['scale']) @ arrays['hidden.weight'].T + arrays['hidden.bias'])
    raw = hidden @ arrays['output.weight'].T + arrays['output.bias']
    return STEP_M*raw/np.sqrt(1.+np.square(raw).sum(axis=-1,keepdims=True))


def split_rows(rows):
    """Roles are fixed by new physical parents; diagnostics are never TRAIN."""
    need(rows and all(r.get('data_role') in ('TRAIN','DEV') for r in rows),
         'NEW_TRAIN_DEV_ONLY_NO_DIAGNOSTIC_RELABEL')
    roles = {}
    for row in rows:
        parent = row['parent_id']; role = row['data_role']
        need(parent not in roles or roles[parent] == role, 'PHYSICAL_PARENT_CROSSES_SPLIT')
        roles[parent] = role
    train = [i for i,r in enumerate(rows) if r['data_role'] == 'TRAIN']
    dev = [i for i,r in enumerate(rows) if r['data_role'] == 'DEV']
    need(train and dev, 'INDEPENDENT_TRAIN_AND_DEV_REQUIRED')
    return np.asarray(train,dtype=np.int64),np.asarray(dev,dtype=np.int64)


class Sampler:
    """Checkpointed permutation sampling, independent of global NumPy RNG."""
    def __init__(self,size,seed):
        need(type(size) is int and size > 0, 'NONEMPTY_TRAIN_SAMPLER')
        self.rng = np.random.default_rng(seed)
        self.permutation = self.rng.permutation(size)
        self.cursor,self.epoch = 0,0

    def take(self,count):
        need(type(count) is int and count > 0, 'POSITIVE_BATCH')
        result = []
        while len(result) < count:
            if self.cursor == len(self.permutation):
                self.permutation = self.rng.permutation(len(self.permutation))
                self.cursor = 0; self.epoch += 1
            end = min(len(self.permutation),self.cursor+count-len(result))
            result.extend(self.permutation[self.cursor:end].tolist());self.cursor = end
        return np.asarray(result,dtype=np.int64)

    def state(self):
        return dict(rng=copy.deepcopy(self.rng.bit_generator.state),
            permutation=self.permutation.tolist(),cursor=self.cursor,epoch=self.epoch)

    def restore(self,value):
        need(sorted(value['permutation']) == list(range(len(self.permutation)))
             and 0 <= value['cursor'] <= len(self.permutation) and value['epoch'] >= 0,
             'EXACT_SAMPLER_DATASET_AND_CURSOR')
        self.rng.bit_generator.state = copy.deepcopy(value['rng'])
        self.permutation = np.asarray(value['permutation'],dtype=np.int64)
        self.cursor,self.epoch = value['cursor'],value['epoch']


def save_checkpoint(path,model,optimizer,sampler,*,step,provenance,training_state=None):
    import torch
    from .neural_training import rng_state
    need(type(step) is int and step > 0 and isinstance(provenance,dict) and provenance,
         'NONZERO_CANDIDATE_UPDATES_AND_PROVENANCE')
    exported = {k:torch.from_numpy(v) for k,v in export_arrays(model).items() if k != 'schema'}
    with Path(path).open('xb') as stream:
        torch.save(dict(schema=SCHEMA,model=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler=sampler.state(),rng=rng_state(),step=step,provenance=provenance,
            export_schema=SCHEMA,export_weights=exported,GR00T_updates=0,
            training_state=copy.deepcopy(training_state)),stream)
        stream.flush();os.fsync(stream.fileno())


def restore_checkpoint(path,model,optimizer,sampler,*,expected_provenance,device='cpu'):
    import torch
    from .neural_training import restore_rng
    value = torch.load(path,map_location=device,weights_only=True)
    need(value['schema'] == value['export_schema'] == SCHEMA
         and value['provenance'] == expected_provenance and value['GR00T_updates'] == 0,
         'EXACT_CANDIDATE_DATA_AND_FROZEN_BASELINE')
    model.load_state_dict(value['model']);optimizer.load_state_dict(value['optimizer'])
    need(set(value['export_weights']) == set(model.state_dict())
         and all(torch.equal(v.cpu(),model.state_dict()[k].cpu())
                 for k,v in value['export_weights'].items()), 'EXACT_SAVED_EXPORT_WEIGHTS')
    sampler.restore(value['sampler']);restore_rng(value['rng'])
    return value['step']
