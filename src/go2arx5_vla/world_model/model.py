"""FP32 residual ensemble with legal SO(3) transitions and event logits."""
import torch
from torch import nn
from .features import STATE_DIM, ROTATIONS, FIELDS, SLICES


def exp_rotation(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    cross = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)
    angle = torch.linalg.vector_norm(vector, dim=-1)
    a = torch.sinc(angle/torch.pi)[..., None, None]
    b = (.5*torch.sinc(angle/(2*torch.pi))**2)[..., None, None]
    return torch.eye(3, dtype=vector.dtype, device=vector.device)+a*cross+b*(cross@cross)


def rotation_distance(left, right):
    relative = left@right.transpose(-1, -2)
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2
    skew = torch.stack((relative[..., 2, 1]-relative[..., 1, 2],
        relative[..., 0, 2]-relative[..., 2, 0], relative[..., 1, 0]-relative[..., 0, 1]), -1)/2
    sine = torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sine, cosine)


class Member(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        self.input = nn.Linear(STATE_DIM+56+3, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.delta = nn.Linear(width, STATE_DIM)
        self.rotation = nn.Linear(width, 3*len(ROTATIONS))
        self.events = nn.Linear(width, 4)
        self.activation = nn.SiLU()

    def forward(self, state, action, goal, mean, scale):
        # Commands retain physical units; constant scaling is shared across runs.
        action_scale = action.new_tensor([.01]*3+[.05]*3+[1.]).repeat(8)
        h = self.activation(self.input(torch.cat(((state-mean)/scale, action/action_scale, goal), -1)))
        for layer in self.hidden:
            h = h+self.activation(layer(h))
        delta = self.delta(h)*scale
        proposed = state+delta
        rotations = self.rotation(h).reshape(*state.shape[:-1], len(ROTATIONS), 3)
        # Avoid in-place edits to tensors needed by autograd through four steps.
        parts = []
        r = 0
        for name, _, group in FIELDS:
            sl = SLICES[name]
            if group == 'rotation':
                updated = exp_rotation(rotations[..., r, :])@state[..., sl].reshape(*state.shape[:-1], 3, 3)
                parts.append(updated.flatten(-2)); r += 1
            elif name == 'time_s':
                parts.append(state[..., sl]+.4)
            else:
                parts.append(proposed[..., sl])
        return torch.cat(parts, -1), self.events(h)


class Ensemble(nn.Module):
    def __init__(self, mean, scale, *, members=3, width=512):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32).clone())
        self.members = nn.ModuleList([Member(width) for _ in range(members)])

    def forward(self, state, action, goal):
        return [m(state, action, goal, self.mean, self.scale) for m in self.members]


def state_loss(prediction, target, scale):
    groups = {}
    for name, _, group in FIELDS:
        if group == 'time':
            continue
        sl = SLICES[name]
        if group == 'rotation':
            loss = rotation_distance(prediction[..., sl].reshape(-1, 3, 3),
                                     target[..., sl].reshape(-1, 3, 3)).square().mean()
        else:
            loss = ((prediction[..., sl]-target[..., sl])/scale[sl]).square().mean()
        groups.setdefault(group, []).append(loss)
    return torch.stack([torch.stack(values).mean() for values in groups.values()]).mean()


def multistep_loss(model, member, states, actions, goals, targets, events):
    first, logits = member(states[:, 0], actions[:, 0], goals, model.mean, model.scale)
    one = state_loss(first, targets[:, 0], model.scale)
    imagined = states[:, 0]
    losses = []
    for k in range(actions.shape[1]):
        imagined, _ = member(imagined, actions[:, k], goals, model.mean, model.scale)
        losses.append(state_loss(imagined, targets[:, k], model.scale))
    rollout = torch.stack(losses).mean()
    event = nn.functional.binary_cross_entropy_with_logits(logits, events[:, 0])
    return one+rollout+.1*event, dict(one=one.detach(), rollout=rollout.detach(), event=event.detach())
