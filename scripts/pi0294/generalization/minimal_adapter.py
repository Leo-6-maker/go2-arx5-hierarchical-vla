"""Cold E1/E2 parameter isolation for the pinned N1.7 action head.

No loader, launcher, authority, model call or ledger mutation lives here. Apply
only to a separately qualified, fully loaded model before creating its optimizer.
The seven category-linear leaves preserve NVIDIA's W[in,out] / bmm convention.
"""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib

import torch
from torch import nn


SCHEMA = 'ARX5_MINIMAL_ADAPTER_PARAMETER_CONTRACT_V1'
UPSTREAM = '51d4c89f72fda44cbf77285c6a8114b52676b8a1'
ARMS = ('E1_ADAPTER_ONLY', 'E2_SHARED_ACTION_HEAD')
LEAVES = tuple('action_head.' + p for p in (
    'state_encoder.layer1', 'state_encoder.layer2',
    'action_encoder.W1', 'action_encoder.W2', 'action_encoder.W3',
    'action_decoder.layer1', 'action_decoder.layer2'))
SHARED = ('action_head.model.', 'action_head.vlln.', 'action_head.vl_self_attention.')
MODE = 'MATCHED_ACTION_HEAD_AND_BACKBONE_EVAL_WITH_AUTOGRAD_V1'


def need(condition, message):
    if not condition:
        raise ValueError(message)


class TargetCategoryLinear(nn.Module):
    """Only the selected category is an optimizer parameter, never a masked row.

    Keep full frozen tables for exact other-category dispatch and restoration.
    These include a redundant original target slice; report this storage honestly.
    Export reconstructs ordinary upstream W/b with the current target substituted.
    """
    def __init__(self, original, target):
        super().__init__()
        w, b = original.W, original.b
        need(isinstance(w, nn.Parameter) and isinstance(b, nn.Parameter)
             and w.ndim == 3 and b.shape == (w.shape[0], w.shape[2])
             and w.device == b.device and w.dtype == b.dtype
             and w.device.type != 'meta', 'REAL_LOADED_CATEGORY_TABLES_REQUIRED')
        need(type(target) is int and 0 <= target < w.shape[0], 'EXACT_TARGET_CATEGORY_REQUIRED')
        need(set(dict(original.named_parameters())) == {'W', 'b'}
             and not list(original.named_buffers()), 'ONLY_PINNED_CATEGORY_LINEAR_LAYOUT')
        self.target = target
        self.num_categories = w.shape[0]
        # The already-qualified upstream leaf type is needed for ordinary
        # inference export. It is not a child module or saved tensor.
        self._official_linear_type = type(original)
        self.register_buffer('frozen_W', w.detach().clone())
        self.register_buffer('frozen_b', b.detach().clone())
        self.target_W = nn.Parameter(w[target].detach().clone())
        self.target_b = nn.Parameter(b[target].detach().clone())

    def retain_frozen_tables_on_cpu(self):
        """Explicit target-only execution; retain every frozen category value."""
        self.frozen_W = self.frozen_W.to('cpu')
        self.frozen_b = self.frozen_b.to('cpu')
        self._cpu_frozen_table_dispatch = True
        return sum(x.numel()*x.element_size() for x in (self.frozen_W,self.frozen_b))

    def _apply(self, fn, recurse=True):
        # Trainer/Accelerate may place an already placed model again. The
        # explicitly retained tables must stay on CPU through that placement.
        kept = {}
        if getattr(self,'_cpu_frozen_table_dispatch',False):
            kept = {name:self._buffers.pop(name) for name in ('frozen_W','frozen_b')}
        try:
            return super()._apply(fn,recurse=recurse)
        finally:
            self._buffers.update(kept)

    def forward(self, x, cat_ids):
        need(x.ndim == 3 and cat_ids.ndim == 1 and cat_ids.shape[0] == x.shape[0]
             and cat_ids.dtype == torch.int64 and cat_ids.device == x.device
             and x.device == (self.target_W.device if getattr(self,'_cpu_frozen_table_dispatch',False)
                              else self.frozen_W.device) and x.shape[2] == self.frozen_W.shape[1],
             'PINNED_CATEGORY_BATCH_SHAPE_AND_DEVICE_REQUIRED')
        need(bool(((cat_ids >= 0) & (cat_ids < self.num_categories)).all()),
             'NO_NEGATIVE_OR_UNKNOWN_CATEGORY')
        selected = cat_ids == self.target
        if getattr(self,'_cpu_frozen_table_dispatch',False):
            need(self.frozen_W.device.type=='cpu' and self.frozen_b.device.type=='cpu'
                 and bool(selected.all()),'CPU_FROZEN_TABLES_REQUIRE_EXACT_TARGET_ONLY_DISPATCH')
            w = self.target_W[None].expand(x.shape[0],-1,-1)
            b = self.target_b[None].expand(x.shape[0],-1)
            return torch.bmm(x,w)+b.unsqueeze(1)
        w = torch.where(selected[:, None, None], self.target_W[None], self.frozen_W[cat_ids])
        b = torch.where(selected[:, None], self.target_b[None], self.frozen_b[cat_ids])
        # No no_grad: a frozen shared DiT must still backpropagate to this input.
        return torch.bmm(x, w) + b.unsqueeze(1)

    def official_tables(self):
        w, b = self.frozen_W.detach().clone(), self.frozen_b.detach().clone()
        w[self.target] = self.target_W.detach()
        b[self.target] = self.target_b.detach()
        return w, b


def _eval_before_forward(module, args):
    # Trainer.train() can recursively re-enable training mode every step. This
    # hook reasserts the SAME forward semantics in both arms, including E2.
    # eval() suppresses stochastic module updates, but DOES NOT disable autograd.
    module.eval()


class ParameterContract:
    def __init__(self, model, arm, target):
        need(arm in ARMS and type(target) is int, 'EXPLICIT_MATCHED_ARM_AND_TARGET')
        need(not any(isinstance(m, TargetCategoryLinear) for m in model.modules()),
             'ADAPTER_ALREADY_INSTALLED_NO_REAPPLICATION')
        head = model.get_submodule('action_head')
        need(head.state_dropout_prob == 0, 'ORIGINAL_STATE_DROPOUT_ZERO_REQUIRED')
        for attr in ('model', 'vlln', 'vl_self_attention'):
            head.get_submodule(attr)
        model.get_submodule('backbone')
        leaves = [(path, model.get_submodule(path)) for path in LEAVES]
        # Complete prevalidation before touching even requires_grad or modules.
        need(len({id(m) for _, m in leaves}) == len(LEAVES), 'NO_ALIASED_CATEGORY_LEAVES')
        parameter_ids = [id(p) for _, m in leaves for p in m.parameters()]
        need(len(set(parameter_ids)) == 2 * len(LEAVES), 'NO_TIED_CATEGORY_PARAMETERS')
        replacements = [(path, TargetCategoryLinear(m, target)) for path, m in leaves]
        model.requires_grad_(False)
        for path, replacement in replacements:
            parent, _, name = path.rpartition('.')
            setattr(model.get_submodule(parent), name, replacement)
        if arm == ARMS[1]:
            for name, p in model.named_parameters():
                if name.startswith(SHARED):
                    p.requires_grad_(True)
        self.model, self.arm, self.target = model, arm, target
        self.trainable_names = tuple(n for n, p in model.named_parameters() if p.requires_grad)
        target_names = {path + '.' + n for path in LEAVES for n in ('target_W', 'target_b')}
        need(target_names <= set(self.trainable_names), 'EVERY_TARGET_SLICE_IS_TRAINABLE')
        need(all(n in target_names or (arm == ARMS[1] and n.startswith(SHARED))
                 for n in self.trainable_names), 'NO_UNDECLARED_TRAINABLE_PARAMETERS')
        self.hooks = [model.get_submodule(name).register_forward_pre_hook(_eval_before_forward)
                      for name in ('action_head', 'backbone')]
        self.enforce_modes()

    def enforce_modes(self):
        self.model.action_head.eval()
        self.model.backbone.eval()

    def parameters(self):
        self.validate()
        return [p for _, p in self.model.named_parameters() if p.requires_grad]

    def validate(self):
        need(self.model.action_head.state_dropout_prob == 0,
             'ORIGINAL_STATE_DROPOUT_ZERO_REQUIRED')
        current = tuple(n for n, p in self.model.named_parameters() if p.requires_grad)
        need(current == self.trainable_names, 'TRAINABLE_WHITELIST_CHANGED')
        for name in LEAVES:
            module = self.model.get_submodule(name)
            need(isinstance(module, TargetCategoryLinear) and module.target == self.target,
                 'CATEGORY_DISPATCH_CHANGED')

    def describe(self):
        self.validate()
        params = dict(self.model.named_parameters())
        return dict(schema=SCHEMA, upstream_commit=UPSTREAM, arm=self.arm,
                    target_category=self.target, module_mode=MODE,
                    trainable=[dict(name=n, shape=list(params[n].shape), numel=params[n].numel())
                               for n in self.trainable_names],
                    trainable_numel=sum(params[n].numel() for n in self.trainable_names),
                    parameter_numel_including_target_slices=sum(p.numel() for p in params.values()),
                    frozen_buffer_numel=sum(b.numel() for b in self.model.buffers()),
                    real_model_qualification=False, native_adopted=False)


def export_official_state(model):
    """Return a detached ordinary upstream state dict, no adapter-only artifact.

    The caller must still bind config/processor/R0/category/source/plan and prove
    actual full-model reconstruction equivalence before scientific delivery.
    """
    out = OrderedDict((n, t.detach().clone()) for n, t in model.state_dict().items())
    for path, module in model.named_modules():
        if isinstance(module, TargetCategoryLinear):
            prefix = path + '.' if path else ''
            for key in ('frozen_W', 'frozen_b', 'target_W', 'target_b'):
                del out[prefix + key]
            out[prefix + 'W'], out[prefix + 'b'] = module.official_tables()
    return out


def consume_for_official_inference(contract):
    """One-way ordinary upstream reconstruction after the full save.

    This does no inference, disk write, checkpoint admission or accounting.
    The native caller must first seal the full wrapped training checkpoint,
    debit a load, save/reload the ordinary inference package, then use the
    original fixed_draw for the reserved same-input/seed output comparison.
    Reusing this consumed contract for any training or restore is forbidden.

    Reuse frozen table storage after substituting the learned target. Building
    a second whole12GB model/state_dict beside AdamW would inflate peak memory.
    The original full-state checkpoint on disk remains unchanged.
    """
    contract.validate()
    need(getattr(contract,'_training_session_used',False)
         and not getattr(contract,'_official_inference_consumed',False),
         'ONLY_COMPLETED_TRAINING_CONTRACT_ONCE_FOR_INFERENCE')
    leaves=[(name,contract.model.get_submodule(name))for name in LEAVES]
    replacements=[]
    # Validate all ordinary leaf constructors before consuming any table.
    for path,wrapped in leaves:
        with torch.random.fork_rng(devices=[]),torch.device('meta'):
            ordinary=wrapped._official_linear_type(wrapped.num_categories,
                wrapped.frozen_W.shape[1],wrapped.frozen_W.shape[2])
        need(set(dict(ordinary.named_parameters()))=={'W','b'}
             and not list(ordinary.named_buffers())
             and ordinary.W.shape==wrapped.frozen_W.shape
             and ordinary.b.shape==wrapped.frozen_b.shape,
             'EXACT_ORIGINAL_UPSTREAM_CATEGORY_LINEAR_EXPORT')
        replacements.append((path,wrapped,ordinary))
    contract._official_inference_consumed=True
    contract.model.requires_grad_(False)
    with torch.no_grad():
        for path,wrapped,ordinary in replacements:
            wrapped.frozen_W[wrapped.target].copy_(wrapped.target_W)
            wrapped.frozen_b[wrapped.target].copy_(wrapped.target_b)
            ordinary.W=nn.Parameter(wrapped.frozen_W.detach(),requires_grad=False)
            ordinary.b=nn.Parameter(wrapped.frozen_b.detach(),requires_grad=False)
            parent,_,name=path.rpartition('.')
            setattr(contract.model.get_submodule(parent),name,ordinary)
    for handle in contract.hooks:handle.remove()
    contract.hooks=[]
    contract.model.eval()
    need(not any(isinstance(m,TargetCategoryLinear)for m in contract.model.modules())
         and not any(p.requires_grad for p in contract.model.parameters()),
         'OFFICIAL_INFERENCE_HAS_NO_WRAPPER_OR_TRAINABLE_STATE')
    return contract.model


def restore_parameter_state(contract, state, saved_contract):
    """Restore the adapter architecture only after exact arm/slot-map binding.

    Optimizer/scheduler/RNG/sampler restoration remains the original Trainer's
    job. This guard prevents a tensor-compatible wrong-category/wrong-arm load.
    """
    need(saved_contract == contract.describe(), 'EXACT_ARM_SLOT_WHITELIST_RESTORE_REQUIRED')
    contract.model.load_state_dict(state, strict=True)
    contract.validate()
    contract.enforce_modes()


def frozen_fingerprint(model):
    """All frozen parameters AND buffers, not only optimizer parameter names."""
    tensors = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    tensors += [('buffer:' + n, b) for n, b in model.named_buffers()]
    result = {}
    for name, value in tensors:
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        result[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                            sha256=hashlib.sha256(raw).hexdigest())
    return result


def parameter_state_fingerprint(model):
    """Full wrapped model identity, including trainable slices and all buffers."""
    result = frozen_fingerprint(model)
    for name, value in model.named_parameters():
        if value.requires_grad:
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            result['trainable:' + name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                              sha256=hashlib.sha256(raw).hexdigest())
    return result


@contextmanager
def matched_flow_rng(pair_seed, *, step, microbatch, cuda_devices=()):
    """Arm-independent CPU Beta and device flow-noise stream per accumulation.

    Same total seed alone is insufficient if the arms execute different random
    operations. Include microbatch to avoid reusing a flow draw 16 times/step.
    Hold this context through backward when activation recomputation is used.
    Sampling/cursor RNG is separate. No real-model calls occur here.
    """
    need(all(type(x) is int and x >= 0 for x in (pair_seed, step, microbatch)),
         'EXACT_NONNEGATIVE_FLOW_COORDINATES')
    seed = int.from_bytes(hashlib.sha256(
        f'{SCHEMA}:flow:{pair_seed}:{step}:{microbatch}'.encode()).digest()[:8], 'big') % (2**63)
    devices = list(cuda_devices)
    need(len(devices) == len(set(devices)) and all(type(d) is int and d >= 0 for d in devices),
         'EXPLICIT_UNIQUE_CUDA_DEVICES')
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            torch.cuda.default_generators[device].manual_seed(seed)
        yield seed
