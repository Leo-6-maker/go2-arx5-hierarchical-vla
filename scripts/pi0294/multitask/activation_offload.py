"""Lossless saved-activation storage; no parameter/optimizer/recipe change.

PyTorch saved_tensors_hooks moves eligible intermediates to host RAM and returns
the same values, dtype, shape and device on backward. Parameter/buffer storage
(including views) stays on device. Copies are synchronous, without quantization,
recomputation, RNG use, pinned-memory pools or a second model.
"""
from contextlib import nullcontext

POLICY = dict(schema='M32_LOSSLESS_SAVED_ACTIVATION_CPU_V1',
              minimum_tensor_bytes=65536, parameter_and_buffer_storage_offload=False,
              quantization=False, gradient_checkpointing=False,
              same_policy_on_both_devices_and_matched_arms=True)
LOW_RESIDENCY_POLICY = {k:v for k,v in POLICY.items() if k!='parameter_and_buffer_storage_offload'}
LOW_RESIDENCY_POLICY.update(schema='M32_CPU_FROZEN_CATEGORIES_BOUNDED_ACTIVATIONS_V2',
    parameter_storage_offload=False,buffer_storage_policy='RETAIN_ALL_FROZEN_CATEGORY_TABLES_ON_CPU',
    target_category_dispatch_only=10,per_microbatch_activation_offload_bytes=1024**3)
RESIDENT_ACTIVATION_POLICY = dict(LOW_RESIDENCY_POLICY,
    schema='M32_CPU_FROZEN_CATEGORIES_GPU_ACTIVATIONS_V1',
    activation_storage='CUDA_RESIDENT',per_microbatch_activation_offload_bytes=0)


class SavedActivationOffload:
    def __init__(self, model, *, device_type='cuda', minimum_bytes=65536, policy=None):
        import torch
        self.torch = torch
        self.device_type = device_type
        self.minimum_bytes = minimum_bytes
        self.policy = POLICY if policy is None else policy
        self.microbatch_bytes = 0
        self.cpu_category_table_bytes = 0
        self.excluded = {self.storage(t) for t in (*model.parameters(), *model.buffers())}
        self.copied_tensors = self.copied_bytes = self.restored_tensors = 0

    @staticmethod
    def storage(tensor):
        return (str(tensor.device), tensor.untyped_storage().data_ptr())

    def pack(self, tensor):
        detached = tensor.detach()
        size=tensor.numel()*tensor.element_size()
        maximum=self.policy.get('per_microbatch_activation_offload_bytes')
        if (tensor.device.type != self.device_type
                or tensor.numel()*tensor.element_size() < self.minimum_bytes
                or self.storage(tensor) in self.excluded
                or maximum is not None and self.microbatch_bytes+size>maximum):
            return (None, detached)
        self.copied_tensors += 1
        self.copied_bytes += tensor.numel()*tensor.element_size()
        self.microbatch_bytes += size
        return (tensor.device, detached.to('cpu', copy=True))

    def unpack(self, packed):
        device, tensor = packed
        if device is None:
            return tensor
        self.restored_tensors += 1
        return tensor.to(device)

    def context(self):
        self.microbatch_bytes = 0
        if self.policy.get('activation_storage')=='CUDA_RESIDENT':
            return nullcontext()
        return self.torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    def audit(self):
        return dict(policy=self.policy, copied_tensors=self.copied_tensors,
                    copied_bytes=self.copied_bytes, restored_tensors=self.restored_tensors,
                    parameter_and_buffer_storages_excluded=len(self.excluded),
                    retained_frozen_category_bytes_on_CPU=self.cpu_category_table_bytes)


def execution(root, manifest, model):
    """Opt in only through a separately sealed native execution document."""
    from scripts.pi0294.native_dataset import sha
    from scripts.pi0292.durable import read
    path = root/'M32_MEMORY_EXECUTION.json'
    pin = manifest.get('memory_execution_sha256')
    if pin is None:
        if path.exists():
            raise PermissionError('UNSEALED_MEMORY_EXECUTION')
        return None
    policy=read(path)
    if sha(path) != pin or manifest['sealed_inputs'].get(path.name) != pin or policy not in (POLICY,LOW_RESIDENCY_POLICY,RESIDENT_ACTIVATION_POLICY):
        raise PermissionError('EXACT_SEALED_LOSSLESS_MEMORY_POLICY')
    retained=0
    if policy in (LOW_RESIDENCY_POLICY,RESIDENT_ACTIVATION_POLICY):
        from scripts.pi0294.generalization.minimal_adapter import LEAVES,TargetCategoryLinear
        leaves=[model.get_submodule(name) for name in LEAVES]
        if any(not isinstance(m,TargetCategoryLinear) or m.target!=10 for m in leaves):
            raise PermissionError('EXACT_SEVEN_RETAINED_TARGET_CATEGORY_TABLES')
        retained=sum(m.retain_frozen_tables_on_cpu() for m in leaves)
        import torch
        torch.cuda.synchronize();torch.cuda.empty_cache()
    result=SavedActivationOffload(model,policy=policy)
    result.cpu_category_table_bytes=retained
    return result


def context(offload):
    return nullcontext() if offload is None else offload.context()
