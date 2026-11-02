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


class SavedActivationOffload:
    def __init__(self, model, *, device_type='cuda', minimum_bytes=65536):
        import torch
        self.torch = torch
        self.device_type = device_type
        self.minimum_bytes = minimum_bytes
        self.excluded = {self.storage(t) for t in (*model.parameters(), *model.buffers())}
        self.copied_tensors = self.copied_bytes = self.restored_tensors = 0

    @staticmethod
    def storage(tensor):
        return (str(tensor.device), tensor.untyped_storage().data_ptr())

    def pack(self, tensor):
        detached = tensor.detach()
        if (tensor.device.type != self.device_type
                or tensor.numel()*tensor.element_size() < self.minimum_bytes
                or self.storage(tensor) in self.excluded):
            return (None, detached)
        self.copied_tensors += 1
        self.copied_bytes += tensor.numel()*tensor.element_size()
        return (tensor.device, detached.to('cpu', copy=True))

    def unpack(self, packed):
        device, tensor = packed
        if device is None:
            return tensor
        self.restored_tensors += 1
        return tensor.to(device)

    def context(self):
        return self.torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    def audit(self):
        return dict(policy=POLICY, copied_tensors=self.copied_tensors,
                    copied_bytes=self.copied_bytes, restored_tensors=self.restored_tensors,
                    parameter_and_buffer_storages_excluded=len(self.excluded))


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
    if sha(path) != pin or manifest['sealed_inputs'].get(path.name) != pin or read(path) != POLICY:
        raise PermissionError('EXACT_SEALED_LOSSLESS_MEMORY_POLICY')
    return SavedActivationOffload(model)


def context(offload):
    return nullcontext() if offload is None else offload.context()
