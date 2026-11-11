"""Move only frozen backbone storage out of the AdamW temporary peak.

Every original forward still executes on the declared device. No feature cache,
optimizer implementation, parameter dtype, gradient or RNG operation changes.
The site envelope and a complete fresh-process restore qualify the GPU route.
"""
from time import monotonic
from contextlib import contextmanager
from .target_gr00t_profile_v1 import need

MEMORY='CORE_FROZEN_BACKBONE_OPTIMIZER_SWAP_V3'


class BackboneOptimizerSwap:
    def __init__(self,model,activation_storage,*,device='cuda'):
        import torch
        self.torch=torch;self.model=model;self.storage=activation_storage
        self.backbone=model.backbone;self.device=device;self.resident=True
        self.to_host_count=self.to_device_count=0
        self.to_host_wall_s=self.to_device_wall_s=0.
        self.optimizer_state_restores=0
        self._frozen()
        self.bytes=sum(t.numel()*t.element_size()for t in
            (*self.backbone.parameters(),*self.backbone.buffers()))
        self.handle=self.backbone.register_forward_pre_hook(self.before_forward)

    def _frozen(self):
        need(all(not p.requires_grad and p.grad is None for p in self.backbone.parameters()),
            'ONLY_FROZEN_GRADIENT_FREE_BACKBONE_STORAGE_CAN_MOVE')

    def _move(self,device):
        self._frozen()
        tensors=(*self.backbone.parameters(),*self.backbone.buffers())
        identity=[(tuple(t.shape),str(t.dtype))for t in tensors]
        self.backbone.to(device)  # Device only: preserve every value and dtype.
        after=(*self.backbone.parameters(),*self.backbone.buffers())
        need(identity==[(tuple(t.shape),str(t.dtype))for t in after],
            'FROZEN_BACKBONE_SWAP_NO_DTYPE_OR_SHAPE_CHANGE')
        # Saved-activation exclusion keys contain storage addresses. Refresh
        # them after moving the backbone, while keeping action storage resident.
        if self.storage is not None:
            self.storage.excluded={self.storage.storage(t)for t in
                (*self.model.parameters(),*self.model.buffers())}
        if self.device=='cuda':self.torch.cuda.synchronize()

    def before_forward(self,module,args):
        need(module is self.backbone,'ORIGINAL_BACKBONE_ONLY')
        self.restore()

    def restore(self):
        if not self.resident:
            started=monotonic()
            if self.device=='cuda':self.torch.cuda.empty_cache()
            self._move(self.device);self.resident=True;self.to_device_count+=1
            self.to_device_wall_s+=monotonic()-started

    def before_optimizer(self):
        if self.resident:
            started=monotonic()
            self._move('cpu');self.resident=False;self.to_host_count+=1
            if self.device=='cuda':self.torch.cuda.empty_cache()
            self.to_host_wall_s+=monotonic()-started

    def callback(self):
        from transformers import TrainerCallback
        owner=self
        class StorageOnly(TrainerCallback):
            def on_step_begin(self,args,state,control,**kwargs):
                owner.restore()
            def on_pre_optimizer_step(self,args,state,control,**kwargs):
                owner.before_optimizer()
            def on_step_end(self,args,state,control,**kwargs):
                # Original prepare_input consults model.device before calling
                # backbone.forward; get_action bypasses model.forward hooks.
                # Restore after AdamW/zero_grad, before either entry is used.
                owner.restore()
        return StorageOnly()

    @contextmanager
    def optimizer_restore_storage(self):
        resident=self.resident
        self.optimizer_state_restores+=1
        self.before_optimizer()
        try:yield
        finally:
            if resident:self.restore()

    def audit(self):
        return dict(schema=MEMORY,backbone_bytes=self.bytes,
            to_host_before_optimizer=self.to_host_count,
            to_device_before_original_forward=self.to_device_count,
            to_host_wall_s=self.to_host_wall_s,to_device_wall_s=self.to_device_wall_s,
            optimizer_state_restores=self.optimizer_state_restores,
            original_backbone_forward_each_microbatch=True,
            feature_cache=False,optimizer_changed=False,precision_changed=False,
            backbone_resident=self.resident)
