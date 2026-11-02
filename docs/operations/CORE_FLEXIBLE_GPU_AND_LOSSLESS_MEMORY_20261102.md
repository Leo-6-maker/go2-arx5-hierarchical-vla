# Flexible GPU selection at cold boundaries

The user authorizes GPU1 while GPU0 is busy and asks future scripts to select
either card. Existing running capsules remain immutable. Retire an unstarted
project waiter only after its exact process identity, empty native job set and
idle metadata boundary are independently proven; preserve its costs and state.

Use `night10.gpu_selection.choose_device` inside the existing resident owner.
Supply actual UUIDs, current free VRAM, host RAM, stable observations, sharing
permission and a workload/device-specific qualification. Formal training cannot
select an unqualified device. Native qualification may select a candidate.
Record the chosen UUID before registration and bind `CUDA_VISIBLE_DEVICES` to
that UUID. Recheck actual capacity at registration and launch. Never migrate a
live model, combine the two cards' memory or signal another user's process.

For this Core, GPU0's conservative training entry remains 24 GiB. GPU1's new
native qualification candidate entry is 15 GiB, with no existing model on that
card, sufficient host RAM and the same original **2 GiB measured training
margin**. The lower entry alone is not formal training qualification. Old GPU1
native10 and new D2 native10 completed updates/reloads but failed headroom;
neither is relabeled as a pass. Each new memory-policy/arm/device qualification
is finite, one native10, with all actual costs and failures retained.

`activation_offload.POLICY` is a separately sealed execution option. Only saved
autograd intermediates of at least 64 KiB are copied losslessly to CPU memory;
all parameter/buffer storage, including views, stays on GPU. Backward copies the
same values/dtype/shape to their original device. No model precision, effective
batch, accumulation, loss, sampler, optimizer, schedule, RNG or activation
checkpointing change is introduced. Both matched arms and both GPUs use the same
policy. Host memory and copy time must be measured. No persistent pinned pool or
new dependency/environment is introduced.

The mechanism uses PyTorch 2.9's documented
[saved tensor hooks](https://docs.pytorch.org/docs/2.9/autograd.html#torch.autograd.graph.saved_tensors_hooks).
CPU tests check exact losses, gradients, AdamW updates, RNG preservation, dtype
and shape roundtrips and parameter-view exclusion. The actual registered neural
environment passed both tests without initializing CUDA. This is **not** native
GPU training qualification. Adoption still requires real updates, full model /
AdamW / scheduler / RNG / sampler restoration and the original measured margin.

The native run seals `M32_MEMORY_EXECUTION.json`, logs actual copy counts and
includes that document in each checkpoint. Old manifests without the explicit
pin use their original path. An unsealed or changed policy is rejected.

Cold successor order: complete the original D2 scientific recipe using a
qualified available device, then fixed-D2 CONTROL and BOUNDARY, each fresh4096
after native10 on its selected device. A known closed failure isolates that arm;
unknown writes/ownership prevent another start. No old failed role is reopened.
Reset DEV and the shared raw-parent baseline remain separate authorized work;
this memory/selection change does not claim either is already deployed.
