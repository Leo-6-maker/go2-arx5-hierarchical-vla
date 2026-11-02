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

The worker itself calls `book_gpu_capacity.entry_requirement`. Its new explicit
training branch preserves the default 24 GiB path and the separate inference
certificates. A sealed lossless native10 request may use the 15 GiB candidate
entry. A formal4096 request additionally verifies the original native10 result,
actual capacity, full restore, copies and CLOSED lease on the **same GPU** and
the exact matched recipe/parent/data/statistics. Below 2 GiB, different devices,
incomplete restore, changed bytes or an inference-only certificate do not admit
formal training. Outer registration/launch and the actual worker use the same
entry. Tests exercise this actual entry, including a forged formal probe and a
one-byte-below-margin certificate. An earlier unarmed staged capsule is retained;
it must not be launched with the old worker's fixed 24 GiB entry on GPU1.

## Measured GPU1 limit and low-residency V2

The real V1 GPU1 native10 completed all updates and full42-state restoration,
then closed. Its minimum free VRAM was 1,431,109,632 bytes, below the original
2 GiB floor; formal training was not admitted. The resident retained its
qualified GPU0 fallback. A separate GPU0 native10 showed the lossless copies
preserved all four weight/AdamW/scheduler/RNG files byte for byte and all ten
loss/LR/cursor records, but that does not by itself qualify GPU1.

The retained full parent's safetensors headers identify 1,303,134,720 bytes in
seven frozen category W/b tables (all32 categories). V2 retains these complete
tables on CPU; the selected trainable category10 parameters stay on GPU. Its
target-only bmm dispatch rejects a different category rather than silently
dropping other weights. All category values remain available for reconstruction,
and value-based frozen fingerprints/checkpoint restoration remain strict.

Trainer/Accelerate may place a model more than once. The explicit retained-table
mode preserves those CPU buffers through generic module placement, while the
default E1/E2 and other existing paths retain their original behavior. CPU tests
exercise exact output/input-gradient/parameter-gradient/AdamW equality, retained
buffer values and repeated placement. V2 also bounds saved-activation copies to
1 GiB per microbatch to reduce the V1 copy cost; this changes storage location,
not accumulation, effective batch or the native loss.

V2 has a distinct sealed execution identity. Its formal job needs a closed
same-GPU **V2** native10; a V1 or inference certificate cannot admit it. Both PB
arms use the same execution policy. CPU proof is not GPU admission: actual
memory, every update and complete restoration must still be measured before
formal adoption. Preserve all V1 failures, costs and checkpoints.
