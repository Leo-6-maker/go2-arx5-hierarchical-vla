# Raw shared four-task Trainer entry

The separate `V7_SHARED_RAW_M32_FINITE_LEARNING` worker now calls the existing
`book_fit.run` / native `Gr00tTrainer`, original masked flow loss, Processor and
continuous four-step decoder. It does not restore book128, b248, E1, an old
optimizer or a qualification run's weights. Existing book recipes and active
server capsules are unchanged.

The new runnable recipe is `SHARED4_RAW_M32_TASK8_TRAINER8192_V2`. The earlier
V1 input/preparation recipe remains unchanged and does not admit a model. V2
binds the actual frozen TRAIN59 manifest, statistics, population and complete
68-condition DEV reference; it adds sixteen fixed TRAIN diagnostic windows,
four per task at anchors 0/48/200/400. Diagnostic inputs/noise are the same
before training and at declared curve points. All task means are reported
separately; none is a closed-loop task score.

Effective batch32 remains 8:8:8:8. The native10 consumes exactly the first320
draws of the8192 schedule, including warmup256; formal work starts fresh from
raw and saves2048/4096/6144/8192. Every save uses the original full
optimizer/scheduler/RNG/cursor/trace checks. The raw-dependent delta carries
its own schema, retained parent reference, full state and unchanged statistics
and population. It cannot be loaded as a standalone model or through an old
book checkpoint header. A reconstruction consumer is provided for subsequent
evaluation, but the owner must first qualify the actual evaluation lifecycle.

Raw parameter mapping selects the original category10 **without transposing**
seven W/b pairs and all28 tensors of DiT blocks30/31. At the initial model
load, all42 current parameters must equal these raw values after the model's
dtype conversion. Restore re-reads those selected tensors and reapplies the
saved delta while separately checking every frozen parameter/buffer. The
existing full-seven-file audit is reused only while all original native stat
identities remain unchanged.

The same-device capacity consumer accepts only this raw recipe's own CLOSED
native10,32 completed diagnostic generations,42 AdamW states and complete
restore, with at least the original2GiB measured headroom and the explicit V2
lossless CPU-storage policy. D2/book native10 or inference certificates cannot
authorize shared training. GPU0 and GPU1 keep separate qualifications. No
second scheduler, owner, ledger, loss or decoding intervention is introduced.

## Actual validation and deployment boundary

The PC read the actual pinned raw files, completed a full stream/stat audit,
then read42 selected BF16 tensors:77,847,684 values /155,695,368 bytes, without
loading a neural model or initializing CUDA. The actual inventory is in
[the CPU receipt](../../reports/PI0294/design/controlled-finetuning-20261102/SHARED4_RAW_TRAINER_PC42_ACTUAL_20261102.json).
Sixty-six targeted CPU checks cover the raw mapping, recipe/old-path separation,
sampler, delta restoration and capacity-certificate rejection, alongside the
existing book/phase regressions. Compilation and fixtures do not constitute
native GR00T forward/backward or a successful policy.

Core CPU adoption, raw-model native10, four-task evaluation qualification and
resident successor integration must be reported from actual receipts before
calling this READY. The current D2→CONTROL→BOUNDARY resident training queue is
not hotpatched or replaced by this change. Independent reset DEV qualification
and shared training are still outside its mounted scope. Shared endpoints,
valid DEV conditions and successes remain zero.
