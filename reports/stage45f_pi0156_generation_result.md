# PI0156 fixed-input generation-path comparison

Status: **COMPLETED**  
Token: `CORELAB_PI0156_FIXED_INPUT_PATH_COMPARISON_FOR_PI_REVIEW`  
Authority: Issue #11 comment `5650620343`; PR #69 comment `5650623655`

## Execution and fixed identities

The repair was executed at `40b74b91c1a2557985d600b7a2c334d0cc88fa94` after
the CPU regression and local admission.  The metadata-only evidence patch was
committed afterward at `f3dd147ab12c4b4818352cc466a1aa5c0ba983fe`; no model
query was repeated.  The run used `corelab-4080s`, GPU0 UUID
`GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad` (RTX 4080 SUPER, 32,091 MiB free
at admission), Python 3.12.3, Torch `2.9.0+cu128`, CUDA 12.8, and bf16
parameters.

The N2 seed2223@2000 checkpoint tree is
`09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152` (19
files, 20,889,823,145 bytes).  N2 stats file/payload are
`0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb` /
`508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572`.
The saved float32 noise array is
`5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd` (file
SHA `83590e27b22847260a5c7cc2af365c088b352ccafc056ad741418a6e98168922`).
Cosmos remained offline at revision
`9ce19a195e423419c349abfc86fd07178b230561`, tree
`1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9`.

## CPU and runtime boundary repair

The complete query wrapper now places both the C1 `_sample_one` path and the
deployment worker `model.get_action(**inputs)` path in one
`torch.inference_mode()` scope.  The scope includes prepared inputs,
conditioning, all four updates, action-head calls, and raw extraction.  A CPU
stub first reproduced the old error by passing an inference-produced feature
to a parameterized `Linear` with autograd re-enabled; the real wrapper then
passed the same producer-to-parameterized-operator test with finite output,
no `grad_fn`, and no parameter update.  Existing tuple/bf16, lazy-import,
target-exclusion, and processor-builder regressions also passed.

Both real processor builders passed in isolated CPU processes for TRAIN and
C0.  Their model-input keys and prepared summaries were equal, and
`target_position_world_m` was absent from model inputs.  The C1 path used the
fixed noise as a CPU direct argument; the worker injected the same array at
`cuda:0`.  Source float32, effective numeric float32, and effective raw-dtype
hashes were equal (`586d0c0bd6ff4728c7eaae7f05e7cad0ea5f76bcc26ff93669e9268b6ac98591`
and `3e7f45776a019440957e623f2e169729c5193089b04782406b98102f711f3946` for
the effective values), so the device placement did not change the result.

## Four fixed queries

| input | path | initial error (m) | minimum error (m) | final error (m) | progress (m) | best progress (m) | path length (m) | physical/raw pair delta |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TRAIN_a_expert-0000 | C1_SAMPLE_ONE | 0.0200680564 | 0.0196378478 | 0.1772810062 | -0.1572129498 | 0.0004302085 | 1.9079431877 | — |
| TRAIN_a_expert-0000 | WORKER_GET_ACTION | 0.0200680564 | 0.0196378478 | 0.1772810062 | -0.1572129498 | 0.0004302085 | 1.9079431877 | — |
| C0_NOMINAL | C1_SAMPLE_ONE | 0.2261557984 | 0.1464257575 | 0.3233401175 | -0.0971843191 | 0.0797300409 | 1.8551533024 | — |
| C0_NOMINAL | WORKER_GET_ACTION | 0.2261557984 | 0.1464257575 | 0.3233401175 | -0.0971843191 | 0.0797300409 | 1.8551533024 | — |

For both input pairs, C1 versus worker is byte-identical: raw max-absolute
delta, raw RMSE, active-190 delta, XYZ/rotation/timing/gripper/padding deltas,
decoded physical max-absolute delta, and decoded physical RMSE are all exactly
`0.0`.  The decoded physical hashes are
`8068fba8144e1f574fae6584b860be0f4e15fea0e3c211a37ec353bd88182e09` (TRAIN)
and `40650d7011de39e163c3a3915eec0c458db2cdf090bec5b8b43affb4e5d6185f`
(C0), identical across each path.  Every decoded p0 has zero residual to
`StateV1[:3]`.

The four raw arrays remain on the node-local evidence root
`/var/cache/users/liuyu-nfs/stage45f_pi0156_generation_probe_20260913`:

- TRAIN C1 and worker: file SHA `cd05fdc101cdd2f54b173a98f72d9bf80d4143ddeda703d618659988ccd1b7c2`, array SHA `2c4fab5f17b8d8d6621278be210849de2fffe44e6107d8bea9dfb250dcc6c47c`.
- C0 C1 and worker: file SHA `64a7c832ff7e290575ca8ad52c735b16009884a2af9bc00b2ee2d18301c4e834`, array SHA `e58e5d31a50a506e6840fd6b08aca2d1214d836c5ce279add53b016afc71f690`.

The original runtime packet is
`pi0153_generation_probe.json` SHA
`d1749c957758697b1dfeab2c95a6290840a99b1595d27e4dc4aad00a8d932a64`;
the compact report and CSV in this branch are CPU-derived evidence views, not
new model outputs.

## Attribution and boundary

Under fixed checkpoint, stats, inputs, seed, noise, model, and processor
semantics, the two generation paths are equivalent for both TRAIN and C0.
The previous failure was a lifecycle/autograd-boundary defect, not a strategy
negative result.  After restoring the complete inference boundary, the model
still produces negative endpoint progress on both inputs (TRAIN `-0.1572 m`,
C0 `-0.0972 m`), while each reaches a small positive best-progress excursion.
This is a policy-output observation; it is not a WBC, closed-loop, or task
success claim.  No path-specific cause can be assigned from these data because
the paired outputs are exactly equal.

New calls were 4 queries, 4 conditioning passes, and 16 action-head calls (20
model-side calls), all completed; one model construction and one worker
attempt succeeded, with zero exceptions, retries, or unknown completions.  The
previous PI0155 failure remains counted: cumulative totals are 5 query starts,
5 conditioning starts, 16 action-head calls, and 21 model-side calls against
the PI0156 cumulative limits of 6, 6, 20, and 26 respectively.  No training,
backward, ICL, WBC, simulator, hardware, PBS, Validation-E, or Holdout-F was
performed.

**Single recommendation:** before any new training or closed-loop run, PI
should authorize one controlled model-side input/conditioning probe to locate
why this fixed policy's decoded trajectory has negative target progress; do not
split the already-equivalent C1 and worker paths again.
