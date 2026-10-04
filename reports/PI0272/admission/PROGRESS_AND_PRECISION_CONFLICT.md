# PI0272: actual source admission and one unresolved precision contract

Authority: release `e62e6a469dce1f4d3df5d49b38bad4a3c627e969`, protocol blob `111260a1c3371ef817c84313ce87415290a7a91e`, Issue #11 canonical [5979236384](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5979236384).

## Current execution

**Training has not started.** New optimizer updates, training heads, quality queries and PBS submissions are all zero. There is no PI0272 quality verdict. PI0270/71 remain closed and unchanged; selected/deployment remain NONE.

- Reused PI0271's complete closeout and fixed donor receipts. Freshly hashed the four AGG4096 model/config files against revision `f862abb26041c8e7d355470d6e9d48899f78b9ce`; no model download or optimizer inheritance.
- Froze the exact 296 consumed conditions, paired 2048-microbatch plans and 2148-query/6228-head evaluation membership before training. All original source samples and codec receipts matched. Exact processed condition plus label overlap with NEW_FIT192 is zero.
- Actual Hopper CPU processor traversed **2048 microbatches per arm**, all paired input and label identities matched. Each arm exposes 272 conditions seven times and 24 conditions six times. This proves planned encoding, not neural training consumption.
- CPU constructed DIRECT/checkpoint gradient, mode/RNG restoration, inherited temporal loss and unclamped noncommuting rotation fixtures passed. Constructed BF16 toy features do not establish the actual model's feature dtype.
- Bound the initializer's names and shapes: 471 trainable tensors stored as FP32; 559 frozen tensors with mixed BF16/FP32 storage. A model-wide BF16 assertion would be incorrect.
- Hopper has no current PI0272 job. The observed queued `645844` job is unrelated and untouched. Actual scratch quota was 687.7 GB of 1.0 TB; the conservative retained-canary/formal-checkpoint/write-overlap plan plus 10 GiB reserve fits. No cleanup was needed. Project credit lookup was unavailable through the interactive accounting interface; no credit availability is claimed.

## Exact contradiction, discovered before any GPU work

The release's sections 3/4 describe S4's generated query and feature dtype as BF16 and require DIRECT's zero query to use that same BF16 feature dtype. They also require preserving the qualified S4 precision/path.

The **actual completed AGG4096 audit contains 8192/8192 identical dtype signatures**:

| Quantity | Actual dtype |
| --- | --- |
| Native flow epsilon | `torch.bfloat16` |
| Processed VL features | `torch.float32` |
| S4 initial query after conversion to feature dtype | `torch.float32` |
| Each of the four Euler states | `torch.float32` |

Actual stock S4 parity PASS has raw and decoded XYZ maximum absolute differences **0.0**. The source converts native epsilon to `vl_embeds.dtype`; there is no evidence that the qualified S4 query was BF16. The existing completed training audit explicitly validates this mixed precision signature.

Source `S4_audit.jsonl` SHA256: `0d2bd6253c433425d6e7dda3963d81a06da19abb8321fac9f667508e02f9b917`.

The complete source-bound conflict receipt is [A2_PRECISION_CONTRACT_CONFLICT.json](A2_PRECISION_CONTRACT_CONFLICT.json), SHA256 `4e928d8337e2d5979023c3b2dab9a7b92fc939bd644704f9b047a7df45bb059c`. It includes first actual tensor identities, exact source hashes and the unchanged stock parity receipt. No neural query was used to diagnose this.

## Proposed resolution — not executed

**Phase-specific clarification:** preserve **FP32 generated query during training and BF16 query during the existing Corelab quality evaluation**. DIRECT's zero query follows the actual feature dtype in each phase. Native BF16 epsilon/autocast, FP32 physical losses and the original mixed parameter precision remain unchanged. Do not interpret the earlier FP32 proposal as permission to convert evaluation to FP32.

Fresh inspection of all **1168/1168** closed AGG A1/A2 quality receipts finds BF16 query, native S4 and four heads; every receipt matches the closed output index hash. The actual qualified quality worker (`7e70551b613b218986efae1dfe6f2fa9be2d347779352166274e8d31686ceeb4`) explicitly loads the evaluation model with `.eval().to('cuda:0', dtype=torch.bfloat16)`. This contrasts with the training model's mixed parameter precision. See [A2_PHASE_PRECISION_EVIDENCE.json](A2_PHASE_PRECISION_EVIDENCE.json), SHA256 `e7484efc9aec9a300166dfb3ba6d57eab9e6fddc3b9b8e72332e39874886aaa3`. No old prediction was regenerated.

This is not permission to cast the S4 path to BF16, refit statistics, change losses, reset budgets or collect new conditions. Section 2 requires reporting a concrete semantic contradiction; section 8 does not classify scientific precision changes as ordinary engineering repair. Explicit clarification was requested from the user before proceeding.

The frozen requested-mode cards remain preserved. The new training entry requires an explicit resolution receipt; it is not launch-admitted. No real canary has run. The isolated source is a prepared implementation, not a claim of model correctness or training completion.

After clarification: bind corrected versioned mode evidence, finish actual runtime admission/seal, run both real 8-versus-4+4 canaries, then S4FIT1024 and DIRECT1024 serially, followed by the fixed endpoint panel, CPU replay and complete new Trainer HF readbacks. No routine GO is needed after the precise contract is resolved.

## Independent preparation completed while clarification is pending

The per-condition comparison helper has a passing constructed CPU self-check. It computes each DIRECT-minus-mean-S4 MSE difference before the group median, distinguishes /condition and /noise-output direction denominators, retains finite REJECT results, leaves undefined directions undefined, and marks an incomplete numerical bank unavailable without reducing its noise count. The check explicitly distinguishes a median of paired differences (-1 in its fixture) from a difference of marginal medians (-78). These are synthetic arithmetic values, not PI0272 model results.

The new formal-checkpoint archive entry reuses the qualified fixed-revision whole-file HF upload/readback helper and requires the new mode sidecar. It has not uploaded any checkpoint because no PI0272 Trainer exists. The original canonical was re-read and still refers to the same unrevised BF16 wording; no explicit precision resolution has been received.
