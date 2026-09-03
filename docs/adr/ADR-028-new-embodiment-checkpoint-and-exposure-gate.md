# ADR-028: Exact New-Embodiment Checkpoint and Exposure Gate

Date: 2026-09-03

Status: Accepted for Stage24 static audit; future Hopper execution requires a
separate PI authorization.

## Context

Stage23 O1 was a bounded TINY1 H40 negative. Its full free-running profile was
oscillatory, but the Phase-A relay did not include the exact base checkpoint
weights. Source-level statements about `NEW_EMBODIMENT` therefore could not
answer whether category ID 10 was actually initialized, pretrained, or
already adapted in the checkpoint used by O1.

The exact base tree is now available on Corelab and matches the frozen index,
shard, and tree hashes. The audit also shows that the O1 profile defect is
visible in truncated H4/H8 views, while the H2 physical translation
statistics are nondegenerate after the structural first relative point.

## Decision

1. Use the exact checkpoint tree, not constructor defaults or a directory name,
   for category-row claims.
2. Classify category ID 10 from multiple audited weight blocks and compare its
   10D physical and 122D inactive partitions. The current result is
   `ID10_UNUSED_LIKE`, with a static-weight-only claim boundary.
3. Do not call the 122D energy fraction causal evidence. An intervention needs
   a separately authorized controlled experiment.
4. Do not promote H4/H8 prefix results into a causal horizon claim; they are
   truncated views of sealed H40 outputs.
5. Because dense H40 relative EEF has an upstream N1.7 precedent, translation
   normalization has no demonstrated catastrophic mismatch, ID10 is
   initializer-like, and O1 exposure is 320× below the pinned SO100 nominal
   presentation budget, prepare exactly one future candidate:
   `H2_LARGER_STOCHASTIC_FLOW_EXPOSURE_SAME_SAMPLE_H40_OVERFIT`.

The prepared candidate retains the current 10D/132D ABI, TINY1_H2STATS, B1,
training seed 2222, four-step sampler, and frozen visual/LLM flags, and
proposes a bounded N=2000 same-sample run. The longer cosine schedule is a
known coupled consequence and must be reported. It is not an isolated causal
claim and it is not executed by Stage24.

## Consequences

The exact checkpoint recovery closes the former infrastructure blocker while
preserving the earlier fail-closed history. Stage24 ends with a scientifically
bounded next question rather than an unreviewed Hopper job. A future result
must still distinguish objective fit, free-running profile, and any downstream
WBC behavior; no real-robot or generalization claim follows from this ADR.

## Evidence

```text
reports/STAGE24_STATIC_CHECKPOINT_AUDIT.md
reports/STAGE24_PREFIX_AUDIT.md
reports/STAGE24_NORMALIZATION_AUDIT.md
reports/STAGE24_RECIPE_UPSTREAM_AUDIT.md
reports/STAGE24_NEXT_HOPPER_PROTOCOL.md
```
