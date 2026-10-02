# PI0268 preparation complete

**PI0268_PREPARATION_COMPLETE_AWAITING_SEPARATE_SCIENTIFIC_LAUNCH**

Authority: release80554763aaeeecdc631da8c5ded96074884ff434 / Issue11 comment5955686136. This isolated preparation does not complete or take ownership of the main PI0267 lane. selected/deployment remain **NONE**.

## Findings

1. Actual old statistics map X/Y TRAIN recovery rotation into up to~120k inputs/targets. Roundtrip is accurate and Gram–Schmidt is well-conditioned; reversibility is not a conditioning guarantee.
2. Actual R late X/Y native losses remain~4e7–8e8; accumulated gradient norms reach1.37e8 and all4096 updates clip. Donor/N tolerate the same amplified input state, so the causal mechanism remains unproved.
3. All2088 saved offline traces and192 closed virtual episodes were analyzed. R M1 original first hits30/32, M3 0/32, one collision; donor/N M1 first hits13/32 and26/32. All69 successful M1 runs later finish outside original pose bounds. Original qualification and late failures are preserved.
4. 112 truly consumed TRAIN inputs and304 factor specifications are ready as manifests only. No altered state was paired with old RGB. Existing packet bytes/keys were verified. Planned1584 queries/304 frame pairs are **NOT_AUTHORIZED**.
5. Recommend one combined state/action rotation normalization comparison, retaining all raw data, loss formulas and other contracts. Per-contract step0 is essential. [NEXT_RUN_DRAFT.json](NEXT_RUN_DRAFT.json) remains `launch_allowed=false`, `NOT_AUTHORIZED`.

## Deliverables

- [Codec conditioning](CODEC_CONDITIONING_REPORT.md), feature/horizon tables and fixed sensitivity probes.
- [Actual exposure, optimization and saved S4 traces](TRAIN_EXPOSURE_AND_OPTIMIZATION.md), complete consumed index and hashes.
- [Closed-point prefix, entry, retention and safety](CLOSED_LOOP_GAP_REPORT.md), all18,988 query rows and fixed-array figure.
- [Hypotheses and counterevidence](HYPOTHESIS_MATRIX.md), [112 seen inputs](SEEN_TRAIN_112_MANIFEST.json), [304 factors](FACTOR_304_MANIFEST.json), [single training draft](NEXT_RUN_DRAFT.md).
- [CPU checks](CPU_CHECKS.json):31 positive/negative checks, including four real processor/collator calls with exact existing input identities, wrong units/frame/labels, missing/duplicate/heldout inputs, preserved gate rejection, partial-point exclusion and denied model/launch paths. **Static/processor checks, not a model canary.**
- [Source/concurrency/resource snapshot](SOURCE_SNAPSHOT.json), [original log identities](TRAIN_LOG_SOURCE_FILES.json), [file inventory](DELIVERY_INVENTORY.json).

## Resources and omissions

No new model or gradient queries, optimizer updates, renderer/physics/WBC calls or PBS submissions. No full model copied. One finite CPU analysis at a time, CUDA hidden, four-thread library caps, nice19/ionice3. The first process's threads were pinned to four cores after startup (before/after receipt retained); later processes inherit four-core affinity at startup. Peak observed RSS1.242GiB; working data at collection~159.1MiB, derived numeric data~64.3MiB. No GPU evaluator process touched. Small hash/source reads on Hopper only.

Completed24-family source arrays were streamed once into a small CPU numeric cache. Subsequent arithmetic uses that cache; four bounded processor fixtures use their matching original frames. The880 missing legacy prefix columns were finished from hash-matched existing labels/outputs. Original failed analysis logs remain; no scientific query was repeated for a report error.

UNOBSERVED: neural fit on the proposed112 TRAIN inputs, per-module activation/gradient attribution and unsaved training Euler/epsilon tensor values. Main secondary WBC status is not polled or taken over. No normalization was fitted to EVAL, and examined EVAL remains development evidence. No live statistic, checkpoint, scorer, gate, original output or selected status changed.

Large plain metadata tables are losslessly gzip-compressed; inventory includes both stored and uncompressed hashes. Use `gzip.open` to inspect. No weights or image corpus are in this return.
