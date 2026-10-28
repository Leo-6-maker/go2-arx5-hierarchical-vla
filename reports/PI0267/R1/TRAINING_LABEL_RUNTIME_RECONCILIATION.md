# PI0267 R1 — original runtime label identity reconciliation

## Preserved failure

The original strict main audit failed at R unit3/micro0. Of8192 consumed R microbatches,1359 M1 recovery labels differ in normalized FP32 bytes from the Corelab CPU preflight plan. N's8192 rows match that plan exactly. The original plan, actual consumption logs, failure inventory, training fileset and processor remain unchanged. Cross-host normalized label byte identity is **false**.

## Actual zero-model evidence

- All5632 distinct consumed R raw labels/states and their source proposal files agree across hosts. Every other consumed input/metadata field agrees with the frozen plan.
- The original Hopper encoder exactly reproduces all8192 actual consumed R label/mask hashes. The original Corelab encoder independently reproduces all8192 preflight label/mask hashes.
- Source code, package versions and frozen statistics agree. The first intermediate divergence is the FP32 three-vector squared norm inside the original row6D Gram–Schmidt operation; hosts expose different CPU instruction sets. No codec, CPU dispatch, precision, statistics or training setting was changed.
- Maximum normalized component difference is0.0078125. This is reported, not hidden behind a byte-equality claim.
- Hopper-produced labels decoded on Corelab have maximum physical errors6.048941909381865e-9m /2.886563610568817e-8rad against their original raw labels. Cross-host decoded disagreement is at most6.602841196309624e-9m /2.8624204230924836e-8rad. Every source satisfies the **original**1e-5m /1e-4rad codec bounds.
- Both original real exact-resume canaries remain unchanged. These compare actual training runtime identities, not Corelab versus Hopper numerical-library dispatch.

Full evidence: `evidence/R_RUNTIME_LABEL_REPLAY.json`, `evidence/R_CROSS_HOST_LABEL_RECONCILIATION.json`, original `evidence/R_INPUT_PLAN_FAILURE_20261002.json`, and the original per-host numerical probes. Full replay arrays remain on both servers and enter the final public evidence archive.

## Narrow auditor correction

The default audit still enforces its original exact comparison. Explicit `--reconcile-r-labels` first requires the source-bound full replay receipts and original physical bounds, then compares actual R labels **exactly** to their independently replayed original-runtime bytes. Every other input field remains exact. The frozen Corelab plan is not rewritten. A regression reproduces the original strict failure and rejects altered label or condition hashes.

This corrects the auditor's unsupported cross-CPU byte assumption. It does not authorize arbitrary numerical equivalence, changed labels, regenerated training, precision changes or new tolerance. The amended auditor lives outside the original sealed training code on Hopper. Main admission additionally requires the unchanged random-packet, S4/loss/gradient/mode, budget and four complete Trainer checks; these are not replaced by this receipt.

All four complete Trainers have fixed PUBLIC16-file readbacks. Their saved471 optimizer states each have exactly the checkpoint's2048/4096 steps, and scheduler epochs/rates match the original warmup10/cosine4096 schedule. No new model, gradient or optimizer query was used for this reconciliation. Quality results remain a separate future decision.

`selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`.
