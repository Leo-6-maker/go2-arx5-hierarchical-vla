# NIGHT1 continuation: production DEV, 200+100 recovery, and an actual late-decision panel

The primary round's cold-boundary index now records **38 eligible production episodes / 3,504 new task atoms**, 475,921 native ticks including initialization/failures, and **700 main WM updates + 4 recovery probes** across four model runs. This is a measured increment over the pre-NIGHT1 campaign, not a new campaign. The full original eight-parent × four long-episode panel remains partial: failed paired attempts were excluded, not replayed to manufacture a complete panel. The minimum collection and 200+100 restoration milestones are achieved; contact/full manipulation and the remaining scientific work continue under the original deadline.

## New production fit and strict continuation

Both runs use source `90486476eb726f74a191478fad8e8dc7baa4907d`, frozen native `24a4417db45ad858a356bea4e1aae3938d417a62e533fbaee473ead6734e6561`, controller `9e8ce46837d09bc39ae04d4b49f1ce4dd8c8c2d31bea39eac56c5806d108c1f3` and the same qualified RootSafe initial-workspace family. No WBC parameters or native safety rules change.

| Item | Fresh production A | Continuation on sealed new TRAIN from B |
|---|---|---|
| Run | `night1-wm-receiver3-first-01` | `night1-wm-receiver3-second-sealed-train-01` |
| Main updates / extra restore probe | 200 / 1 | 100 / 1; cumulative main=300 |
| Replay SHA-256 | `d9ab55da89ce2a530b70c6c7cd0a2bcfd8bcd8ece9a9ac91650e353706f5ef76` | `a545cb4fa0c94e6e177bf2a74817afbd6e1c4cd0a2979f5d6aa12e9e82089fe1` |
| Checkpoint SHA-256 | `28ecaf68144777178befa17c109fce0eada02773f3ada584920cf913a569472d` | `c32cd16ac30782d3e1efe135c922f3134ae1dd100c4515a48de49dcfb20f438f` |
| Fit wall / process CPU seconds | 4.449183 / 4.346155 | 2.828051 / 2.766089 |
| Main CUDA-event seconds | 2.923647 | 1.550845 |
| Entire-run peak allocated / reserved bytes | 85,086,720 / 96,468,992 | 85,086,720 / 96,468,992 |

Both fits exited0/CLOSED, 74,986 parameters, 106 state coordinates. Model/optimizer/RNG/cursor/extra-update comparisons all report zero error. The continuation keeps A's normalizer and optimizer/RNG state. It evaluates the previously frozen model on the new TRAIN batch before updates.

A has eight complete production episodes (six TRAIN parents / two DEV parents). B contributes six new complete TRAIN episodes. Its fourth pair failed at native step4,255 with `REGISTERED_JOINT_SAFETY`; the peer stopped at4,153. Both trajectories and all their prefixes are excluded. Every owned process in that failed pair exited0, and both leases closed, so this is distinct from the earlier shutdown defects. The 100-update replay reuses the two sealed A DEV episodes. **It does not contain new B DEV episodes.** The two replays therefore represent 14 unique complete episodes / 1,344 atoms, not 16 unique episodes / 1,536 atoms.

## A limited positive now uses production DEV

On the two A production DEV parents, 0.8s TCP coordinate RMSE improves from **0.832828 mm after200** to **0.273214 mm after200+100**. Persistence is 1.170377 mm and numeric velocity is 6.212372 mm. The two parents individually improve from 1.021480→0.327128 mm and 0.586331→0.205619 mm. These are two independent held-out parents with overlapping windows, not146 independent trials or a significance claim.

For a stronger available-data comparison, a separate zero-WM-update review built the union of all12 TRAIN and2 DEV episodes. Its replay SHA-256 is `8469c3a02312feb6e99ae687a8fab876bb66cc1cb2a0682235d6b62c98e6eadb`. Both closed-form fits use TRAIN only. At0.8s the original reference-residual ridge reaches **1.059145 mm** and the new standardized, action/time-aware recursive ARX reaches **0.435573 mm**. The WM's0.273214 mm remains better on this pooled endpoint. ARX gives0.341328/0.512779 mm on the two DEV parents, compared with WM0.327128/0.205619 mm. At0.05s the stronger baselines beat the WM: ridge0.130419 mm and ARX0.148363 mm versus WM0.188681 mm. Static object and finger errors remain negative controls; see [full selected fields](PRODUCTION_WM_02.json).

The ARX strengthening is an exploratory backtest here. Its fixed lambda1, causal features and recursive prediction were separately committed **before the first native late-decision episode**, and will be used prospectively for that next experiment. It never receives future achieved states, future measured reference ages, scene labels or DEV normalization. This result establishes finite prediction utility on this dataset; no selected-action or manipulation benefit is claimed.

## Actual next experiment, already running

Preregistration source `47c7278dec17161eb4c111a4aa64ed3fc1f66e50` and [the RootSafe action protocol](../../../../experiments/wm02/night1_late_decision_rootsafe_v4.json) define an actual pre-response exam: eight50ms hold atoms, then installation of a preregistered hold/+X/-X/+Y1mm action, then16 predicted atoms. There is no preview of the future chosen action in the initial controller prefix. All eight single-scene pattern qualifications and the two-environment1.2s pattern qualification passed with clean owned exits. The balanced32-episode production panel has started. Its data and three matched RAW/scaled/NO_ACTION models are **not yet completed at this report snapshot**.

All three models are predeclared as fresh2000-update+one-probe fits on the same sealed TRAIN data, seed and normalizer. New physical parents P09–P16 are held for subsequent zero-update prospective evaluation after all three models freeze. This is an action-information test; it is not exact hidden-state branching, oracle regret or formal Actor training. Contact qualification remains blocked by the absence of a qualified native approach prefix; unrelated qualified science continues.

## Commands, index and validation

The actual collector/replay/baseline/training CLI entries are unchanged from [PROGRESS_01](PROGRESS_01.md). The continuation's immutable training spec binds the prior200 checkpoint and `new_batch_tag=NIGHT1_RECEIVER_SAFE_BATCH_B`; the output receipt records100 real new optimizer updates plus one separate reload probe. Additional implemented zero-physics controls and frozen-model evaluation entries were run with exit0:

```bash
PYTHONPATH=src "$WM02_TORCH_PYTHON" -B -m scripts.wm02.evaluate_target_wm \
  --spec "$WM02_PRIVATE/FROZEN_EVAL_SPEC.private.json" --output "$WM02_PRIVATE/PER_PARENT_METRICS.json"
PYTHONPATH=src "$WM02_TORCH_PYTHON" -B -m scripts.wm02.fit_target_arx \
  --replay "$WM02_PRIVATE/UNION_REPLAY.npz" --output "$WM02_PRIVATE/ARX_REVIEW"
```

The existing canonical AUTO1 state index was updated through its original owner and onceIO at the verified cold boundary. Its `night1` entry contains current counters and sealed model/data identities; the older top-level global counters are explicitly retained as the pre-NIGHT1 snapshot. No second ledger, daemon or campaign was started.

The updated full component suite passes124 tests; ARX causality, DEV exclusion and recursive plant tests also pass with Python-O. This is component validation, not a replacement for the recorded native checks. No GitHub CI result or monetary/power charge has been measured. The next executable entries are the already registered finite late-decision collectors and then `scripts.wm02.train_target_wm` on the sealed V3 replay, under the same original dispatch cutoff.
