# Stage23 Phase B Attribution Audit

- Classification: `PASS_WITH_PROVENANCE_DEVIATION`
- Scientific evidence classification: `OBJECTIVE_NOT_DEMONSTRABLY_FIT`
- Scope: sealed evidence analysis only; no model forward, inference, training, PBS, WBC, hardware, or holdout-f access.

## 1. Evidence seal

- Archive SHA256: `720aca9a1a1c7146919620df4000a76cc228a71729b057a2a0d155787ea50dcd` — `PASS`
- Payload manifest SHA256: `dfeb6b44410b0042e53bac261e49acf7dbaa23ee2a61839ab65d4171b6c42b6a` — `PASS`
- Payload files: `252/252` hashes verified
- Credentials absent by sealed declarations: `PASS`
- Sealed no-new-training/inference declaration: `PASS`

## 2. Training fit reconstruction

These are five independent fresh BASE → N experiments. Loss and learning-rate rows are not one continuous curve.

| run | global step | log entries | first loss | minimum loss | final loss | final LR | grad norm final |
|---|---:|---:|---:|---:|---:|---:|---:|
| O1@10 | 10 | 10 | 0.9684 | 0.9677 | 1.011 | 3.015369e-06 | 0.7196 |
| O1@25 | 25 | 25 | 0.9684 | 0.8947 | 1.059 | 4.657027e-07 | 0.6726 |
| O1@50 | 50 | 50 | 0.9684 | 0.8825 | 1.02 | 1.1165607e-07 | 0.526 |
| O1@100 | 100 | 100 | 0.9684 | 0.8052 | 1.008 | 2.7337133e-08 | 0.3894 |
| O1@200 | 200 | 200 | 0.9684 | 0.8129 | 0.9737 | 6.8347504e-09 | 0.3193 |

Exact global-step evidence: `PASS`. Direct optimizer.step callback in the bundle: `FAIL`.

## 3. Normalization

H2 relative q01/q99 cells: `360`; zero-width under the official `np.isclose` rule: `243`.
Regenerated TINY1 relative q01/q99 zero-width cells: `360` / `360`.

The one-episode regenerated tiny statistics collapse per-time action ranges to zero width, so TINY1_TINYSTATS is a degenerate diagnostic. H2 frozen Stage22 statistics retain nonzero positional relative-action ranges; orientation rows are constant by task design.

The regenerated one-episode statistics are therefore retained as a diagnostic warning, not used to explain the authoritative H2 result.

## 4. 132D inactive-noise mechanism audit

Analytic inactive/active noise-only energy ratio: `12.2x`.
Expected inactive noisy energy over H=40: `2093`.
TINY1_H2STATS expected normalized target signal energy: `1.545`.

Official source confirms that the full [40,132] noisy trajectory enters the action encoder/DiT and only the first ten dimensions are masked in the loss. This is mechanistically plausible nuisance burden, not a demonstrated bug.

## 5. Trajectory profile recomputation

No smoothing, clipping, retiming, or trajectory correction was applied. High-frequency DFT is the declared band `min(k,N-k) >= ceil(N/4)`.

| recipe | median path (m) | median tortuosity | median 2nd diff (m) | median direction cosine | median back/forth | median high-freq Δ DFT | SafetyGate |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 3.1309 | 29.352 | 0.13877 | -0.60173 | 0.7895 | 0.8404 | REJECT=3 |
| O1@10 | 3.0514 | 20.752 | 0.13802 | -0.55054 | 0.7368 | 0.8473 | REJECT=3 |
| O1@25 | 3.159 | 21.998 | 0.14185 | -0.56043 | 0.7632 | 0.8393 | REJECT=3 |
| O1@50 | 3.1284 | 23.382 | 0.14017 | -0.59423 | 0.7632 | 0.8484 | REJECT=3 |
| O1@100 | 3.1309 | 22.446 | 0.14071 | -0.54066 | 0.7632 | 0.8443 | REJECT=3 |
| O1@200 | 3.1368 | 22.513 | 0.14092 | -0.54928 | 0.7632 | 0.8436 | REJECT=3 |

The O1@200 profile remains about 25× the expert path on the sealed metrics, with negative adjacent direction cosines and high first-difference high-frequency energy. This strongly supports a free-running temporal profile failure / oscillation phenotype.

## 6. Attribution matrix

| hypothesis | classification | evidence boundary |
|---|---|---|
| input drift | `NOT_SUPPORTED_BY_SEALED_IDENTITY` | Frozen input tree, base index, dataset path, timing, model flags, GR00T SHA and simulation-only flags are consistent across candidate manifests. |
| GPU/resource | `RULED_OUT_FOR_THIS_RESULT` | All five candidates completed on H200 with no CUDA OOM; peak reserved memory was about 32.0 GiB against about 143.0 GiB total and cleanup returned about 65 MiB. |
| optimizer/LR | `PLAUSIBLE_CONTRIBUTOR_NOT_IDENTIFIED` | Independent runs have different cosine/warmup schedules; final losses span 0.9737-1.0591, and final learning rates decay to the 1e-9 scale by @200. |
| flow objective fit | `OBJECTIVE_NOT_DEMONSTRABLY_FIT` | The trainer reached exact requested steps, but the minimum losses only reached 0.8052 and the resulting one-example free-running profiles did not approach the expert; loss alone does not establish target fit. |
| normalization | `SECONDARY_PLAUSIBLE_FOR_TINYSTATS_ONLY; NOT A PRIMARY H2 EXPLANATION` | TINY1_TINYSTATS has 400 zero normalized active values out of 400 and is a degenerate one-episode diagnostic. H2 uses frozen Stage22 statistics and must not be conflated with this variant. |
| temporal oscillation | `STRONGLY_SUPPORTED` | O1@200 has median path ratio 25.57, median back-and-forth rate 0.763, and first-difference high-frequency DFT fraction 0.844; the expert path is near-straight and low-frequency. |
| 132D inactive-noise burden | `MECHANISTICALLY_PLAUSIBLE_NOT_PROVEN` | Official source shows full-width noisy trajectory input before a first-10-dimension loss mask. Analytic inactive/active noise-energy ratio is 12.2. |
| 4-step Euler sampler mismatch | `UNTESTED_PLAUSIBLE` | The frozen source/config uses four Euler inference steps; Phase A contains no 4-vs-8-vs-16-vs-32 comparison, so sampler causality cannot be assigned. |
| data scarcity/generalization | `PLAUSIBLE_UNRESOLVED` | The experiment uses one training snapshot and has no held-out read in the sealed bundle; the result diagnoses learnability on TINY1, not generalization. |

## 7. Required limitation

`PARAMETER_DELTA_NUMERIC_AUDIT_NOT_AVAILABLE_FROM_PHASE_A_BUNDLE` — full checkpoint weights were not transferred, so no numerical parameter-delta claim is made.

The evidence supports `OBJECTIVE_NOT_DEMONSTRABLY_FIT` as the conservative primary classification. It does not prove that the objective is the sole cause: temporal oscillation is strongly observed, while normalization (for H2), inactive noise, sampler mismatch, optimizer/LR, and data scarcity remain attribution hypotheses with different evidence strength.

## 8. Stop decision

No additional Hopper work, fixed-noise teacher-forced flow diagnostic, sampler-step comparison, WBC execution, or closed-loop run was performed or authorized by this Phase B audit.

## 9. Provenance deviations

- The sealed phase_a_scope authorization field references PR8_COMMENT_5518486291, while the current PI directive is PR8 comment 5519501397; the immutable bundle is reported as received and not rewritten.
- The sealed bundle contains jobs 597960, 597965, and 597966; earlier informal references to other job IDs are not substituted for the bundle's own immutable records.
