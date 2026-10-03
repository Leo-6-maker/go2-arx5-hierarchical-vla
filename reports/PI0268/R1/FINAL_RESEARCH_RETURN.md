# PI0268-R1 normalization study: final research return

Both authorized 2048-update arms and the fixed four-policy evaluation are complete. **Quality: VALID_QUALITY_FAIL.** There is no qualified virtual research candidate; selected/deployment remain NONE. Final preservation is tracked separately in the summary archive receipt and final closeout receipt.

## Three prespecified verdicts

| Verdict | Result | Evidence |
|---|---|---|
| Numerical treatment effect | FAIL | ROT final improves both tasks' prespecified coupled X/Y median XYZ, but legacy median XYZ and SO3 regress |
| Original virtual qualification | FAIL for both endpoints | OLD M1 passes both backend subgroups, but M3 fails; ROT M1 and M3 fail |
| Reliable virtual qualification | FAIL for all four policies | Every policy has 0/64 episodes meeting original completion and complete final-five-second retention |

| Policy | M1 original /32 | M3 original /32 | Reliable /64 | Gate-refused episodes /64 | Dangerous collisions /64 |
|---|---:|---:|---:|---:|---:|
| OLD0 | 13 | 0 | 0 | 0 | 0 |
| ROT0 | 0 | 0 | 0 | 64 | 0 |
| OLD2048 | 30 | 0 | 0 | 0 | 0 |
| ROT2048 | 22 | 0 | 0 | 0 | 0 |

OLD2048 M1 is 15/16 under each backend, command counts 7/8 and 8/8. ROT2048 M1 is 11/16 under each backend, with 7/8 for command0 and 4/8 for command1. The failed command1 subgroup is retained, not hidden by a pooled rate. Each M3 task/backend subgroup is 0/16. Original V3 remains primary and original V2 is preserved in the row-level side report.

## What the normalization treatment achieved

| Physical panel, final models | OLD median XYZ | ROT median XYZ | OLD median SO3 | ROT median SO3 |
|---|---:|---:|---:|---:|
| Seen TRAIN fit224 | 17.570 mm | 9.940 mm | 0.048520 rad | 0.036871 rad |
| Factor304 | 16.506 mm | 10.977 mm | 0.038531 rad | 0.037925 rad |
| Legacy440 | 13.941 mm | 15.121 mm | 0.023426 rad | 0.046878 rad |

The prespecified X/Y coupled subset has 32 rows per task. M1 median XYZ changes from 16.782 to 7.392 mm; M3 changes from 18.650 to 18.407 mm. Both OLD controls already have zero REJECT, so the preregistered zero-control exception applies. Legacy regression nevertheless makes the numerical-treatment verdict FAIL. Both final policies have zero offline REJECT and zero nonfinite rows across all 968 conditions each.

Same donor weights with the new processor give ROT0, a different effective initial policy. Its offline orientation errors and 64 virtual gate-refused episodes demonstrate that an inference-time processor change is consequential. ROT's improvements relative to ROT0 describe learning within its own contract. Final OLD versus final ROT measures the combined state/action numerical-contract treatment, including changed effective native supervision weighting; it does not isolate state scaling, action scaling or a single loss term.

The actual matched training audit verifies raw examples, order, random packets, full-S4 losses/gradients/mode and the fixed frozen scope. OLD numerical maxima are about 119928.68 for state rotation and 117504.29 for action rotation, versus approximately 1 under ROT. Recorded maximum preclip norms are 13883616 for OLD and 4411.61377 for ROT; both arms clip 2048/2048 updates. These observations support numerical conditioning and changed optimization behavior, but clipping frequency alone does not explain the task failures. Dense training Euler arrays remain UNOBSERVED; saved hashes and same-pass component losses are preserved.

## Saved behavior: entry, progression and late drift

All 13 OLD0, 30 OLD2048 and 22 ROT2048 successful M1 episodes subsequently leave the original joint pose bounds. Median time from success to first exit is respectively 2.54, 6.28 and 2.86 seconds. These medians use each point's own successful subset and are not matched causal effects. All 32 M1 episodes of each of those three policies are outside the combined position/orientation bounds at every one of the 251 final-window ticks. Original first attainment therefore does not establish persistent retention.

For M3, OLD2048 has broad readiness in 8/32 and V3 entry in 6/32. All six entered episodes complete one ordered bin and register reverse failure; three also register shortcut failure. ROT2048 has broad readiness in 3/32 and entry in 2/32; one entered episode completes one bin and one completes two, and both register reverse and shortcut failure. These overlapping frozen labels show both an entry limitation and failure after entry. No gate or entry threshold was relaxed.

The initial H40 first-position bias is small for both endpoints: median M1 bias 0.451 mm OLD and 0.272 mm ROT; M3 0.349 and 0.271 mm. Median actual net motion in the first 0.4 seconds is M1 5.945/3.951 mm and M3 6.232/5.237 mm OLD/ROT. First-position bias is reference-to-initial-state distance, not target error. Net displacement is not goal-directed progress. These measured quantities do not support a unique first-point explanation for later path and retention failure.

`VIRTUAL_CASE_MECHANISMS.csv` retains all 256 condition identities and source hashes. `VIRTUAL256.csv` separately records reference endpoint, actual endpoint, same-tick tracking, entry/bins, V2/V3, first-success exit, collision/refusal and complete final-window retention. `OFFLINE3872.csv`, `OFFLINE_SUBGROUPS.csv` and `PAIRED_OFFLINE_CHANGES.csv` retain task/command/variant/family/noise identities and the four distinct contrasts. All fixed denominators and failures remain present.

## Actual execution and preservation

| Consumption | Actual | Ceiling |
|---|---:|---:|
| Main optimizer updates | 4096 | 4096 |
| Raw updates including canaries and technical recovery | 4136 | 4384 |
| Training heads | 74504 | 90000 |
| New inference queries | 16326 | 24576 |
| Inference S4 heads | 65304 | 98304 |
| New virtual starts | 192 | 208 |
| New virtual queries | 12894 | 19200 |
| New factor + virtual frame pairs | 13086 | 20000 |
| New WBC/physics/hardware starts | 0 | 0 |

3872 logical offline rows include 440 exact cached OLD0 rows; 256 logical virtual episodes include 64 exact cached OLD0 episodes. Both final policies use 6400 virtual queries each. ROT0 ends early with 94 queries. Every recorded inference attempt has four S4 heads and a saved prediction receipt, reconciled to the independent CPU replay inventory. No valid result was retried after the VPN interruption; remote evaluation had already completed when connectivity returned.

Four full Trainers have public fixed-revision full-file readbacks, matching all 18 files of each actual checkpoint inventory:

| Trainer | HF revision |
|---|---|
| OLD1024 | `9eee805c54833b89199c4041d1c0feafb64f29f9` |
| OLD2048 | `b95514340120cd50d81381b7b9fa37e90979bad3` |
| ROT1024 | `f02ced9302a31a4aada4951ace504d86f692687f` |
| ROT2048 | `97d8bf027f07241e2206b05e07597e4719c891e4` |

Their prefixes are `arms/PI0268/{OLD|ROT}/checkpoint-{1024|2048}` in `leoyuer/pi0246-gate-pass-campaign`. Full tree hashes, file hashes and optimizer/scheduler/RNG identities are in the four full-readback receipts and `PRIMARY_COMPLETION_AUDIT.json`. Local complete recovery sources remain retained. The 169-file training evidence package preserves actual journals, failed engineering attempts and source identities; no Trainer weights were copied to the PC or Corelab.

Full CPU replay covers 256 episode arrays, 3872 offline metrics and all 16326 new native BF16 S4 predictions. Original task replay shares the frozen V3 scorer; entry geometry and physical metric paths are independent. Four videos use the fixed first family, all eight task/command/backend cases and the entire saved observation stream, with VIRTUAL/NOT_WBC/NON_REALTIME labels. All 2410 frames decode; visual inspection covers each chapter's first/middle/last frame. No new rendering or policy query was used to make the videos.

PI0267 remains `PRIMARY_COMPLETE / VALID_QUALITY_FAIL`; its four full Trainers and five result archives are preserved. Secondary WBC remains explicitly PARTIAL: eight valid experts, one engineering-incomplete model attempt with three queries and a missing final post-step state, and 23 deferred model conditions under the shared integrity dependency. Deferred conditions are not counted as quality failures or completions.

The bounded study ends at `STOP_FOR_RELIABILITY_OR_WIDER_TRANSFER_DECISION` once final archive/return receipts close. Improved fit has not produced a qualified persistent policy. The protocol's subsequent learning decision requires its own authorization; no extra normalization search, updates, seed, candidate, WBC or hardware experiment is launched here.
