# Why three saved release choices were worse

This is a **post-hoc mechanism audit of already-opened prospective DEV8**, not new blind validation, a new deployable selector or evidence of full-task failure. No model was reloaded, no inference/training ran and no physics was repeated. The original rule already reaches the sampled oracle at these anchors.

Actual diagnostic source **`af71b25beb00954d0200cde0f166d7e2d07c18a2`**, exit0. It SHA-checks66 saved files against original stream manifest **`d6593c1b37432164ca38fd88da1c60036b20b2b7126b7e9f767acce856c38c01`**, then recomputes the original score and measured Q for all8 parents×5 candidates. Original frozen prediction-model derivative `cba5a6b162c9e162614855dfe9983dbaad10a2f630f5e3c4be6eb07e14a5d674`, parent checkpoint `9676312c5b810a757c465174ddc4302fc67b1f6896f8ba66dbed5242dee77f7b`. Model tensors, commands, scores and outcomes are unchanged.

The original release proxy is `(1 - predicted_grasp) * predicted_gripper_fraction`, clipped to the physical opening range. Actual diagnostic Q additionally requires native support and no native grasp. All40 candidate endpoints have native support, so the missing support head **does not explain these particular ranking errors**. It remains a broader deployment limitation.

| Parent | Original WM choice | Predicted score advantage over immediate release | Actual Q disadvantage |
| --- | --- | ---: | ---: |
| 41 | delay2 | .000262082 | .045322515 |
| 43 | delay2 | .000361502 | .045322897 |
| 46 | delay2 | .000225842 | .045322339 |

The actual immediate-release opening is about.72223 versus delay2 about.67691. The model predicts both near.617–.620 on these3 parents and reverses their opening order by only.0006–.0008. Thus a substantial actual opening difference is compressed and reversed before ranking; the final score margins are tiny. This localizes an important error to gripper action/time response rather than establishing a unique architectural or training cause.

We substituted one **future truth component at a time, only for explanation**:

| Diagnostic score | Parents with positive sampled-oracle regret |
| --- | ---: |
| Original saved WM score | 3/8 |
| PRIVILEGED true opening + predicted grasp | 0/8 |
| Predicted opening + PRIVILEGED true grasp | 6/8 |
| PRIVILEGED true opening + true grasp | 0/8 |

True opening removes the3 errors. True grasp alone worsens the ranking to6 errors because the opening ordering is already wrong on6 parents, and small predicted grasp differences partly offset that error on3 of them. Consequently, low release grasp-Brier or simply replacing its event head is insufficient to qualify the selector. These are truth-substitution explanations, never inputs available to a deployed Actor, and never a new strategy PASS.

The first audit source `20df8f43b1615e17cef24b73e469831612160545` exited1 before output because it expected a later optional `study` field. Original new8 records predate that field. The fix verifies the original SHA-pinned40–47 DEV cards/partition and24 intended anchors instead; no original metadata or results were edited. Three focused tests pass in ordinary and `-O` mode, including a TEST-partition rejection and component-confounding counterexample.

The complete updated world-model suite runs **59 tests without skips** in the existing isolated PC native/model environment: normal4.253s, `-O`4.152s, both exit0. The preceding system-Python checks each collected59 but skipped1 module because MuJoCo headers/library were unavailable; those logs are preserved and not counted as59 passes. No packages or system environment were changed to obtain the complete check.

Actual command, with private variables resolved to the already-audited original paths:

```bash
PYTHONPATH=src:. python3 -m scripts.wm02.diagnose_saved_release \
  --root "$ORIGINAL_NEW8/validation-01" \
  --manifest "$PRIVATE/audit-new8-01/STREAM_MANIFEST.private.json" \
  --manifest-sha256 d6593c1b37432164ca38fd88da1c60036b20b2b7126b7e9f767acce856c38c01 \
  --output "$PRIVATE/POSTHOC_RELEASE_RANK02.private.json"
```

Measured diagnostic CPU.326164364s, elapsed.326181091s; failed-read/test/startup overhead not separately metered. New GPU/S/T/WM/Actor/full tasks all0. Full arrays, input SHA pins and privileged substitutions are in `POSTHOC_RELEASE_RANK.json`.

**Decision:** retain the original release rule; do not train Actor from these small proxy margins or start another neural run merely to improve this already-opened exam. A future model/decision gate must cover measured gripper response and ranking margins as well as object/TCP and event metrics. New independent validation with actual oracle headroom is required for a capability claim. Target dynamic acceptance remains independently owner-blocked.
