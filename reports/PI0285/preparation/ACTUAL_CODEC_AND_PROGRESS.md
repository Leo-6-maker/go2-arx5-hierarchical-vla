# PI0285 actual source/data progress and frozen-codec gap

Authority remains canonical6058329368 / release7e840324c71aef47b401bd861b22710774334c97. This is execution evidence, not a new scientific decision. PI0284 and all original PI0285 receipts/code seals remain intact.

| Stage | T1 | T2 |
|---|---:|---:|
| PILOT expert | 8/8 | 8/8 |
| TRAIN expert | 64/64 | 64/64 |
| DEV expert | 16/16 | 16/16 |
| Learned policy | NOT_EXECUTED | NOT_EXECUTED |

The original collection completed exit0. Source TRAIN/DEV CPU audits and immutable data build completed exit0. The original full-codec admission exited1 before native mixed-input/two-worker checks, so there is no DATA_ADMISSION_COMPLETE receipt and no healthy training job to restart.

## Actual codec evidence

The sealed original code iterates all128 TRAIN episodes before the first DEV episode. All76928 TRAIN H40 assertions completed before the failure at T1-DEV-285200 / anchor20. Each assertion enforces XYZ<=1e-5m, SO3<=1e-4rad, gripper<=1e-5 and finite/nondegenerate rotation. Aggregate TRAIN maxima were not saved: UNOBSERVED. This is control-flow and original-log evidence, not a newly issued complete codec PASS receipt.

Read-only CPU decomposition of all192 originally required DEV representative windows found T1:95PASS/1FAIL and T2:95PASS/1FAIL. Both failures are layout285200 / anchor20 / phase0. SO3 max0.035746579rad; XYZ max~2.12e-8m; gripper error0. Relative rot6d channel8 exceeds the frozen TRAIN range, with34 coefficients outside its minmax bound and preclip magnitude1.4482065. Independent matrix/SciPy calculation confirms the metric. Unclipped counterfactual algebra roundtrips within~1.65e-8rad; it is diagnostic only and is not used for supervision, inference or a changed decoder.

All data/processor/statistics hashes are unchanged. No label exclusion, normalizer refit, decoder modification or replacement teacher collection occurred. The two DEV failures remain visible in their original denominators.

## Scope decision and independent next work

Protocol section5 says valid labels and representative events must roundtrip, then specifically protects supervision/recovery labels from silent clipping. DEV is not supervised. Whether the two non-supervised DEV target windows should block training is a scientific scope clarification, not a routine GO. An explicit user clarification was requested; until received, launch_allowed=false. Neither interpretation changes TRAIN/recovery tolerances or the frozen mapping.

The existing native input checker was factored into an independent input-only path. Its standalone CPU run checks real episode images/state/language, single versus mixed padding/masks, paired initial observations, source/processor cache identity and actual ordered two-worker batches. It cannot write DATA_CODEC_ADMISSION. A wrapper syntax failure occurred before child execution, was retained as an engineering failure, and was repaired without new scientific queries. The actual repaired process is separately adopted and awaited.

## Resource/access truth

Corelab access is restored through the authenticated alternate login address and existing internal GPU hop. The old wired interface retains its address but cannot resolve its gateway/peer neighbors; SSH itself is listening. Login home has~1.6TB free (57% used), so a filled home is not supported by evidence. No remote network/service changes were made. Hopper qstat currently reports no user jobs; no PI0285 submission/allocation/data-readback exists.

No new model, gradient, optimizer, target-WBC or hardware calls occurred. No confirmation layouts were opened. selected/deployment=NONE. The original finite PLACE-first, then confirmed SHARED route remains the objective; preparation and expert success do not establish model quality.
