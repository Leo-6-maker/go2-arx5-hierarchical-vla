# CR2: Fixed-R1 Pilot Closed And Delivered

Status: **CR2_FIXED_R1_P12_READY_AND_SCOPE_CLOSED**. This is a successful Teacher/data qualification pilot, not a trained-policy result.

## Actual Results

- All 24 first attempts completed on 12 new TRAIN geometries: T1 12/12 and T2 12/12 actual task successes, independently audited and admitted under the fixed R1 codec.
- All 12 geometries retain both tasks. Every AP/direction/task stratum has 2/2 admitted parents. No physical replacements, native safety faults, unknown effects or R0 PASS to R1 FAIL regressions.
- The same saved raw data produced 14 R0 PASS to R1 PASS and 10 R0 FAIL to R1 PASS transitions. R1 did not control the Teacher. These transitions concern encoding fidelity, not a causal task-success or policy A/B improvement.
- Each parent has 641 actual dual-camera observations and 601 checked H40 windows. The bank contains 30,768 RGB images and 14,424 overlapping windows, not 14,424 independent demonstrations. Both codec versions used six real official-reader anchors per parent; every admitted parent also entered an actual R1 cross-parent CPU batch.
- Original 4mm Teacher, actual-state inputs, committed EEF10 labels, WBC/plant/gripper PD/S1/scoring and both fixed statistics identities were preserved. No new fitting, training, GPU, PBS, neural inference or BASE labels.

## Fixed HF Delivery

[Open the fixed, fully read-back bank](https://huggingface.co/datasets/leoyuer/pi0294-public-artifacts/tree/9435ba48fa98336b6e88431ed145382724b16b44/PI0294/20261020T160340Z-localpc-cr2/CR2_FIXED_R1_P12).

- Data revision: `3538419e8092a9c792793259ba286f3a2ab12607`.
- Final READY revision: `9435ba48fa98336b6e88431ed145382724b16b44`.
- Dataset manifest SHA256: `3b818f7c53e155a6b47cc2e2116abc34b921a73ea26d6665a623f2bb9e699f44`.
- Archive index SHA256: `758047304df975e8d64cfeb0237b36469b4011af8d04d0ce11f5af967be88e15`.
- Payload: 1,138,087,675 bytes. The complete payload was independently downloaded on CoreLab and every archive member verified before READY was uploaded. Final READY and final fixed-revision content identities were read back again.
- Only this new bank was transferred. No old models/banks, private WBC code, mesh or credentials entered the payload. Raw video originals remain on CoreLab with the separate 24-entry video index. This is not a claim of actual Hopper consumption or neural qualification.

The first delivery failed at HF preupload with HTTP 401 because its private input binding omitted the existing writer locator. The worker and original intent were preserved. Independent public reads confirmed the repository HEAD unchanged and the prefix absent, resolving the unknown commit outcome as zero remote commit. The existing fine-grained credential had verified own-namespace write permission. A new sealed delivery child explicitly reused that locator and completed; no new credential was requested or created. The failed attempt and its full conservative transfer allowance remain charged.

## Resource Terminal State

The original independent T0 and all cutoffs were retained. Closeout occurred after about 2.91 hours of calendar time, with 4.1544 accounted CoreLab core-hours out of 32. Accounted CPU includes conservative control charges; actual child CPU is listed separately in the resource receipt.

Actual integration totals: 384,000 live plus 384,000 shadow ticks. No reservation remains. All ten child leases, including the preserved zero-effect failed delivery, and the control lease are CLOSED; all registered processes are terminal. Only the finished CR2 control watcher was stopped. The accounted storage upper is 5,205,956,347 bytes versus 48 GiB; the logical transfer upper, including the failed delivery allowance, is 4,753,677,292 bytes versus 16 GiB. These are accounting bounds, not measured TCP wire bytes.

## Claim Boundary And Next Work

This supports producing a small complete fixed-R1 bank on fresh source-domain geometries without refitting R1. It does not demonstrate GR00T improvement, DEV generalization, Go2 transfer or real-robot success. The optional CR1 archive union was not built.

PV1 R0 and its original 139-parent bank were not changed. DEC1's already-authorized original R0 source A/B work remains independent; this new R1 bank is not hot-injected into it. Any R1 neural experiment needs its own explicit scientific binding. `selected/deployment=NONE`.

See the adjacent population, all-24 paired attempts, bank/reader, video and final-resource JSON receipts. Actual capture/export source was `520f5a5a8db536237c5da2bbe9bbef2dc990d8db`; actual bank and delivery source was `89ac3f7cc13dbb1134daa119a905f59d3bc6eb82`, with exact Actions `39476731231` / job `118814509721` successful. Later reusable authentication guards were published separately and were not hot-swapped into the running delivery.
