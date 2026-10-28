# Q0 Inventory And Freeze

Adopted CPU handoff:4a9bbf38cbde90c2e08c4f4270364d801534cdf1. Existing owner
read the complete same-ref handoff, protocol and machine card, plus AGENTS,
START_HERE, entry policy and frozen-WBC scope. #11/#47 were refreshed through
their final paginated pages; no second owner was created.

Actual read-only boundary:2026-10-16 19:10:01–19:10:03 SGT, epochs
1792149001.1746328 and1792149003.0333529. This is a timestamped snapshot,
not a permanent claim about the live scheduler.

| Item | Actual Fact |
|---|---|
| E0 run | 20261016T083821Z-localpc-e0 |
| Native job | 670049.hopper-m-02, Q;1 submission,0 model loads,0 queries |
| Both E0 leases | ACTIVE, same exact run UUID |
| Core main work | 0 starts/0 integrations/0 reservations |
| Both immutable capsules | 428 members individually verified, runtime9a699d28c6a585df1abb5aa906e41e8d8c025bdc |
| Seal | d9e8d2f7ce2faa92353e8a8f052f46163adaa162f111774771c3a92c77c63a87 |
| Core manifest | 01a445db7b06ba7aa0d305639773777229f831c58adb68dedcf90163a4c7dfe5 |
| Hopper manifest | 5c671d27cb8d2b8a36845882991c3ee3cc311737d30b53db94e3b14a56f5da85 |
| FlowDAgger in E0 capsule | false on both nodes |
| TF-R3 | STOP_R3_CURRENT_VERSION_NATIVE_SAFETY present; resource/source-boundary locks unheld |
| TF-R3 stop hash | 22d554d5f39bd30e3b8fddf1bd44f0b6dbb1865e888ba6dbd1c0f9f196ba6b79 |

No queue cancellation/reordering, second qsub, source upgrade, TF-R3 restart,
lock removal or data cleanup occurred. E0 still has priority and its original
clock/2PBS/6H200h/oneGPU/TRAIN4->DEV8/S1 limits remain intact.

Fixed existing source input: v3.0 158 admitted episodes,126TRAIN/32DEV, EEF10.
v3.1 is295 including v3.0, separate/not hot-mixed. M2048 completed training;
historical16 evaluation conditions remain NOT_RUN. Fullbody successful seeds0.
Statistics fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7;
manifest b8ddcb9808e6dfac4f3ac5fa5a4828128fa9644a330c17d1c57bf833ce5889df.
Actual model/processor metadata, fixed upstream blobs and versions are in
`third_party/flowdagger/UPSTREAM_SOURCE_LOCK.json`. No weights were loaded or
redownloaded for this preparation.

The five original E0 preparation files were rehashed against the existing
adoption list and all remained byte-identical. They stay under the E0 report;
we did not rebuild those materials or export/refit the dataset again.

CPU preparation uses a separate local management checkout, not either runtime.
Final receipt generation is confined to2 local CPUs,1 tensor/BLAS thread,
600CPU seconds and no CUDA; no new remote prep worker or lease was created.
Early development tests were not affinity-confined (Torch reported14 threads);
their exact cumulative CPU is unavailable. This limitation and a conservative
upper charge are disclosed in PREP_FINAL_REPORT, not treated as measured2-core
execution. No destructive cleanup was done.
