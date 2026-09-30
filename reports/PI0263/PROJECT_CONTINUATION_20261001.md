# PI0263 lawful project continuation — 2026-10-01

Private account login and VPN are now restored. Live `hpc project date` and the authenticated Budgets project record confirm CFP03-CF-004 ended on 2026-09-30 despite its remaining 904.3 GPU-hours. This confirms the previous scheduler expiry error; positive balance alone did not establish eligibility.

CFP05-CF-052 is active, includes e1192548, starts 2026-09-28, ends 2027-09-30, and currently has 1,000,000 credits / 10,000 H200 GPU-hours. The resource continuation uses this authorized current allocation. Each queued request remains one H200 for at most six hours.

At 2026-10-01 01:25:06 +08, native qalter changed ONLY the project attribute of existing jobs 640396 (F main) and 640412 (X canary) to CFP05-CF-052. The readback confirms both project values. Both already originated through auto and remain routed to small. F remains Q; X remains system-held afterok:640396. No duplicate qsub, new hold, job deletion, staged file edits, donor changes, or scientific changes occurred.

Future PI0263 submissions must explicitly pass `-P CFP05-CF-052` to override the old project directive in the sealed PBS files. Keeping their bytes intact preserves the already-sealed execution identity. Archive/report problems never justify retraining.

F actual recovery canary PASS remains valid. Main updates remain 0/4000 at this check; raw canary updates16/heads304 retained. Initializer P2000, plan, templates, RNG, loss, optimizer, evaluation budgets, historical FAIL/MIXED and selected/deployment NONE unchanged.

The budget credential block is resolved. Continue existing F→X canary→X main sequence as resources become available; no routine scientific GO required. Latest scheduler comment at project change was insufficient ngpus, not a training result.
