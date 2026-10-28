# E0 M2048 Source Evaluation: Partial, No Task Success

Final status: **E0_PARTIAL_WITH_EXPLICIT_MISSING_CONDITIONS**.
All 12 fixed conditions were attempted. Eleven have valid scientific negative
results; one TRAIN condition is engineering-incomplete, not a policy failure.
Both PBS allocations and all own workers are terminal; both E0 leases are CLOSED.
No training, WBC/S1 change, v3.1 mixture, BASE fabrication or deployment occurred.

## Actual Coverage

| Panel | Attempted / Planned | Scientifically Adjudicable | Successes | S1 Rejections | Engineering Incomplete |
|---|---:|---:|---:|---:|---:|
| TRAIN | 4 / 4 | 3 | 0 / 3 | 3 | 1 |
| DEV | 8 / 8 | 8 | 0 / 8 | 8 | 0 |
| T1 | 6 / 6 | 6 | 0 / 6 | 6 | 0 |
| T2 | 6 / 6 | 5 | 0 / 5 | 5 | 1 |
| Total | 12 / 12 | 11 | 0 / 11 | 11 | 1 |

The 11 rejected conditions remain in the policy denominator. Do not report
0/12 valid failures or complete scientific coverage. The original historical
16 NOT_RUN records remain unchanged. TRAIN task success was not an extra gate:
after settling the relay interruption, the unchanged pipeline continued to DEV8.

| Condition | Actual Duration (s) | Main Steps | Generations | Terminal |
|---|---:|---:|---:|---|
| TRAIN F0/000 T1 | 1.2 | 600 | 4 | Valid S1 rejection |
| TRAIN F0/000 T2 | 14.8 | 7400 | 38 | Engineering-incomplete relay interruption |
| TRAIN F1/000 T1 | 1.2 | 600 | 4 | Valid S1 rejection |
| TRAIN F1/000 T2 | 12.4 | 6200 | 32 | Valid S1 rejection |
| DEV F0/000 T1 | 6.0 | 3000 | 16 | Valid S1 rejection |
| DEV F0/000 T2 | 11.2 | 5600 | 29 | Valid S1 rejection |
| DEV F1/000 T1 | 1.2 | 600 | 4 | Valid S1 rejection |
| DEV F1/000 T2 | 9.2 | 4600 | 24 | Valid S1 rejection |
| DEV F1/001 T1 | 1.6 | 800 | 5 | Valid S1 rejection |
| DEV F1/001 T2 | 11.2 | 5600 | 29 | Valid S1 rejection |
| DEV F2/000 T1 | 1.6 | 800 | 5 | Valid S1 rejection |
| DEV F2/000 T2 | 10.4 | 5200 | 27 | Valid S1 rejection |

## What Actually Happened

Each model query used a new actual observation after the preceding prefix ACK.
Across the panel: 217 fresh observation/generation intents, 216 bundles submitted
to the original gate/S1, and 205 prefixes actually executed and acknowledged.
The remaining 11 submitted bundles were rejected before live execution. One
completed generation in the interrupted condition was never submitted.
Saved source seeds, identities, observation hashes, actions, commands and ACKs
were independently cross-bound. No intent was resent or resampled.

All 11 rejection forecasts reported `ILLEGAL_CONTACT_OR_PENETRATION` in the
unchanged privileged S1 shadow. These are predicted violations, not observed
live violations. Actual high-rate native safety flags and live S1 failures were
zero in all 12 attempts. Maximum live-shadow difference was about 2.28e-13.
The guard did not correct actions or substitute a Teacher.

Most conditions stopped before useful object interaction. DEV F0/000 T1 did
reach actual contact and bilateral-grasp flags: 2206 contact ticks and 1643
grasp-flag ticks, with saved object displacement about 23.48 mm and maximum lift
about 0.986 mm. These are observed totals, not a claim of continuous stable
grasp, successful lifting, release, placement or hold. The interrupted TRAIN
push had 15 contact ticks and about 0.336 mm saved displacement, but cannot
supply a final policy outcome. No condition completed 32 seconds or the task.

Short-trajectory scorer fields such as `clock_valid=false` and, for incomplete
T2, `highrate_contact_complete=false`/`safety_valid=false` do not themselves
establish actual safety violations. Their original values remain preserved;
the independent audit explicitly covers each actual partial high-rate array.
The old worker's `model_calls=0` is a CPU-worker-local field; the real neural
count is 217 from Hopper's durable reservations and generation receipts.

## Engineering Failures Preserved

Allocation 01 (`670049.hopper-m-02`) failed before model load: the owner-side
fresh project query returned 15006/token-expired and no START release was made.
The launcher waited 180 seconds and exited. Native GPU time was 183 seconds,
conservatively charged 188. A separate latent parser defect was found because
native qstat returns English site-local dates rather than only integer epochs.
The narrow parser repair passed 8 E0 and 188 PI0294 CPU tests.

At a verified pre-model/pre-physics boundary, allocation 02 sealed runtime
`ce1a14a56ddfe9f322e159eef8a2f8054da22525`, preserving old capsules and bindings.
All 15 original numeric files and `ResidentModel.generate` AST were unchanged.
Both endpoint seals and original scene/episode/noise bindings passed. The fixed
protocol remains `22e5f0baeed0af662a7cf1c97edae1f4060947fe`.

Allocation 02 (`670345.hopper-m-02`) loaded M2048 once, then the local foreground
relay process disappeared during TRAIN F0/000 T2. The local record ended in
WAIT_ACTION at sequence 37; Hopper had completed that generation, but CoreLab
had no command/intent/ACK for it. CoreLab timed out safely at 7400 steps.
All 37 preceding commands/ACKs and their physical receipts were reconciled;
there was no pending neural intent or unknown executed prefix. The exact raw
engineering result was not relabelled or retried. Its engineering stop marker
was archived with a resolution receipt before the ten untouched conditions
continued in the same resident allocation. No actual safety stop was cleared.

## Evidence And Resource Terminal

- [Per-condition results and independent layer audit](E0_MODEL_BEHAVIOR_RESULTS.json).
- [TRAIN4 interim and explicit engineering resolution](E0_TRAIN4_INTERIM.json).
- [Original interruption settlement](OWNER_INTERRUPTION_RESOLUTION.json).
- [Raw hashes and 12 private video IDs](E0_RAW_EVIDENCE_AND_PRIVATE_VIDEO_INDEX.json).
- [Native resource and closed-lease receipt](E0_RESOURCE_FINAL.json).

All 12 saved-state score/raw-hash/S1/native-clock audits passed. All 12 paired
videos were fully decoded at their actual lengths, totalling 1652 paired frames;
none was padded to 32 seconds. RGB, videos and model weights remain private.

Both native jobs are F, with exits 1 then 0. Total scheduler GPU time is 1708 s;
the conservative native start-to-obit charge is 1723 s, below 21600 s. PBS 2/2
is consumed. Main starts 12, main steps 41000, shadow steps 42075, generations
217, all reservations zero. Core/Hopper CPU accounting is 6378.43/11003.97 s,
including actual work and deliberately conservative management bounds, not
utilization telemetry. Final owned logical bytes are 227241679/12615903797;
this is a file-size snapshot, not continuous peak sampling or shared free space.
All recorded own workers are terminal. Both closed-lease readbacks reverified
438 sealed source members; the separate TF-R3 safety stop hash is unchanged.

## Interpretation And Next Boundary

M2048 has not demonstrated either complete source task on this fixed development
panel. This does not prove no learned behavior at all, explain the training
failure's unique cause, or say anything positive or negative about full Go2,
target WBC execution, another embodiment or a real robot. E0's remaining gap is
the one interrupted TRAIN result; no third PBS or silent retry is authorized.

The next useful work is a bounded source-learning diagnosis using TRAIN-only
inputs, plus qualification of the prepared Recovery Oracle on safe actual
student-visited TRAIN states before any DAgger labels or FlowDAgger training.
Do not tune on E0 DEV, refit v3.0 statistics, mix v3.1 or fabricate BASE labels.
Any real recovery physics, model inversion, optimizer work or replacement of
the incomplete condition needs a separately adopted scope. The already-published
CPU FlowDAgger package remains PREP_CPU_READY_NOT_DAGGER_EXECUTED.

E0 execution is stopped and closed. **Selected/deployment: NONE.**
