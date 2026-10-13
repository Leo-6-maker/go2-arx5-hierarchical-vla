# Next Data Goal Draft

Status: **NOT_AUTHORIZED**. No run ID, new science T0 or resource grant is adopted.
User will discuss the task with GPT before a fixed science card is issued.

Suggested first scope: widen object/target layouts within the existing qualified
ARX5 fixed-base task; admit visual variations only after separate teacher/contact
and camera qualification. Multi-object language selection, pushing and stacking
need distinct teachers and scoring and are not implicit additions.

The next card must specify:

- Scientific PI/run identity, fixed protocol/GO/source/entry release and leases.
- Task/scene families, robot/assets, instruction, teacher and safety/score gates.
- Disjoint pre-registered CANARY/TRAIN/DEV/confirmation episode IDs and seeds.
  Counts are **TBD**, not inherited from the completed 64 TRAIN / 16 DEV bank.
- CPU core-hours, renderer threads/concurrent cores, wall cutoff/closeout,
  integration and observation counts, disk/quota headroom and failure policy.
- A serial canary then bounded 1-vs-2-worker comparison, with all first attempts
  and failures retained; no silent retries or changes to dt/safety/physics.
- Separate immutable next-version raw data, numeric export, TRAIN statistics and
  fixed HF revision/prefix/member hashes. Do not refit on DEV or feed new data into
  an ongoing v2 optimizer. Reuse existing authentication; no full history download.
- Canonical CoreLab entry and single coordinator; process-local MuJoCo state and
  renderer, atomic global reservations, immutable terminal-episode audit/export
  with bounded queues/backpressure. Unknown workers retain resources until proven
  terminal and reconciled. The production adapter still requires implementation.
- Reserve CoreLab capacity for PI0292 S6 fresh-observation relay. New data work
  must yield at a verified boundary rather than interfere with closed-loop timing.
- Hopper only in legal owned PBS allocation, freshly rechecked CFP04-CF-007 and
  personal quota. Do not alter current PI0292 recipe or reopen its deadline.

Engineering options for a later training run: compare loader workers 0/2/4 and
the bounded verified decode cache against the native baseline. Accept only actual
byte/order/resume/model-update equivalence and measured time/RSS/scratch benefit.
Persistent workers and prefetch=2 already exist in PI0292. VLM feature caching,
kernel/compile changes, DDP and LoRA are deferred, not silently enabled.

This fixed-base task remains an independent data environment. Success does not
establish Go2/WBC integration or real-robot success. Selected/deployment are NONE.
