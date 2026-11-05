# PILOT1 TRAIN coverage and zero-integration startup evidence

The original P18–P20 TRAIN panel is being completed without reusing consumed TEST parents. All three scene qualifications passed. The first coordinator stopped before feedback admission because its private builder omitted `scene_qualification` for `TRAIN_FEEDBACK`; the intent was reconciled as aborted, with no feedback physics. The corrected builder passed all three real-scene dry runs.

The second coordinator completed 27 feedback conditions at 4,103 native ticks and 64 task atoms each. Its 28th attempted condition remained at zero ticks and timed out waiting for step 25. Pair 13 slot 0 completed; slot 1 failed. Two later conditions were never started. The failed two-slot batch is retained as failed, regardless of its successful child. Its coordinator additionally failed while reading the absent score from the zero-tick child; that reporting defect must not erase this attempt from the denominator. All involved worker leases closed.

The zero-tick child reached native HOLD/RLLocomotion, constructed MPC/MRT/planner before clock zero, and kept reporting `running=false`, time zero. There was no native `Simulation Start`, physics journal, safety STOP or outer rejection. The earlier adapter published one ENTER without observing that publisher's matched subscription count. Native source also uses a single shared keyboard event and clears it each control cycle. Existing evidence cannot distinguish DDS delivery from native event consumption. Root cause remains unresolved; no physics/model failure is inferred.

The new adapter waits for exactly one matched clock-key subscriber and records the single ENTER request. It still requires the original exact native pause and never retries an unacknowledged toggle. A timeout preserves matched-count, control and pause diagnostics. This adds a graph readiness check and evidence; it does not establish that the historical cause was DDS or repair native keyboard concurrency. Three seam tests cover delayed discovery, no subscriber and ambiguous subscribers. A fresh cold-start check is required before completing only the three unexecuted conditions.

A matched old-coverage Actor control independently finished on GPU1:

- Source `5dbe7d6c2eebc899e4e00c8d84a445e174aef5d6`.
- Original frozen task `5e0653617bc735dd39ec75cac1598d4b76488d1c6d72ecc50e30ae6d6d6c7b24`.
- Replay `ea11b27e568443b5f9964f36da7563040a59e4dfb012391c0708a9b747d04734`, 24 TRAIN / 8 DEV windows.
- Checkpoint `87230a2341a7c79d4686ab14fe001aec9c6de8add06dcbfe249f2bb717a94cf3`.
- Portable Actor `e3ee91086a19922e072bedbe4f1028bab4f80df1676b44a58bc20df9731935fe`.
- 2,000 main optimizer updates, one separate restore probe; full restore and endpoint reload errors both zero. Worker exit 0, lease CLOSED.
- DEV teacher agreement 0.59912109375, mean model regret 0.1957324857 mm. No native control benefit is established by distillation.
- Worker 6.329826684 s, child CPU 4.848049 s; trainer 3.830409055 s, CUDA 2.504238281 s, peak allocated 147,341,312 B / reserved 341,835,776 B.

A fresh actual GPU query found another user's compute on GPU0. It is left untouched. Local native collection remains two qualified CPU slots; local neural jobs use GPU1. No GPU-physics backend, four-world rollout, distributed job, WBC change or safety relaxation is introduced.
