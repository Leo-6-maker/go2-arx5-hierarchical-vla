# H200 engineering selection and first formal training allocation

At 16:55 SGT on 21 November, Hopper PBS had released the first formal training job `727341.hopper-m-02` under project `CFP04-CF-007`. PBS reported Q and insufficient available GPUs. The job requests one H200, 12 CPUs, 225 GB host memory and six hours. Its first segment covers phase 0–100. Formal updates, visits, sealed endpoints and student requests were all zero at this readback.

## Engineering result

| Backend | FM / auxiliary microbatch | Adam updates | Relative L2 error of total first-step gradient | Outcome |
|---|---:|---:|---:|---|
| E1 | 1 / 1 | 16 | 0.000000 | Selected reference |
| E2 | 4 / 1 | 0 | 0.007891 | First-step rejection |
| E3 | 8 / 1 | 0 | 0.008064 | First-step rejection |
| E4 | 16 / 1 | 0 | 0.008128 | First-step rejection |
| E5 | 1 / 2 | 0 | 0.053658 | First-step rejection |
| E6 | 1 / 4 | 0 | 0.046715 | First-step rejection |

The E1 worker in `727311.hopper-m-02` committed 16 updates with 1,024 Go2, 512 RoboDojo and 512 LIBERO visits and 64 training H40 generations. Twelve measured warm updates took 16.26 seconds at p50 and 17.56 seconds at p95. The original H40 and seam checks passed for 13 of 32 target outputs. All 48 ordinary outputs matched after independent cold restore. This is an engineering panel. Each rejected candidate consumed a further 64/32/32 first-step inputs and attempted four training H40 generations before Adam. All five had matched inputs. The first technical interruption and the successful engineering allocation charged 23 and 1,536 H200 seconds respectively.

Formal MAIN restores all 140 arrays from ADAPT2K `2bff547b6f2c3aca0e7e2fd8a790ce6b30ef1c5f33b004c6897e3083647358d5`. The E1 engineering checkpoint remains separate. Scientific implementation `b9581196d0acd02fb748222988e5a5543ce1f232` and the frozen data and sampler digests are recorded in the JSON.

## Persistent execution

The reconciled VANDA CPU dispatcher `1601905.stdct-mgmt-02` reported the real queued Hopper job on two separate polls. The prior dispatcher `1601885.stdct-mgmt-02` had stopped during a login-context check before submission. The terminal receipt was retained. There was one formal GPU submission. VANDA CPU jobs `1601870`, `1601877` and `1601890` reported waiting for private backup, independent formal evaluation and public Go2 weight publication respectively. Core's receiver PID 1960000 reported zero received checkpoints in an independent connection. Its service runs under user linger and takes the inference GPU lock only when an eligible package arrives and capacity is available. GPU generation, formal evaluation, strict Core RGB16 and student DEV/CONF results are still pending.

## Core expert collection

At 16:55:04 SGT, TRAIN had 23 terminal requests with 14 grasps, 14 lifts of at least 20 mm and 7 admitted successes. Fourteen stopped for illegal contact and two exhausted the carry budget. VAL had 12 terminal requests with eight grasps, eight lifts and four admitted successes. Eight stopped for illegal contact. Closed TRAIN and VAL accounted for 660,463 and 346,115 physics ticks respectively. All 22 hard stops had zero subsequent ticks. One additional TRAIN request was active. The cumulative published TRAIN snapshot contained four of the seven admitted examples. VANDA A40 job `1599370.stdct-mgmt-02` independently reported phase 2,797 at 16:53:41 SGT, with phase 2,750 sealed.

The next verifiable events are the actual H200 worker start, the phase-zero complete-state seal and the first committed MAIN update. Independent formal evaluation can consume the privately backed-up seal. Hopper REF16 and Core RGB16 have their own receipts. Only model checkpoint arrays and digest-only comparison metadata cross into Core through the authorized public repository. Campaign cutoff is 5 December at 12:41:52 SGT.
