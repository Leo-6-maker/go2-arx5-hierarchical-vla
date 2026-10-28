## PI0261 C2000 fixed quality milestone — valid FAIL, P still required

C completed its frozen 2000 optimizer updates and 4000 supervised microbatches (`639370` + `639399`, both exit 0). The complete 16-file final Trainer is public and passed full-file fixed-revision readback: `leoyuer/pi0246-gate-pass-campaign@ce5122232b3577005ecabb4cdd95f252cde5b2a6/arms/PI0261/C/checkpoint-2000`, tree `c476ded1519fe792372c5de50ca19bed1e71e60a1eb484efa5b1a5da71626f33`.

C's fixed final-only 680 logical rows all have valid saved cells. Independent CPU replay passed (`42495eaedf4eda2a8c3146823d9fc4953c0ab7d0830d19d617f88c8a5da4af97`). No C retry or interim checkpoint screening occurred.

| Original KIN-VAL task/noise | median XYZ mm | p95 XYZ mm | median SO3 rad | REJECT | Verdict |
|---|---:|---:|---:|---:|---|
| M1/B0 | 5.1 | 11.5 | .0081 | 0 | PASS |
| M1/B1 | 7.3 | 12.9 | .0098 | 0 | PASS |
| M3/B0 | 7.5 | 10.3 | .0019 | 0 | PASS |
| M3/B1 | 5.1 | 11.2 | .0051 | 0 | PASS |

| Task/wording | B0 correct/12 | B1 correct/12 | Original gate |
|---|---:|---:|---|
| M1 instruction | 6 | 7 | FAIL |
| M1 paraphrase | 6 | 6 | FAIL |
| M3 instruction | 12 | 11 | PASS |
| M3 paraphrase | 12 | 12 | PASS |

M1 TRAIN entry is 50/60 B0 and 51/60 B1, versus donor 41/60 in each noise. This observed within-TRAIN gain did not transfer to the original M1 instruction/paraphrase gates. It cannot establish a pairing effect because C is the ordinary-window control. C is a valid quality FAIL, not an integrity failure.

Global new-query ledger after donor and C: 870 attempts and 3480 S4 head forwards, within the fixed 1728/6912 cap. Donor original440 remain reused. P1000 is queued as `639525.hopper-m-02`; P recipe, 2000-update budget and evaluation card remain unchanged. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`. No WBC/physics/TEST/DEV-F/hardware work.
