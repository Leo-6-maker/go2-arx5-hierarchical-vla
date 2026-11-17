# B0 saved-prediction baseline — no new generation

This CPU-only audit reuses the exact frozen SHIFT7 outputs and original evaluators. It does not execute a trajectory or establish a new grasp success. Analysis source: `9c0413c14cdda409aa2e8e34624cd99c714b564a`; package manifest: `ef530bceebcc1691d9af9b5b1ea4ea89731f0c1d5d8f3e2287c3618dff7806eb`.

| Bank | Complete H40 passed | CLOSE timing | Terminal intent mismatches |
|---|---:|---:|---:|
| TRAIN96 | 96/96 | 20/20 within100 ms | 0 |
| CN38 | 180/180 | 36/36 within100 ms | 0 |
| ROLL96 | 93/96 | No same-state expert labels | Not applicable |

The three saved failures remain `02-student-f2/q2` (tail .2542523654 m/s²), `04-student-f2/q12` (head .2514066458), and `06-student-f2/q21` (head .2946091929), against the original .25 limit. Thus B0 currently fails the saved-reject3 promotion gate. Its eight E2 conditions remain mandatory; this finding does not filter or replace any control.

The JSON also reports head/tail/interior margins, previous-command seam speed, H40 progress and EEF error distributions. TRAIN/CN38 EEF errors use their aligned expert plans. ROLL96 EEF differences use saved B0 predictions only and are not expert supervision. All values are predictions, not native consumption. Current-scope RGB16 restoration remains pending.

Additional effects from this audit: **0 model loads, 0 new generations, 0 updates, 0 native requests, 0 integration**. Historical PR403 and old failures remain unchanged.
