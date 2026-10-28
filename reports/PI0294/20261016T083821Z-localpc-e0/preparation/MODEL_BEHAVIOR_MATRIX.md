# Original Model Panel: Unobserved, Not Zero Success Rate

Source: FIXED_MODEL_RESULTS_CPU.json and FINAL_MODEL_DEVELOPMENT_REPORT.md in R3 run20261014T121640Z-incoming-pi0294-r3 at review commit b408ee2882c9d517557be8aa0236719163475f40.

| Model | Split | Task | Planned | Observed | Status | Observed Success Rate |
| --- | --- | --- | ---: | ---: | --- | --- |
| M512 | TRAIN | T1 | 2 | 0 | NOT_RUN / UNOBSERVED | Undefined |
| M512 | TRAIN | T2 | 2 | 0 | NOT_RUN / UNOBSERVED | Undefined |
| M2048 | TRAIN | T1 | 2 | 0 | NOT_RUN / UNOBSERVED | Undefined |
| M2048 | TRAIN | T2 | 2 | 0 | NOT_RUN / UNOBSERVED | Undefined |
| M2048 | DEV | T1 | 4 | 0 | NOT_RUN / UNOBSERVED | Undefined |
| M2048 | DEV | T2 | 4 | 0 | NOT_RUN / UNOBSERVED | Undefined |

Total16 planned,0 observed,0 CPU-recomputed episodes. No observed early rejects or live native failures in this panel. Original closeout reason ORIGINAL_NEW_WORK_CUTOFF_PARTIAL. This is not0/16 task quality and does not erase earlier R1 actual M512 safety failure under its separate identity.

Original groups, read from R3 design task card: TRAIN F0/000 and F1/000, each T1/T2 at both checkpoints; DEV F0/000, F1/000, F2/000 and F1/001, each T1/T2 at M2048. Keep private exact cards/initial-state hashes from the original registered panel, not newly sampled substitutes.

New evaluation must have new attempt IDs and its own results, leaving this table historical. Suggested first priority is M2048 TRAIN4, then the fixed DEV8 if permitted by actual safety/resources; M512 TRAIN4 is a matched comparison when budget permits. Changing the original execution order must be disclosed, not described as a byte-exact replay of the old run.

Each actual row needs status, exposure, early reject, live native safety, model raw H40 hash, installed prefix hash, post-action observation/ACK identity, approach/contact/grasp-or-push/release events, final object outcome, actual CPU/GPU/integration/query cost and scoring receipt. S1 rejection or incomplete exposed tail counts as task failure in the predeclared attempted denominator. Unstarted conditions remain NOT_RUN; infrastructure failures retain their own outcome and exposure, never quietly disappear or become policy successes. Report attempted, fully observed and originally planned denominators separately.

No new model behavior has been measured in this preparation package.
