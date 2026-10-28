# PI0274 — Pose equilibrium and rotational recovery

Execution completion and quality qualification are separate. One new EQUIL5120 arm; closed AGG4096/BROAD5120 outputs reused.
All results are development virtual EEF experiments. E0/E1 are mathematical executors; no WBC, dynamics, physical transfer, blind TEST or real-time claim.
Formal selected_checkpoint=NONE; deployment_selected_checkpoint=NONE.

## Original-start task, reliability and safety

| Point | Task/backend | First completion /16 | Reliable /16 | Reliable commands /8 | Collisions | Gate refusals |
|---|---|---:|---:|---|---:|---:|
| AGG4096 | M1/E0 | 16 | 1 | 1, 0 | 0 | 0 |
| AGG4096 | M1/E1 | 16 | 3 | 1, 2 | 0 | 0 |
| AGG4096 | M3/E0 | 1 | 1 | 1, 0 | 0 | 0 |
| AGG4096 | M3/E1 | 1 | 1 | 1, 0 | 0 | 0 |
| BROAD5120 | M1/E0 | 16 | 2 | 1, 1 | 0 | 0 |
| BROAD5120 | M1/E1 | 16 | 3 | 2, 1 | 0 | 0 |
| BROAD5120 | M3/E0 | 11 | 1 | 1, 0 | 0 | 0 |
| BROAD5120 | M3/E1 | 5 | 3 | 0, 3 | 0 | 0 |
| EQUIL5120 | M1/E0 | 16 | 16 | 8, 8 | 0 | 0 |
| EQUIL5120 | M1/E1 | 16 | 16 | 8, 8 | 0 | 0 |
| EQUIL5120 | M3/E0 | 6 | 6 | 1, 5 | 0 | 0 |
| EQUIL5120 | M3/E1 | 6 | 6 | 1, 5 | 0 | 0 |

## Fixed local field: per draw, world-left, reference only

| Point | Task | Center XYZ median (mm) | Center SO3 median (rad) | Toward-center angular increments /576 | rho>1 matrices /96 | largest singular value>1 /96 |
|---|---|---:|---:|---:|---:|---:|
| BROAD5120 | M1 | 1.901750963551398 | 0.004245706427897651 | 466 | 30 | 94 |
| BROAD5120 | M3 | 2.3519296202254423 | 0.002365373328522363 | 538 | 77 | 96 |
| EQUIL5120 | M1 | 1.2147262303756479 | 0.005069596482160876 | 576 | 2 | 34 |
| EQUIL5120 | M3 | 0.7910880862248234 | 0.004172362035296237 | 576 | 63 | 94 |

These are finite-scale response estimates from two fixed noise draws and 24 TRAIN families. They are not analytic Jacobians or stochastic/WBC stability certificates. Matrix denominators remain 96 per task; incomplete/rejected neighborhoods do not acquire eigenvalues.
Finite REJECT references remain in physical-error tables but were never executed. Zero-denominator center ratios are undefined. Translation-only field poses were excluded from supplemental training.
The same-input teacher map and its actual projection metadata are retained for every cell. exp(-0.8) applies only when the actual teacher projection is terminal.

## Matched-case gains, losses and retention

| Comparator | Metric | Task/backend | Retained | Lost | Gained | Both fail |
|---|---|---|---:|---:|---:|---:|
| AGG4096 | success | M1/E0 | 16 | 0 | 0 | 0 |
| AGG4096 | success | M1/E1 | 16 | 0 | 0 | 0 |
| AGG4096 | success | M3/E0 | 1 | 0 | 5 | 10 |
| AGG4096 | success | M3/E1 | 1 | 0 | 5 | 10 |
| AGG4096 | reliable | M1/E0 | 1 | 0 | 15 | 0 |
| AGG4096 | reliable | M1/E1 | 3 | 0 | 13 | 0 |
| AGG4096 | reliable | M3/E0 | 1 | 0 | 5 | 10 |
| AGG4096 | reliable | M3/E1 | 1 | 0 | 5 | 10 |
| BROAD5120 | success | M1/E0 | 16 | 0 | 0 | 0 |
| BROAD5120 | success | M1/E1 | 16 | 0 | 0 | 0 |
| BROAD5120 | success | M3/E0 | 6 | 5 | 0 | 5 |
| BROAD5120 | success | M3/E1 | 5 | 0 | 1 | 10 |
| BROAD5120 | reliable | M1/E0 | 2 | 0 | 14 | 0 |
| BROAD5120 | reliable | M1/E1 | 3 | 0 | 13 | 0 |
| BROAD5120 | reliable | M3/E0 | 0 | 1 | 6 | 9 |
| BROAD5120 | reliable | M3/E1 | 3 | 0 | 3 | 10 |

## Qualification and interpretation

- BROAD5120: original=False; reliable=False; safe=True; original ROT LEGACY retained=True.
- EQUIL5120: original=False; reliable=False; safe=True; original ROT LEGACY retained=True.

EQUIL minus BROAD original/reliable/collision counts: {'M1': {'original': 0, 'reliable': 27, 'collisions': 0}, 'M3': {'original': -4, 'reliable': 8, 'collisions': 0}}.
Local field improvement alone does not establish repair: it must be read with original-start M1 post-arrival SO3 departures, complete 35–40 s holding, M3 entry/order, safety and legacy.
If M1 improves while M3 remains unqualified, the result is a targeted M1 advance. If local maps improve without reliable M1 holding, goal-local coverage is insufficient to solve the complete failure.
Any qualified candidate is virtual research only. No automatic structural feedback intervention or physical transfer is authorized.

## Provenance and retained failures

A0 preserves all 128 BROAD/TARGET trajectories and onset/visibility evidence; new EQUIL onset covers all 64 original-start episodes. Same-tick reference, installed reference and achieved pose remain separate.
All fixed panel rows, REJECT/nonfinite/collision failures and component exclusions are preserved. No quality retries, new seeds or outcome-based selection are permitted.
CPU replay checks saved four-step BF16 Euler arithmetic, actual noise, FP32 decoder boundary and metrics. Gate, scorer, renderer and executor sources are shared frozen implementations; this is not an independent physics backend.
The first frozen EVAL family ALL8 video uses saved actual RGB/timestamps and includes all failures. No rerendering or score-based clip selection.
Detailed tables: OFFLINE2904, NEW_FIT576, LATE_CONTEXT2368, VIRTUAL192, COMPONENT96, PAIRED_FIELD2496 and MATCHED_CASE_GAIN_LOSS256. S4FIT local rows are historical context, not a new candidate.
Final artifacts, literal cumulative budgets, two complete Trainer readbacks and dataset/output/summary revisions are closed by the final audit and terminal canonical.
