# Stage45F future PI gate templates

Status: **PREPARATION ONLY — none of these templates authorizes execution.**

These templates exist so a future result does not cause a rushed, post-hoc experiment design. Every template must be instantiated with exact identities/hashes and posted as a new PI gate before execution. Validation-E/Holdout-F remain locked unless the specific future gate explicitly changes that state.

## Template A — canonical seed replication

Use only after a candidate recipe is frozen and the scientific question is optimizer/initialization stability.

Required bindings:
- candidate recipe identity;
- exact BASE tree;
- exact dataset tree;
- normalization/loss identities;
- canonical seed family and already-completed valid seeds;
- optimizer/scheduler/precision/effective batch/trainable modules;
- max steps/checkpoint steps;
- no metric-based seed replacement;
- retain every scientifically valid run;
- TRAIN-only evaluation protocol and call budget;
- Validation-E/Holdout-F locked.

Forbidden automatic changes:
- LR/scheduler/batch/precision change between seeds;
- extra seed chosen after seeing results;
- deleting a worse valid replicate;
- extending steps on only one seed without a new gate.

## Template B — longer-training / exposure study

Use only after `stage45f_scaling_admission.py` has classified the proposal and PI has resolved scheduler-horizon and fresh-vs-resume semantics.

Required bindings:
- frozen recipe/dataset/stats/loss/model/trainable modules;
- baseline and proposed optimizer-step budgets;
- scheduler horizon binding and exact LR trajectory policy;
- fresh-BASE or resume choice;
- if resume: checkpoint + optimizer + scheduler + RNG state hashes;
- exact optimizer-step exposure instrumentation plan;
- predeclared checkpoints;
- predeclared stop/extension rule;
- no adaptive extension without a new PI gate.

Claim boundary:
A `max_steps` increase is not automatically a pure exposure intervention, and steps are not epochs.

## Template C — static training-data expansion

Use only after dataset-union, support, sampling-mix and data-compute audits are complete.

Required bindings:
- accepted FULL24 tree as base lineage;
- exact 24 base TRAIN windows preserved;
- exact added-window manifest;
- independent expert/oracle label provenance;
- exclusion of Validation-E/Holdout-F identities;
- current frozen-stats support audit across XYZ and all active T16 channels;
- explicit stats route: retain current stats or separately classify as `DATA_PLUS_NORMALIZATION`;
- sampling policy and nominal group weights;
- exact realized exposure trace plan;
- optimizer-step budget decision separated from data-cardinality decision.

Forbidden automatic changes:
- recomputing q01/q99 while calling the arm data-only;
- duplicating old windows and calling it diversity;
- changing sampling weights without naming that intervention;
- using locked-split states or labels.

## Template D — rollout-state / DAgger-style augmentation

Use only when evidence supports state-distribution shift (e.g. teacher-forced field acceptable while free-running degrades).

Required bindings:
- frozen state-generating policy checkpoint/recipe/sampler;
- WBC identity if WBC participates in state visitation;
- TRAIN-only collection scope;
- visited-state artifact identities;
- independent simulator expert/oracle implementation, environment and label-policy identities;
- oracle-label artifact identities;
- fixed state-collection and relabel budgets;
- no GR00T self-labeling;
- explicit mixture/sampling policy with static expert data.

Forbidden:
- copying GR00T predictions back as labels;
- collecting from Validation-E/Holdout-F;
- changing policy, WBC, expert and normalization simultaneously without a multifactor design.

## Template E — N2 × L1 factorial interaction

Use only if single-factor evidence justifies investigating interaction. `N2+L1` remains a new scientific arm.

Conceptual cells:

```text
A: FULL24 normalization + legacy loss
B: N2 normalization     + legacy loss
C: FULL24 normalization + L1 loss
D: N2 normalization     + L1 loss
```

Required bindings:
- matched dataset/base/model/optimizer/scheduler/precision/effective batch;
- matched canonical seeds;
- exact N2 contract;
- exact L1 contract;
- fresh BASE for D unless a future gate explicitly defines another design;
- matched TRAIN evaluation protocol;
- descriptive interaction contrast only after D exists.

Do not infer D from A/B/C or assume additivity.

## Template F — first Validation-E access

Use only after one PI-selected candidate is frozen from TRAIN evidence.

Required bindings:
- recipe freeze ledger;
- exact selected checkpoint tree;
- TRAIN decision-packet SHA;
- dataset/stats/loss/sampler/decoder/metric identities;
- Validation-E access generation = 1;
- immutable fresh evidence root;
- no recipe change after the read;
- Holdout-F remains locked.

The first Validation-E read must be preserved even if weak. Any later candidate change creates a new selection generation and cannot be described as the same unseen validation.

## Template G — large-scale fine-tuning campaign

Use only after `stage45f_large_scale_campaign_admission.py` passes structurally and a new PI gate explicitly authorizes execution.

Required bindings:
- PI-selected frozen candidate;
- predeclared seed family;
- predeclared optimizer-step tiers;
- predeclared checkpoint retention plan;
- scheduler-horizon policy;
- resource-port audit and fixed effective batch/precision semantics;
- compute/storage telemetry;
- exact optimizer-step exposure instrumentation;
- all valid runs retained;
- no metric-based retry selection;
- no adaptive tier extension without a new gate.

## Minimal future completion evidence for any training arm

Every future training arm should seal, at minimum:
- exact Git HEAD/branch;
- exact BASE/dataset/stats/loss identities;
- training recipe and seed;
- runtime device/precision/effective batch;
- optimizer/scheduler policy;
- full-state checkpoint tree hashes;
- exact optimizer-step count;
- optimizer-step-bound sample exposure trace (when instrumentation is authorized);
- peak resource telemetry;
- event/counter closure;
- locked-split/downstream zero counters;
- immutable evidence inventory/seal;
- no automatic winner claim.
