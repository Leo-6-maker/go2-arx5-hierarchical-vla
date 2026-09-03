# Stage24 Final Report — New-Embodiment Adaptation Audit

Date: 2026-09-03 (Asia/Singapore)

## A. Classification

```text
PASS WITH DEVIATION
```

Stage24 A0–A5 static evidence is complete and A6/A7 has one bounded Hopper
candidate prepared. The deviation is intentional: no new model forward,
inference, training, PBS submission, or WBC execution was performed in this
stage. The result is a readiness decision, not a model-qualification result.

```text
STAGE24_STATIC_ADAPTATION_AUDIT: COMPLETE
NEXT_HOPPER_EXPERIMENT: PREPARED_NOT_EXECUTED
```

## B. Git and access

```text
branch: stage24/new-embodiment-adaptation-audit
audit input HEAD: 9567a7e0948eff11b751d9a4dcd782bb6b255a63
origin: git@github.com:Leo-6-maker/go2-arx5-hierarchical-vla.git
delivery commit: recorded by the final Git handoff after this report
```

The exact branch was created from the Stage23 lineage without rewriting
history. Only reusable audit scripts and small JSON/Markdown reports are in
scope; no checkpoint, archive, WBC source, video, dataset, credential, or
private runtime cache is staged.

Current read-only access probes from `corelab-amd-server` succeeded for all
three configured compute aliases. At probe time:

| host | GPU state observed |
|---|---|
| `corelab-4080s` | RTX 4080 SUPER 32 GB: 153 MiB used; RTX 4080 SUPER 16 GB: 38 MiB used |
| `corelab-5080-1` | RTX 5080 16 GB: 9,084 MiB used; no workload was started |
| `corelab-5080-2` | RTX 5080 16 GB: 91 MiB used |

No process was killed or altered. The Stage24 exact-checkpoint and static
statistics work ran read-only on `corelab-5080-2`; no GPU workload was
started. The earlier SSH-auth recovery reports are historical and are
superseded by the successful current probes.
Machine-readable access evidence is in
`reports/STAGE24_CORELAB_ACCESS_PROBE.json`.

## C. A0/A1 exact checkpoint and static coupling

```text
A0_EXACT_CHECKPOINT: PASS
A1_STATIC_10D_122D_AUDIT: PASS_WITH_STATIC_CLAIM_BOUNDARY
ID10_CLASSIFICATION: ID10_UNUSED_LIKE
```

The exact Stage10 base tree matched:

```text
index SHA256: 407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59
tree SHA256:  8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4
shard 1:      8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d
shard 2:      c3f61940deb2007ba1ad7743013b57f0f8462356151db9655175d7aca2d40661
```

ID10 is present but initializer-like across five category-specific blocks.
Its W1 active/inactive RMS is `0.020060/0.020015`; decoder layer2 is
`0.019959/0.019971`. The inactive energy fraction is about `0.924` in both
blocks, consistent with 122 versus 10 columns and not causal proof. The
weight-derived finding does not assert training history and does not justify
an inactive-channel intervention without a separately authorized experiment.

Detailed evidence: `reports/STAGE24_STATIC_CHECKPOINT_AUDIT.md` and `.json`.

## D. A2 temporal-prefix audit

The sealed Phase-A prediction artifacts yielded 126 records: six experiment
conditions, three inference seeds, and seven truncated horizons. No output
was corrected. For O1@200, the median path ratio was already `2.343` at H4
and `2.837` at H8, increasing to `25.566` at H40; H40 had 3/3 test-only
linear-speed SafetyGate rejects. This is evidence that the profile phenotype
is present early in the sampled H40 output, not proof of a causal horizon
effect. A true H16 architecture/loss experiment remains unrun.

Detailed evidence: `reports/STAGE24_PREFIX_AUDIT.md` and `.json`.

## E. A3 normalization and scale

```text
H2_STATISTICS_IDENTITY: PASS
PHYSICAL_TRANSLATION_DEGENERACY: STRUCTURAL_FIRST_POINT_ONLY
RELATIVE_EEF_SCALE: SHIFTED_BUT_REASONABLE_ON_TRANSLATION
ROTATION_SCALE: NOT_COMPARABLE_TASK_CONSTANT_FIXTURE
```

The exact H2 stats hashes are:

```text
stats.json:          fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23
relative_stats.json: 8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1
```

Relative EEF q01/q99 had 243/360 zero-width cells, but the physical XYZ
partition had 117/120 nonzero cells; only the first relative point in each
translation component was structurally zero. All 240 rotation cells and the
single gripper cell were constant by fixture design. The H2 translation width
median was `0.096105`, just below but close to the pinned upstream observed
cell envelope `0.107890..0.689208`; no catastrophic translation scale mismatch
was identified. This remains a descriptive train-fixture result, not a
future-data or real-robot guarantee.

The candidate metadata wrapper with SHA
`28ef7a15fc7c540dd38733eddbf55ebdc3fa2f62c3bafa8b75bfff8bbee54450` was not
mistaken for the frozen Stage22 stats.

Detailed evidence: `reports/STAGE24_NORMALIZATION_AUDIT.md` and `.json`.

## F. A4/A5 recipe and upstream precedent

The sealed O1 recipe is:

```text
NEW_EMBODIMENT; H40; 10D physical / 132D universal
relative EEF XYZ+rot6d; absolute gripper
projector=true; diffusion=true; visual=false; LLM=false; VLLN=false
B=1; gradient accumulation=1; N=200
AdamW; learning rate=1e-4; cosine; warmup_ratio=0.05
bf16; four Euler inference steps; state_dropout effective value 0.0
0.05 s TEST_ONLY_STAGE22_SIMULATION
```

The pinned upstream SO100 custom example is B32/N2000, or 64,000 nominal
presentations versus O1's 200 (320×). These counts are not unique-window
counts. The pinned upstream source also contains the dense H40
`OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` EEF XYZ+rot6d precedent. Neither fact
proves Go2+ARX5 distributional compatibility.

Detailed evidence: `reports/STAGE24_RECIPE_UPSTREAM_AUDIT.md` and `.json`.

## G. A6 decision and A7 boundary

The single prepared candidate is:

```text
H2_LARGER_STOCHASTIC_FLOW_EXPOSURE_SAME_SAMPLE_H40_OVERFIT
```

The proposed future protocol retains the exact TINY1_H2STATS sample, H40,
10D/132D ABI, B1, seed 2222, four-step sampler, and frozen visual/LLM flags,
with a bounded proposed increase to N=2000. Because a longer cosine schedule
also changes schedule time scale, the future result must not be described as a
perfectly isolated exposure intervention. H1 fixed-noise, H3 true H16, and H4
inactive-channel intervention remain unexecuted alternatives.

```text
Hopper/PBS submitted: false
new model forward/inference: false
new training: false
validation e read: false
holdout f read: false
WBC/native rollout: false
hardware/ROS/ZMQ/P1M: false
```

Protocol: `reports/STAGE24_NEXT_HOPPER_PROTOCOL.md`.

## H. Final scientific statement

The exact custom embodiment slot is not supported by the checkpoint audit as a
trained-like slot; the current O1 profile failure is visible from the earliest
audited prefix; normalization has no demonstrated catastrophic translation
defect; and the O1 exposure is far below the pinned upstream custom-example
budget. The next scientifically justified question is therefore whether
greater same-sample H40 flow exposure can make the custom slot learn a healthy
profile. That question is prepared for PI review only. Stage24 does not claim
task performance, generalization, real-robot readiness, or WBC qualification.
