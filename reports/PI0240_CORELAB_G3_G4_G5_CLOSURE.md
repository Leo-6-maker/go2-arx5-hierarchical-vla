# PI0240 Corelab G3-G5 closure

## Decision

`SCIENCE_FAIL_TRAIN_FIT — STOP_FOR_PI_RECIPE_DECISION`

The fixed PI0240 G4 diagnostic completed all 256 unique scientific cells on
Corelab.  The fresh-10000 model improved substantially over baseline-1000 in
expert-relative XYZ and SO(3) error, but every TRAIN and CALIB cell still
triggered the unchanged SafetyGate linear-speed rejection.  Because the
failure is present on TRAIN and CALIB at nearly the same level, the supported
attribution is a training-fit deficit, not a TRAIN-to-CALIB generalization
gap.  `selected_checkpoint=NONE`.

## Frozen execution identity

- Protocol: `ae57d2824d2e97770382393d265dce57c472a334`
- Execution branch: `pi0240/corelab-first-execution-goals`
- G4/G5 code head: `bebbe90eb59def4ffbc1ec536b3993fc1af4d0a6`
- GR00T source: `51d4c89f72fda44cbf77285c6a8114b52676b8a1`
- Device: Corelab GPU0, RTX 4080 SUPER,
  `GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad`
- G4 admission SHA256:
  `6945c89018b88e380318f764d16072b811706c5648c903e28fd606de98dbdef4`
- Dataset canonical manifest SHA256:
  `267c6d9fccb524ff109f48fb4b0f90c9c483fb8d2089db9b058fbe97a0e6dce1`
- PI0237 evidence canonical manifest SHA256:
  `5641a7ae338d4582c4b3d44f90bf1cca0ed06632d911d9b3c705d9c1e7a44926`

## G3 contract closure

- Status: `PASS_READ_ONLY`
- Rows: 142 (fixed 128 cohort plus 14 preregistered extras)
- Real path: dataset/collator `[142,40,132]` -> slots/mask/state anchor ->
  independent evaluator decode -> unchanged SafetyGate -> artifact write/read
- Active action slots: 0-9
- Physical parity: XYZ max `1.5305370383095253e-07 m`; SO(3) max
  `1.425034124872992e-05 rad`; gripper max `0`
- Expert gate: ACCEPT 9 / WARN 121 / REJECT 12, retained without alteration
- Model loads, model forwards, and `get_action` calls: 0
- Actual-batch trace SHA256:
  `cc585ddd2fb708a268394d8e8bc733420c839a80f4d38b5447ebdbcc3a72580a`
- Artifact round-trip SHA256:
  `016ef530e18597d2accff1b64301a0f6e1e7e4de0821c398527c3d87b1b89891`
- stdout SHA256:
  `932297f6baab4345364af781d162d326eb34e5053cd23e12e5f73b6e3fc7fd75`

## G4 fixed-cohort results

Each model completed the same frozen 64 TRAIN and 64 CALIB rows.  There were
256 actual `get_action` calls and 256 first-valid results.  A preserved
pre-query engineering ledger entry from attempt 2 did not call the model and
is not a scientific cell.

| Model / split | Gate | mean XYZ error | p95 XYZ error | mean SO(3) error | mean max linear speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline-1000 TRAIN | 64 REJECT | 0.496990 m | 0.657035 m | 0.510579 rad | 4.4814028 m/s |
| baseline-1000 CALIB | 64 REJECT | 0.516374 m | 0.746676 m | 0.489551 rad | 4.4951143 m/s |
| fresh-10000 TRAIN | 64 REJECT | 0.151497 m | 0.267144 m | 0.075521 rad | 3.0721757 m/s |
| fresh-10000 CALIB | 64 REJECT | 0.150062 m | 0.317672 m | 0.077615 rad | 2.9845185 m/s |

Paired fresh-minus-baseline mean changes over 128 rows were `-0.355902 m`
for mean XYZ error, `-0.423497 rad` for mean SO(3) error, and
`-1.459911 m/s` for maximum linear speed.  Nevertheless, fresh-10000 retained
`LINEAR_SPEED_REJECT` on 64/64 TRAIN and 64/64 CALIB rows.  The model-reference
minimum target distance also worsened on 126/128 rows; this diagnostic does
not support proceeding to WBC.

Artifact SHA256 values:

- baseline summary:
  `b5a436ea70db5288444e8af0a8ea4c84782a3c4facbf227747025081c994d150`
- baseline records:
  `ee1e72e044f65287ef19bb798f7fb814fc2e308c2ebcf192339b261c0e7b43c3`
- fresh summary:
  `399066d3fb15c301b5a4cc4ad83504261b9f308badff1341258d35a55e00a5c2`
- fresh records:
  `118218c965a31c015879ce6e857c3acf23333ff1ba2fc37f0a46eb842d61672f`
- G5 attribution:
  `c2ee264aaec526c2b2d338d7c5bdb20d3daf82f561e961f15f0798c321557cd2`
- G5 paired CSV:
  `f333aa6d35787960926c7dde7d0bbeb90791b7c8bc7db1dacf1c58cee128d8bb`

Corelab evidence root:
`/var/cache/users/liuyu-nfs/pi0240_corelab_g4_20260920`

## Engineering recovery accounting

Three zero-query engineering failures were preserved before the first valid
G4 result: offline Cosmos cache binding, `BatchFeature` mapping support, and
reuse of the mature Stage20S frame reader.  Each was repaired at the shared
root cause and checked before continuing.  No valid scientific cell was
repeated or selected by metric.

The earlier Hopper job `624616.hopper-m-02` is preserved as an engineering-only
failure: it made zero model calls and was not resubmitted.  G3 and G4 were
completed on Corelab after the cheaper local/Corelab test chain passed.

## Holds and counters

- training/backward/optimizer steps: 0
- WBC/simulator/receding steps: 0
- DEV-E reads: 0
- Final-F reads: 0
- selector freeze: not run
- selected checkpoint: `NONE`
- PR #77 merge: 0

The G5 report contains a fixed-16 memorization feasibility run card for PI
review only.  It is explicitly **not authorized** by PI0240 and was not run.
