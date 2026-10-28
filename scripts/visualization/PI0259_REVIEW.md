# PI0259 Agent V: finite, read-only evidence review

Authority: `240dedfd3954dd4cbbb48645a2aad5b19738e2c4`, canonical
`5892082658`. PI0257-v2 code is imported unchanged from
`df5517d89561ffc32e41ae2fe6b315460853c8a5`. Its existing entry remains
`GUARD_STUB_ONLY` for physical execution. Use the new entry for Agent V work:

```powershell
py -3.12 scripts/visualization/pi0259_review.py
py -3.12 scripts/visualization/test_pi0259_review.py
py -3.12 scripts/visualization/pi0259_review.py H0 --manifest C:/sealed/H0/manifest.json --sha256 <producer-delivered-SHA256> --out C:/reviews/H0-v1
```

Linux uses `python` instead of `py -3.12`. Existing NumPy/OpenCV are enough;
pure FK imports repository NumPy helpers. No model/runtime installation.
`H1` and `H2` use the same arguments. No packet means the default `plan`
returns `WAITING_FOR_SEALED_HANDOFF`, with no search, polling or mutation.
The output must be a new directory outside the entire sealed packet tree.
No pickle, producer Python import, model, scene constructor or physics call.
The reused import guard rejects Torch/Transformers/Isaac imports.

The actual received `PI0259_H0_V1` layout is supported directly, without
rewriting its manifest or arrays:

```powershell
py -3.12 scripts/visualization/pi0259_review.py H0 --manifest ../H0_v1/manifest.json --sha256 94954516f9b1e9159e7fe468c04d7155e6cc20eb4d21c601b8389e8ab2dcac62 --urdf ../private_binding/go2_x5.urdf --out ../new-H0-review
```

`pi0259_h0.py` verifies its `entries`, independently parses the hash-bound
URDF, reconstructs FK/current-state/full camera rotations, mask counts and
RGB axis colors, and backprojects actual depth onto cue sphere geometry.
It separately counts missing registered poses and failed visible samples.
The private URDF is never copied to the output. All 144 received raster
records are exposed; the producer's PASS booleans are not the decision rule.

## Input binding, not a new data production format

The protocol defines packet contents but not concrete array keys. The adapter
uses a small JSON view of those fields. M may retain its existing field names:
each array binding is `{ "path": "shard.npz", "key": "actual_saved_key" }`.
Binding files must themselves be sealed. Do not fabricate missing arrays,
upsample camera observations to pretend they were captured at other times,
or rewrite original producer outputs. A differing actual packet layout needs
only a read-only adapter once that immutable layout exists.

Manifest fields consumed by this entry: `sealed: true`,
`producer_role: "Agent M"`, `protocol_commit`, `code_commit`, `stage`,
`files: [{path, bytes, sha256}]`, `review_records` (sealed JSON list filename),
and `source_ids` (complete outcome index). All paths are relative to the
manifest directory; escaping paths and unlisted files are rejected.
The complete protocol §8 provenance (schema/version, all data/processor/
camera/reference/IK identities, counters, missing fields, cohort/split and
evidence class) remains required for final human-reviewed stage closure.
Missing schema bindings cannot become a PASS.

H0/H1 view rows bind:

- Source/task/command/instruction/colors, original outcome, target ID and
  command-target map; M1 `target0/target1`, M2 `goal`, M3 destination
  `white_B/black_A`. Language is parsed independently of the command number.
- `frame: world`, `state_frame: native`, units `{position:m, rotation:rad,
  time:s}`, `state_origin: KINEMATIC_FK_NOT_DYNAMIC_MEASUREMENT`,
  `time_origin: synthetic_reference_time`, declared RGB/BGR order.
- NPZ `arrays`: `time` (N), `fk_world`, `reference_world`, `render_tcp_world`,
  `root_world` (N,4,4), `T_world_native` (4,4), `StateV1` (N,10), `joints`
  (N,J), `active` (N). Every body/wrist stream binds `*_rgb` (N,96,96,3),
  `*_segmentation` and `*_depth` (N,96,96), `*_time`, `*_world`, `*_view`,
  `*_projection`. This is one aligned observation view; retain full-rate
  solver/geometry arrays separately for the complete H1 gate replay.
- Sealed `urdf` filename, `joint_names` containing every movable joint,
  `tcp_link`, `tcp_offset_m`, `camera_links`, complete homogeneous
  `camera_attachments`, `camera_convention` (`row_vector_opengl` or
  `column_vector_opengl`) and depth convention (`negative_camera_z_m` or
  `positive_camera_z_m`). Unknown conventions stop this check; never guess.
- `cues`: `{id,name,role,world_m}`; M3 contour sectors additionally carry
  `sector: 0..11`. `required_cues` names and `complete_sequence` are explicit.
  Counts come from raw masks; projected centers are separate diagnostics.
  RGB cue means and projection-to-raster centroid differences are retained.
- Saved solver residuals, velocity/collision flags and raw/codec gate arrays
  are shown as diagnostics, explicitly `MISSING` when absent. Flags alone
  are not an independently recomputed collision/gate verdict.

H1 adds complete geometry tuple, fixed root/DOF, camera and asset hashes,
split, actual `collator_fields`, `collator_arrays` and `collator_language`,
plus `paired_prefix_indices`. Window diagnostics bind `window_start_index`,
`window_geometry_valid`, `raw_gate_reject`, `codec_gate_reject`.
Grouping excludes colors, task and family names; all aliases stay together.
Pair RGB/state/time are compared as actual arrays. The fixed protocol
admission thresholds and observed counts are emitted. `array_qualified`
counts are provisional, conditioned on saved window flags; they do **not**
grant admission. Original full-rate geometry/collision/gate replay, the
registered split algorithm/denominators, TRAIN-only statistics and actual
collator source must be reviewed before `DATA_QA_PASS`.

H2 adds `evidence_class: OFFLINE_MODEL_PREDICTION`, `model` (MT-Base/MT-G),
`step` (0/6000 only), task/instruction, complete checkpoint/processor hashes,
initializer revision, raw-noise hash (SHA256 of contiguous array bytes),
sealed `cpu_receipt`, `first_valid_or_reuse` (FIRST_VALID/EXACT_REUSE),
`inference_latency_s`, `frame: native`. Arrays are `prediction` and
`reference` (40,4,4), `time` (0:.05:1.95), and original `noise`.
The entry recomputes XYZ/SO3, emits source-labelled cards and per-task/
worst-task summaries. It does not load the checkpoint, decode a raw model
head, validate the producer scorer by its PASS string, or choose deployment.

## Receipts and honest scope

Every accepted package is size/SHA-bound; output receipts bind the manifest,
reviewer code, all originals, results, missing checks and chosen examples.
H0/H1 representative cards use source-ID hash order, cover every available
task/command/quarter/outcome class, and keep diagnostic masks/depth separate
from native RGB. Source pixels are displayed at labelled nearest-neighbor
scale. H2 displays all supplied rows without score selection.

The automatic entry emits `CORRECTNESS_BLOCKED` for concrete inconsistencies
or `H0/H1/H2_NUMERIC_REVIEW_PARTIAL`. This deliberate boundary prevents an
untested schema or a producer flag from awarding a scientific PASS. A final
stage receipt requires the missing source/semantic checks and actual natural
RGB inspection on that exact packet. H0 camera discernibility, H1 admission,
and H2 decoder/semantics are different checks. Video styling is never one.

At least one complete correct/opposite-command sequence per admitted task
must accompany eventual review. If a packet has only calibration snapshots,
the tool reports that scope; it does not invent intermediate frames/video.
CPU test cards are fixtures, never actual H0 evidence. H0_v1 arrived during
tool preparation and was reviewed with the native-layout adapter. H1/H2
schemas will bind the actual sealed fields when those stages exist; the
generic view currently provides CPU checks and named partial receipts only.
