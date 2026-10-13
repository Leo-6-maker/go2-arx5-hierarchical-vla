# PI0294 Shared Model Case Registration

`scripts.pi0294.relay_transport.model_registry` prepares local-owner case
metadata for the existing CoreLab endpoint and Hopper resident. It does not
authorize, load, select, train, infer, launch physics, or submit PBS.

The planner requires an actual complete main checkpoint inventory and its
verified private HF READY-last delivery. Engineering M4 and incomplete or
unverified artifacts are rejected. The relay model revision is explicitly the
private full-state HF artifact revision, not a fabricated model commit. Model
config, every delivered model/resume file, current frozen dataset/statistics,
processor and both runtime source/manifest identities are bound separately.

M512 uses the already frozen four TRAIN development cases; M2048/M4096 and an
actually authorized M6144 use the same frozen eight DEV cases. Original card
hashes, language, split, denominator and initial state must match. Paired tasks
share their original layout ID for inference noise and one checkpoint identity,
one processor/statistics and one Hopper resident directory. Only instruction
and audit task identity differ; no model is chosen by task ID. Confirmation is
not opened by this planner. Selection and deployment remain NONE.

Use the existing admitted same-allocation parent and endpoint components to
execute the resulting sealed requests. The parent must be live in the exact
own native PBS job; a registry alone is not execution authority. Do not create a
resident with placeholder weights while waiting for M512, or restart a stopped
query/physical intent. Runtime CoreLab and Hopper capsules are not hot-updated
when this local-owner planning helper is published.

CPU fixture tests cover registry construction and rejection. They are not
actual main-model learning, resident loading, closed-loop behavior or scientific
acceptance evidence.
