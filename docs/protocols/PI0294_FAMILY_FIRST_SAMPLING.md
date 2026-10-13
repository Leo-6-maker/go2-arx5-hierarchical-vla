# PI0294 Family-First Formal Sampling

The registered recipe balances each task's scene families before sampling an
episode and offline phase. The earlier implementation chose T1 old/new origin
first. Because the preserved old bank contains F0 only, that would allocate one
third of all T1 positions to F0 before the new-bank family draw and distort family
balance. No formal optimizer update has executed under that implementation.

Formal multitask sampling now draws the family first, then allows an old T1
episode only at the existing one-in-three eligible T1 positions and only when
that family has old data. Actual old-bank exposure can be below one third; the
protocol says at most one third, not a minimum or exact quota. Episode sampling
and offline phase weights remain unchanged, and T1/T2 still alternate 50/50.
No family, origin, task ID, phase, object truth, or score enters the model input.

The position-addressable state records algorithm
`PI0294_POSITION_FAMILY_FIRST_OLD_EXPOSURE_CAP_V2`. Historical single-task legacy
engineering smoke preserves both the exact old index sequence and its original
V1 state identity. Failed checkpoint provenance is not rewritten. New multitask
Hopper 0/2/4-worker CPU qualification must run the new exact source before final
model admission; the older CoreLab bank's official codec/processor qualification
is data-format evidence, not a claim that this new sampler already ran there.

Statistics are not refitted, dataset/splits are not changed, and no failed episode
is replaced. Main training remains zero at publication. No extra engineering
updates, PBS calls, GPU hours, deadline extension, or science-threshold change is
granted by this correction. Selected/deployment remain NONE.
