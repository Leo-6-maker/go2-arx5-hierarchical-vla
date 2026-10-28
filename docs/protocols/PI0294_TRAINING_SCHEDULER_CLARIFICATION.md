# PI0294 Fixed Training Scheduler Clarification

This addendum specifies the implementation before any formal main training.
It does not rewrite the original protocol or authorize a new allocation.

The native main Trainer uses one cosine horizon of 6144 optimizer updates and
308 warmup updates (ceil(5% of 6144)). The normal stopping point is M4096.
M512 and M2048 are full-state intermediate checkpoints; M4096 is also a
full-state checkpoint. A separately admitted M6144 extension is permitted
only after the existing fixed actual DEV improvement and budget gates pass.

The recipe, sampler stop, scheduler horizon and warmup are identical before
and after M4096. Resume loads the original full optimizer, scheduler, RNG,
sampler cursor, processor and frozen statistics. There is no second warmup,
post-DEV scheduler retargeting, source hot replacement or pristine restart.
The executable update cap is normally 4096 and becomes 6144 only with the
preregistered extension admission; the scheduler horizon is not that cap.

Consequently M4096 has not reached the cosine floor. Its learning rate is
approximately 0.2743 of the initial rate under this schedule. It is not a
4096-horizon cosine experiment and must not be reported as directly schedule-
equivalent to one. All runs and checkpoints disclose the fixed 6144 horizon.

The sealed microbatch benchmark is exactly three two-update candidates in
the fixed order 2x16, 4x8, 8x4. These six real optimizer updates are charged
against the smoke limit: eight comparison updates plus six benchmark updates
equals fourteen. Benchmark weights are never model candidates. CPU loader
qualification actually exercises ordered consumption and resume with 0, 2
and 4 workers; the highest measured initial consumption rate selects workers,
with smaller worker count breaking ties. Among successful GPU candidates,
highest real synchronized two-update rate selects microbatch, with smaller
microbatch breaking ties. Rates are short measured benchmarks, not steady-
state performance claims. Any candidate failure stops without automatic retry.

The benchmark driver exits after recording its actual profile. It does not
retain the H200 while waiting for new data. Resource release and subsequent
allocation admission remain with the sole local-PC owner.
