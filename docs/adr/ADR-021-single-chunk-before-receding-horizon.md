# ADR-021: Keep Stage20T Single-Chunk Attribution Separate from Receding Horizon

- Status: Accepted for Stage20T and Stage21 planning
- Date: 2026-09-01

## Decision

Stage20T is a single-chunk experiment:

```text
one observation -> one 40-point chunk -> native WBC execution
```

It may replay the already sealed actual GR00T output, but it must not claim a
closed-loop VLA system because no new image/state observation is fed back into
GR00T during the chunk.

Stage21 is the separate same-scene receding-horizon gate:

```text
native scene observation
 -> GR00T chunk
 -> execute prefix through WBC
 -> new native observation
 -> GR00T chunk
```

## Rationale

Combining chunk timing attribution with fresh-observation scheduling would
make a failure ambiguous. Stage20T first isolates reference profile, timing,
and reset effects. Stage21 can then test chunk age, freshness, stale and
out-of-order handling, native-scene vision, and multi-chunk task behavior.

Neither stage authorizes hardware, network transport, or real-robot commands.
