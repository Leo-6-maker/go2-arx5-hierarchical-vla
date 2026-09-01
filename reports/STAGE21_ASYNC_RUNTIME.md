# Stage21 Asynchronous Runtime Status

```text
ASYNC_RUNTIME_NOT_QUALIFIED
```

Stage21 intentionally used the permitted
`SYNCHRONOUS_PAUSED_SIM_RECEDING_HORIZON` mode. Physics was paused while the
actual model worker produced a fresh chunk, then resumed for the next prefix.
This proves feedback/re-observation/replanning orchestration without
pretending that the current transport is deployable.

No asynchronous worker, ZMQ, ROS2, shared-memory production transport, or
PolicyServer path was introduced. The shared NFS mailbox was an offline test
harness. Its approximately 29.7 s response visibility delay is incompatible
with a real-time claim, while the measured model forward path remains a
separate approximately 0.063 s median / 0.580 s maximum diagnostic.

The lifecycle contract records observation, receive, and execute timestamps,
chunk age, and `FRESH`/`AGING`/`STALE`/`OUT_OF_ORDER` semantics. The actual
synchronous run had six zero-age executions and five accepted `FRESH`
updates; this does not qualify asynchronous freshness under a real
transport.
