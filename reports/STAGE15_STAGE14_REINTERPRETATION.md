# Stage15 reinterpretation of Stage14

Stage14's external injection was mechanically executable but semantically
unaligned. Its seven references carried `SIM_REFERENCE_ONLY` placement/TCP
semantics, while the native WBC added a reset-dependent shoulder/reach anchor.
Therefore its tracking numbers are retained as:

```text
UNALIGNED_REFERENCE_INJECTION_SMOKE
```

They are not native General-WBC tracking evidence.

Stage15 closes the specific frame/anchor/TCP ambiguity with
`TrajectoryShapeV1`, `PlacementProfileV1`, and `EpisodeReferenceBindingV1`.
The independent implementation reproduces the native preview over 12 resets.
This permits aligned native diagnostics, but it does not retroactively promote
Stage14 numbers.

The Stage13C P1M values remain a separate transfer result:

- `93/100` arm targets clipped;
- position mean/p95 `0.527/0.864 m`;
- rotation mean/p95 `1.761/3.018 rad`;
- label: `UNQUALIFIED_BRIDGE_TRACKING`.

Those values diagnose the P1M embodiment bridge and must not be compared as
native WBC performance.
