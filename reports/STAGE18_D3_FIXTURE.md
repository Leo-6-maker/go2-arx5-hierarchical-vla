# Stage18 D3 Simulated Fixture

## Result

The non-destructive fixture builder ran on `corelab-5080-2` using the existing
Stage10 Python environment with `CUDA_VISIBLE_DEVICES=""`.

- selected six deterministic `*_V0` episodes;
- 6/6 converted to `RawEpisodeV1`, synchronized, and derived to
  `GR00TTrainingExampleV1` with `H=40`;
- 6/6 quality decisions were `WARN`/accepted, not rejected;
- state statistics shape: `[6,10]`;
- future action statistics shape: `[240,10]`;
- each example retains 40 explicit native source timestamps;
- target labels came from `action_target`; achieved EEF stayed in raw audit
  data and was not used as a training label;
- body and wrist PNG paths were read for the processor fixture, not copied.

Selected episodes:

```text
d3_A_d2_ep_00_V0
d3_A_d2_ep_01_V0
d3_A_d2_ep_02_V0
d3_A_d2_ep_03_V0
d3_B_d2_ep_07_V0
d3_B_d2_ep_08_V0
```

## Known D3 issue

Raw state timestamps contain 121 samples from 0.0 to 2.4 s at 0.02 s
spacing. Each camera contains 61 samples from 0.0 to 2.4 s at 0.04 s
spacing. Thus the effective camera rate is 25 Hz, while the camera metadata
declares 20 Hz. The builder preserves 0.04 s timestamps and emits
`D3_CAMERA_FPS_METADATA_MISMATCH`; it does not rewrite the source.

## Artifact boundary

Node-local output:

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage18/d3_fixture/
```

It contains JSON manifests/records/statistics and references to external image
paths only. The D3 source tree was not modified, and no D3 data/image was
staged in Git.
