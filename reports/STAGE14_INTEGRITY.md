# Stage14 integrity and stop-gate record

## Immutable inputs

| input | expected SHA256 | status |
|---|---|---|
| delivered WBC archive | `b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167` | verified unchanged |
| P1M canonical USD | `92d64f697d3f5ea6f17bbeaab72142607019f0309b5068c47f37a623499b9a53` | verified unchanged |

The shared Isaac-GR00T repository remains at commit
`51d4c89f72fda44cbf77285c6a8114b52676b8a1` with its pre-existing user
modifications preserved. The SONIC reference repository was not modified.

## Processes and resources

Stage14 used the user-owned native runtime on `corelab-4080s`. The external
workload policy was respected: no other user's process was killed, signaled,
reniced, attached to, or modified. At the stop gate no Stage14 Isaac process,
ROS process, ZMQ process, PolicyServer, or project-owned listener on ports
5550, 5556, or 5557 remained.

No physical robot, camera, writer, GR00T model, P1M simulation, or hardware ROS
path was started in the Stage14 native baseline.

## Repository hygiene

Private/large inputs are outside Git and referenced only by manifests. The
working tree contains the canonical source and documentation changes only;
generated Python caches and pytest cache are ignored. No secrets, tokens, SSH
credentials, checkpoints, lab source, demos, datasets, USD, or run JSONL were
staged.

## Tests

- pytest: `17 passed`;
- CPU contract smoke: `CPU_CONTRACT_SMOKE_PASS`;
- artifact manifests: `validated 5 artifact manifests`.
