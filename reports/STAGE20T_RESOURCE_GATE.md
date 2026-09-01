# Stage20T Resource and Integrity Gate

Date: 2026-09-01 (Asia/Singapore)

## Execution

- Native WBC runs executed on `corelab-4080s` using the previously validated
  Python 3.8.20 / PyTorch 2.3.1 / CUDA 12.1 native IsaacGym runtime.
- The selected visible CUDA device was used through the existing native
  environment. No 5080 native WBC run was attempted because the checked
  5080 nodes did not contain the validated native IsaacGym runtime.
- 33 Stage20T native runs completed. No OOM, process timeout, gate rejection,
  or unexplained launcher failure occurred.
- No other user's process was killed, signalled, reniced, or attached. The
  remote shell emitted a pre-existing missing `sonic-core` profile-file
  warning, but the SSH commands and workloads executed successfully.

## Scope restrictions

Stage20T used a sealed Stage20S model output as a replayed input. It did not
perform new GR00T inference, training, finetuning, P1M execution, hardware
control, camera inference, ROS, ZMQ, PolicyServer, or real-data collection.

The native WBC checkpoint, archive, URDF, and source were external read-only
inputs. The repository contains only the project-owned harness, telemetry,
tests, reports, and ADRs; node-local candidates, traces, NPZ files, logs, and
private runtime inputs remain outside Git.

## Node-local artifact root

```text
/var/cache/users/liuyu-nfs/go2-arx5-vla-stage20t/timing_sensitivity_v1_full/
```

The manifest records the Stage20S model-output SHA256
`0c1ea36801cfc438775550b9f3274edae1ad60b9c6eff180ddfe7c1764034bcc`, the
Stage16 candidate SHA256
`0baaa2ac6f9f0536a98033e26a2f45927b997e6b6503464bff42ed6132106b48`, the
checkpoint paths, runtime metadata, and per-run artifact paths.

## Immutable input hashes

The Stage16 integrity gate remains the authority for the delivered inputs:

| Input | SHA256 |
|---|---|
| WBC archive | `b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167` |
| native Go2+ARX5 URDF | `96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25` |
| native arm checkpoint | `99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a` |
| native dog checkpoint | `51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670` |
| native config | `56132d4ff9b866ce196bbc1b1353900b967c54f0f48d2d4e4b82c3555769a0bb` |
| P1M canonical USD | `92d64f697d3f5ea6f17bbeaab72142607019f0309b5068c47f37a623499b9a53` |

No immutable input was modified by Stage20T.
