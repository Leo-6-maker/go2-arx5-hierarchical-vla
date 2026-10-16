# FlowDAgger Source Adaptation

Upstream: Microsoft FlowDAgger, MIT License, fixed commit
`3426f246510fd67a9e39c3c08bd5b02d586ad688`.
Original copyright and permission notice: [LICENSE](LICENSE).
Exact retrieved Git blob identities: [UPSTREAM_SOURCE_LOCK.json](UPSTREAM_SOURCE_LOCK.json).

Adapted components in `src/go2arx5_vla/flowdagger/`:

- `inversion.py`: forward Euler and per-step fixed-point inverse, with injected
  velocity callback, explicit work budget, finite/device/dtype checks and costs.
- `native.py`: N1.7 velocity computation, with our fixed original source route,
  exact model-space checks and frozen/eval assertions. Feature extraction uses
  the original head's feature encoder, not an incidental full policy draw.
- `steering.py`: MLP outputting the full initial latent; explicit detached
  observation features and inversion targets. No optimizer/online run entry.

We did not import LIBERO's 7D action encoder, expert, environment, gripper sign
convention, setup commands, controller patches or full training script. Our
EEF10 codec delegates to the existing frozen processor. No submodule, external
runtime dependency installation, checkpoint download or upstream code execution
was performed for this preparation.

The upstream example pins another GR00T revision. Source inspection and
synthetic CPU tests are not actual M2048 compatibility or task qualification.
