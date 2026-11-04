# WM02 NIGHT1: real target data and a limited held-out predictive gain

This is an intermediate execution delivery, not completion of N0–N6. The existing AUTO1 owner remains in charge. The original window ends at **2026-11-05 08:56:08.657 Asia/Singapore**, with no new dispatch after 08:36:08.657. The user's later uncapped engineering instruction removes cumulative engineering attempt/update ceilings for new immutable workers; it does not remove ownership, measured capacity, original safety, finite worker endpoints or this deadline.

## Measured increment

At the sealed reconciliation immediately after `night1-mrt3-panel-a-r2-pair4`:

| Quantity | Actual new NIGHT1 result |
|---|---:|
| Native attempts, including failures and qualifications | 115 |
| Native integration ticks, including initialization and failed attempts | 384,162 |
| Eligible production episodes | 24 |
| Eligible production task atoms, 25 actual 2 ms ticks each | 2,160 |
| WM main optimizer updates | 400 |
| Additional exact-restore optimizer probes | 2 |
| Actor updates / qualified full manipulation tasks | 0 / 0 |
| Live NIGHT1 workers at that reconciliation | 0 |
| Sum of recorded NIGHT1 systemd service CPU time | 5,262.879781 s |

The production total is 48 atoms from N1, 960 from five complete C05 paired batches, 576 from three complete RootSafe/controller-9f batches, and 576 from three complete RootSafe/controller-9e8 batches. A physically complete child of a failed pair is excluded with its sibling. Qualification episodes are not production. These are free-space finite collection episodes, not successful full tasks. Historical pre-NIGHT1 data/300 updates are not counted here.

## Held-out predictive result, with limits

Run `night1-wm-rootsafe-qualified-dev-first-01` used six production TRAIN episodes from six parents and two complete **scene-qualification** DEV episodes from two separately assigned parents. There were 768 task atoms, 438 overlapping TRAIN windows and 146 overlapping DEV windows. DEV was neither fitted nor used for normalization. The window count is not an independent sample count.

At a 0.8 s horizon, pooled DEV TCP position coordinate RMSE was **0.338631 mm** for the WM, **0.968800 mm** for persistence, **3.354907 mm** for local numeric velocity and **0.494576 mm** for the TRAIN-only reference-residual ridge baseline. At 0.05 s and 0.4 s the WM TCP RMSE was 0.112155 mm and 0.247809 mm. This is a limited independent-parent hold-trajectory prediction result; the DEV episodes do not constitute a passed DEV production action panel or an action-selection experiment.

Negative outcomes remain: a stationary object's persistence baseline is exact, while the WM drifts by 0.355544 mm at 0.8 s; finger position prediction is also worse than persistence/ridge. Short-horizon orientation is not uniformly improved. No contact success, learned action preference, policy improvement or exact hidden-controller counterfactual restoration is established.

| Identity | Full SHA-256 / source commit |
|---|---|
| Run source | `d40c9df7dc31d09cc4fa11c07ab589400b50e9f9` |
| Replay | `22c1b031d58af446c51b0026a3c22785090f4529e7f8001c78408b31c3bf43f1` |
| Checkpoint, 200 main updates | `31880bd439a7eb4319b76cd644e37cc8bf858c6b93e40e0292c064749bc30ff6` |
| Earlier C05 replay, 1,008 production atoms | `e2df2876d9a4cab030ef03e7f72dd5c2e21762ce6ca878b1d648d6b327232165` |
| Earlier C05 checkpoint, separate fresh 200 updates | `ebd3ffc9b2af9ff512a6722edbd202c17d4fd312c27ce800c87cbc7b0f4d4e9b` |

The held-out run has 74,986 parameters and 106 state coordinates. First reload, endpoint reload, optimizer, RNG, cursor and the extra continuation update all match exactly (reported errors zero). Its 200 main updates took 3.069985 CUDA-event seconds; whole fit wall time was 4.116820 s, CPU 4.060298 s, peak allocated 85,086,720 bytes, peak reserved 96,468,992 bytes. The C05 fit took 3.0493 CUDA-event seconds. These boundaries exclude dataset preparation/import/service startup. Power and monetary cost were not measured. Raw data/checkpoints remain private.

## Repairs, unchanged safety and retained failures

1. The first object-layout implementation changed qpos0 while native reset used the home keyframe. Its apparent static diversity was invalidated and was not counted as independent scene evidence. V2 binds the actual native reset. The original V2 P07 attempt triggered native base/table contact and remains failed.
2. A separately registered V3 initial workspace moves only the initial home root XY by -0.10 m, alongside preregistered object XY variants. Original root height/orientation/joints, robot/table geometry, dynamics, policy, gains, solver, dt and native safety remain unchanged. This is a different physical initial condition, not a retry relabeled as the old condition. All eight RootSafe scene qualifications passed 4.8 s. Scene-parent TRAIN/DEV assignment predates sampling.
3. Constructor readiness is now observed while native time is zero; reference clocks use native nanoseconds; initialization uses qualified 25-tick quota/pause windows. Live JSONL readers wait for newline-complete records and still reject malformed complete records. A signal-traced diagnostic was not treated as production timing qualification.
4. Original controller teardown had a real MRT/shared-Node ownership cycle. Explicit MRT worker joining and release precede ROS context destruction. Source `d52c4e435bff6ab26868eae9bbce2e6133b1126f` produced controller `9e8ce46837d09bc39ae04d4b49f1ce4dd8c8c2d31bea39eac56c5806d108c1f3`. Original control-equation text is preserved; lifecycle methods change, so the controller class is **not** byte-identical. Cold50, single4.8s and dual4.8s all passed, with clean owned exits. Three later production pairs also passed.
5. The fourth controller-9e8 production pair completed both physical trajectories without STOP, but its Python reference receiver exited1 during owned SIGINT (`take_message` conversion error). Controller/solver/simulator all exited0. The whole pair remains excluded. Source `90486476eb726f74a191478fad8e8dc7baa4907d` now explicitly stops the receiver executor before context shutdown and loads that receiver from the current verified source, rather than an older PYTHONPATH capsule. Its subsequent single4.8s and dual4.8s native regressions both passed with every owned process exiting0 and leases CLOSED; the next production panel has started under a new identity. Live callback exceptions remain fatal; there is no catch-and-pass for RuntimeError.

Preserved failures include `night1-scene-p01-q01`, `night1-fallback48-06`, the original gripper-safety duration attempts, the signal-traced failed pair, `night1-parent-p07-q01`, `night1-rootsafe-panel-a-r1-pair4` and `night1-mrt3-panel-a-r2-pair4`. New identities follow evidenced implementation/condition changes. No failed pair is silently converted into training data.

The readonly native contact inspection found a real approximately 0.25 m approach gap and no qualified approach prefix. Supported contact, grasp/release/withdrawal and standing manipulation task scoring remain unqualified. Free-space collection does not waive those requirements. Controller selected-policy identity is derived from real target/policy evidence; complete internal MPC/WBC/receiver hidden state export and exact midpoint fork are not claimed.

## Actual commands and next executable entry

Commands below use the already bound private paths to the immutable run specifications. They are the actual CLI entries called once by the existing Journal worker; no new scheduler/owner is introduced.

```bash
PYTHONPATH=src "$WM02_RUNTIME_PYTHON" -B -m scripts.wm02.parallel_target \
  --spec "$WM02_RUN/EXECUTION_SPEC.json" --output "$WM02_RUN/batch-01" --run-id batch-01
PYTHONPATH=src "$WM02_RUNTIME_PYTHON" -B -m scripts.wm02.build_target_replay \
  --manifest "$WM02_MODEL_RUN/data/MANIFEST.private.json" --output "$WM02_MODEL_RUN/data/replay-01"
PYTHONPATH=src "$WM02_TORCH_PYTHON" -B -m scripts.wm02.fit_target_baselines \
  --replay "$WM02_MODEL_RUN/data/replay-01/REPLAY.npz" --output "$WM02_MODEL_RUN/baselines-01"
PYTHONPATH=src "$WM02_TORCH_PYTHON" -B -m scripts.wm02.train_target_wm \
  --spec "$WM02_MODEL_RUN/EXECUTION_SPEC.json" --output "$WM02_MODEL_RUN/fit-01" --run-id fit-01
```

The two sealed fits and all replay/baseline builds exited0; the failed pairs exited1. Worker receipts retain exact expanded argv, source seal, dependency/binary pins and systemd exits privately. Before new work, resolve the exact current worker to terminal/CLOSED rather than replaying these commands against an existing output.

Next: continue the newly registered unstarted scene/action combinations under new immutable identities, build a fresh same-executor production TRAIN/DEV replay, and complete 200 updates plus a separate new-batch 100-update strict continuation. The implemented late-decision V3 path holds eight atoms before installing the chosen action and prohibits future goal preview; its native qualification and matched action/no-action training remain NOT_RUN at this snapshot. Continue the original goal instead of stopping at this report.

The full component suite passed 121 tests in the PC native/model environment; receiver-specific tests also passed with Python `-O`. Component fixtures are not native qualification. GitHub CI has not been observed for this execution.

## Exact code publication

The API publication retains every source tree byte-for-byte and preserves the concurrent `59aa9e4d12208e3e51de8fc0d84ab7fa93ebe6cb` successor-plan update by a real merge. API commit metadata produces different publication SHAs. [PUBLICATION_SOURCE_MAP.json](PUBLICATION_SOURCE_MAP.json) records original raw commit objects and exact trees for the first 27 new/history commits; verification passed with zero ref/checkout/physics changes. Restore exact run-source objects after fetching the published history:

```bash
python3 -m scripts.wm02.restore_source_commit_objects --repo "$WM02_SOURCE" \
  --manifest "$WM02_SOURCE/reports/WM02/night1-20261105/PUBLICATION_SOURCE_MAP.json" --write
```

No private scene assets, raw datasets, checkpoints, WBC/vendor binaries or host-bound execution specifications are published.
