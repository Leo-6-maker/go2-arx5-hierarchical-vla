# WM02 trace evidence repair · actual build-only result

2026-11-04 SGT. This continuation fixes recording defects discovered during A0. It does not establish the cause of T2 leg-limit failure or qualify target initialization. New S transitions, target integrations, Policy queries, WM/Actor updates and complete tasks are all **0**. Earlier charter totals remain6,752 S transitions and400+2 WM updates; no action-selection advantage over the original rule has been established.

## Defects and falsifiable checks

1. The old Python reader discarded oversized INPUT/HISTORY references because their raw `n=0`. It now preserves slot/generation/original-count, streams the f32 snapshots and verifies token/kind/length/hash. Finite first4/recent28 retention losses are reported independently of the event-drop counter.
2. PUBLISH lacked the position/velocity channel. One scalar now records channel0/1, matching CONSUME. Instrumentation removal reproduces the original controller/SDK text exactly; no command, queue operation, WBC, gain, solver, safety or physics arithmetic was changed.
3. The first normal55-test regression failed with a missing APPLY row. A deterministic C++ fixture holds the first disk write until destructor join, forcing events to arrive after the writer's head snapshot. At source `af16c5da667b0f61f4e9d76efe1b9ef61bdac83d`, **2 events enqueued,1 written, dropped0**. The old trace fails the independent event-count audit. This failure and original trace are retained.

The corrected drain compares against final head after finish and records a separate written count. The identical fixture at source **`82439355dd5edabba9dc8999a577f435f4a877a4`** writes **2/2, dropped0**. Harness SHA256 `66a3fdc1a2a9e169722d8305732cf39fe66a73d2b7448b86af71ccfc84b7bd2d`; corrected C++ SHA256 `063d04ec049c94a6ed29f22f7a6758f08ccd7a817a365a203c165f91c44cde5b`. This is a synthetic logger/concurrency test with no target model, controller or physics. Its audit correctly remains INSUFFICIENT because it has no native steps or two completed policies.

Preregistration: `experiments/wm02/init_trace_evidence_v2.json`, committed at `af16c5da667b0f61f4e9d76efe1b9ef61bdac83d`. Byte fixtures cover retained history, publication channels, missing/truncated payloads, mixed versions, post-STOP step tokens, empty traces and nonfinite/truncated inputs. They are not native qualification.

## Actual Core candidate

Fresh immutable run `wm02-init-trace-evidence-build-20261104-01`, source **`82439355dd5edabba9dc8999a577f435f4a877a4`**. New binary SHA256 **`5e2c7d290f6323eb1322757a217fd08755b9c50058a7fd2ee01fcdf53c2c5cf7`**. Old7ae/7828 candidate and its request are preserved separately.

All14 commands exited0: trace/main/SDK compilation, archive replacement, native link+ldd, two outer wrappers+observer compilation, loader doctor compile, both actual dlsym-chain checks and original R1 arithmetic harness compile/run. Selected chain:

| Artifact | SHA256 |
| --- | --- |
| task_actuation_r1.so | `ce28213b5419cb0d34761cdafb8393f062b779b4c2251eead31c5a64c92d669f` |
| observer.so | `55a5ea33811df22cccc5165a093b337943639647070fa5e30431a948f96c18ad` |
| libwm02_init_trace.so | `99874dda096f3c458dea7654797d8d99a619d0e9be1f267c511c5789fab41372` |
| frozen libmujoco.so.3.2.7 | `848a26ad3daccac3a56d7269ac68167ed3497f048223f8720f4e56eb432eb6c6` |

Both outer alternatives resolve to observer, then the frozen MuJoCo library. They were loaded by the loader doctor, **not by a native simulator execution**. Their actual empty trace header SHA256 is `66cb7288ccf1f1d3c1e8c4da86b4eb0fc164ebd79ea16512a1bf744f50394434`; summary SHA256 `6ce0b7829cb99ad28c39289a24897331ab7a83cdbfab275f37c899c7bbd0a85f`. Both audits return **INSUFFICIENT_RECORDED_EVIDENCE**, events/written/dropped0, STOP **NOT_TESTED**, target qualification **NOT_ESTABLISHED**. No trace overhead or native PD scheduling measurement is claimed.

The old target lease remains CLOSED with SHA256 `db2c30f675acb9e8dc280e3fc69d277e1bf91db10cfc2476881eeac2895be407`. The build's independent lease is CLOSED; MainPID0, restarts0, OS exit0. `TARGET_SCOPE_REQUEST_TRACE_V2.json` binds this new candidate for review only, retaining staged2×50→1000→2500 ticks under the original owner/safety mechanisms. It grants no dynamic rights. Cold starts,2s/5s holds, actual Policy/PD inputs, contacts, mechanical limits and STOP remain NOT_RUN/NOT_TESTED.

## Actual commands and exits

These commands describe completed immutable executions. Private variables stand for their saved resolved paths; full14 argv/exit records remain in private `TRACE_V2_COMMANDS.private.json`, whose SHA is in `TRACE_EVIDENCE_V2_ACTUAL.json`. Do not rerun CLOSED capsules.

```bash
# Actual Core build worker entry; source82439355..., exit0.
"$CORE_NATIVE_PYTHON" -B "$TRACE_BUILD_RUN/worker.py"
# Worker calls implemented build_init_candidate.main with these actual arguments:
# --spec "$TRACE_BUILD_RUN/EXECUTION_SPEC.json"
# --output "$TRACE_BUILD_RUN/candidate-01" --run-id candidate-01

# Actual corrected standalone C++ fixture compilation and execution, both exit0.
g++ -std=c++17 -O2 -pthread -rdynamic \
  -I src/go2arx5_vla/target_mujoco "$REPRO/harness.cpp" \
  src/go2arx5_vla/target_mujoco/init_trace.cpp -ldl -o "$REPRO/harness"
WM02_INIT_TRACE="$REPRO/trace" "$REPRO/harness"

# Actual normal/-O world-model regression commands;56 tests each, exit0.
PYTHONPATH=src:. "$PC_MODEL_PYTHON" -m unittest discover -s tests/world_model -v
PYTHONPATH=src:. "$PC_MODEL_PYTHON" -O -m unittest discover -s tests/world_model -v

# Actual PC audit of copied Core loader bytes, exit0; status INSUFFICIENT.
PYTHONPATH=src:. python3 -m scripts.wm02.inspect_init_trace \
  --trace "$LOADER_FILES/task_actuation_r1-doctor.trace" --audit --max-native-ticks 50
```

Targeted8 tests pass; normal56 in4.282s and optimized56 in4.150s. The preserved initial normal55-test failure is explicitly included, rather than erased by the corrected result. All-repository CI was not run by these suites. Inspection CLI exit0 means the inspection completed, not that native qualification passed.

Measured new cost: Core build cgroup **26.963767 CPU seconds**,28.73704 elapsed seconds,peak2,164,297,728 bytes RAM; corrected standalone fixture children .495887 CPU seconds. Combined measured subtotal27.459654 CPU seconds. No new GPU worker. Initial failed fixture, tests and transport/publication overhead are not separately CPU-metered; no whole-turn utilization claim is made. Core capsule35,142,370 apparent bytes; explicit new PC artifact snapshot9,703,072 bytes, excluding later report/Git/transport copies. No currency cost is fabricated.

## Next phase

**A0 recording evidence repaired; A1 OWNER_BLOCKED.** The available source decision exam still provides no reason to train Actor or blindly extend WM. A valid new scope from the original owner is required before any target construction/integration. Under that scope, run one cold start of at most50 native2ms ticks and inspect all original safety and trace evidence before the second start; do not invent a target launch command before the scope/executor is bound.

Next implemented read-only command from this checkout:

```bash
PYTHONPATH=src:. python3 -m scripts.wm02.review_autonomous_round \
  --report-dir reports/WM02/autonomous-recovery
```

Actual run-source commits and API publication commits are distinct; `PUBLICATION_SOURCE_MAP.json` preserves exact original commit objects and verifies identical full trees. No force push, active-source edit, old lease reopening or other user's GPU interruption occurred.
