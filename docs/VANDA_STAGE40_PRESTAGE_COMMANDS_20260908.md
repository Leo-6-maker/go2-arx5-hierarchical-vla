# Vanda A40 Stage40 pre-stage execution plan

Gate authority: PI_GATE_STATE_0060, Issue #11 comment 5573400717.
Branch: stage40-dev/a40-bringup.
Executable code head at planning time: 02700906bcaac62f9a0b9d669bf15a8970a94a86.

This file stages commands and admission predicates only. V3/V4/V5 are not submitted by this document.

## Common PBS/accounting contract

Every A40 runner must contain:

    #PBS -q interactive_gpu
    #PBS -l select=1:ncpus=12:mem=64gb:ngpus=1:gpu_model=A40
    #PBS -l walltime=00:30:00

Every runner must omit both the #PBS -P directive and a qsub -P argument. The system may attach its own accounting project after acceptance; that scheduler field is not a user-side project request.

Common immutable root:

    /scratch/leonliu/leon-storage/stage40_frozen_inputs_20260907_v1/materialized

Common output roots, one per stage:

    V2: /scratch/leonliu/leon-storage/runs/vanda_stage40_v2_$PBS_JOBID
    V3: /scratch/leonliu/leon-storage/runs/vanda_stage40_v3_q0_$PBS_JOBID
    V4: /scratch/leonliu/leon-storage/runs/vanda_stage40_v4_c0_$PBS_JOBID
    V5: /scratch/leonliu/leon-storage/runs/vanda_stage40_v5_one_handoff_$PBS_JOBID

All stages use DEV_SEED=4242. Formal seeds 2223/2228/2233/2238 are forbidden. No stage may read/write Validation-E, Holdout-F, training, hardware, ROS/ZMQ/P1M, or H200 paths.

## V2 A40 runtime qualification

Runner:

    /home/svu/leonliu/projects/go2-arx5-hierarchical-vla/hpc/pbs/vanda_stage40_v2_runtime.pbs

Exact submission command:

    qsub /home/svu/leonliu/projects/go2-arx5-hierarchical-vla/hpc/pbs/vanda_stage40_v2_runtime.pbs

Current V2 runner is the only executable PBS in this plan. It performs native camera/StateV1 smoke and exact GR00T model construction with forward_count=0 and wbc_forward_count=0. It records telemetry phases baseline, after_native_camera, after_gr00t_construction, and final.

Required V2 admission:
- qsub accepted in interactive_gpu;
- execution host is GN-A40-*;
- GPU is NVIDIA A40 with approximately 48 GB;
- Apptainer/--nv path passes;
- camera/StateV1 status is PASS;
- exact GR00T construction is PASS;
- forward_count=0 and wbc_forward_count=0;
- stage40_telemetry.jsonl contains the four phase rows;
- no formal seed, no scientific rows.

## V3 development Q0 admission

Planned runner (must be committed and statically audited before use):

    /home/svu/leonliu/projects/go2-arx5-hierarchical-vla/hpc/pbs/vanda_stage40_v3_q0.pbs

Exact command after the runner exists and V2 PASS is sealed:

    qsub /home/svu/leonliu/projects/go2-arx5-hierarchical-vla/hpc/pbs/vanda_stage40_v3_q0.pbs

Required admission predicate:
- V2 result/seal is present and PASS;
- source/artifact/runtime identities equal the frozen bundle;
- fresh body/wrist RGB and finite StateV1;
- DEV_SEED=4242 only;
- exact STRUCTURED@1000, D1 StructuredActionV1 decoder, H40/T16/D132 -> physical D10;
- SafetyGate is ACCEPT or WARN; REJECT stops;
- one Q0 model forward is recorded;
- no WBC stepping and no Q1/handoff in V3;
- result is NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE.

## V4 semantic C0-dev

Planned runner (must be committed and statically audited before use):

    /home/svu/leonliu/projects/go2-arx5-hierarchical-vla/hpc/pbs/vanda_stage40_v4_c0_dev.pbs

Exact command after the runner exists, V2 and V3 predicates pass, and source remains frozen:

    qsub /home/svu/leonliu/projects/go2-arx5-hierarchical-vla/hpc/pbs/vanda_stage40_v4_c0_dev.pbs

Required admission predicate:
- V2/V3 evidence paths and exact identities still PASS;
- both native and GR00T runtimes are resident;
- fresh Q0 decode is D1 StructuredActionV1 and finite;
- SafetyGate is ACCEPT or WARN;
- initial reference bind succeeds before any WBC boundary;
- executor uses round(HANDOFF_S/native_dt) and no semantic tuning occurs;
- no reset, teleport, policy-history clear, or multi-handoff;
- A40 UUID and VRAM baseline/peak are recorded;
- result is NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE.

## V5 complete one-handoff engineering smoke

Planned runner (must be committed and statically audited before use):

    /home/svu/leonliu/projects/go2-arx5-hierarchical-vla/hpc/pbs/vanda_stage40_v5_one_handoff.pbs

Exact command only after V2/V3/V4 predicates pass and the bundle is still frozen:

    qsub /home/svu/leonliu/projects/go2-arx5-hierarchical-vla/hpc/pbs/vanda_stage40_v5_one_handoff.pbs

Required admission predicate:
- fresh Q0 -> exact decoder -> SafetyGate -> initial bind;
- native steps to the nearest frozen 0.9750000000000001 s handoff tick;
- pause; fresh Q1 -> exact decoder -> SafetyGate;
- exactly one reference-only update;
- continue to completion; no reset/teleport/history clear;
- result schema has actual_switch_s, switch_tick_error_s, executed_prefix_steps, q0/q1/wbc counters, artifact/runtime/git identities, GPU UUID, VRAM baseline/peak;
- result seal passes vanda_stage40_telemetry.py;
- after PASS, stop for GPT/PI review.

## Freeze/rebind rule

Before any V2/V3/V4/V5 qsub:

    git status --porcelain --untracked-files=all
    git rev-parse HEAD
    git rev-parse @{u}
    git diff --check
    bash -n hpc/pbs/<runner>.pbs
    grep -nE "PBS -P|qsub[^$]*-P|personal-leonliu" hpc/pbs/<runner>.pbs

The queue job must be cancelled/rebound if tracked source or runner bytes change after submission. Never rely on a mutable working directory silently.

Scheduler checks are low frequency only. Do not submit duplicate speculative GPU jobs.
