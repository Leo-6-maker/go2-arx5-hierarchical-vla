# V5 execution — 2026-11-14

The execution branch continues from reviewed runtime `cb09e714`, using the plan at `f61305f9c1f847e460851fd067892160f71ca432`. This snapshot is 2026-11-14 01:00:47 SGT. It is not a completed V5 result.

## Actual native policy endpoint

The fixed R0-then-R1 population reuses `r1_train8_inputs_v2.json`: four TRAIN layouts, two new restarts per method, sixteen total requested conditions, unchanged seeds, original endpoint basis, native gates and scoring. It is distinct from historical S1 TRAIN4/DEV16. Each method still has the original 500 updates/16,000 target visits; evaluation performs no training.

| Condition | Physical start | Terminal | Raw H40 | Installed candidate H40 | Consumed candidate H40 | Result |
|---|---:|---:|---:|---:|---:|---|
| R0 TRAIN01 restart3 | 1 | 1 | 1 | 0 | 0 | Valid policy failure: nominal gripper outside original range |
| R0 TRAIN01 restart4 | 1 | 0 | Not yet observed | Not yet observed | Not yet observed | Native startup in progress |
| Other R0 conditions | 0/6 | 0 | 0 | 0 | 0 | Unstarted |
| R1 conditions | 0/8 | 0 | 0 | 0 | 0 | Unstarted; fixed order retained |

The first condition reached measured post-hold readiness and 5,703 actual native integrations. Its first RGB query generated one complete H40. The unchanged whole-proposal gate rejected it before any candidate command installation. There was no clipping, teacher fallback or retry. Task successes are 0/1 terminal, with 2/16 physical requests started; the in-flight condition is not a failure or success. No old DEV/TEST condition was rerun.

Native result SHA256: `9037178bbe00b7dce3cecde522477285815e5f817155682f8d8e4fde582bfbcf`; neural result SHA256: `e9356a7f8febb54aaa7f120d39a7907e1311902453c1d42510d65dda24fd6f13`. The first native worker used 455.41 wall seconds; the paired neural worker used 476.61 wall seconds, with one 7.24-second generation. Total measured pair CPU was 353.89 CPU seconds. GPU allocator peak was 7,475,250,688 bytes. Neural wall time includes loading and waiting and is not GPU utilization time.

The R1 bridge binds the actual H200500/16,000 result, byte-verified transferred checkpoint, exact logit statistics/tokenizer, Core32 fresh RGB reconstruction and single sigmoid decoder. Two concrete preparation defects were fixed: H200's temporary parent path needed its byte-identical Core path; the native supervisor's metadata check imported a model-only SciPy dependency. The failed supervisor launch had zero model loads, zero registered pairs and zero physical requests. Its actual exit1 is retained and reconciled. Both policy admissions then passed in the actual native Python before the cold successor. No RGB qualification or training was repeated.

## New spatial and data operations

A finite Core GPU0 operation has started for a new dense spatial seam, with two alpha0 updates followed by two independent clean-parent alpha0.5 updates. This snapshot has **0 confirmed new updates**, 0 confirmed new optimization visits and no completed spatial checkpoint. Dispatch and model preparation do not establish gradients. GPU1's current native inference job is preserved.

The implementation reuses all122 frozen target windows, stages the target bank once to GPU, and keeps common parameters, Adam state, projector and features resident. Alpha0 must compare all64 microbatch common gradients and both Adam updates exactly against an independent FM state. Positive alpha must produce nonzero gradients in all five deployed VLM LoRA-B leaves. Both arms save complete initial/endpoint state. This is a Core micro1 qualification, not a claimed H200 micro8 or3K spatial candidate.

A separate finite CPU collection operation targets48 fresh PICK_HOLD resets: the same8 TRAIN layouts ×2 restarts at actor counts1/2/4. It reuses the frozen qualified expert, persistent body/wrist camera, bounded asynchronous writer, P10 command stream and actual2s success tail. Completed engineering16 and the original2EP are not relabelled. At this snapshot new complete demonstrations, newly sealed commands/windows and formal D100/D300/PICK_PLACE admission are all **0**. Its first preparation failed before physics because a compiler *receipt* was read as a preparation spec. That terminal0-request failure is preserved; the cold successor uses the actual preparation spec and is rebuilding TRAIN geometry without integration.

Throughput must use all16 attempts per actor count, with per-reset initialization and archiving, and separately account for the shared static preparation cost. No actor count has a measured throughput result yet. No expansion to8/16 actors is authorized by dispatch alone.

## H200 and retained evidence

Four read-only attempts to inspect the same job `716146.hopper-m-02` failed because the existing SSH endpoint closed the connection. This is not scheduler terminal evidence. The job was not cancelled, restarted or duplicated. Its last *dated* observation remains513 candidate updates/16,416 target visits (2026-11-13 23:35 SGT), with the complete500/16,000 checkpoint SHA256 `ba52032ec238ad2cb53f9bef7b4af3a840f862fcbee477e597bffefeb19e5d45`. No3K completion is claimed.

The original S1 remains500/16,000, TRAIN4 0/4 and DEV15/16 terminal with0 successes. Original S2 candidate updates remain0. Frozen teacher122 cache preparation is complete; a cache forward is not a spatial training update. The default five RoboTwin archives contain250 episodes, with20 previously audited; missing clock/desired-command/TCP/current-gripper semantics remain unresolved and S_cmd admission stays0. Completed camera, SFT2000, GR00T qualifications and all historical negatives remain intact.

## Next evidence

1. Read actual terminals of the fixed native population; preserve valid failures and stop for any concrete engineering/safety failure.
2. Read alpha0/positive actual gradients and complete states before scheduling a clean-base matched spatial candidate.
3. Admit new complete consumed expert episodes only after native success, immutable shard and all-window readback; use the measured1/2/4 throughput before expanding concurrency.
4. Recover read-only access to the same H200 job, accept its actual saved endpoint, then implement the dense pilot's separately correct raw/P10 consumer. The R1 endpoint consumer is not that dense consumer.

51 relevant local CPU tests pass. They validate software boundaries and regressions, not robot capability. WBC, leg policy, gains, costs, solver,2ms dynamics and native safety remain frozen. Raw data, weights, site paths and credentials remain private.
