# PI0255 requirement audit — partial terminal, not READY

Authority: `eac535e7c4016df39821798b111fae7ad79e8253`, read completely again at closeout. Sections1,5,10 explicitly permit a partial scientific terminal, prohibit valid-failure retries, and require all three admissions for Motion3 readiness. The original full goal is **not achieved**. This audit does not replace its scope with the completed subset.

| Required item | Inspected evidence | Result |
|---|---|---|
| Actual platform/WBC/frame/TCP/camera/gripper capability | S0/source readback, resolved native configuration; CAPABILITY_MATRIX.md | Source-backed complete; no verified operational contact/gripper claim |
| Separate push and grasp/place readiness | CONTACT_EXTENSION_READINESS.md | Complete gap specification; contact physical trials unexecuted |
| Immutable Motion3 registry/expert/scorer/split before physics | design/execution_seal.json, pre-execution canonical5885426386, runtime hashes | Complete; original identity retained |
| Fixed M1/M2/M3 expert pilots | PILOT/pilot_stage_summary.json, actual physics ledger in fixed HF archive | All36 first-valid; each task0/12, admission requires10/12 |
| Counterfactual reset identity | PILOT/pilot_cpu_replay.json | All18 pairs match actual state/RNG/initial images; six families are not36 independent replications |
| Actual desired/achieved/time/SE3/handoff records | Raw789-member archive and recording deviations | Raw failed preparation retained. Original24 lack complete handoff chunks; remaining12 prospectively recorded V2 and CPU-replayed. No reconstructed original PASS |
| Task-phase outcomes and representative behavior videos | PILOT_RESULT.md, four fixed recorded examples | No task phase reached; videos show actual preparation failures, not Motion3 demonstrations |
| Fixed80TRAIN/20VAL per admitted task and minimum usable pairs/episodes | READINESS_VERDICT_WITH_HF.json, DATASET_CARD.md | All300 collection attempts prohibited by failed admissions; successful TRAIN/VAL and eligible windows0 |
| All-window actual codec/gate and observation sufficiency | Existing CPU loader/allowlist/boundary receipts; pilot replay | CPU primitives checked; actual TRAIN/VAL window QA and task-phase observability incomplete, not vacuously PASS |
| Whole-family split, TEST metadata only, privileged-input exclusion | Frozen registry/split; actual loader CPU checks | Complete registered boundaries; TEST renders/episodes/statistics0 |
| Public fixed-revision failure evidence and licences | HF/pilot_HF_public_receipt.json, DATASET_CARD.md, source licence notice | Anonymous full2-HF-file and789-member readback PASS at c6bc2a041323788a4b19b7b8ef11227f9950a0a6; private assets/weights excluded |
| Base/G exact architecture/keys/tokenizer/licence/slot | INITIALIZER_COMPARISON.md, model_metadata receipts | Metadata verified;1030 common shapes, one tied alias,537 recorded dtype differences, existing slot10. No model weight load |
| Actual new TRAIN support under C_T; numerical common C_MT/adapter | model_drafts/NORMALIZATION_PROPOSAL.json | Incomplete: TRAIN0, numerical statistics null. Proposed mapping cannot be called an executed common processor |
| Matched6000 cards, task sampler, RNG, guard, safe CLI | model_drafts cards, draft_cpu_receipt.json, cli_cpu_receipt.json | Draft CPU checks complete; synthetic schedule has0 real rows; train guard rejects unapproved execution |
| Real sample/augmentation pairing and full Trainer resume | model_drafts/RESUME_FIXTURE_FOLLOWUP.md and resume_s4_cpu_fixture_receipt.json | Qualified S4 primitive/scalar fixture PASS only; real GR00T Trainer/data/processor binding incomplete |
| Fixed future evaluation and separate budgets | FUTURE_EVALUATION_DRAFT.md | Draft complete, no model evaluation executed; no TEST exposure |
| Exact counters/protected independent study/resources | READINESS_VERDICT_WITH_HF.json; live U/R scheduler and Corelab process refresh | PI0255 expert starts36, invalid0, new GR00T loads/queries/backward/optimizer0; policy sim0; selected/deploymentNONE; quotaUNKNOWN |

## Specific dependency and handoff

All mandatory pilots failed during frozen settling:18 expert gate rejects and18 physical terminations. This tests registered preparation, not task-phase feasibility or learned-policy generalization. The immutable authority does not permit changing expert preparation, seed, gate, WBC, task semantics or turning these valid failures into engineering retries. No admitted independent task remains to collect.

Safe independent source/model/CPU/evaluation preparation and public failure preservation are completed. Genuine motion data, TRAIN-only support/statistics, real common processor and real6000 Trainer bindings remain missing. A new versioned scientific decision is required to address preparation; no training or new physics is launched under this audit. Status remains **PARTIAL_MULTITASK_PREPARATION_BLOCKED**, never DATA_MODEL_READY or MT_PILOT_TRAINING_ADMITTED.

PI0254 is separate: U-final4608 has384/384 CPU-verified cells, passes E64/P64 relative research gates but fails all four O64/old16 retention groups. R has0 valid updates and exhausted its two submissions. That negative/partial result neither changes PI0255 admission nor provides a multitask initializer substitution. Its raw training counter includes the failed initial attempt (41477 heads); inference total1088 queries/4352 heads. See the separate PI0254 final canonical report.

No active PI0255 collector/dispatcher or PI0254 evaluator remains on Corelab. U637538 is terminal F; R637808 is terminal F/Exit1. Other GPU workloads remain present and unchanged. Mail/monitor/merge and old DEV/F stay closed.
