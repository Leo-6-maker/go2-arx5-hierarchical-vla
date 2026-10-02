# PI0268-R1 execution continuation

Authority: release `e5ec6d52100f6bb2e32be5b9661c756b0adb61f4`, Issue #11 canonical 5957403226. This is a progress record, not a canary, training or quality PASS.

## Actual training attempts

| Job | Actual result | Completed updates | Training heads |
|---|---|---:|---:|
| 643647 | Import resolved the older same-name helper; failed before model construction | 0 | 0 |
| 643670 | OLD continuous canary reached 8; native save completed model/optimizer/scheduler/RNG, then the new metadata check read the wrong directory | 8 | 152 |
| 643705 | Replacement submitted after CPU repair; last observed Q, insufficient GPU availability | pending | pending |

The qualified PI0267 helper is now explicitly bound. Native processor/config sidecars are copied from the actual training output into every checkpoint and verified against the arm's effective statistics. No metadata, labels or model output are invented. First-update S4 gradient evidence is now persisted immediately after the existing backward pass, without another gradient query.

The failed continuous-8 payload and logs remain under `OLD/failed-643670-continuous`; its first-update gradient receipt is UNOBSERVED. Because that required canary evidence was lost before the final reporting line, continuous-8 alone must be repeated within the 256-update technical allowance. The original 8 updates remain in raw accounting. Main training has not started; both 2048 endpoints and all quality evaluations remain pending. Both-arm loss, sample plan, random packets, precision and budgets remain unchanged.

Current training fileset SHA256: `7eb259ea284c80f6ee67085bee21b389605674e246624c5846f9aa6bb7af0be5`. Previous failed filesets and scheduler logs are retained. No healthy job was cancelled or duplicated.

## Actual evaluation preparation

- FIT112 is rebound to actual first2048 consumption; its two packets yield 224 rows per point.
- All 304 fixed factor conditions passed both physical codec checks. 112 exact frame pairs were reused and 192 newly rendered; no physics steps.
- Actual train/eval conditioning equality passed 416 inputs per contract, 832 checks total; RGB/tokens match across contracts, targets are absent from policy inputs.
- OLD0 exact cache binding passed for 64 virtual episodes and 440 legacy rows; ROT0 reuse across processors remains prohibited.
- Native virtual-prefix/original-gate CPU checks and prospective final-five-second/group safety boundary tests passed. The finite evaluator and original dependency hashes are sealed before any new quality query.
- New quality inference attempts remain 0 at this snapshot. Prepared evaluation code is not evidence that evaluation has run.

## PI0267 closeout, separately

`PI0267_PRIMARY_COMPLETE_SECONDARY_PARTIAL_ARCHIVES_COMPLETE`

192 virtual outcomes, 2088 offline rows, full CPU replay and all four full Trainer readbacks are present. Primary quality remains `VALID_QUALITY_FAIL`, no virtual research candidate. Historical raw training accounting remains 8228 updates; 18,988 primary virtual queries are preserved.

Secondary: eight valid expert outcomes; one model attempt with three actual queries ended in a native TCP/FK cache assertion. It has 43 raw control-tick starts but only 42 saved post-step records; the final post-step event is UNOBSERVED. It is `ENGINEERING_INCOMPLETE`, not a valid quality failure. The other 23 model rows are explicitly deferred under PI0268-R1 section 3. No new PI0268 WBC budget was used.

Fixed readback revisions for newly completed old archives:

- R4096 results: `254e20488e8b4c3544cc64357e995cb704eef087`
- Secondary: `85ea64286e229863748268665bde4b54541eb7fd`
- Summary/media: `c7cfa81c6ede174748adce28fb4613b71a194f1e`

The separate closeout receipt retains the exact old four Trainer revisions, five result archive receipts and secondary continuation details. Old V2/V3 scores are unchanged. Formal selected/deployment remain NONE.
