# PI0253 four fixed FULL live side reports

Four first-valid L/Nlive/Q queries,16 native heads,0 optimizer updates,0 physical episodes. Separate CPU process reproduced all four saved arrays/decoder/gates/scalars. Actual input/processor/frame/noise identities match the frozen G bridge. Baseline G side query is reused, never repeated. All four original gates ACCEPT; this is not physical success or a qualification override.

|FULL step|Reference minimum target m|Reference final target m|Expert XYZ RMSE m|Expert SO3 RMSE rad|Gate|
|---|---:|---:|---:|---:|---|
|2500|0.057385|0.093443|0.096836|0.036807|ACCEPT|
|5000|0.064729|0.212770|0.159008|0.076658|ACCEPT|
|7500|0.049392|0.153700|0.115839|0.039720|ACCEPT|
|10000|0.056401|0.127726|0.106985|0.031950|ACCEPT|

Reference world positions are native decoded XYZ plus the actual fixed anchor. Expert comparison uses the corrected A_attempt2 expert sampled at the original real times. These offline reports contain no actual TCP measurement; actual_TCP=UNOBSERVED_NO_PHYSICS_EPISODE. No minimum-distance subtraction is called tracking. Old G actual TRAIN-A/C0 valid FAIL remains untouched.

The side reports do not affect training, candidate identities, CALIB ranking or original thresholds. Original FULL family and retention are complete. Only now may original1088-row CALIB be read: exact source treef53e85e8245735ec0f9a3ee9d1008d20385a4cfd8100eb66a15c015d7ecaf371,68 source rollouts. Original comparison baseline tree12c83e6695c28be3b4a91ef6df201b18f6a1a2877d3e80f7db15ef656176ab3b is model-only, not a complete Trainer resume. Source tree validation and fixed HF materialization are transport/admission work,0 model queries.

selected_checkpoint=NONE until original selector freezes one winner. All four candidates remain, including intermediate quality FAILs. Winner must independently pass its original all64/old16 B0/B1 retention before any new physical episode; no runnerup substitution. No DEV/F, DAgger, hardware, WBC/gate changes, merge, mail or monitor.