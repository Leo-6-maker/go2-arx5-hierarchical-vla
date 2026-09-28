# FULL7500 fixed retention observation

Fixed revision c25a4252675fae8616d0dc80e4251c91a81c319b. 128 first-valid cells,512 native head evaluations; independent CPU replay PASS. Valid intermediate quality FAIL retained. Candidate set and10000 endpoint unchanged.

|Group|REJECT|median XYZ m|p95 XYZ m|median SO3 rad|Quality|
|---|---:|---:|---:|---:|---|
|all64/B0|0|0.054862|0.190460|0.033288|FAIL|
|all64/B1|0|0.055453|0.189959|0.031960|FAIL|
|old16/B0|0|0.051210|0.087701|0.038645|FAIL|
|old16/B1|0|0.051171|0.088903|0.038392|FAIL|

Original thresholds remain zeroREJECT / medianXYZ<=.020m / p95XYZ<=.035m / medianSO3<=.050rad. All four groups fail XYZ median/p95. No score-based retry, training change, candidate retirement or premature selection.

Completed retention scope: G0/B0 reused64 + new FULL1000/B064 +2500/5000/7500 B0/B1 each128 =512 logical cells,448 new queries,1792 new head evaluations. FULL10000 remaining128; formal total640 logical/576 new remains unchanged.

FULL5000 raw evidence PUBLIC archive full391-file readback: revision4d0827decdf4d707cd0360aa8b72fbcc1f1523b8, prefix evidence/PI0253/full5000-20260929, bytes24,587,981, treeb9bef66b3a1b0343bac47de5ae10569badff5b694bf665bd9cfabc72ee52cb6e. No new query/optimizer update from archival.

selected_checkpoint=NONE; CALIB and live side reports wait for complete FULL10000 family/integrity and final retention.

FULL7500 raw evidence PUBLIC archive full391-file readback: revision5922ab1fbafdca3dd061ba5281d1af8206ca93b0, prefix evidence/PI0253/full7500-20260929, bytes24,583,563, tree8849f21483ff69f85c0b1a2ec958b6c1f0e3f99bbe006dc9d98bd7f6ff039715. Original first-valid tensors and CPU report preserved; archival adds0 queries/updates.
