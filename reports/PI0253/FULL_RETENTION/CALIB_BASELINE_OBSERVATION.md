# Original CALIB baseline first-valid result

The comparison-only original adapted1000 baseline completed all 1088 distinct CALIB sources, 1088 VALID cells / 4352 native S4 head forwards. A separate CPU process reproduced decoder, gate and scalar records from the saved arrays without model queries. The original safety gate rejected all 1088 cells; all include LINEAR_SPEED_REJECT. Additional rejection issues occurred in 63 ANGULAR_SPEED_REJECT, 13 PATH_LENGTH_SANITY and 11 INITIAL_POSITION_DISCONTINUITY_REJECT cells. Issue counts overlap.

This is a valid baseline quality result, not an integrity failure or a formal FULL candidate. No retry, threshold change or runner replacement follows from it. The five-profile sequence continues with the four formal FULL candidates and the original selector. selected_checkpoint=NONE until that unique freeze; no physical episode has been run.

- Profile SHA256: e211dd1cd45157684fc6df38b028f8090a67711fbfbe10333b3a2dc3eab2679d.
- Raw archive: leoyuer/pi0246-gate-pass-campaign, revision d8bae31f87bc40a1ae4c99d5dcbf848331dcc5ce, prefix evidence/PI0253/calib-baseline-20260929.
- Nine archived files, 139086896 bytes; tree 34e824108f9dc077c38223564821c7689aa4d7cfc927219cdf60f9625d63acea. Every archived file was streamed and hashed from that revision.
- Lossless profile-1000.tar contains all 3267 original profile files, including 1088 raw NPZ, attempts and receipts; each member content was checked against PACKAGE.json before upload. Profiles/card/CPU/bindings/runtime source are also separately indexed.
- CALIB data, processor/statistics and model source identities remain bound to the original authorization. Accepted-cell ranking statistics are empty because every baseline cell is REJECT; no score is substituted.
