# M static preparation — not native task qualification

Core's finite CPU service `wm02-sla-v6-multitask-static-scene-20261118-03` terminated CLOSED/exit0. Source `145bf72085f0d91cac83bab130f1176ed4c2fd6f`, spec `42ea427f4b0f739013d5f32705cadc28efeb5d12854c907f380314f432a81522`, former PID1562043. CPU affinity2, MemoryMax4GiB, no integration API or native controller process was called. Wall1.505s; see JSON for actual CPU cost.

Both candidate scenes have two free physical cubes, both noncontact destination regions, ABI41/38/20, and20 unchanged actuators. The positive compiled inventory verifies original robot body/joint/actuator/sensor IDs, properties, mesh data, control parameters, initial robot states and physics options. Both protected robot fingerprints are `67d4b804c438beabec9790ecf40b60584d2b45ec0c5f1799f377bf784f316bc8`. Scene content is independent of which of the six instructions is selected.

- F1 reuses the previously compiled and hashed valid static candidate; no second compilation.
- The initial F2 lateral prototype failed full support geometry: blue center43.5mm outside the required box footprint/support margin. That negative remains in static02. Before any Teacher execution, the new F2 prototype adds85mm worldX to its scene origin; the table, robot and control parameters are unchanged. The new static candidate passes its original positive robot inventory. This does not relabel the earlier prototype PASS.
- static01's missing asset_root field is a separate retained zero-native technical failure. The implementation now resolves exactly the two already-frozen XML paths/hashes, without a filesystem search or asset replacement.

Existing single-object readers/cameras still require34/32/20. The existing contact predicate correctly rejects the new blue-object/table name, table/pad contact and wrong-object/pad contact; no guard has been disabled or changed. An explicit two-object native binding and initial-pose/contact coverage must be verified before physics.

Actual updates/visits/model generations/native requests/native integrations/Teacher attempts: **all0**. These are CPU scene candidates, not simulated grasp/place success. The15 G7/scoring checks and6 scene/asset-binding tests likewise establish no physical qualification. T/S work remains independent and untouched.
