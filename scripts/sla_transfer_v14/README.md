# Frozen SLA V1.4 CPU replay

Run these checks on the host with the V1.2/V1.3 sealed data, receiver source, episode manifests, ROS-generated message bindings and V1.3 CPU replay environment. The V1.4 scripts and the archive member hashes can be kept in a separate directory. Set `TRANSFER_ROOT` to the prior experiment root and `OUTPUT` to a new private directory.

```bash
umask 077
mkdir "$OUTPUT"
python3 -B identity.py "$TRANSFER_ROOT" "$OUTPUT/B0_BACKEND_IDENTITY_AND_SOURCE_ACTUAL.json"
python3 -B launch.py shadow "$TRANSFER_ROOT" "$OUTPUT/b1" --archive ARCHIVE_ABI.json
python3 -B launch.py regression "$TRANSFER_ROOT" "$OUTPUT/angular"
```

The shadow replay uses the recorded Python environment and original archive and site callbacks. It replaces graph transport with an in-memory ROS serialization sink and disables the RViz display. The synthetic observation clock and TCP positions are explicit fixtures. The CDR padding may differ between runs; compare all three typed message fields. SHA256 identifies an individual run's receipt. A shadow HOLD is a request captured by the sink.

The numerical regression uses V1.3's canonical gate replay and the original per-window H40/G7 checks. Its angular-speed diagnostic uses the same scalar geodesic operation as the frozen gate. The isolated process keeps the original directed angular-velocity vector calculation for angular acceleration. The V1.3 receipt remains available for comparison.

For the clock guard and B2 preparation, run these CPU checks against the same sealed root. Each output must be a fresh directory. The historical tracking check also needs the two predecessor M0 batches identified by the teacher execution card. The technical shadow uses their recorded start pose only as a test fixture.

```bash
python3 -B launch.py clock_replay "$TRANSFER_ROOT" "$OUTPUT/clock"
python3 -B launch.py historical_tracking "$TRANSFER_ROOT" "$OUTPUT/history"
python3 -B launch.py technical_registration "$TRANSFER_ROOT" "$OUTPUT/cards"
python3 -B launch.py technical_shadow "$TRANSFER_ROOT" "$OUTPUT/tech_shadow" --registration "$OUTPUT/cards/B2_TECHNICAL_REGISTRATION.json"
```

The two approved B2 technical cards used this clock guard in separate native simulator sessions. The live run and its evidence are described in [the B2 result](../../reports/SLA_MEMORY_V14/B2_NATIVE_TECHNICAL_CARDS.md). On the registered Corelab site, `b2_owner.py preflight TRANSFER_ROOT` checks the frozen files and exercises the actual clock-fault callback offline. The finite `execute` entry also runs those checks and refuses to reuse an existing campaign. To audit the two closed cards, use `b2_readout.py GROUP NEW_OUTPUT_JSON`. Gripper ownership and stopping under continuous motion remain separate experiments.
