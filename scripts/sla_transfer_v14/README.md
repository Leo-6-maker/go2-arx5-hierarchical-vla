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
