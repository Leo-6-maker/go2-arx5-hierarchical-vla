# Recording-only continuation V2

The frozen registry and expert/scorer source remain byte-identical. No geometry, instruction, seed, settling, timing, gate, controller or metric changes are made.

The original V1 dispatcher is paused between episodes; its already-running child completes unchanged. All 23 original first-valid results are retained and never repeated. The remaining registered pilot IDs use a separately sealed runtime_v2 directory. Original source and execution seal remain available.

Raw tick JSON bytes are unchanged. Each row is flushed; the completed raw ledger is synced before final result publication, replacing costly per-row NFS fsync. Exposure, handoff and final result ledgers remain durable. A CPU check verifies identical serialized bytes and selective sync.

Future handoffs additionally save the actual full chunk, achieved pose, simulation timestamp and frame transform before the unchanged gate evaluation. Each gate row binds its NPZ hash, and each episode retains its execution seal. Missing original V1 full chunk arrays are an explicit evidence deviation: they are not reconstructed, and no exact full gate/codec parity is claimed for those original rows.

Native Python CPU checks precede immutable code publication, new execution seal and the first V2 physics step. There are no new GR00T loads, queries, gradients or optimizer updates.
