# PI0294 One-Shot SSH Endpoints

`relay_endpoint.Endpoint` provides file operations for the existing local owner
and paused-observation ACK exchange. It does not open a server, call physics,
load a model, authenticate a budget account, or submit PBS. Host addresses and
SSH configuration remain in the existing private runmap.

Both endpoint roles require the actual registered run manifest, full source SHA,
and a hash-bound case registry. CoreLab exposes fresh observation, Teacher-only
answer, and actual ACK reads; installation verifies the original observation,
previous ACK and fixed prefix gate before one atomic, non-replayable bundle write.
Hopper Teacher queries persist a one-shot intent and bind the next query to the
previous physical ACK. B1 queries can only enter the already admitted resident's
inbox; this endpoint does not perform inference and rejects a Teacher answer.

Read-only response polling is permitted. An uncertain query or installation must
not be sent again. The endpoint registry, CPU accounting, transport timeout and
actual process identity remain required in the private approved launcher.

The endpoint tests are file fixtures only. They do not establish actual dual-SSH,
Teacher contact success, parallel-source continuity, or model policy qualification.
Deployment and two actual Teacher episodes with independent physical audits are
still required before any `LIVE_READINESS` PASS claim.
