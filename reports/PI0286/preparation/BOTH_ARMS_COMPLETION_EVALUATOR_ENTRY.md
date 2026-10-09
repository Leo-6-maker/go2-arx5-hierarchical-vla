# PI0286: authorized completion of both endpoints

The owner forwarded the user's explicit reply: “完成两臂，允许越过14:00”. The old 14:00 calendar cutoff is superseded. The experiment recipe, cumulative query/head/physical caps, cumulative model GPU time limit and minimum 45 minute confirmation reserve remain unchanged.

Core's stopped-boundary time binding has actual CPU and seal checks: `USER_BOTH_ARMS_COMPLETION_PRIORITY_NO_1400_CUTOFF.json`. Calendar ceiling: MODEL/UPDATE 18:00 SGT, ARM/CONFIRM start 17:15 SGT, 2026-10-09. These calendar values do not override cumulative time accounting. Only `r2.py` changed. Old bytes and seal remain archived.

## Actual C0 result

C0 completed once: fixed TRAIN8 T1 success 2/8, independent source CPU PASS and DIAG96 CPU PASS. The original TRAIN gate omitted DEV. CONFIRM was not queried. Selected/deployment NONE.

Actual complete stage timings: transfer and native reconstruction 702.947 seconds; TRAIN policy 310.699 seconds; source CPU 4.486 seconds; diagnostic run 98.386 seconds; diagnostic CPU 32.175 seconds. Dispatch through final closure: 1151.385 seconds. Original stage receipts are retained.

## C1 once-only entry

From the workspace, use the prepared finite owner entry after actual Hopper C1-1024 READY **and** independent training CPU PASS:

```powershell
py -3 scripts/pi0286_execution/start_endpoint_evaluation.py --point C1_CORRECT1024-1024
```

This entry checks the actual checkpoint cursor, canary comparison, export/READY/CPU identities, current seal and cumulative guards. It adopts an existing same-point process and rejects parallel own endpoint workers. The unrelated healthy GPU consumer is permitted under explicit user coexistence authority and remains untouched.

C1's exact transfer partial directory and BF16 model directory have empty, fixed aliases to owned local SSD storage. The logical paths and immutable point binding remain unchanged. Mature transfer hashes every declared byte, materializes and audits the native BF16 view, and retires its consumed changed-tensor duplicate. No FP32 master file is written to disk. Estimated simultaneous delta plus BF16 payload is 11,958,254,896 bytes; this is an estimate, not a future actual transfer receipt.

The existing serial endpoint evaluates fixed TRAIN8, independent CPU, DEV16 only if TRAIN>=7, independent CPU, DIAG96 and independent CPU, then closes the route. It never selects or starts confirmation. C0's quality failure does not cancel C1.

## Confirmation and saved-image evidence

The owner alone closes comparison, applies the original unique rank, freezes one eligible candidate and authorizes fresh CONFIRM32. The minimum reserve remains max(2700 seconds, 1.25 times fresh episode p90 times 32 plus measured extra), subject to actual cumulative time availability. No confirmation is presumed from preparation.

Saved RGB videos are prepared separately in `video_saved_endpoints.py`. Run only after the scientific mainline closes. They use saved cameras, full fixed condition coverage, actual T1 and original stack hold labels, source Panda, NON-REALTIME, no WBC. They generate no new render, physics or model query and preserve earlier videos.
