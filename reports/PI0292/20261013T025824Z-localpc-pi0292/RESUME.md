# PI0292 Resume

## Latest Hold: 2026-10-13 13:56 UTC

Read `S3_COMPUTE_ADMISSION_HOLD.md` and `S3_SMOKE03_TERMINAL_RECEIPT.json` first.
Login authentication/project qualification is restored. Both actual S3-smoke-02
and the single explicitly authorized engineering retry S3-smoke-03 are terminal,
reconciled pre-model compute-admission failures. Total allocated GPU time is 17
seconds; model loads and updates remain zero. PBS calls are 3/4 under the direct
human revision in `ENGINEERING_PBS_REVISION.json`. No PI0292 allocation is active.
Do not submit either old request again or start main without the actual S3 gate.
Another automatic engineering smoke is not authorized. Preserve the one remaining
submission and original T0/cutoff/deadline until explicit review of the compute
project-attestation repair and submission plan. The healthy peer is untouched.

The remainder below is the older first-submission recovery snapshot, preserved
for provenance. Its authentication, capacity and 3-call-cap observations do not
supersede the current receipts.

Keep the original run UUID/T0/deadline. All24 supplement attempts are terminal; do not relaunch them.

## Current S3 Hold

First qsub call was rejected because CFP03-CF-004 ended. It has no job ID and
counts conservatively as 1/3 submission calls; no allocation or model execution
occurred. Do not rerun `submit-smoke`: the original request, bindings, manifest
and failure are preserved. The status-only command is:

```bash
python3 work/private/pi0292_pbs_control.py status --source c187d9573bdff302783fb5f41d06fcaa1753d2a8
```

Before preparing a distinct retry, the user must refresh Hopper's existing
account-manager login (`amgr login`, no credentials in chat), and the owner must
verify a currently authorized unexpired project. The personal quota report also
fails the prepared 80 GiB smoke peak +20 GiB margin: 940.7GB/1.0TB. Do not infer
permission or capacity from historical project lists or shared df output. Obtain
verified additional quota or an explicitly authorized storage binding; do not
delete historical checkpoints/data under this science GO.

The retry must retain the first failure and original T0/counters. Only two
submission calls remain under conservative accounting. Plan any later main and
evaluation allocation accordingly, without adding a fourth submission or
changing 4096-step candidate, full checkpoints, thresholds, or evaluation scope.
If prerequisites cannot be met within original time/budget, deliver the exact
partial outcome. No model work on login/Corelab is a fallback.

The new relay packet code has only synthetic CPU coverage. Actual SSH compute
reachability, endpoints, real observation provenance and teacher 2/2 physics
qualification remain to implement/verify at the original stage gates.

Read the durable CPU/data stages from the existing local workspace:
```bash
python3 work/private/pi0292_bank_control.py status bank
python3 work/private/pi0292_bank_control.py status transfer
python3 work/private/pi0292_bank_hopper.py status
```

Only launch a stage after proving its prior gate, absence of its earlier attempt and no competing owner. Launch acknowledgements are not completion. On failures preserve all files, receipts and budgets; diagnose before an explicit engineering repair. Original56 have been fully consumed on Hopper, with actual cross-host CPU tensors verified. Never rerun its workspace initializer or original56 consumer.

After bank CPU/probe and immutable private HF producer readback pass, complete the new-bank full Hopper consumer and actual TRAIN/DEV CPU comparisons. S3 requires those gates and a new own single-H200 PBS allocation. The prepared PBS smoke wrapper only runs the 4-vs-2+2 engineering chain, not main training. Reconcile actual scheduler allocation time and complete-state storage before further submissions. S4 additionally requires exact smoke equivalence and full4096 throughput/capacity qualification. Continue S0-S6 within fixed authority, without resetting budget or substituting another candidate.

Use private runmap bindings and existing credentials only. GitHub files are sanitized projections; full raw images/numeric data, private bindings and complete checkpoint assets do not belong in this repository.
