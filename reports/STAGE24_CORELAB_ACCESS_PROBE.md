# Stage24 Corelab Access Probe

Date: 2026-09-03 (Asia/Singapore)

The configured compute-node SSH access from `corelab-amd-server` is restored.
The probe was read-only (`hostname`, identity, and GPU inventory); no process
was killed, signalled, reniced, or started.

| node | result | observed GPU memory |
|---|---|---|
| `corelab-4080s` | SSH PASS | 32 GB card: 153 MiB; 16 GB card: 38 MiB |
| `corelab-5080-1` | SSH PASS | 9,084 MiB / 16,303 MiB used |
| `corelab-5080-2` | SSH PASS | 91 MiB / 16,303 MiB used |

This closes the earlier access blocker for the Stage24 exact-checkpoint audit.
The figures are time-specific observations and do not authorize a future
workload by themselves. Machine-readable evidence, including GPU UUIDs, is in
`reports/STAGE24_CORELAB_ACCESS_PROBE.json`.
