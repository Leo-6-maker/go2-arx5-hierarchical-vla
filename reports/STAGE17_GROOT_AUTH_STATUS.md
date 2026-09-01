# Stage17 GR00T Authentication Status

Date: 2026-09-01 (Asia/Singapore)

## Result

`AUTHENTICATED`

The Stage10 runtime on `corelab-5080-2` was checked with the non-secret
identity command `hf auth whoami`. It returned:

```text
user: leoyuer
```

The authenticated account has read permission. The gated
`nvidia/Cosmos-Reason2-2B` processor dependency also loaded successfully in
the Stage10 environment. This closes the earlier `AUTH_PENDING` deviation for
the actual-model subtask.

## Runtime

- Stage10 root: `/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10`
- Python: `3.12.3`
- PyTorch: `2.9.0+cu128`
- CUDA build: `12.8`
- Hugging Face Hub: `0.36.2`
- host: `corelab-5080-2`

An existing other-user GPU workload was present on the node. It was not
killed, signaled, reprioritized, or attached to. The one approved actual
inference therefore used `device=cpu`; this is a resource-safety choice and
does not imply that the model is CPU-qualified for deployment.

## Secret handling

Only the account identity was queried. The token was not requested, printed,
copied, inspected, or stored in the repository or Stage17 manifests.

## Scope boundary

Authentication and processor access establish runtime availability only. They
do not establish Go2+ARX5 task performance, robot-specific adaptation, or
permission to execute an unadapted model trajectory through WBC.
