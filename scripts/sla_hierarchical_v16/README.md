# V1.6 high-level SLA science preflight (read-only)

This folder contains a **CPU-only** evidence-consistency gate. Its output is
not a resource permit or a complete source authentication. New native and GPU
work stays `NOT_AUTHORIZED` until a separately registered resource owner
approves an explicit roster, per-run tick caps, model forward/update caps,
controller identity, collision/STOP qualification and exclusive lease.

## Files

- [Complete V1.6 plan](../../docs/protocols/SLA_MEMORY_GO2_HIERARCHICAL_AUTONOMY_V16_PLAN_20261124.md)
- [Versioned stage/gate registry](../../docs/protocols/SLA_MEMORY_GO2_HIERARCHICAL_AUTONOMY_V16_REGISTRATION_20261124.json)
- `preflight_public.py`: checks public O1/O3/F2 summaries for misleading
  science claims. It intentionally does not validate private checkpoint
  artifacts or attest MPC/WBC internal policy identity.
- `../../tests/test_sla_hierarchical_v16_preflight.py`: positive and
  adversarial unit tests based on synthetic fixtures.

Run from the repository root where the parent #439 and #437 artifacts are
present:

```bash
python3 -m unittest discover -s tests -p 'test_sla_hierarchical_v16_preflight.py' -v
python3 scripts/sla_hierarchical_v16/preflight_public.py \
    --o1 reports/SLA_OFFICIAL_GO2_TRANSFER/O1_ACTUAL.json \
    --o3 reports/SLA_OFFICIAL_GO2_TRANSFER/O3_RECEIVER.json \
    --f2 reports/SLA_FROZEN_FEEDBACK/DECISION_ACTUAL.json \
    --registration docs/protocols/SLA_MEMORY_GO2_HIERARCHICAL_AUTONOMY_V16_REGISTRATION_20261124.json \
    --output /tmp/v16_public_preflight.json
```

The expected result is `PUBLIC_EVIDENCE_CONSISTENT` with
`native_or_GPU_permission_granted=false` and blocked student control.
It never changes historical numerators, denominators, checkpoint weights,
OCS2 binaries or native safety limits.

## Next meaningful work

Begin V0 (source identity read-only) and V1 (path geometry/speed/acceleration,
source-task events, P10 prefix ownership) on the **existing 16 sealed windows**,
while V2 validates independent student reset/gripper command ownership and V3
profiles known dynamic A1 offline. V4 evaluates the **SLA-proposed** phase and
object-relative EEF goals, separating a rule-scripted supervisor baseline from
a learned high-level planner.

No student gripper/contact/native/GPU work is implicitly authorized by
passing this check. All true held-out manipulation rates remain unmeasured.
