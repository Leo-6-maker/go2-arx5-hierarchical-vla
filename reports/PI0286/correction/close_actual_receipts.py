"""Copy closed metadata receipts and describe existing CPU results; no raw arrays."""
import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/pi0276_execution'))
from coordinator import remote

DEST = Path(__file__).resolve().parent
d = json.loads((DEST / 'ACTUAL_CORRECTION_ADMISSION_AND_FAILURES.json').read_text())
names = ['correction/COLLECTION_CLOCK.json', 'correction/C1_PLAN_AUDIT.json',
         'source/evidence/CORRECTIONS32.json',
         'correction/CORRECTION_TRANSFER_MANIFEST.json', 'runtime/CORRECTION_CHAIN_002_STATE.json',
         'runtime/CORRECTION_CHAIN_002_collect_COMPLETE.json',
         'runtime/CORRECTION_CHAIN_002_assemble_COMPLETE.json',
         'runtime/CORRECTION_CHAIN_002_manifest_COMPLETE.json']
script = "import base64,json\nfrom pathlib import Path\nr=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286')\n"
script += f"names={names!r}\nprint(json.dumps({{n:base64.b64encode((r/n).read_bytes()).decode() for n in names}}))\n"
raw = json.loads(remote('core', script))
copied = {}
for name, encoded in raw.items():
    b = base64.b64decode(encoded)
    h = hashlib.sha256(b).hexdigest()
    if name in d['receipts']:
        assert h == d['receipts'][name]['sha256']
        assert len(b) == d['receipts'][name]['bytes']
    target = DEST / 'closed_receipts' / Path(name).name
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        assert target.read_bytes() == b
    else:
        target.write_bytes(b)
    copied[name] = dict(sha256=h, bytes=len(b), local=str(target))

groups = {}
for seconds, tick in [(3,80),(12,260)]:
    rows = [r for r in d['slots'] if r['takeover_tick'] == tick]
    checked = [r for r in rows if r['codec_maxima']]
    reasons = {}
    for r in rows:
        for k, count in r['codec_reason_window_counts'].items():
            reasons[k] = reasons.get(k,0) + count
    groups[str(seconds)] = dict(requested=len(rows), faithful_restores=len(rows),
        physical_success=sum(r['physical_success'] for r in rows),
        physical_failure=sum(not r['physical_success'] for r in rows),
        codec_pass=sum(r['admitted'] for r in rows),
        codec_failure=sum(bool(r['failed_H40']) for r in rows),
        codec_not_tested=sum(r['codec_status']=='NOT_TESTED_PHYSICAL_FAILURE' for r in rows),
        codec_failure_window_reasons=reasons,
        native_preclip_value_occurrences=sum(r['native_preclip_failed_window_values'] for r in rows),
        maxima={k:max(r['codec_maxima'][k] for r in checked)
                for k in ['XYZ_max_m','SO3_max_rad','gripper_max_fraction']},
        codec_failed_slots=[r['slot'] for r in rows if r['failed_H40']],
        first_failed_anchor_by_slot={r['slot']:r['first_codec_failure']['anchor']
                                    for r in rows if r['failed_H40']})
out = dict(status=d['summary']['status'], by_takeover_seconds=groups,
           exact_metadata_copies=copied, raw_arrays_read=False,
           recomputed_codec=False, model_calls=0, physics_calls=0)
(DEST / 'ACTUAL_CORRECTION_FAILURES_BY_TAKEOVER.json').write_text(json.dumps(out,indent=2)+'\n')

md = '''# Actual PI0286 correction admission

**C1 NOT_ADMITTED.** The fixed 32 slots closed once: 32 faithful restorations, 28 physical successes, and 15 branches passing the unchanged strict codec. The required count is 24. No C1 optimizer launch or training transfer is allowed; the transfer manifest contains zero files.

| Takeover | Requested | Faithful restore | Physical success | Physical failure | Strict codec pass | Codec failure | Codec not tested |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 s | 16 | 16 | 15 | 1 | 14 | 1 | 1 |
| 12 s | 16 | 16 | 13 | 3 | 1 | 12 | 3 |
| Total | 32 | 32 | 28 | 4 | 15 | 13 | 4 |

## Codec failures

The unchanged gates are XYZ <= 1e-5 m, relative SO3 <= 1e-4 rad, gripper <= 1e-5, finite valid rotations, and zero native preclip out-of-range values. The 13 failed branches contain 15 failed H40 windows. Preclip and SO3 counts overlap; they must not be added as distinct failed windows.

- **3 s:** only 285118 fails. Its three failed windows trigger both preclip and SO3 gates. Branch maxima: XYZ 2.686345013103164e-8 m; SO3 0.010654909013566369 rad; gripper 0. The failed windows contain 98 preclip out-of-range value occurrences in total.
- **12 s:** 285108, 285109, 285111, 285112, 285115, 285116, 285117, 285119, 285120, 285121, 285122 and 285123 fail. Each has one failed window, with 80 preclip out-of-range values; only the zero-preclip gate fails. Their measured XYZ, SO3 and gripper roundtrip errors satisfy the physical error thresholds. This is still a strict-codec failure under the frozen rule.
- Every failed branch first fails at suffix anchor 0 (original tick 80 for 3 s; tick 260 for 12 s). No XYZ or gripper threshold failures occur. No reported nonfinite or degenerate-rotation failure occurs.

| Codec maximum across tested branches | 3 s | 12 s | Gate |
|---|---:|---:|---:|
| XYZ (m) | 3.1828893298779803e-8 | 2.60770320892334e-8 | 1e-5 |
| SO3 (rad) | 0.010654909013566369 | 6.956590542729907e-8 | 1e-4 |
| Gripper fraction | 0 | 0 | 1e-5 |

The stored audits record preclip counts, not the maximum excess magnitude or affected native channels. Those details are unknown in this metadata-only report. No normalization or gate was changed.

## Physical failures

| Slot | Missing sustained history | Failed terminal hold predicates |
|---|---|---|
| 285113 at 3 s | lift; horizontal transfer | zone 0/41; TCP_clear 0/41 |
| 285113 at 12 s | lift; horizontal transfer | zone 0/41; TCP_clear 0/41 |
| 285114 at 12 s | grasp; lift | zone 0/41; TCP_clear 0/41 |
| 285118 at 12 s | grasp; lift; horizontal transfer | zone 0/41; TCP_clear 0/41 |

All four ran to the original 32 s completion boundary. They pass the other stored terminal hold predicates. Their suffixes were not admitted or codec-tested for training. Full original score fields, restoration predicates and source-result hashes are retained in the JSON and CSV.

## Closed evidence and limits

Collection began 10:48:33 SGT; the collection/CPU assembly/manifest chain closed 11:04:31 SGT. Each slot used one physical pass, with zero quality retries. There are no missing or duplicate slots or duplicate branch states. The correction chain made zero new learner model queries. This report reads existing closed metadata only and performs no additional physics, model, teacher or codec execution.

Bank: `CORRECTION_NOT_ADMITTED`; plan: `CORRECTION_NOT_ADMITTED`; manifest: `CORRECTION_NOT_ADMITTED_NO_TRAINING_TRANSFER`, files 0. The sampler was not run after bank rejection; no actual sampler infeasibility is claimed. Raw evidence and failed branches remain preserved. Learning causality is UNPROVEN; the 28/32 physical rate makes no generalization or source-policy qualification claim.

Local companions: `ACTUAL_CORRECTION_ADMISSION_AND_FAILURES.json`, `ACTUAL_CORRECTION_CLOSED_RECEIPTS.json`, `ACTUAL_CORRECTION_SLOT_DECOMPOSITION.csv`, `ACTUAL_CORRECTION_FAILURES_BY_TAKEOVER.json`, and exact byte copies in `closed_receipts/`.
'''
(DEST / 'ACTUAL_CORRECTION_ADMISSION_AND_FAILURES.md').write_text(md,encoding='utf-8')
print(json.dumps(dict(summary=d['summary'],groups=groups,exact_copies=len(copied)),indent=2))
