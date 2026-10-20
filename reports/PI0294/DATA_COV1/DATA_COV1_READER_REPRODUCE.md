# DATA-COV1 Saved Reader And Receipt Reproduction

This document reproduces evidence, not a collection/training command. It must
not reopen either closed lease or create physical replacement attempts.

## Public Receipt Checks

From an authorized repository checkout, this CPU-only snippet reads the small
public reports. It loads no model, private raw data, network service or physics:

```python
import hashlib
import json
from pathlib import Path

r = Path('reports/PI0294/DATA_COV1')
read = lambda name: json.loads((r / name).read_text())
terminal = read('DATA_COV1_P24_TERMINAL.json')
for name, expected in terminal['original_native_report_file_hashes_also_matched_by_public_JSON_files'].items():
    assert hashlib.sha256((r / name).read_bytes()).hexdigest() == expected
saved = r / 'DATA_COV1_SAVED_CLOSEOUT_ACTUAL.json'
assert hashlib.sha256(saved.read_bytes()).hexdigest() == '1dc3b4c5974d246fedbaa143a5b10b9d842b670771f37ce216c25ce92c3d3b75'
actual = read(saved.name)
mixed = actual['cross_parent_batch']
assert mixed['distinct_parent_count'] == 2
for batch in mixed['batches']:
    assert len(batch['actual_parent_anchor_rows']) == 12
    assert batch['deterministic_repeat'] and not batch['model_forward']
assert not actual['first_four_gate_pass'] and not actual['stage_bank_READY']
rows = read('DATA_COV1_P24_ALL_ATTEMPTS.json')['rows']
assert len(rows) == 48 and sum(v['status'] == 'NOT_RUN' for v in rows) == 44
assert read('DATA_COV1_P24_FIRST_FOUR_GATE.json')['statuses'] == {
    'ENGINEERING_INCOMPLETE': 1, 'CODEC_REJECT': 1, 'ADMITTED': 2}
```

The four original public JSON files match their exact native file bytes.
`DATA_COV1_RESOURCE_FINAL.json` is a compact public view, not a byte-identical
copy of the larger native per-render-call timing receipt. It gives that full
native SHA separately and records independent closed-lease/hash readback.

## Actual Native Consumption Already Performed

The saved audit used pinned own source
`16a4c2d629fe79dfa1f9289004307694e1ba6185`, official GR00T source
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`, original local Cosmos tokenizer
assets and fixed statistics
`fb5f5bc6969fae47f51b8a6acefdd7312ed948a0fad4b9f9a79e1fbb9aa325c7`.
Existing authentication/assets were reused; no historical models downloaded.

`scripts.pi0294.data_cov1.closeout.run` first validates the cold manifest,
canonical registry, new read-only audit lease, original closed source hashes,
remaining original P24 CPU budget/clock, disjoint CPU and exact source/CI seal.
It then calls the existing `parent_files` and `mixed_batch.qualify_all` on two
separately saved export roots. Six real anchors per parent enter the official
processor/collator; hashes/shapes in the actual result prove this consumption.
The two roots refer to distinct episodes, not two windows of a single parent.

Private root/asset/lease locators remain in the original runmap and bound input
files, not this repository. Repeating native qualification would require a
separately valid CPU audit admission and budget; the existing audit is already
CLOSED and its launcher correctly refuses reuse. The public snippet above
requires no such native rerun.

The saved action/physics/event indices retain original committed references,
not achieved future poses or imagined S1-rejected transitions. Original raw
audits, all601-window codec verdicts and source manifests remain authoritative.
The failed first-four gate prohibits bank READY, HF release and any expansion
regardless of this later successful reader audit.
