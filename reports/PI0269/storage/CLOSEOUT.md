# PI0269 storage-only retirement, 2026-10-03

User authorized low-frequency training wait, evaluation preparation and direct deletion of scientifically retired weights. The active science remains PI0269.

| Host | Logical deleted GiB | Unique allocated GiB unlinked | Free before GiB | Free after GiB | Whole-window free change GiB |
|---|---:|---:|---:|---:|---:|
| Hopper | 252.825 | 233.377 | 123.056 | 336.983 | 213.928 |
| Corelab | 17.750 | 17.750 | 755.081 | 772.830 | 17.749 |

Hopper:13 retired checkpoint trees,39 large file paths. PI0268 OLD1024/2048, ROT1024, superseded failed OLD canary; PI0267 discarded N/R canaries; PI0264 B2048. Corelab:OLD2048 and B2048 model-only views,4 shards. Two N checkpoint-4 trees shared exactly3 inodes with link count2; both links were within the explicit retired set, so deleting both removed one physical payload. No external shared blob was deleted.

Logical deletion and physical allocation differ for hardlinks. Concurrent UNIFORM2048 saving consumes about19.450 GiB during the whole cleanup window; free change is measured separately and is not equated to unlink volume. Hopper uses exact user-fileset `/scratch/e1192548`, not the global filesystem. Quota headroom remains UNKNOWN.

All39 Hopper and4 Corelab deleted paths were checked absent. Small configs, model indexes, statistics, trainer states, RNG/scheduler metadata, training audits, raw accounting and results remain. New local-retirement markers explicitly classify these directories as metadata-only, unsuitable for resume/model loading. Previously published historical HF revisions remain; no useless model was newly uploaded to justify deletion.

Protected:active PI0269 runtime and all four official candidates, complete ROT2048 donor, original L2048 donor, PI0267 N/R final models while secondary dependencies remain unresolved, data/processors/statistics and all unique scientific evidence. Other old payloads were not deleted based on names or age. Corelab frozen evaluation fileset35/35 hashes passed after cleanup; no evaluation query occurred.

UNIFORM2048 completed its actual18-file Trainer save and retained its full recovery source. New Trainer HF archiving is a separate CPU-only operation; upload/readback is still PENDING. Training remains healthy. selected/deployment remain NONE.
