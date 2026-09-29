# Future fixed evaluation proposal — no current queries

Status: DRAFT_NOT_EXECUTION_AUTHORIZED. Actual source IDs and common processor/runtime cannot be sealed until successful data and observability admission exist. This proposal does not authorize any future exposure.

Use both MT-Base and MT-G at3000/6000, a shared checkpoint across all tasks, and two actual saved generation-noise tensors per input. No per-task checkpoint switch or best-of-noise selection. Bind source inputs and noise bytes before querying. Select future fitting inputs by fixed SHA256 order in namespace PI0255_MT_FIT_V1, independently of scores:32 eligible windows from distinct TRAIN episodes per task,96 total. Old O64 remains regression only.

| Future comparison | Fixed allocation | Proposed maximum |
|---|---|---:|
| TRAIN raw trajectory fitting |96 inputs x2 noises x4 checkpoints|768 queries|
| O64 regression |64 inputs x2 noises x4 checkpoints|512 queries|
| Reserved paraphrase contrast |12 MT-VAL inputs,4/task, one alternate paraphrase x2 noises x4 checkpoints|96 extra queries|
| Empty-language ablation |same12 inputs x2 noises x4 checkpoints|96 extra queries|
| MT-VAL closed-loop outcomes |60 registered episodes x2 noises x4 checkpoints|480 physical starts|
| MT-VAL receding policy calls |at most100 calls per physical episode,20s/.20s; no call beyond frozen task end|48000 queries|

Comparison maximum:49472 model queries and197888 S4 inference-head calls,480 model-controlled physical starts. These are separate from future training forwards/backward/update accounting and from today's expert-only360 cap. Invalid-result retry reserve is zero in this draft; any later reserve requires an explicit bounded field. Valid failures remain first-valid outcomes. Counterfactual commands share actual reset state/image/RNG and actual generation noise; alternate wording and empty language are additional interventions, not free diagnostics.

After a single model/checkpoint/processor/runtime is frozen under a later protocol, future MT-TEST contains40 manifest-only scenario families x2 commands x3 tasks x2 noises =480 physical starts, at most48000 queries/192000 S4 heads. Its20 unseen-geometry families and20 composition families remain unrendered/unexecuted here. This separate blind budget cannot be spent during selection or repeatedly reused for tuning.

Report per-task raw legality/precision; actual same-tick reference tracking; and registered achieved task outcome separately. Equal-weight task macro-average and worst-task outcome accompany all per-task/condition/noise counts. Paired semantic correctness uses requested target/side/direction and actual results, not output variation alone. Confidence intervals, if admitted later, resample scenario families, not overlapping windows. Two noises are not independent training seeds.

The immutable PI0255 scorer fixtures cover wrong target/route/direction/order, false sector crossing, contact, reference-only hits, timeout, duplicate times and rotated+translated SE(3). Full task evaluation remains unvalidated on actual successful episodes because current pilots have not passed settling. Observation phase ambiguity, especially full contour completion under current no-history inputs, remains a required admission gap. No current deployable, multitask, manipulation or blind-generalization PASS is claimed.
