# PI0263 F2000 quality result; X main running

Authority remains commit `1c574b6e6e9d6a7851e64ca57e89c468a4c9eb28`, Issue #11 canonical5913659202. This is a partial scientific delivery, not the pilot closeout. F's valid quality FAIL does not alter X's schedule, loss, noise, optimizer or endpoint.

## Execution and identity

F completed2000 optimizer units/4000 microbatches, job640396 exit0. The complete actual source/template/color/nonlanguage/label-mask/random-packet/mode audit PASS was reported previously. Both full Trainer checkpoints are HF PUBLIC with complete fixed-revision16-file readback:

| Checkpoint | Revision | Tree SHA256 |
|---|---|---|
| F1000 | `768cc6b74700d6e3a849064e3c91694f645bdf0a` | `cdf5155478c885ab77a93e8ff3c823c032c66afb45162ed5e766d69d2fdb1194` |
| F2000 | `4e5f40a4f6fe516003e5ee322a8c92eb7b03a6a4` | `ac62469c76725ed28fd6f7eba99da159b940d89865cdc1ac2f9844d68ac6cd0b` |

Repo `leoyuer/pi0246-gate-pass-campaign`; prefixes `arms/PI0263/F/checkpoint-1000` and `arms/PI0263/F/checkpoint-2000`. Only F2000's four model-view files were materialized on Corelab, with each file hash checked; optimizer/Trainer payloads were not copied to Corelab or PC. No1000 quality queries were made.

F2000's unchanged fixed inference completed1152 VALID unique rows, preserving680 legacy and640 crossed memberships with168 exact overlap. Actual fresh attempts1116, S4 heads4464, exact forward reuses36, no invalid-result reserve or quality retries. P2000's1152 donor rows stay reused without new queries.

The independent computation-path CPU replay of all1152 F rows PASS checked source conditioning, reference/state/noise bytes, saved-tensor hashes, original decoder, metrics and gate; all panel indices/denominators retained. Reviewer is the same sole producer using an independent computation path, not an external signoff. CPU helper gained only a single-point output option; scoring and tolerance code remained unchanged. It writes a separate `F2000_cpu` report and cannot nominate a candidate before both endpoints. Actual code SHA256 `6df59294a2b8937137adb0335351c98a67e10e3741eb0a3afa57bd8afc35492f`.

## Quality: original gate and new gate separately FAIL

Original four KIN-VAL numerical groups all PASS with zero REJECT. Original eight semantic/paraphrase groups pass5/8: M1 semantic B0/B1=9/12,9/12; M1 paraphrase B0/B1=9/12,10/12; M3 semantic=11/12,11/12 and paraphrase=11/12,12/12. Thus `legacy_full_quality_pass=false`, preserving the original>=10/12 gate and all historical failures.

Crossed M1 correct target choices (strict own/opposite H40 XYZ margin, ties wrong):

| Split/template | P2000 B0 | P2000 B1 | F2000 B0 | F2000 B1 |
|---|---:|---:|---:|---:|
| TRAIN T0 /60 |55|57|57|58|
| TRAIN T1 /60 |44|49|48|47|
| TRAIN V0 /60 |53|52|53|49|
| TRAIN V1 /60 |33|36|42|50|
| KIN-VAL T0 /20 |17|14|18|18|
| KIN-VAL T1 /20 |12|14|15|16|
| KIN-VAL V0 /20 |14|16|15|15|
| KIN-VAL V1 /20 |11|12|15|16|

All eight F crossed KIN-VAL numerical groups PASS with zero REJECT, but only T0 B0/B1 meet>=17/20. Therefore `crossed_M1_robustness_pass=false` and `combined_pass=false`. V1 remains unequal across commands: B0 c0/c1=10/10,5/10; B1=9/10,7/10. Both-command family counts for T0/T1/V0/V1 are B0=8/5/6/5 and B1=8/6/5/7 out of10; an aggregate correct count does not establish both-target control.

The original M3 language counts remain the donor's11/11/11/12. M3 original KIN-VAL median XYZ B0/B1=.007915/.005257m, p95=.012574/.012951m. O64 remains poor: median XYZ B0/B1=.182331/.184615m and median SO3=1.092583/1.097824rad, compared with donor XYZ=.183806/.185710m and SO3=1.095150/1.099635rad. Small O64 changes do not establish transfer. Complete per-family/template/color/command/noise rows, wording flips, target-choice g/b decomposition and noise-boundary crossings are retained in `F2000_cpu/`.

F shows gains on some held wording without counterbalancing, and changes of both signs elsewhere. This is a continuation control; extra optimization and the common repeat-pair design can affect it. Counterbalancing's comparative effect cannot be judged until X finishes. These are30 TRAIN/10 development-validation geometry families with correlated template/noise measurements. V0/V1 remain absent from supervision, but this is development validation, not blind TEST, physical or generalist readiness.

## X continuation and actual accounting

Existing X job640622 started03:26:29+08 on Hopper H200 hopper-19 under valid CFP05-CF-052, original P2000 donor/fresh optimizer. No duplicate submit. First-unit actual stock/native S4 parity PASS: raw/decoded/four-step velocity differences0, gate equality, no expert-conditioning. Both real recovery canaries already PASS.

At the live snapshot accompanying this report X completed311/2000, with raw completed2343/4288 (F2000+X311+discarded canaries32). Raw training heads42230/100000 were a separately sampled live counter and may include in-flight work; final totals await terminal accounting. Technical completed-update reserve used0. F inference budget1116/2432 attempts and4464/9728 heads is separate from training heads. Hopper user-root df free494GiB; quota remains UNKNOWN. Corelab GPU0 returned to the other projects'7490MiB after F evaluation, GPU1 unchanged7583MiB; no unrelated process stopped.

The closed F/canary small evidence bundle (17MiB, no checkpoint weights) was relayed to Corelab with SHA256 `7da965726bc668d795df491cb4ac9ef64486a4acfa84d325b4ac89c52658233e` matching Hopper, PC relay and Corelab. An extraction attempted before relay completion failed; after transfer exit0/hash agreement the identical complete bundle was extracted successfully. This was evidence IO only, with no repeated training/query.

Remaining: X2000 and both main packet reconciliation; X1000/2000 complete HF readbacks; X's1152 fixed rows; final3456 CPU/report; prospective saved-array P/F/X video and public evidence readback. Pilot IN_PROGRESS. `synthetic_research_candidate=NONE`, `selected_checkpoint=NONE`, `deployment_selected_checkpoint=NONE`; physics/WBC/TEST/DEV-F/hardware/mail/monitor/merge remain0.
