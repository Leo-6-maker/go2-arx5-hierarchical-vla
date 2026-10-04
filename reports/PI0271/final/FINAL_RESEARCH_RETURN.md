# PI0271 — Frozen-policy late-support and visited-state diagnosis

Authority: `PI0271_FROZEN_POLICY_LATE_SUPPORT_AND_FEEDBACK_DIAGNOSIS_GO_V1`; release `cbdcaf85ab6c25999fe054bf5da5285fadc394a3`; Issue #11 canonical5977478618. Date: 2026-10-04 SGT. This report is interpreted together with the final summary archive and terminal-publication receipts, which are recorded separately to avoid a self-referential archive hash.

## 1. Decision and limits

**The fixed diagnostic experiment is complete at the prediction, label, CPU-replay and support/visited-preservation levels. The evidence is mixed: substantial learning on consumed late support, residual terminal reference errors, and a distinct M1 gap on actual AGG-visited development states.** Final completion additionally requires the summary archive/readback and terminal canonical, as recorded in the final closeout receipt.

AGG's late-support improvement is real and extends beyond the earlier Q0/Q1 panel. It is not sufficient for reliable feedback: the closed PI0270 qualification remains VALID_QUALITY_FAIL, with6/64 reliable cases. Eight-noise Q3 banks retain predominantly empirical mean-reference error, although their sampling dispersion increases. Neither a unique neural root cause nor a remedy by changing seeds is established.

The strongest separate observations are:

- Both tasks improve on Q2/Q3; Q3 prefix MSE and empirical mean-error squared improve in all24 family summaries for each task.
- M3 terminal support remains problematic despite having been consumed by AGG:35/48 two-noise outputs point against the teacher's local displacement, versus46/48 for PREFIX. This component panel covers only two TRAIN families.
- On the two fixed AGG-visited development families, AGG has worse prefix XYZ error than PREFIX on all31 usable M1 observations. M3 visited-state position errors usually improve, while later orientation errors and progress remain mixed.
- The Q3 eight-noise empirical mean-error term accounts for a median90.5% of AGG M1 prefix MSE and92.0% for M3. This is a descriptive finite-bank ratio, not an unbiased population squared-bias estimate, a statement about every state, or a stability guarantee.

**One next intervention is recommended, not executed:** a bounded, TRAIN-only, same-input direct trajectory-head feasibility comparison against the current S4 formulation. Its question is whether the residual corrective-reference map on already-consumed terminal/late support can be represented and fitted more faithfully. Keep the data, exact labels, processor, conditioning, frozen-backbone scope and matched budget fixed; do not combine this with new collection, history, loss, camera or WBC changes. A new release must specify and authorize that comparison. `launch_allowed=false`; `NOT_AUTHORIZED`. This choice is motivated by residual errors on consumed support, not by proof that deterministic execution would solve the M1 visited-state gap. No EVAL diagnosis label may enter its training.

## 2. Predecessor closure and actual A0

PI0270 completed all fixed training and evaluation before any PI0271 neural or teacher call. Its terminal canonical is [5978357203](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/issues/11#issuecomment-5978357203), source commit `ab939289864206833b6d7462dcc84c911be3075c`. Complete final summary revision is `c62f4d010328c69f2b8f22fb91bb696cb3bd3539`; final-publication revision is `3f16360d09a53733b66550f40ef573ce88c3cfdb`. Closed model, collection, video and predecessor numerical readbacks were reused.

Actual A0 selected `PROCEED_REGISTERED_DIAGNOSIS`: original/reliable main qualification failed, full64-point safety passed and original ROT-relative LEGACY passed. Its SHA256 is `2effe1b62c67db378f6b4912536ee068fadd80aec552daa4b63a7aa5de0d9261`. The actual460-file execution seal SHA256 is `07d534a679a5128e2525bd47a2de8f993e3ca8d8e6527adc3615a75bc04005cf`. No predecessor quality failure was rewritten.

PI0270's4096 official plus16 canary updates,74040 training heads,27592 inference attempts/110368 heads,288 virtual starts,26240 frame pairs and6784 teacher calls remain on its original ledger. They are not PI0271 expenditure.

## 3. Exact policies, sources and execution

Public model repository: `leoyuer/pi0246-gate-pass-campaign`.

| Policy | Fixed HF revision | Prefix | Complete Trainer tree |
|---|---|---|---|
| PREFIX4096 |953bb358f37e4d788188c9bedcbeef1780f33e5e|arms/PI0269/PREFIX/checkpoint-4096|12aff760d05dd05e0e11a024be3353fb45a5cb98a8b110ae7c7a0da7199609a5|
| AGG4096 |f862abb26041c8e7d355470d6e9d48899f78b9ce|arms/PI0270/AGG/checkpoint-4096|0427b2be852f572ab4f3c4fa343cc56104acbbf0cca9db2e61d60abad3b15f66|

Existing model-only views and complete Trainer readback receipts were reused. Effective ROT statistics SHA256 is `0ea96eaf9ba6f2bfb5c16681542723e9ee0e82281dd07c4e33ed9189799580e9`. Recovery mapping remains `53fe049e406f079e079d03af71e25c0ffd339b587fab7d8e5dc80bebe35fd633`.

All296 A1 inputs have verified actual AGG consumption:96 cells each contribute their first consumed Q2/Q3 source, plus all104 consumed component records. Only8/296 have exact PREFIX consumption evidence;288 do not. Selection precedes any new prediction. Body/wrist RGB, StateV1 and language are the only policy inputs. Source labels/geometry remain outside conditioning. The original FP32 noise bytes and BF16 initial states, native four-step S4, FP32 relative SE(3)/row-rot6d decode,40 samples at0.05s, and full-H40 gate are unchanged.

The original finite evaluator loaded PREFIX then AGG serially and finished all2400 registered outputs. No healthy worker was modified or restarted. The64 saved-state teacher calls ran once with hidden CUDA/four CPU threads; no renderer or episode was created.

## 4. A1: consumed late and component support

The table uses j0/j1 only:1184 total rows,592 per policy. Values are marginal medians in **mm**, except SO3 explicitly given in **rad**. Paired deltas are computed per exact input/noise, not by subtracting these marginal medians.

| Task/source | Rows per policy / families | Full-H40 XYZ PREFIX→AGG mm | 0–0.4s XYZ PREFIX→AGG mm | Paired median prefix change mm |
|---|---:|---:|---:|---:|
| M1 Q2 |96 /24|101.461→10.946|44.872→5.121|−40.624|
| M1 Q3 |96 /24|139.586→14.813|62.112→7.401|−56.152|
| M1 terminal |64 /2|27.250→5.405|9.874→2.007|−8.206|
| M3 Q2 |96 /24|22.383→3.989|7.224→1.779|−5.452|
| M3 Q3 |96 /24|22.719→3.720|7.291→1.680|−5.323|
| M3 entry |48 /2|10.747→8.462|3.364→3.010|−0.727|
| M3 midarc |48 /2|17.714→4.207|4.838→2.327|−2.777|
| M3 terminal |48 /2|37.760→24.059|11.809→8.954|−2.689|

Full-H40 SO3 Q3 medians improve .348465→.046096rad for M1 and .022931→.011979rad for M3. Component improvements are descriptive two-family results, not24 independent families or new task qualification. Existing Q0/Q1 results remain a separate panel with a distinct noise registry: their comparison to A1 is not a same-noise causal contrast. PI0270 Q0 XYZ had small regressions while Q1 improved strongly; the new result does not erase that heterogeneity.

Commands also differ. M1 Q3 command0 prefix XYZ is58.148→5.455mm and command1 is129.267→10.060mm (48 rows each). M3 Q3 command0 is8.476→1.544mm and command1 is6.672→1.690mm. Complete command/family/noise denominators, suffix, endpoint, rotation and signed displacement remain in the tables.

AGG's M3 terminal signed direction is still negative in35/48 rows:20/24 for command0 and15/24 for command1; median cosines are−.658 and−.258 respectively. PREFIX has46/48 negative directions. All48 directions are defined under the fixed1e-9m reporting guard; no undefined hold case was assigned a favorable cosine. This is a residual local-reference defect on consumed component support, not proof that these directions alone cause every M3 entry failure.

First-point error is a real tradeoff: M1 Q3 median actual-state XYZ bias increases0.279→1.905mm, while its .4s endpoint teacher error improves96.948→9.740mm. M3 Q3 first-point bias increases0.126→0.256mm. No active value was overwritten, projected or smoothed. The first-point effect is insufficient by itself to explain the whole closed-loop failure.

## 5. A2: finite eight-noise error decomposition

All96 Q3 inputs have all8 draws for both policies. A1 j0/j1 are reused;1152 additional queries supply j2..7. All192 banks have finite valid rotations and the FP64 identity is checked per horizon/region. Finite gate rejection remains in the bank. No K reduction, resampling, action averaging or deployment change occurred.

Each entry below is the **median across48 input banks per task** in mm². The three separate medians need not sum; the identity holds within every individual bank. SO3 remains separate geodesic error in radians.

| Task / region / policy | MSE mm² | Empirical mean error² mm² | Within-bank dispersion mm² | Median per-bank mean-error fraction |
|---|---:|---:|---:|---:|
| M1 prefix PREFIX |3850.249|3850.115|.109|99.997%|
| M1 prefix AGG |48.459|40.529|4.261|90.510%|
| M1 .4s endpoint PREFIX |9363.558|9363.353|.251|99.998%|
| M1 .4s endpoint AGG |115.286|92.564|6.597|91.557%|
| M3 prefix PREFIX |52.871|52.761|.112|99.793%|
| M3 prefix AGG |2.705|2.500|.181|91.996%|
| M3 .4s endpoint PREFIX |136.313|136.132|.243|99.817%|
| M3 .4s endpoint AGG |6.079|5.534|.412|92.205%|

For first/full/suffix/per-horizon quantities see `K8_BANKS192.json` and `K8_REGIONS960.csv`. Full-H40 AGG mean-error fractions are94.654% for M1 and86.080% for M3. For each task, all24 family-level paired medians improve in prefix, endpoint and full MSE and empirical mean error², while all24 show increased dispersion. These are family-level descriptive counts, with two commands combined within each family; no significance test treats noise or horizon points as independent scenes.

The residual error has a strong empirical mean component under this fixed bank. Dispersion is nevertheless nonzero and increases, so calling sampling irrelevant would be wrong. K8 was measured on consumed Q3 inputs, not all component states or the A3 visited states; its percentages cannot be transferred to those panels.

## 6. A3: actual AGG-visited states

The first two unchanged development-EVAL families028/029 provide all64 saved observations at query0/25/50/99, times0/10/20/39.6s. AGG's64 original predictions/noises are reused exactly; PREFIX supplies64 new same-input/same-noise predictions. Actual camera/TCP/time/StateV1 identities are verified. These are local comparisons, never a counterfactual PREFIX rollout.

Original same-pose teacher labelling yields63 usable and1 unusable proposal. Family028/M1/c0/E0/q99 has a label maximum linear speed2.782519m/s, above the unchanged2.0m/s gate. Its observation, label payload, both policy predictions and actual trajectory remain preserved; teacher-relative metrics are unavailable. The denominator remains64 proposals/128 policy rows. The63 usable labels and the rejected one are permanently DIAGNOSIS_ONLY, never TRAIN or blind TEST.

| Task/time | Usable per policy / proposed | Prefix XYZ PREFIX→AGG mm | Prefix SO3 PREFIX→AGG rad |
|---|---:|---:|---:|
| M1 0s |8/8|1.937→3.003|.020865→.005369|
| M1 10s |8/8|6.725→10.320|.011941→.015285|
| M1 20s |8/8|7.967→13.219|.017313→.030599|
| M1 39.6s |7/8|6.409→12.056|.044713→.059832|
| M3 0s |8/8|1.818→2.348|.036337→.009150|
| M3 10s |8/8|4.790→2.675|.014324→.013056|
| M3 20s |8/8|8.586→5.850|.025370→.031236|
| M3 39.6s |8/8|11.161→7.491|.033941→.043624|

M1 prefix XYZ worsens in all31 usable paired observations:15/15 in family028,16/16 in029. Median paired increases are5.187mm and3.939mm respectively. For later M1 observations, both commands show worse XYZ/SO3: command0 has11 usable proposals and command1 has12. AGG's local direction is negative in8/11 and6/12, compared with PREFIX1/11 and0/12. At10s/20s, AGG's median predicted terminal-distance changes are+.663mm/+.400mm, while the teacher changes are−14.510mm/−20.347mm. At39.6s AGG median progress is only−.347mm versus teacher−18.233mm on the7 usable proposals.

M3 is mixed: family028 improves prefix XYZ in10/16 pairs and029 in13/16, but their prefix SO3 paired medians worsen. At39.6s AGG's median signed teacher-direction progress is approximately−.016mm, despite improved position error relative to PREFIX. Terminal distance and a cosine never substitute for ordered M3 path qualification.

The A3 result supports a remaining M1 state/conditioning-distribution gap. Counterevidence to a broad failure claim is strong improvement on consumed late support and improved M3 visited-state position. Only two opened development families and one saved noise per observation were used. Scene transfer, visited-state distribution and conditioning behavior are not causally separated here. This does not prove a unique representation bug or authorize training on these labels.

## 7. Reference, installation, achieved state and prior reliability

`VISITED128.csv` retains original AGG prefix-executed flags, actual achieved target-error changes, same-tick tracking and installed endpoint errors. The full6400-row predecessor transition table is reused by hash. PREFIX's counterfactual output is never presented as an achieved state. A teacher endpoint is a diagnostic target, never installed to repair a model prediction.

PI0270's E0 same-tick XYZ/SO3 tracking is exactly zero in all32 AGG episodes despite failed reliability. E1 episode RMS tracking median is1.503mm/.002263rad. Tracking error is therefore not necessary to explain E0 failures; neither mathematical executor establishes WBC transfer. The existing M1 late drift, M3 missing-entry28/32 and two first-completion successes with later recorded reversal remain unchanged. Neither local fitting nor the finite error identity supersedes these actual task outcomes.

## 8. Safety, literal budgets and numerical evidence

A1 has PREFIX592 ACCEPT, AGG584 ACCEPT/8 WARN. Across all2336 support predictions, PREFIX has1168 ACCEPT; AGG1133 ACCEPT/34 WARN/**1 REJECT**. All128 A3 policy outcomes are ACCEPT. The single rejected model output is AGG `u0311-m1`, j6, family009/M1/c0/Q3: angular speed4.427936rad/s exceeds the original4.0rad/s rejection threshold. It remains in the finite K8 calculation, every error table and the archive; no favorable retry or rollout installation occurred.

| PI0271 actual consumption | Actual | Planned / hard ceiling |
|---|---:|---:|
| A1 new predictions |1184|1184 fixed|
| A2 additional predictions |1152|1152 fixed|
| A3 new PREFIX predictions |64|64 fixed|
| All neural attempts |2400|2400 /2560|
| Native S4 heads |9600|9600 /10240|
| Teacher calls |64|64 /80|
| Technical neural retries |0|reserve unused|
| New training/canary/backward |0|0|
| New episodes/renders/WBC/physics/hardware |0|0|

CPU replay independently reconstructs all2400 new S4/decode arrays and physical metrics; the original gate is shared and this limitation is explicit. Old AGG A3 numerical replay is reused, not charged as64 new queries. All192 K8 banks are available. `FINAL_ARTIFACT_AUDIT.json` matches the full query plan, attempts/heads, first-valid output receipts, all64 teacher records, table dimensions/hashes, all predictions and label payloads against the two complete public archives. Summary archive and terminal publication are separate requirements.

## 9. Preservation and evidence index

Public repository: `leoyuer/pi0246-gate-pass-campaign`. Both parts have full anonymous fixed-revision file and tar-member readback; no historical model archive was redownloaded.

| Part | Fixed revision | Prefix suffix | Members | Tree SHA256 |
|---|---|---|---:|---|
| Support + frozen noises |5b5dc9bc65dec326c9f4d758a8920a4268d00a29|support|5840|e40157c532d2fe880f14c8ffa8b1a518571f4c2ef0524dd93827a31353a6a1f8|
| Visited predictions + all labels |5fa779e7b8dcbe2ec5949baf9c169a377b15a921|visited|256|fb4fd3ab7871cbaeb83b6cb3134caa8bb7e5649a6dd70abfcc5ddb9df5634c30|

Prefix root: `evidence/PI0271/frozen-policy-diagnosis-20261004/`. The summary archive contains the full delivery tables, source/cards/contracts/model bindings, execution and archive proofs, and this report. Its receipt is separate. Actual space checks passed; user quota remains UNKNOWN. There was no cleanup, model-copy retirement, monitor, email or merge. PI0270's saved-video artifact is reused with its VIRTUAL/NOT_WBC/NON_REALTIME labels; PI0271 generated no new video or motion episode.

Key SHA256 identities:

- Query plan: `e0af41a3710280b7a870816bb43fae228b4597ce167265a6e5d18ca384e466ac`.
- All2400 output completion: `5b43ee8368b5f95ad39e2b7aa828206da9def8915e9f74f94af08f9491e22875`.
- All64 label completion: `d27ac5495594e9d982325e050e123822293f9cf03d8e422c211f4734c568c53e`.
- Full new CPU replay: `e384c4613df573d12d332d8b6ac9a558c0734586888022f340a905485eff6d31`.
- Full artifact/budget audit: `efc1d2bb3312423c522c94d264832ae89ff4d05e9172772fb15631edca8b8927`.
- K8 bank arrays: `5d24827a25a262f77d1764989c5aef86735072ff783df16552c13ff5ecfba5e4`.
- Descriptive source-bound summary: `3b2fc704ec71cfb7bedb94503df3d89c341658a49687bfc819c3023b8250668f`.

The complete row files are `A1_SUPPORT1184`, `SUPPORT_ALL_NOISES2336`, `A1_PAIRED592`, `PAIRED_SUPPORT1168`, `K8_REGIONS960`, `PAIRED_K8_REGIONS480`, `VISITED128`, `PAIRED_VISITED64`, plus strata/family tables and `K8_BANKS192.json`. All denominators and finite/invalid outcomes are retained. Direction vectors and continuous progress are available alongside signs; signs alone are not a new success criterion.

## 10. Terminal scope

After the required summary/readback and single source-bound terminal canonical, return:

**PI0271_LATE_SUPPORT_FEEDBACK_DIAGNOSIS_COMPLETE — STOP_FOR_SINGLE_TARGETED_REPAIR_OR_TRANSFER_DECISION**.

This is completed diagnostic science with unresolved feedback limitations. It does not convert PI0270 or PI0269 valid quality FAIL, PI0268 FAIL, or PI0267 secondary WBC PARTIAL into PASS. It does not authorize the recommended next intervention, independent transfer, another seed, new training, physical execution or real-time claims. `selected_checkpoint=NONE`; `deployment_selected_checkpoint=NONE`.
