# PILOT1 contact model1: real training, poor independent-parent generalization

2026-11-05, Asia/Singapore. This is the successor to the
[sealed local-loop / first-contact report](PILOT1_LOCAL_LOOP_AND_CONTACT.md).
Frozen plan remains `fec841e74d406b3b556c84b733d2de1615581bf7`.

C03 independently reproduced the new touch/withdraw curriculum. Its cube starts
1mm further in world X than TRAIN C02, with unchanged robot, support, WBC and
original safety. Both actions ran from independent native initializations under
source `b1a0216dd86d6b444abc976c69126d796de974e9`; qualification retained source
`b7c75b045b09aff60ec568b2ea4d34cb6b5e6c99`. Qualification and both controls passed
native checks, exited0, shut down cleanly and closed their leases.

| C03 action | Away-Y TCP motion | Terminal noncontact | Table support | Fixed exit predicate |
|---|---:|---:|---|---|
| HOLD | 0.546349mm | 0.056s | true | fail |
| +Y1mm | 0.970603mm | 1.048s | true | pass |

Each control contains128 actual task atoms /5,703 total native ticks.
HOLD has99 positive /29 negative task atoms; +Y has72 /56. This is a second
physical-parent action contrast, not a trained-policy evaluation, exact restored
fork or statistically reliable estimate. C03 was designated DEV before this
execution. [Actual C03 results and costs](CONTACT_C03_DEV_SUMMARY.json).

## Actual model, data and complete checkpoint recovery

Model source: `e547d150ca6c10a151bf5a3b89619b6d02b36646`.
It uses nine actual observation rows and a known next reference to predict the
next50ms TCP/object displacement and contact-any event. Current contact features
use actual native forces, available equally to the learned heads and persistence
baseline. The next reference was fixed by the native program before integration;
future achieved state, receiver/solver memory, phase, outcome and score are not
inputs. The choice-blind control replaces only the new next reference with the
past reference, preserving all actual history. It does not erase past actions.

The two64-wide MLP arms used matched seed36003, AdamW0.001, batch32 and2,000
main updates each. Normalization is TRAIN only. All actual25 native contact
labels per atom are checked against the original0.05N threshold. This is a
one-step, partial task-space model, not a full-state or free-rollout simulator.
No robot rotation, joint-state dynamics or complete MPC/WBC state is fabricated.

| Immutable artifact | SHA256 |
|---|---|
| C02 TRAIN bundle,3 real episodes /352 raw atoms | `281d8421bc5532dc1f48367558d8c1832b29a11c68e2e1ae3e0d313cdd41ba3f` |
| C03 DEV bundle,2 real episodes /256 raw atoms | `c7dc2ea71f904bf0a6a70e19aefdd0b08f886842299b46aba97d75db9bbe57cd` |
| Action-conditioned checkpoint2000 | `75ae8bb65879f7bed10bc768e3e9b49307995c170ea8ce57d5cfaf233fa41100` |
| No-new-reference checkpoint2000 | `bc44d5147db9d9dd81c48fc8578d7e4076aa2e1258e84f490268b10a52479523` |
| Actual execution spec | `c7eded54fc6e75539e8dfef82fee02a37eea6c62ad90034ad31482af268ff566` |

After nine-row history requirements, there are325 TRAIN transitions from one
physical parent and238 DEV transitions from a separate parent. TRAIN has237
positive /88 negative labels; DEV has171 /67. Five correlated episodes do not
constitute563 independent trials. No TEST was loaded or fitted. Raw bundles,
checkpoints and host-bound specifications remain private.

Both arms completed2,000 actual main updates plus one separate restore-probe
optimizer call each. Reconstructing model, optimizer, normalizer, RNG and cursor
from checkpoint1999 reproduced2000 exactly; full continuation and endpoint
reload errors were zero. The fresh original-registry PC model-only job exited0
and its lease is CLOSED. There were0 native steps and0 Actor updates in this
training job. Actual implemented CLI:

```bash
PYTHONPATH=src:. "$MODEL_PYTHON" -B -m scripts.wm02.train_pilot1_contact_wm --spec "$SPEC" --output "$FIT"
```

Training CPU32.420606s, wall17.389138s, peak RSS329,260KiB. Including worker
startup/imports:37.085196 child CPU seconds /22.077456 wall seconds. This was
CPU training under the existing owner-granted LOCAL_PC_PILOT1_CPU route, not a
claim of GPU execution. No Core filesystem or GPU admission guard was bypassed.

## Negative predictive result

| Independent C03 DEV predictor | TCP displacement RMSE, mm | Object displacement RMSE, mm | Contact Brier ↓ | Contact accuracy |
|---|---:|---:|---:|---:|
| Action-conditioned | 0.417965 | 0.182973 | 0.295737 | 69.33% |
| No new reference | 0.258026 | 0.166578 | 0.279301 | 70.59% |
| State / event persistence | 0.292349 | 0.003200 | 0.025210 | 97.48% |
| Constant velocity / event persistence | 0.273490 | 0.002756 | 0.025210 | 97.48% |

Both learned heads fit TRAIN contact labels100% while missing70–73 DEV
positives. This is poor generalization; neither head is deployed for action
selection. State persistence is a strong baseline and most intervals have no
event change. A next comparison must report event transitions separately from
steady intervals, rather than treating aggregate accuracy as decision utility.
[Full model and baseline results](CONTACT_WM1_SUMMARY.json).

A readonly feature audit found some DEV object angular-velocity features at
about12.7 TRAIN standard deviations versus TRAIN maxima about2.9. Other large
normalized features were also large in TRAIN; an outlier maximum alone is not
a causal explanation. The observed contact-point/dynamic distribution shift
and one-parent TRAIN support a new coverage hypothesis, not a WBC repair or a
reason to delete this negative model.

## Cold-boundary totals and next work

Through completed C03 controls:36 actual native attempts,35 native PASS and one
preserved recorder failure;153,333 ticks /2,529 raw task atoms. Native
own-plus-child CPU2,395.427350s; summed worker wall1,291.068144s. All36 leases
are CLOSED. Registration failures retain their separate0-physics denominator.
Actor totals remain4,000 main updates +2 restore probes. WM totals are8,000
main updates +4 restore probes (4,000 residual +4,000 contact). Full manipulation
successes0; contact-policy training/evaluation0; H200 / GPU physics jobs0.

175 world-model tests and12 subtests passed;8 affected tests passed under-O.
The latter includes complete training/recovery and future-achieved-state leak
tests. These checks are not native qualification.

The successor coverage panel adds TRAIN C04/C05 with cube X0.264/0.267m,
keeps exposed DEV C03 at0.266m, and reserves unseen TEST C06 at0.2655m. Other
cube coordinates and support remain unchanged. C04/C05 separately passed new
static checks; their native qualification/action coverage is additional work
outside the totals above. Contact execution remains one native environment,
with actual fresh owner registration, original safety, finite endpoints and no
hot capsule edit. The original free-space TEST P23/P24 remains unexposed too.

Next: seal the finite C04 qualification, audit actual entry before admitting its
HOLD/+Y controls, then continue C05. Expand TRAIN coverage and preregister the
next model comparison. Multi-step predictive qualification is still needed
before a learned contact policy can rely on this model for the3.2s exit task.
