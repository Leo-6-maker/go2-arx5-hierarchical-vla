# F1 saved-query diagnostics — no new generation

The12 completed F1 episodes were checked using their saved input/action hashes,actual query clock,previous emitted width and complete-atom command records. No new model call,training update or physics integration was made. These student states have no matched expert timing labels;the width0.022m threshold below is a descriptive crossing,not a new omission score.

| Condition | Outcome | FirstH40 crossing query | FirstP10 crossing query | Horizon-only queries | Max reference/actual error(mm) |
|---|---|---:|---:|---:|
| E2-F1-T0-B0 | FAILURE | None | None | 0 | 32.85 |
| E2-F1-T0-B1 | SUCCESS | 58 | 59 | 1 | 42.41 |
| E2-F1-T0-B2 | SUCCESS | 58 | 59 | 1 | 42.41 |
| E2-F1-T1-B0 | FAILURE | None | None | 0 | 30.91 |
| E2-F1-T1-B1 | SUCCESS | 55 | 56 | 1 | 38.44 |
| E2-F1-T1-B2 | SUCCESS | 57 | 58 | 1 | 38.46 |
| E2-F1-T2-B0 | SUCCESS | 57 | 58 | 1 | 39.33 |
| E2-F1-T2-B1 | SUCCESS | 53 | 54 | 1 | 30.95 |
| E2-F1-T2-B2 | FAILURE | None | None | 0 | 31.60 |
| E2-F1-T3-B0 | SUCCESS | 56 | 57 | 1 | 39.27 |
| E2-F1-T3-B1 | FAILURE | None | None | 0 | 27.66 |
| E2-F1-T3-B2 | FAILURE | 58 | None | 1 | 38.50 |

All seven successful F1 episodes have one horizon-only closing query,followed at the next query by a crossing in the firstP10. This does not support repeated horizon-only postponement as the failure mechanism in this subset. B0/T0 never predicted a crossing before contact. B2/T3 predicted one in query58’s suffix but hit contact before closing entered a consumed prefix. The episodes were interrupted;this is not proof of when the model would have closed in an unexecuted continuation.

Initial accepted-reference/actual-EEF errors are approximately27.3–27.7mm in both successes and failures;some successful episodes have larger peak errors than failed episodes. The scalar error alone does not establish a causal WBC or frame defect. Exact controller-selected references and full control-state restoration remain unknown. No reference-error input,action processing,WBC or safety threshold was changed.

Use these concrete states together with the complete24-condition result when deciding E3. They do not authorize correction collection by themselves,and do not turn nearby historical expert plans into valid same-state BC labels. Per-episode summaries and input/result provenance are in [the numeric record](QUERY_DIAGNOSTICS_PREFIX_ACTUAL.json).
