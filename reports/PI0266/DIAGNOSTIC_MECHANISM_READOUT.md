# PI0266 fixed16 actual teacher comparison

Status: diagnostic panel complete; fixed96 FEEDBACK_TRAIN collection in progress. No GR00T queries or optimizer updates. Formal selected/deployment remain NONE. Historical PI0265 PARTIAL/FAIL is unchanged.

| Fixed expert | Valid episodes | M1 physical task criterion | M3 physical task criterion | Safety failures | Valid local windows |
|---|---:|---:|---:|---:|---:|
| CLOCK_DRIVEN | 8 | 4/4 | 0/4 | 0 | Not collected by this diagnostic arm |
| OBSERVABLE_CORRECTION_V1 | 8 | 0/4 | 0/4 | 0 | 800 |

These task counts are not complete candidate qualification. The four successful CLOCK M1 prefixes still have SO3 tracking RMS above0.10rad. Every case and command remains visible in [TEACHER_QUALITY.csv](diagnostics/TEACHER_QUALITY.csv).

## Measurements that constrain the next action

- CLOCK M1 final target distances are3.93–6.42mm; observable M1 final target distances are44.11–137.09mm. The latter has same-tick installed-reference tracking RMS of4.57–9.05mm and SO3 RMS of0.044–0.047rad, while failing the task. Close tracking of a current-anchored local reference is insufficient evidence of reaching the global task target.
- CLOCK M3 final target distances are3.34–4.42mm, but none receives an admitted directed-arc entry or credited bins under prospective V2. Observable M3 finishes43.23–76.08mm from its target and also has zero credited bins. Endpoint proximity, source phase and ordered physical traversal remain separate quantities.
- Observable M1 command1 stays at projected source phase0 in both DEV families. Observable M3 command0 stays in nominal quartile0; command1 lacks nominal quartile2 in both families. Family000 M3 command1 has a12.813-source-second projection jump. This is a teacher projection result, not proof that the robot traversed those intervening phases. Actual TCP displacement and every nominal bucket are retained in [DIAGNOSTIC_PHASE_SUPPLEMENT.json](DIAGNOSTIC_PHASE_SUPPLEMENT.json).
- Safe task failures supplied800 desired-reference windows. They are named WINDOW_VALID_REFERENCE, not successful full-task demonstrations. These DEV windows are excluded from TRAIN. TRAIN admission will use only the separate fixed96 collection, including all actual failures and its own family/task/command/quartile denominator.

The comparison does not identify a unique WBC root cause or establish that feedback fine-tuning works. The frozen expert, controller, geometry and data plan continue unchanged. No full-task diagnostic PASS is required before collecting the authorized TRAIN windows.

## Integrity and media

The same-owner CPU replay reconstructs every saved native state through a distinct full-URDF traversal and reruns the shared fixed scorer. All16 results match; scorer independence or an external reviewer signature is not claimed. The additional source-AST negative check exercises all three actual native termination routes against false post-reset goal hits.

The448-frame video uses actual fresh96x96 body/wrist pixels, explicit4x playback and labeled paused final frames. No pose placement, generated images, model inference or synthetic playback was used. The diagnostic7-file HF PUBLIC package has anonymous full-file readback at revision `34de852ebb889890bde06f1557e863fe8a29fdb8`:

[Actual teacher video](https://huggingface.co/leoyuer/pi0246-gate-pass-campaign/blob/34de852ebb889890bde06f1557e863fe8a29fdb8/evidence/PI0266/diagnostic-panel-20261001/PI0266_ACTUAL_TEACHERS.mp4)

All-attempt raw archival remains pending completion of the fixed96 collection. An SSH driver exit after six valid TRAIN cases was confirmed from process state; the finite driver resumed those first-valid cases without repeating them. This changed transport lifetime, not any scientific setting or budget.
