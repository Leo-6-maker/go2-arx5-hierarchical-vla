# PI0269 — complete original-start panel, final study still pending

Status: `COMPLETE_MAIN192_CPU_REPLAY_COMPONENT_AND_FULL_NATIVE_REPLAY_PENDING`.
This is an actual result return, not the PI0269 completion claim. All three final-policy identities were frozen before quality observation. No policy was rerun for this return.

| Point | M1 E0 /16 (c0,c1 /8) | M1 E1 /16 (c0,c1 /8) | M3 E0 /16 | M3 E1 /16 | Reliable /64 | Full64 safe |
|---|---:|---:|---:|---:|---:|---|
| ROT2048 donor | 11 (7,4) | 11 (7,4) | 0 | 0 | 0 | true |
| UNIFORM4096 | 15 (7,8) | 16 (8,8) | 0 | 0 | 0 | true |
| PREFIX4096 | 15 (7,8) | 15 (7,8) | 0 | 0 | 0 | true |

Each task/backend retains 16 cases and each command 8. Original qualification needs every task/backend >=13/16 and each command >=6/8, plus full64 safety. Both new points meet the M1 subgroups but fail original full-task qualification because M3 fails. All three fail the unchanged completed-by35/full40s/every-tick35..40s reliability rule. Safety passing does not imply reliable task completion.

The new CPU pass verifies PREFIX64 saved array and record identities, exact shared V3 score reproduction, independent M3 entry acquisition, original V2 side scores and unchanged reliability. It reuses the already-closed donor/UNIFORM128 CPU receipt after checking all 128 current record/array hashes. Shared-scorer limitations remain: this is not a wholly independent V3 implementation. Full native S4/decoder replay and the complete component panel remain pending in the fixed final pipeline.

The separately closed 2904-row offline panel showed paired PREFIX-minus-UNIFORM FIT prefix-XYZ medians of -0.220427mm (M1) and -0.295505mm (M3). This supports the preregistered local-fit contrast; it does not support reliable task qualification. Both new points regress donor legacy440 median XYZ/SO3. Common donor-to-final M1 improvements include shared continuation/exposure effects; the sole matched temporal-treatment comparison is PREFIX versus UNIFORM, which has no observed original-start M3 or reliability improvement here. These repeatedly developed conditions do not establish blind-test, independent-scene or physical generalization.

Sources:
- `MAIN192_CPU_PRELIMINARY.json`, SHA256 `588dd1a8f026a6202fe4ff902845a141167b17f1161c6a44a5bdeb1785ce5d59`: 192 exact source identities and condition-level retention receipts.
- `main192_analysis.py`, SHA256 `1025b2d1491c27e1d68a8043aa37a21db7a0306c489dbfc11e3a831b5a9a5354`: only saved-array operations, approved scorer/reliability/qualification helpers, and explicit neural-call/optimizer guards.
- Earlier closed `DONOR_UNIFORM_MAIN_CPU_PRELIMINARY.json` and `OFFLINE_ONLY_PRELIMINARY.json`, referenced by exact receipt/source hashes.
- UNIFORM/PREFIX result archives have full public fixed-revision file/member readbacks. Four complete Trainer readbacks remain closed and local full recovery sources remain retained.

Additional model/gradient/teacher/render/physics calls by this CPU pass: 0. No threshold, scorer, output, task start, noise, teacher or checkpoint changed. Component diagnostics continue; they cannot replace these original-start qualification results. PI0269 remains active. Old FAIL/NONE and the old WBC PARTIAL remain. `selected_checkpoint=NONE`, `deployment_selected_checkpoint=NONE`.
