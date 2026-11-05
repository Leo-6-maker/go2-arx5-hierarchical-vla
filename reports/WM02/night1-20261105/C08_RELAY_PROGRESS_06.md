# C08 native clock acceptance and the fixed successor relay

This is a cold-boundary and in-progress report, not the final NIGHT1 closeout. The original dispatch cutoff08:36:08.657 and deadline08:56:08.657SGT are unchanged. At a sealed boundary the existing AUTO1 owner read successor commit **59aa9e4d12208e3e51de8fc0d84ab7fa93ebe6cb**, recorded primary PARTIAL with the minimum data/recovery milestones achieved, and updated its existing state index with G1–G8. No new campaign, owner, scheduler or active-source update was created.

## Real new native acceptance

The existing SSH endpoints recovered. Both readonly probes verified actual Core UID/host, no C08 root/service/registry identity, and the unchanged active campaign/window. Only then was `night1-c08-clock-build-01` launched once from source **93362d53497fa1aaee58ff03e091153e6c893a6c**. Build exit0, leaseCLOSED, wall28.595596s, systemd CPU27.437140s. The child Python CPU0.048474s excludes compiler children and is not total build CPU.

Actual new native binary SHA-256: **29b261541b9dc2b2ed5c6df065026b33a11c3019181b9a8cfa980a7878d91df2**. Frozen controller remains **9e8ce46837d09bc39ae04d4b49f1ce4dd8c8c2d31bea39eac56c5806d108c1f3**. WBC/leg weights, gains, costs, solver,5ms/20ms control periods,2ms integration and safety limits are unchanged.

All11 new cold/hold/reference stages passed with native exit0/CLOSED. The new original24-atom single/dual, late-decision dual and allP01–P08 single qualifications also passed. That includes P03, where C07 stopped at native step288 before task actions. These are new qualified conditions after a substantive clock repair, not deletion or retry selection of the old failure.

[Actual trace comparison](NATIVE_CLOCK_COMPARISON_06.json): old failed C07 had160 policy begins in25 physical20ms buckets,135 repeats and minimum interval0s. C08 hold5s had250 begins in250 buckets; its EEF-y check271 in271. Both have zero repeated buckets, minimum interval~.020s, maximum PD gap3 native ticks and zero q/dq mixed versions. This supports the timing repair; it is not a formal task-success or hidden-state-restore claim.

Fresh C08 production uses its own scene-asset namespace and executor identity, hold8→chosen action→ramp8,24 atoms per episode, preregistered32 episodes. Passed whole batches only will be sealed; a failed peer invalidates its whole pair. Existing P01–P08 physical parents are not counted again as independent parents. The finite frontend stops preparing new work before the original cutoff with its worker margin; remaining entries become PENDING_NEXT_WINDOW.

## G1: actual full production-population coverage

Audit source **10a32284af686010cfe983819026403846906bdd**; [42-episode input identities and coverage](PRODUCTION_POPULATION_AUDIT_06.json). The snapshot contains all42 then-sealed production episodes /3,600 atoms, excluding10 failed/invalid production episodes. It does not include later C08 production. All original scene/state/command/result byte identities and whole-peer qualification are checked; malformed or nonunit recorded quaternions fail. Maximum unit-quaternion norm error was1.33e-15. Reading uses actual saved state and mj_forward with all integration APIs denied.

There are eight RootSafe physical parents,6TRAIN/2DEV, plus the separately labelled same-parent engineering and original episodes. Maximum task TCP movement40.922530mm, BASE21.681073mm; object movement and all pad/object contact/grasp counts are0. Minimum TTL margin.044s. Thus stationary-object error and historical state prediction can dominate; these data cannot prove a contact model or full task. The audit cost16.572946CPU s /17.609794wall s, zero new integration/update. The original primary full long-episode panel remains PARTIAL.

Final code tests:131 world-model tests pass; both population selection/unit tests also pass in Python-O. The test count is software evidence, not native qualification.

## G4/G8: finite recipes and an actual frozen-model derivative

[Finite C08 relay registration](../../../../experiments/wm02/night1_c08_finite_relay_v1.json) declares two200+1 arms, RAW and scaled-translation input, with one changed mechanism, fixed seed/data/normalizer/exposure. The newly requested finite successor scope already contains700 main updates+4 probes. Therefore neither new arm starts in this relay. The historical three-arm2000-step plan remains preserved and unexecuted; it is not silently treated as two new completed models. GPU1 availability is not the reason for holding these fits.

A zero-update CPU audit loaded the actual C06 endpoint **c32cd16ac30782d3e1efe135c922f3134ae1dd100c4515a48de49dcfb20f438f**, replay **a545cb4fa0c94e6e177bf2a74817afbd6e1c4cd0a2979f5d6aa12e9e82089fe1**, source **90486476eb726f74a191478fad8e8dc7baa4907d**. It fixed the actual past installed goals and differentiated only the hypothetical future goal, on the first saved window of each of the2DEV parents, horizons1/8/16. A float64 diagnostic copy of the frozen float32 weights has finite derivatives; central differences at±0.1mm agree within1.18e-12 absolute m/m. [Complete Jacobians](FROZEN_MODEL_ACTION_DERIVATIVE_06.json).

The largest tested coordinate derivative is only~1.12e-4m/m: a local1mm goal change corresponds to about0.000112mm predicted coordinate change. This is mathematical differentiability with weak local action sensitivity, **not empirically verified native intervention response**, and these future-only interventions differ from the C06 fixed-goal training condition. It does not establish action selection, causal value or a usable Actor gradient; it is not transferred across the changed C08 executor. Measured derivative section CPU.182692s/wall.258790s, excluding imports; no optimizer/native/Actor updates.

G2 still lacks a qualified dynamic approach despite the saved feasible static upper-object geometry. G3 prospective P09–P16 await compatible frozen model/control evaluation; they are not sampled to manufacture a second DEV set. G5 is blocked by unverified intervention response; optional G6 is not run at the expense of collection. G7's two private same-source videos are already complete and are not regenerated. G8 remains **NOT_READY_FOR_FORMAL_POLICY**: deployable observations, native standing-task score binding, contact/release/withdrawal and full controller-state restoration are incomplete. Formal Actor/PPO/GR00T updates and complete tasks remain0.

## Commands and resource snapshot

Actual PC frontends, with private paths already bound:

```bash
python3 "$WM02_PRIVATE/run_c08_g1_panel.py" "$WM02_PRIVATE/C08_CLOCK_G1_G2_PANEL.private.json"
python3 "$WM02_PRIVATE/run_finite_panel.py" "$WM02_PRIVATE/C08_DECISION_QUAL_PANEL_A.private.json"
python3 "$WM02_PRIVATE/run_c08_finite_panel.py" "$WM02_PRIVATE/C08_DECISION_PRODUCTION_PANEL_A.private.json"
python3 "$WM02_CORE_ROUTE" "$WM02_PRIVATE/audit_population_units_core_readonly_once.py" 210
python3 "$WM02_CORE_ROUTE" "$WM02_PRIVATE/audit_frozen_c06_action_gradient_readonly_once.py" 85
```

Qualification frontends and both actual read-only audits exit0; production is still in progress at this snapshot. Full-source storage inspection hit its bounded40s limit and failed without cleanup. The focused output/asset inventory excludes source capsules:31,922 regular files,16,462,217,976 logical bytes /16,554,831,872 allocated bytes at08:28:09SGT. This is neither whole-host usage nor final NIGHT1 storage. GPU snapshot then: GPU0 total32,760MiB/free8,093, GPU1 total16,376/free15,899. No other user's tasks were modified and no new learner was launched. Final total native ticks, CPU and closed-worker reconciliation follow after this production panel; no billing price is assumed.
