"""Read sealed initialization trajectories and evaluate independent archived prefixes with the original matcher."""

import argparse
from bisect import bisect_right
from itertools import combinations
from pathlib import Path

from d1_readiness import B2_PIN, baseline, configure, pinned, read, record, rows, sha


def timeline(folder, identity):
    import numpy as np
    from d1_factorial_reference_matrix import check
    from go2arx5_vla.world_model.init_trace_reader import read_trace
    from go2arx5_vla.world_model.native_replay import packets

    trace_file = folder / "init.trace"
    state_file = folder / "state.bin.journal"
    pinned(trace_file, identity["init.trace"])
    pinned(state_file, identity["state.bin.journal"])
    rec = read_trace(trace_file)
    pd = sorted((r for r in rec if r["kind"] == 9), key=lambda r: r["monotonic_ns"])
    before = {r["token"]: r for r in rec if r["kind"] == 11}
    after = {r["token"]: r for r in rec if r["kind"] == 12}
    clock = [r["monotonic_ns"] for r in pd]
    state = packets(state_file)
    ticks = np.arange(2504, 5704)
    indices = np.array(
        [bisect_right(clock, before[t]["monotonic_ns"]) - 1 for t in ticks]
    )
    overlap = np.array(
        [
            i != bisect_right(clock, after[t]["monotonic_ns"]) - 1
            for t, i in zip(ticks, indices, strict=True)
        ]
    )
    check(all(i >= 0 and pd[i]["n"] == 90 for i in indices), "missing applied PD input")
    q = np.array([state[t - 1]["qpos"] for t in ticks])
    commands = np.array([pd[i]["values"][:18] for i in indices])
    commands[overlap] = np.nan
    check(all(state[t - 1]["step"] == t for t in ticks), "journal tick alignment")
    return ticks, commands, q, indices, overlap


def compare(a, b):
    import numpy as np

    ticks, qa, sa, ia, oa = a
    _, qb, sb, ib, ob = b
    q = abs(qb - qa)
    s = abs(sb - sa)
    result = {}
    channels = {
        "leg_target_deg": (np.degrees(q[:, :12].max(axis=1)), 0.1),
        "arm_target_deg": (np.degrees(q[:, 12:].max(axis=1)), 0.1),
        "leg_actual_deg": (np.degrees(s[:, 7:19].max(axis=1)), 0.1),
        "arm_actual_deg": (np.degrees(s[:, 19:25].max(axis=1)), 0.1),
        "base_origin_mm": (np.linalg.norm(s[:, :3], axis=1) * 1000, 1.0),
    }
    for key, (values, threshold) in channels.items():
        first = ticks[values > threshold]
        result[key] = {
            "threshold": threshold,
            "first_tick_above": int(first[0]) if len(first) else None,
            "max_first_200_ticks": float(np.nanmax(values[:200])),
            "at_5703": float(values[-1]),
        }
    return {
        "channels": result,
        "intra_tick_PD_change_either": int(np.sum(oa | ob)),
        "new_PD_callbacks_first_200_ticks": [int(v[199] - v[0]) for v in (ia, ib)],
    }


def prior_population(root, reg, anchors):
    from d1_factorial_reference_matrix import (
        boundary,
        check,
        initial_features,
        prior_pd,
        verify_initial,
    )
    from go2arx5_vla.world_model.native_replay import packets

    base = root / "v14/history_qualification/run04/HISTORICAL_TRACKING_ACTUAL.json"
    pinned(base, "71aaff6944bdb868b03ce6e813fccf477327cdf781c00be0209f97f294c3e7fb")
    population = read(base)
    s0 = read(root / "v13/run01/S0_ACTUAL.json")
    spec = read(
        next(Path(k) for k in s0["inputs"] if k.endswith("/EXECUTION_SPEC.json"))
    )
    last = Path(spec["mobile_interface_qualification"]["path"]).parents[2]
    panels = [last.with_name("mobile-wbc-sla-boundary-v1-m0-20261115-01"), last]
    sources = {}
    for p, h in zip(panels, population["batch_sha256"], strict=True):
        pinned(p / "RESULT.private.json", h)
        for f in (p / "runs").glob("*/RESULT.private.json"):
            sources[sha(f)] = f.parent
    output = []
    for r in population["cases"]:
        row = {
            "condition": r["condition"],
            "historic_status": r["status"],
            "historic_ticks": r["ticks"],
        }
        output.append(row)
        if r["ticks"] < 5703:
            row["boundary"] = "NOT_REACHED"
            continue
        f = sources[r["result_sha256"]]
        n = f / "native-01"
        sample = n / "MOBILE_SAMPLES.private.jsonl"
        pinned(sample, r["samples_sha256"])
        cfg = read(f / "EXECUTION_SPEC.json")
        pinned(Path(cfg["model_path"]), reg["scene_sha256"])
        st = packets(n / "state.bin.journal")[5702]
        x = next(v for v in rows(sample) if v["native_end_step"] == 5703)
        check(st["step"] == 5703, "missing tick")
        parts, _, _ = boundary(n, x)
        fields = initial_features(
            st, parts, prior_pd(n / "init.trace"), x["installed_EEF10"][9]
        )
        evidence = verify_initial(fields, anchors, reg["pairwise_initial_limits"])
        row.update(
            boundary="OBSERVED",
            matched=evidence["matched"],
            violations=evidence["violations"],
            sample_sha256=sha(sample),
            state_sha256=sha(n / "state.bin.journal"),
            trace_sha256=sha(n / "init.trace"),
        )
    return {
        "registered": len(output),
        "cases": output,
        "prior_manifest_sha256": sha(base),
        "scope": "Historical M0 initial prefixes, original receiver. Heterogeneous panel. No D1 treatment and no change to D1 matching limits.",
    }


def main():
    cli = argparse.ArgumentParser()
    for k in ("root", "out"):
        cli.add_argument(k, type=Path)
    args = cli.parse_args()
    root = args.root.resolve()
    configure(baseline(root))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ready = root / "v15/readiness01"
    reg = ready / "D1_FACTORIAL_4X2_REGISTER.json"
    pinned(reg, "c69cea0c14cf4390d369a5fec9efc44ec29ecfe273df44326f2f39fffb42578a")
    register = read(reg)
    b2 = root / "v14/b2-technical-20261123-01"
    public = b2 / "B2_PUBLIC_READOUT.json"
    pinned(public, B2_PIN)
    b2cases = read(public)["cards"]
    cases = {}
    anchors = []
    for c, h in zip(b2cases, register["prior_anchor_sha256"], strict=True):
        name = c["card"]
        f = b2 / name
        a = ready / (name + "_anchor.private.json")
        pinned(a, h)
        anchors.append(read(a)["initial_features"])
        cases[name] = timeline(
            f / read(f / "RESULT.private.json")["episode"], c["evidence_sha256"]
        )
    wave = root / "v15/sla-v15-d1-native-20261123-01/wave_A"
    oo = wave / "wave_A-D1-OO"
    sealed = read(root / "v15/readout01/D1_WAVE_A_ACTUAL.json")["records"][0]
    pinned(oo / "RESULT.private.json", sealed["result_sha256"])
    cases["D1-OO"] = timeline(
        oo / read(oo / "RESULT.private.json")["episode"], sealed["input_sha256"]
    )
    manifest = read(oo / "EXECUTION_SPEC.json")
    loaded = read(b2 / "B2_PUBLIC_READOUT.json")["frozen_sha256"]
    owner = read(wave / "OWNER.json")
    record(
        out / "D1_PREFIX_AUDIT_ACTUAL.json",
        {
            "stage": "post-stop read-only initialization diagnosis",
            "start_tick": 2504,
            "end_tick": 5703,
            "pairwise": [
                {"pair": [x, y], **compare(cases[x], cases[y])}
                for x, y in combinations(cases, 2)
            ],
            "thresholds": "descriptive localization only: 0.1 degree joint-command/actual and 1 mm base-origin displacement difference",
            "selected_MRT_policy": "UNOBSERVED",
            "clock": "last PD_TARGET before STEP_BEFORE by monotonic time, paired by native after-step journal tick; mixed-step commands excluded",
            "historical_panel": prior_population(root, register, anchors),
            "new_native_requests": 0,
            "new_ticks": 0,
            "new_gpu_forwards": 0,
            "new_optimizer_updates": 0,
        },
    )
    record(
        out / "D1_EXECUTION_IDENTITY.json",
        {
            "owner_code_sha256": owner["source_sha256"],
            "owner_receipt_sha256": sha(wave / "OWNER.json"),
            "wave_registration_sha256": owner["registration_sha256"],
            "source_overlay": manifest["source_sha"],
            "source_parent": manifest["trusted_source_parent"]["source_sha"],
            "loaded_executables_sha256": loaded,
            "frozen_files_sha256": {
                Path(k).name: v for k, v in manifest["frozen_files"].items()
            },
            "scene_sha256": register["scene_sha256"],
            "renderer_environment": manifest["native_environment"].get("MUJOCO_GL"),
            "renderer_binary_identity": "not individually attested; no camera or rendering result used",
            "seed": register["seed"],
            "lease_status": read(oo / "control/LEASE.json")["status"],
            "closed_lease_sha256": sha(oo / "control/LEASE.json"),
            "owner_close_sha256": sha(oo / "CLOSE.private.json"),
        },
    )


if __name__ == "__main__":
    main()
