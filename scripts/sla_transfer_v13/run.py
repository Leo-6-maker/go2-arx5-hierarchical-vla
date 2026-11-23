"""One immutable CPU analysis over the fixed V1.2 panel. Original gate code is executed for every emitted reference."""

import os, sys, time
from acceptance import handover, once_new
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from audit import diagnostic, load_episode, make_context, pinned, read, record, sha
from retime import DT, END, NAMES, SourcePath, geometric_progress, select_timing, twist
from score_target import check
from transfer_frames import convert, transform
from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from go2arx5_vla.world_model import fusion_trajectory_adapter as frozen


def discrepancy(path, phase, mats):
    ref = path.at(phase)
    delta = twist(ref, mats, 1.0)
    # Discrete symmetric Hausdorff on dense samples of the covered portion vs resampled H40 segments.
    from scipy.spatial.transform import Slerp

    query = np.linspace(0, len(phase) - 1, 391)
    dense_source = path.at(np.linspace(phase[0], phase[-1], 391))
    chord = np.stack(
        [np.interp(query, np.arange(len(phase)), mats[:, i, 3]) for i in range(3)], axis=1
    )
    distances = cdist(dense_source[:, :3, 3], chord)
    target = path.at(np.interp(query, np.arange(len(phase)), phase))
    rot = Slerp(np.arange(len(phase)), Rotation.from_matrix(mats[:, :3, :3]))(query)
    return dict(
        on_source_knot_position_peak_m=float(np.linalg.norm(delta[0], axis=1).max()),
        on_source_knot_SO3_peak_rad=float(np.linalg.norm(delta[1], axis=1).max()),
        covered_curve_sampled_Hausdorff_m=float(
            max(distances.min(0).max(), distances.min(1).max())
        ),
        interpolated_SO3_deviation_peak_rad=float(
            (rot.inv() * Rotation.from_matrix(target[:, :3, :3])).magnitude().max()
        ),
        terminal_source_fraction=float(phase[-1] / END),
        sample_points=391,
    )


def compact_gate(g):
    return {
        k: g[k]
        for k in [
            "compiler",
            "H40_G7_pass",
            "G7",
            "full_H40",
            "issues",
            "accepted_join_pass",
            "accepted_boundary_join",
            "previous_actual_emission",
        ]
    }


def metric(row, name, poses, phase, plan, path, teacher, half_edge, context, out, details=None):
    from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
    from go2arx5_vla.sla_adapter.dense_g7_v6 import GRIP_LIMIT

    assert abs(GRIP_LIMIT - 0.040) < 1e-12
    observed = compact_gate(check(plan, context)) if context is not None else None
    if context is None:
        start = poses[0]
        native = analyze(plan, pose10(start, plan[0, 9]), np.eye(4), start)
    else:
        native = observed["full_H40"]
    progress = geometric_progress(path, phase, poses, teacher, half_edge, row["phase"], plan[:, 9])
    arrays = out / "arrays" / (row["id"] + "_" + name + ".npz")
    with arrays.open("xb") as f:
        np.savez_compressed(f, H40=plan, source_phase_s=phase, world=poses)
    original = native["paths"]["nominal"]["channels"]
    pose_pass = all(not original[k]["exceedance_indices"] for k in NAMES)
    return dict(
        method=name,
        shape=[40, 10],
        array_file=arrays.name,
        array_sha256=sha(arrays),
        nominal_pose_channels_pass=pose_pass,
        H40=native["original_full_H40_pass"],
        native_gate=observed,
        intrinsic_only=native if observed is None else None,
        progress=progress,
        path_geometry=discrepancy(path, phase, poses),
        timing=details,
    )


def timings(row, path, plan, teacher, half_edge, ctx, prev, transport, limits, out, ledger):
    base_phase = np.minimum(np.arange(40) * DT, END)
    res = {
        "C0": metric(
            row, "C0", matrix_from10(plan), base_phase, plan, path, teacher, half_edge, ctx, out
        )
    }
    for name, is_uniform in [("UNIFORM_DIAGNOSTIC", True), ("C1", False)]:
        tm = select_timing(path, limits, prev, transport, uniform=is_uniform)
        if tm is None:
            res[name] = dict(
                status="RETIME_INFEASIBLE_IN_REGISTERED_FAMILY", max_duration_s=1046 * DT
            )
            continue
        phase = tm.pop("phase")
        poses = tm.pop("poses")
        full = tm.pop("full_phase")
        h = pose10(poses, plan[:, 9])
        assert np.array_equal(h[:, 9], plan[:, 9])
        tm["full_phase_sha256"] = __import__("hashlib").sha256(full.tobytes()).hexdigest()
        tm["coverage"] = "COMPLETE" if phase[-1] >= END - 1e-10 else "PARTIAL_REQUIRES_CONTINUATION"
        res[name] = metric(row, name, poses, phase, h, path, teacher, half_edge, ctx, out, tm)
    for name, linked in [("C2", "C0"), ("C3", "C1")]:
        reason = handover(ledger) if ledger is not None else "NO_GO2_OWNER_OR_COMMAND_HISTORY"
        if ledger is not None and not reason.startswith("HANDOVER_BLOCKED"):
            raise ValueError(
                "Ownership differs from preregistered teacher-only corpus; inspect receipt"
            )
        res[name] = dict(
            status="HANDOVER_BLOCKED" if ledger is not None else reason,
            reference_arm=linked,
            pose_candidate=res[linked].get("array_sha256"),
            executed_transition=False,
            issued_commands=0,
            admitted=False,
        )
    return res


def main(root, out):
    registration = read(Path(__file__).with_name("registration.json"))
    once_new(out)
    (out / "arrays").mkdir(mode=0o700)
    started = time.monotonic()
    manifest = {}
    V = root / "v12"
    pin = lambda p, h: pinned(p, h, manifest)
    pin(V / "FINAL_V12_DECISION.json", registration["prior_final_sha256"])
    pin(V / "FINAL_V12_DECISION.md", registration["prior_final_md_sha256"])
    seal = read(V / "FINAL_V12_DECISION.json")
    for name, expected in seal["files"].items():
        pin(V / name, expected)
    # Markdown identified by byte readback, JSON and emitted arrays bind the original measurements.
    manifest[str(V / "A4_P2_ABI.md")] = sha(V / "A4_P2_ABI.md")
    prior = read(V / "A4_P2_ABI.json")
    a2 = read(V / "A2_O2_DENSE_CAPTURE.json")
    reg = read(V / "TARGET_REGISTRATION.json")
    roster = read(root / "evidence/P1_REGISTRATION.json")["target_windows"]
    limits = dict(
        zip(
            NAMES,
            [
                frozen.MAX_TRANSLATION_SPEED_M_S,
                frozen.MAX_ANGULAR_SPEED_RAD_S,
                frozen.MAX_TRANSLATION_ACCELERATION_M_S2,
                frozen.MAX_ANGULAR_ACCELERATION_RAD_S2,
            ],
        )
    )
    base = dict(
        protocol=registration["protocol_commit"],
        weights=seal["HF_rev"],
        xpolicy=seal["xpolicylab"],
        registration=sha(Path(__file__).with_name("registration.json")),
        native_gate_source=read(V / "ABI_CPU_PREFLIGHT.json")["frozen_native_modules"],
        owner=os.getuid(),
        physics_ticks=0,
        forward_calls=0,
        training_updates=0,
    )
    records = {}
    cache = {}
    s0 = []
    s1 = []
    s2 = []
    refs = []
    sources = []
    rawfiles = {}
    for population in ["O2", "P2_EXTRA"]:
        folder = V / (population + "_GPU_01")
        for t in map(__import__("json").loads, (folder / "calls.jsonl").read_text().splitlines()):
            if t["event"] == "FORWARD_RETURNED" and not t["repeat"]:
                rawfiles[t["id"]] = pin(folder / t["output_file"], t["file_sha256"])
    for w, old in zip(reg["windows"], prior["windows"], strict=True):
        assert w["id"] == old["window"] and w["episode"] == old["episode"]
        ep = w["episode"]
        bound = a2["windows"][w["id"]]
        if ep not in cache:
            cache[ep] = load_episode(root, ep, roster[w["case_index"]], manifest)
            dp = V / "DENSE" / ep
            receipt = read(pin(dp / "receipt.json", bound["source_sha256"]))
            pin(dp / "frames.jsonl", receipt["frames_sha256"])
        ctx, prev, issued, ach, ledger, q, spec = make_context(root, w, old, cache[ep])
        chain = pin(V / "A4_ABI" / (w["id"] + ".npz"), old["chain_sha256"])
        with np.load(chain, allow_pickle=False) as z:
            raw = z["raw16"]
            plan = z["H40"]
            teacher = z["teacher_H40"]
            selected = z["selected8"]
        assert (
            np.array_equal(raw, np.load(rawfiles[w["id"]]))
            and np.array_equal(selected, raw[:, 8:])
            and np.array_equal(teacher, np.asarray(q["nominal_world_H40"]))
        )
        reproduce = convert(raw, bound["T_W_source_origin"], bound["T_link6_TCP"])
        assert np.array_equal(reproduce["H40"], plan)
        orig = check(plan, ctx)
        te = check(teacher, ctx)
        assert (
            orig == old["student"]
            and te == old["teacher"]
            and te["H40_G7_pass"]
            and not orig["H40_G7_pass"]
        )
        path = SourcePath(selected, bound["T_W_source_origin"], bound["T_link6_TCP"])
        tm = matrix_from10(teacher)
        m = matrix_from10(plan)
        transport = ctx[3][:3, :3] @ ctx[2][:3, :3].T
        vel = twist(m[0], m[1], DT)
        vel = (vel[0], transport @ vel[1])
        key = dict(
            id=w["id"],
            episode=ep,
            seed=read(roster[w["case_index"]]["episode_pin"]["path"])["reset"],
            phase=q["phase"],
        )
        s0.append(
            dict(
                **key,
                previous=ctx[-1],
                teacher=compact_gate(te),
                student=compact_gate(orig),
                input=old["source_input"],
                raw=old["raw_sha256"],
                chain=old["chain_sha256"],
                source_seal=spec["source_sha"],
                execution_profile=spec["execution_clock_profile"],
                acquired_to_infer_to_install_native_s=None,
                latency_scope="offline replay, no student wire issued or installed",
            )
        )
        s2.append(dict(**key, **ledger))
        diag = dict(
            **key,
            raw30=diagnostic(path.raw, 1 / 30, limits),
            resampled_H40=diagnostic(m, DT, limits, True),
            teacher=diagnostic(tm, DT, limits, True),
            incoming_accepted_twist=[v.tolist() for v in prev],
            previous_emissions_twist=[v.tolist() for v in issued],
            achieved_twist=[v.tolist() for v in ach],
            first_student_twist=[v.tolist() for v in vel],
            first_accel_from_consumed=[
                float(np.linalg.norm(a - b) / DT) for a, b in zip(vel, prev)
            ],
            first_accel_from_native_rest=[float(np.linalg.norm(a) / DT) for a in vel],
        )
        s1.append(diag)
        half = spec["multitask_geometry"]["object_half_sizes"]["red"][0]
        records[w["id"]] = (key, path, plan, tm, half, ctx, prev, transport, ledger)
    # Bind all 6 source expert/prediction pairs before analyzing any candidate.
    folder = root / "evidence/O0_PANEL_01"
    oldsource = read(V / "A4_SOURCE_ENVELOPE_EXPLORATORY.json")
    for t in map(__import__("json").loads, (folder / "calls.jsonl").read_text().splitlines()):
        if t["event"] != "FORWARD_RETURNED" or t["mode"] != "primary":
            continue
        f = pin(folder / t["output_file"], t["file_sha256"])
        with np.load(f, allow_pickle=False) as z:
            for kind in ["expert_absolute_arms", "native_absolute_arms"]:
                key = dict(
                    id=f"S{t['call']:02d}_{kind}",
                    episode=t["episode"],
                    phase=t["phase"],
                    frame=t["frame"],
                )
                org = transform(position=reg["source_origin"])
                off = reg["T_link6_TCP"]
                bundle = convert(z[kind], org, off)
                path = SourcePath(bundle["selected8"], org, off)
                sources.append(
                    dict(
                        **key,
                        raw30=diagnostic(path.raw, 1 / 30, limits),
                        resampled_H40=diagnostic(bundle["world_H40"], DT, limits, True),
                        measured_incoming_twist=None,
                        previous_gripper_command=None,
                    )
                )
                records[key["id"]] = (
                    key,
                    path,
                    bundle["H40"],
                    None,
                    0.02,
                    None,
                    (np.zeros(3), np.zeros(3)),
                    np.eye(3),
                    None,
                )
    assert len(s0) == 16 and len(sources) == len(oldsource["rows"]) == 12
    record(
        out / "S0_ACTUAL.json",
        dict(
            **base,
            elapsed_wall_s=time.monotonic() - started,
            status="LOCALLY_AUDITED",
            limits=limits,
            inputs=manifest,
            windows=s0,
        ),
    )
    record(
        out / "S1_ACTUAL.json",
        dict(**base, status="DIAGNOSED", limits=limits, windows=s1, source_controls=sources),
    )
    record(out / "S2_ACTUAL.json", dict(**base, status="SHADOW_AUDITED", windows=s2))
    for key, path, plan, tm, half, ctx, prev, transport, ledger in records.values():
        arms = timings(key, path, plan, tm, half, ctx, prev, transport, limits, out, ledger)
        refs.append(
            dict(**key, population="TARGET" if ctx is not None else "SOURCE_CONTROL", arms=arms)
        )
    record(
        out / "S3_ACTUAL.json",
        dict(
            **base,
            status="FOUR_ARMS_EVALUATED",
            windows=refs,
            elapsed_wall_s=time.monotonic() - started,
            retained_nominal_methods=4,
            uniform_is_diagnostic=True,
        ),
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
