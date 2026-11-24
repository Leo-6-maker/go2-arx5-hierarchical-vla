"""Blind proposal generation and separate scoring on 3 Reach and 16 historical returns."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

BASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(BASE / "scripts" / s)
    for s in (
        "sla_hierarchical_v16",
        "sla_transfer_v15",
        "sla_transfer_v14",
        "sla_transfer_feedback",
    )
]
from compiler import SourceCurve, Compiler, vectors
from demonstrated_grip import pin, read, require
from task_space_intent import geometric_control, propose
from visual_objects import estimate_objects

PROFILES = ("RAW", "REANCHOR", "SHORT", "PROJECT", "GEOMETRY")


def lines(path):
    with path.open() as f:
        yield from map(json.loads, f)


def history(root):
    pins = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]

    def manifest(name):
        p = root / "v12" / name
        return read(pin(p, pins[str(p)]))

    cap = manifest("A2_O2_DENSE_CAPTURE.json")
    archives = {w["window"]: w for w in manifest("A4_P2_ABI.json")["windows"]}
    eps = {r["id"]: r for r in cap["episodes"]}
    groups = read(root / "evidence/P1_REGISTRATION.json")["target_windows"]
    cache = {}
    for r in manifest("TARGET_REGISTRATION.json")["windows"]:
        k, ep, t = r["id"], r["episode"], r["tick"]
        a = archives[k]
        meta = cap["windows"][k]
        color = groups[r["case_index"]]["instruction"].split("-")[1]
        if ep not in cache:
            rec = read(pin(Path(eps[ep]["receipt_path"]), eps[ep]["receipt_sha256"]))
            path = (
                Path(groups[r["case_index"]]["episode_pin"]["path"]).parents[3]
                / "cases"
                / ep
                / "native-01"
                / "MOBILE_SAMPLES.private.jsonl"
            )
            atoms = {
                z["native_end_step"]: z
                for z in lines(pin(path, a["native_samples_sha256"]))
            }
            frames = {
                z["tick"]: z
                for z in lines(
                    pin(root / "v12/DENSE" / ep / "frames.jsonl", rec["frames_sha256"])
                )
                if z["query_clock"]
            }
            cache[ep] = (atoms, frames)
        atoms, frames = cache[ep]
        at, fr = atoms[t], frames[t]
        require(
            np.allclose(at["actual_TCP"], fr["T_W_TCP"], atol=1e-8, rtol=0)
            and at["solver_exact_installed_targets"],
            "unbound historical seam",
        )
        with np.load(
            pin(root / "v12/STUDENT_INPUT" / (k + ".npz"), meta["input_sha256"]),
            allow_pickle=False,
        ) as z:
            obj = estimate_objects(
                z["head"][-1],
                z["right_wrist"],
                fr["calibration"],
                fr["sim_time_s"],
                reject_spanning=True,
            )[color]
        with np.load(
            pin(root / "v12/A4_ABI" / (k + ".npz"), a["chain_sha256"]),
            allow_pickle=False,
        ) as z:
            curve = SourceCurve(
                z["raw16"], meta["T_W_source_origin"], meta["T_link6_TCP"]
            )
            require(
                np.allclose(curve.knots, z["world_knots"], rtol=0, atol=2e-7),
                "source world",
            )
        yield dict(
            id=k,
            episode=ep,
            tick=t,
            condition=r["condition"],
            obj=obj,
            curve=curve,
            at=at,
            standing=False,
        )


def standing(root):
    pubs = read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")
    m = read(pin(root / "v18/sealed-Z1/members.json", pubs["member_manifest_sha256"]))

    def sealed(p):
        return pin(p, m[str(p.relative_to(root / "v18"))])

    geo = read(BASE / "reports/SLA_GO2_V18/Z1_ATTRIBUTION/IMAGE_GEOMETRY.json")
    for slot, panel in (("G00-A", "Z1_REACH_483"), ("G01-A", "Z1_REACH_483_G01")):
        c = root / "v18" / panel / slot
        native = c / read(sealed(c / "NATIVE_REQUEST.json"))["episode"]
        spec = read(sealed(c / "EXECUTION_SPEC.json"))
        entry = read(sealed(c / "REACH_ENTRY.json"))
        records = {
            v["tick"]: v["sensor"]
            for v in lines(sealed(c / "POLICY_OWNER.jsonl"))
            if v["kind"] == "RETURN"
        }
        atoms = {
            z["native_end_step"]: z
            for z in lines(sealed(native / "MOBILE_SAMPLES.private.jsonl"))
            if z["native_end_step"] in records
        }
        for q in sorted(c.glob("query-*")):
            if not (q / "RETURN.json").exists():
                continue
            req = read(sealed(q / "REQUEST.json"))
            t = req["tick"]
            r = next(
                v
                for v in geo["samples"]
                if v["slot"] == slot
                and v["tick"] == t
                and v["input_sha256"] == req["input_sha256"]
            )
            ret = read(sealed(q / "RETURN.json"))
            with np.load(
                pin(sealed(q / "OUTPUT.npz"), ret["output_sha256"]), allow_pickle=False
            ) as z:
                curve = SourceCurve(
                    z["absolute"], records[t]["world_origin"], records[t]["link6_tcp"]
                )
            require(
                atoms[t]["solver_exact_installed_targets"], "missing consumed reference"
            )
            yield dict(
                id=f"{slot}-{t}",
                tick=t,
                episode=native.name,
                condition="standing Reach",
                obj=r["image_objects"]["red"],
                curve=curve,
                at=atoms[t],
                entry=entry,
                spec=spec,
                standing=True,
            )


def compile_path(path, width, actual_ref, previous, limits, raw_curve=None):
    if raw_curve is None:
        # Full H40 previews can extend past the selected prefix; only hold is
        # defined there. No movement after the last measured source index is credited.
        pad = 50 - len(path)
        curve = np.concatenate((path, np.repeat(path[-1:], pad, axis=0)))
        raw = np.zeros((50, 16))
        raw[:, 8:11] = curve[:, :3, 3]
        raw[:, 11:15] = Rotation.from_matrix(curve[:, :3, :3]).as_quat()[
            :, [3, 0, 1, 2]
        ]
        if width is None:
            raw[:, 15] = 1.0
        else:
            raw[:, 15] = np.pad(width[: len(path)], (0, pad), mode="edge")
        raw_curve = SourceCurve(raw, np.eye(4), np.eye(4))
    linear, angular = vectors(np.asarray([previous, actual_ref]), 0.05)
    return Compiler(
        raw_curve,
        actual_ref,
        (linear[0], angular[0]),
        limits,
        cruise_from_width_m=0.042,
    )


def frozen_checks(root, row, path, compiled, ref, previous):
    import mujoco
    from known_dynamic_panel import KnownCurve
    from standing_reach import StateMonitor
    from d1_geometry import sweep_profile
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
    from go2arx5_vla.world_model.native_replay import forbid_integration

    at, spec = row["at"], row["spec"]
    scene = Path(spec["model_path"])
    model = mujoco.MjModel.from_binary_path(
        str(pin(scene, spec["frozen_files"][str(scene)]))
    )
    mon = StateMonitor(model)
    samples = dict(tcp=np.array(at["actual_TCP"]), qpos=np.array(at["qpos"]))
    base = np.asarray(at["actual_mount"])
    guard = binding(spec["native_root"])
    checked = {}
    rejected = None
    # First 100 ms publication plus the full first H40. Later feedback would
    # change qpos and the remaining clearance, which this static replay cannot observe.
    poses, _, _ = compiled.at(np.arange(41) * 0.05)
    poses[-1] = poses[-2]
    with forbid_integration(mujoco) as calls:

        def witness(data):
            clearance = mon.clearance(data)
            if clearance < 0.015:
                raise ValueError(
                    f"robot-cube clearance {1000 * clearance:.4f} mm < 15 mm"
                )

        try:
            KnownCurve.admit(
                poses,
                dict(
                    start_pose=previous,
                    end_pose=ref,
                    finger_fraction=at["target_endpoint_EEF10"][9],
                ),
                samples,
            )
            checked["first_H40_and_consumed_seam"] = True
            for kind, a in (
                ("first_H40_knots", poses),
                ("whole_selected_source_knots", path),
            ):
                checked[kind] = 0
                for p in a:
                    sweep_profile(
                        model,
                        samples,
                        dict(old_ee=ref, old_base=base),
                        p,
                        base,
                        guard,
                        (41, 38, 23),
                        mon.permitted,
                        witness,
                    )
                    checked[kind] += 1
        except (ValueError, RuntimeError, AssertionError) as exc:
            rejected = str(exc)
        require(not calls, "physics during static audit")
    return dict(checked=checked, rejection=rejected)


def evaluate(root, r, limits):
    from scripts.pi0292.eval_contract import matrix10

    at = r["at"]
    actual = np.asarray(at["actual_TCP"])
    ref, prev = map(
        lambda k: matrix10(np.asarray(at[k])),
        ("target_endpoint_EEF10", "installed_EEF10"),
    )
    world = r["curve"].knots
    obj = r["obj"]
    outcomes = {}
    for method in PROFILES:
        try:
            if method == "GEOMETRY":
                path, scale = geometric_control(actual, ref, obj), None
            else:
                path, scale = propose(method, world, actual, ref, obj)
        except ValueError as e:
            outcomes[method] = dict(refusal=str(e))
            continue
        delta = path[:, :3, 3] - ref[:3, 3]
        rel = ref @ np.linalg.solve(actual, world[: len(path)])
        residual = path[:, :3, 3] - rel[:, :3, 3]
        val = dict(
            refusal=None,
            source_knots=0 if method == "GEOMETRY" else len(path),
            trajectory_points=len(path),
            scale=scale,
            endpoint_from_ref_mm=(1000 * delta[-1]).tolist(),
            rewrite_rms_mm=None
            if method == "GEOMETRY"
            else float(1000 * np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
            first_offset_mm=float(1000 * np.linalg.norm(delta[0])),
            last_offset_mm=float(1000 * np.linalg.norm(delta[-1])),
            maximum_orientation_deg=float(
                np.degrees(
                    Rotation.from_matrix(path[:, :3, :3] @ ref[:3, :3].T)
                    .magnitude()
                    .max()
                )
            ),
        )
        if r["standing"]:
            axis = np.asarray(r["entry"]["direction"])
            red = np.array(at["object_pose"])[:3, 3]
            local_end = actual[:3, 3] + delta[-1]
            parallel = delta @ axis
            val.update(
                ending_axial_mm=1000 * parallel[-1],
                maximum_cross_track_mm=1000
                * np.linalg.norm(delta - parallel[:, None] * axis, axis=1).max(),
                local_20mm_endpoint_error_mm=1000
                * np.linalg.norm(delta[-1] - 0.02 * axis),
                local_red_gain_mm=1000
                * (
                    np.linalg.norm(actual[:3, 3] - red)
                    - np.linalg.norm(local_end - red)
                ),
            )
            comp = compile_path(
                path,
                None if method == "GEOMETRY" else r["curve"].arm[:, 7],
                ref,
                prev,
                limits,
                r["curve"] if method == "RAW" else None,
            )
            val["alignment_s"] = comp.lead
            val["static_preview"] = frozen_checks(root, r, path, comp, ref, prev)
        outcomes[method] = val
    return dict(
        id=r["id"],
        tick=r["tick"],
        episode=r["episode"],
        condition=r["condition"],
        vision_status=obj["status"],
        vision_separation_mm=obj.get("separation_mm"),
        outcomes=outcomes,
    )


def main(root):
    limits = read(BASE / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")[
        "limits"
    ]
    rows = []
    for collection in (history, standing):
        for r in collection(root):
            rows.append(evaluate(root, r, limits))
            print(r["id"], file=sys.stderr, flush=True)
    require(
        len(rows) == 19 and len({r["id"] for r in rows}) == 19, "incomplete fixed panel"
    )
    return dict(
        rows=rows, forward_calls=0, native_starts=0, integration_ticks=0, updates=0
    )


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]).resolve()), indent=2, allow_nan=False))
