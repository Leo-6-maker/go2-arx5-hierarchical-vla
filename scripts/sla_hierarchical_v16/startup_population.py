"""Test whether both claimed startup strata have identifiable, pairwise compatible anchors."""

import json
import sys
from pathlib import Path
import numpy as np
from sealed_panel import configure_site


def run(root, repo):
    from audit_sealed import read, pin, digest
    from d1_factorial_reference_matrix import distances

    v0 = read(repo / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")
    g = root / "v15/readiness01/D1_FACTORIAL_4X2_REGISTER.json"
    pin(g, "c69cea0c14cf4390d369a5fec9efc44ec29ecfe273df44326f2f39fffb42578a")
    lim = read(g)["pairwise_initial_limits"]
    feat = {}
    rows = []
    for stage in ("F1", "F2"):
        f = root / f"feedback/{stage}_VERIFIED.json"
        pin(f, v0["source_hashes"][stage])
        campaign = next(
            (root / "feedback").glob("sla-feedback-" + stage.lower() + "-*")
        )
        for rr in read(f)["records"]:
            card = rr["card"]
            case = campaign / card
            ip = case / "INITIAL.private.json"
            rp = case / "RESULT.private.json"
            pin(ip, rr["initial_sha256"])
            pin(rp, rr["result_sha256"])
            feat[card] = read(ip)["features"]
            m = case / read(rp)["episode"] / "MOBILE_SAMPLES.private.jsonl"
            with m.open() as stream:
                samples = [
                    v
                    for l in stream
                    if 5203 <= (v := json.loads(l))["native_end_step"] <= 5703
                ]
            if [r["native_end_step"] for r in samples] != list(range(5203, 5704, 25)):
                raise ValueError("incomplete last startup second")
            actual = np.array([r["actual_TCP"] for r in samples])
            xyz = actual[:, :3, 3] * 1000
            rel = actual[:, :3, :3] @ actual[-1, :3, :3].T
            angle = np.rad2deg(
                np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))
            )
            rows.append(
                dict(
                    card=card,
                    original_B2_admitted=rr["eligible"],
                    initial_sha256=digest(ip),
                    archived_tcp_series_sha256=digest(m),
                    last_second_max_displacement_from_final_mm=float(
                        np.linalg.norm(xyz - xyz[-1], axis=1).max()
                    ),
                    last_second_max_rotation_from_final_deg=float(angle.max()),
                    largest_50ms_step_mm=float(
                        np.linalg.norm(np.diff(xyz, axis=0), axis=1).max()
                    ),
                )
            )
    adjacent = {k: {k} for k in feat}
    edges = []
    for i, a in enumerate(feat):
        for b in list(feat)[i + 1 :]:
            u = distances(feat[a], feat[b])
            v = distances(feat[b], feat[a])
            ratio = {k: max(u[k], v[k]) / n for k, n in lim.items()}
            k = max(ratio, key=ratio.get)
            edges.append(
                dict(
                    a=a,
                    b=b,
                    max_fraction_of_limit=ratio[k],
                    limiting_feature=k,
                    match=ratio[k] <= 1,
                )
            )
            if ratio[k] <= 1:
                adjacent[a].add(b)
                adjacent[b].add(a)
    unseen = set(adjacent)
    components = []
    while unseen:
        stack = [min(unseen)]
        group = set(stack)
        while stack:
            new = adjacent[stack.pop()] - group
            group.update(new)
            stack.extend(new)
        unseen -= group
        components.append(
            dict(
                cards=sorted(group),
                fully_pairwise_compatible=all(group <= adjacent[k] for k in group),
            )
        )
    path = repo / "reports/SLA_HIERARCHICAL_V16/V1_COMPILER_ABLATION_ACTUAL.json"
    peak = {}
    for m in ("C2_EXACT", "C2_BOUNDED"):
        a = [r["modes"][m]["kinematics_5ms"] for r in read(path)["windows"]]
        peak[m] = {k: max(r[k]["speed_peak"] for r in a) for k in ("linear", "angular")}
    return dict(
        startup_rows=rows,
        pairwise=edges,
        components=components,
        limits=lim,
        compiled_speed_envelope=peak,
        source_v1_sha256=digest(path),
        new_physics_ticks=0,
        independent_sensor_noise_estimate=None,
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    print(json.dumps(run(root, repo), indent=2, allow_nan=False))
