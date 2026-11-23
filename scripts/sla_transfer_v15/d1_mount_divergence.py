"""Compare the installed-base origin over all 128 archived initialization atoms."""

import argparse
from pathlib import Path

import numpy as np
from d1_readiness import B2_PIN, pinned, read, record, rows, sha


def positions(folder, pin):
    path = folder / "MOBILE_SAMPLES.private.jsonl"
    pinned(path, pin)
    prefix = [r for r in rows(path) if r["native_end_step"] <= 5703]
    ticks = np.arange(2528, 5704, 25)
    if not np.array_equal([r["native_end_step"] for r in prefix], ticks):
        raise ValueError("incomplete native initialization clock")
    return np.array([r["actual_mount"] for r in prefix])[:, :3, 3]


def main():
    parser = argparse.ArgumentParser()
    for k in ("root", "audit", "output"):
        parser.add_argument(k, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    sealed = read(args.audit)
    panel = sealed["historical_panel"]
    spec = read(
        next(
            Path(k)
            for k in read(root / "v13/run01/S0_ACTUAL.json")["inputs"]
            if k.endswith("/EXECUTION_SPEC.json")
        )
    )
    site = Path(spec["mobile_interface_qualification"]["path"]).parents[2]
    b2 = root / "v14/b2-technical-20261123-01"
    manifest = b2 / "B2_PUBLIC_READOUT.json"
    pinned(manifest, B2_PIN)
    anchors = []
    for c in read(manifest)["cards"]:
        path = b2 / c["card"]
        anchors.append(
            positions(
                path / read(path / "RESULT.private.json")["episode"],
                c["evidence_sha256"]["MOBILE_SAMPLES.private.jsonl"],
            )
        )
    cases = {}
    for path in (site / "runs").glob("*/EXECUTION_SPEC.json"):
        m = read(path)["mobile_condition"]
        cases[(m["axis"], m["eef_mode"])] = path.parent / "native-01"
    register = root / "v15/readiness01/D1_FACTORIAL_4X2_REGISTER.json"
    pinned(register, "c69cea0c14cf4390d369a5fec9efc44ec29ecfe273df44326f2f39fffb42578a")
    curves = []

    def append(label, group, path, sample_hash):
        v = positions(path, sample_hash)
        dist = np.maximum(*[np.linalg.norm(a - v, axis=1) * 1000 for a in anchors])
        curves.append(
            {
                "case": label,
                "group": group,
                "max_to_two_anchors_mm": dist.tolist(),
                "source_samples_sha256": sample_hash,
            }
        )

    for r in panel["cases"]:
        if r["boundary"] == "OBSERVED":
            c = r["condition"]
            key = (c["axis"], c["eef_mode"])
            append(
                "M0 " + " ".join(key),
                "M0 matched" if r["matched"] else "M0 mismatched",
                cases[key],
                r["sample_sha256"],
            )
    run = root / "v15/sla-v15-d1-native-20261123-01/wave_A/wave_A-D1-OO"
    oo = read(root / "v15/readout01/D1_WAVE_A_ACTUAL.json")["records"][0]
    append(
        "D1 OO",
        "D1 OO",
        run / read(run / "RESULT.private.json")["episode"],
        oo["input_sha256"]["MOBILE_SAMPLES.private.jsonl"],
    )
    if not np.isclose(
        curves[-1]["max_to_two_anchors_mm"][-1],
        1000 * oo["matching"]["observed_max"]["measured_base_m"],
        atol=1e-8,
    ):
        raise ValueError("distance differs from actual D1 gate")
    record(
        args.output,
        {
            "unit": "mm",
            "clock": "native tick, post 25-tick initialization atom",
            "gate_applies_only_at_last_tick": True,
            "limit_mm": 1000
            * read(register)["pairwise_initial_limits"]["measured_base_m"],
            "ticks": list(range(2528, 5704, 25)),
            "curves": curves,
            "historical_registered": 12,
            "historical_boundaries": 9,
            "source_audit_sha256": sha(args.audit),
            "new_native_requests": 0,
            "new_ticks": 0,
            "new_model_forwards": 0,
            "new_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
