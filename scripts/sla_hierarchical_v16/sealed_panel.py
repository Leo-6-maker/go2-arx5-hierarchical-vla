"""Open a registered window and its command boundary from the V0 evidence matrix."""

import sys
from pathlib import Path


def configure_site(root, repo):
    sys.path[:0] = [
        str(repo / "scripts/sla_official_pt_transfer"),
        str(repo / "scripts/sla_transfer_v15"),
        str(root / "v13/src"),
        str(root / "v12"),
        str(root / "v12/CPU_LIB"),
    ]
    from audit_sealed import pin, read
    from d1_readiness import configure

    p = root / "v13/run01/S0_ACTUAL.json"
    pin(p, "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9")
    pins = read(p)["inputs"]
    spec = next(Path(k) for k in pins if k.endswith("/EXECUTION_SPEC.json"))
    pin(spec, pins[str(spec)])
    configure(read(spec))


def cases(root, repo):
    import numpy as np
    from audit import load_episode, make_context
    from audit_sealed import check, pin, read
    from compiler import SourceCurve

    v = root / "v12"
    p = root / "v13/run01"
    baseline = read(
        repo / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json"
    )
    o1_path = repo / "reports/SLA_OFFICIAL_GO2_TRANSFER/O1_ACTUAL.json"
    pin(o1_path, baseline["source_hashes"]["O1"])
    o1 = {r["id"]: r for r in read(o1_path)["windows"]}
    pins = read(p / "S0_ACTUAL.json")["inputs"]

    def sealed(file):
        pin(file, pins[str(file)])
        return read(file)

    registered = sealed(v / "TARGET_REGISTRATION.json")["windows"]
    capture = sealed(v / "A2_O2_DENSE_CAPTURE.json")["windows"]
    archive = {r["window"]: r for r in sealed(v / "A4_P2_ABI.json")["windows"]}
    prior = read(p / "S3_ACTUAL.json")
    retime = {
        r["id"]: r["arms"]["C1"]
        for r in prior["windows"]
        if r["population"] == "TARGET"
    }
    groups = read(root / "evidence/P1_REGISTRATION.json")["target_windows"]
    loaded = {}
    for w in registered:
        k, ep = w["id"], w["episode"]
        row = archive[k]
        check(
            w["tick"] == row["tick"] == o1[k]["tick"]
            and ep == row["episode"] == o1[k]["episode"],
            "case clock",
        )
        if ep not in loaded:
            loaded[ep] = load_episode(root, ep, groups[w["case_index"]], {})
        ctx, incoming, _, _, grip, q, spec = make_context(root, w, row, loaded[ep])
        chain = v / "A4_ABI" / (k + ".npz")
        pin(chain, row["chain_sha256"])
        with np.load(chain, allow_pickle=False) as z:
            arrays = {
                key: z[key] for key in ("raw16", "H40", "world_knots", "teacher_H40")
            }
        rawpath = next(
            Path(f)
            for f, h in pins.items()
            if h == row["raw_sha256"] and f.endswith(".npy")
        )
        pin(rawpath, o1[k]["raw_sha256"])
        check(np.array_equal(arrays["raw16"], np.load(rawpath, allow_pickle=False)), k)
        curve = SourceCurve(
            arrays["raw16"], capture[k]["T_W_source_origin"], capture[k]["T_link6_TCP"]
        )
        check(
            np.allclose(curve.knots, arrays["world_knots"], atol=2e-7, rtol=0),
            "TCP path",
        )
        yield dict(
            key=k,
            episode=ep,
            window=w,
            archive=row,
            arrays=arrays,
            curve=curve,
            context=ctx,
            incoming=incoming,
            grip=grip,
            query=q,
            spec=spec,
            prior=o1[k],
            retime=retime[k],
            capture=capture[k],
            raw_file_sha256=row["raw_sha256"],
            limits={
                k: v
                for k, v in baseline["limits"].items()
                if not k.startswith("finger")
            },
        )
