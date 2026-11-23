"""Feed the canonicalizer's re-sampled candidates back through V1.2's commissioned offline checks."""

import sys
from pathlib import Path
import numpy as np
from audit import load_episode, make_context, read, record, sha
from score_target import check
from go2arx5_vla.world_model.task_learning_grasp_v1 import FiftyMsProgram
from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
from scripts.pi0291_arx5_data.native_dataset import matrix_from10


def main(root, out):
    old = read(root / "v12/A4_P2_ABI.json")["windows"]
    reg = read(root / "v12/TARGET_REGISTRATION.json")["windows"]
    roster = read(root / "evidence/P1_REGISTRATION.json")["target_windows"]
    cache = {}
    ctx = {}
    for w, r in zip(reg, old, strict=True):
        ep = w["episode"]
        if ep not in cache:
            cache[ep] = load_episode(root, ep, roster[w["case_index"]], {})
        ctx[w["id"]] = make_context(root, w, r, cache[ep])[0]
    gates = []
    for r in read(out / "CANONICAL_ACTUAL.json")["rows"]:
        summary = {"id": r["id"], "arm": r["arm"], "status": r["status"]}
        if "file" in r:
            f = out / "canonical" / r["file"]
            assert sha(f) == r["sha256"]
            with np.load(f) as z:
                h = z["H40"]
            if r["id"] in ctx:
                g = check(h, ctx[r["id"]])
                summary.update(
                    H40=bool(g["full_H40"]["original_full_H40_pass"]),
                    legal=g["H40_G7_pass"],
                    issues=g["issues"],
                )
            else:
                start = matrix_from10(h[0])
                summary["legal"] = None
                try:
                    FiftyMsProgram(h, start, start, np.array([0, 0, 0, 0, 0, 0, 1.0]))
                    summary["H40"] = True
                except ValueError as error:
                    summary.update(H40=False, compiler_reason=str(error))
                try:
                    g = analyze(h, h[0], np.eye(4), start)
                    assert g["original_full_H40_pass"] == summary["H40"]
                except ValueError as error:
                    summary["attribution_unavailable"] = str(error)
        gates.append(summary)
    record(out / "CANONICAL_GATE.json", dict(rows=gates, native_requests=0))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
