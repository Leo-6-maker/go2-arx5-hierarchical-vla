"""Re-read the deployed gripper receiver identities and join all frozen ownership rows."""

import hashlib
import json
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, repo):
    def read(f):
        return json.loads(f.read_bytes())

    reports = repo / "reports"
    v0 = read(reports / "SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")
    o3 = reports / "SLA_OFFICIAL_GO2_TRANSFER/O3_RECEIVER.json"
    if sha(o3) != v0["source_hashes"]["O3"]:
        raise ValueError("changed O3 archive")
    archive = read(o3)
    pins = read(root / "v13/run01/S0_ACTUAL.json")["inputs"]
    teacher_spec = next(Path(p) for p in pins if p.endswith("/EXECUTION_SPEC.json"))
    if sha(teacher_spec) != pins[str(teacher_spec)]:
        raise ValueError("teacher spec identity")
    spec = read(teacher_spec)
    prior = (
        Path(spec["mobile_interface_qualification"]["path"]).parent
        / "EXECUTION_SPEC.json"
    )
    receivers = {}
    for key, s in (
        ("teacher_receiver", spec),
        ("prior_interface_qualification_receiver", read(prior)),
    ):
        lib = Path(s["outer_library"])
        source = lib.with_suffix(".cpp")
        identity = dict(library_sha256=sha(lib), source_sha256=sha(source))
        if identity != archive[key]:
            raise ValueError("receiver source or binary identity")
        text = source.read_text()
        rec = text.split("void record(", 1)[1].split('extern "C"', 1)[0]
        if (
            "put(f,steps)" not in rec
            or "f.write(" not in rec
            or "sequence" in rec
            or "owner" in rec
        ):
            raise ValueError("native recording schema changed, re-audit")
        receivers[key] = identity
    rows = read(reports / "SLA_HIERARCHICAL_V16/V1_COMPILER_ABLATION_ACTUAL.json")[
        "windows"
    ]
    if len(rows) != 16 or len({w["id"] for w in rows}) != 16:
        raise ValueError("window denominator")
    return dict(
        scope="frozen target TRAIN, offline CPU",
        receivers=receivers,
        historical_teacher_intervals=archive["historical_teacher_intervals"],
        historical_teacher_ticks=archive["historical_teacher_ticks"],
        windows=[
            dict(
                id=w["id"],
                held=w["grasp"],
                owner=w["old_owner"],
                seam_m_s=w["modes"]["C2_EXACT"]["g7_first_m_s"],
                limit_m_s=0.040,
            )
            for w in rows
        ],
        actual_student_reset_receipts=archive["archived_student_owner_receipts"],
        unavailable=archive["missing"],
        new_native_requests=0,
        real_student_task_attempts=0,
        decision="V2_GRIPPER_BLOCKED",
    )


if __name__ == "__main__":
    print(
        json.dumps(
            main(Path(sys.argv[1]), Path(__file__).resolve().parents[2]),
            indent=2,
            allow_nan=False,
        )
    )
