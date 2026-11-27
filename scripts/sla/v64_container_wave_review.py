"""Read closed Container qualifications without re-running a physical case."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def review(case, layout):
    lease = read(case / "control/LEASE.json")
    spec = case / "EXECUTION_SPEC.json"
    if lease["status"] != "CLOSED" or lease["spec_sha256"] != sha(spec):
        raise ValueError("Read only finished, sealed native cases")
    terminal = case / "RESULT.private.json"
    result = read(terminal)
    if result["physical_requests"] != 1 or result["actual_native_integrations"] > 50703:
        raise ValueError("Unexpected Container request or integration accounting")
    detail = result.get("detail") or {}
    score = detail.get("score") or {}
    raw = case / "native-01"
    trajectory = []
    center = np.array(layout["fixture_world_xyz"][:2])
    log = raw / "MOBILE_SAMPLES.private.jsonl"
    if log.exists():
        with log.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["native_end_step"] <= 5703:
                    continue
                position = np.array(row["object_pose"])[:3, 3]
                trajectory.append(dict(tick=row["native_end_step"], object_xyz=position.tolist(),
                    center_distance_mm=1000*float(np.linalg.norm(position[:2]-center)),
                    gripper_fraction=row["installed_EEF10"][9], held=row["native_grasp"]))
    held = [row for row in trajectory if row["held"]]
    contact = None
    physics = raw / "physics.jsonl"
    if result["unsafe"] and physics.exists():
        with physics.open() as stream:
            header = json.loads(next(stream))
            geometry = {item["id"]: item for item in header["geoms"]}
            for row in map(json.loads, stream):
                if row.get("contact_stop") or row.get("joint_stop"):
                    contact = dict(step=row["step"], contact_stop=row["contact_stop"], joint_stop=row["joint_stop"],
                        pairs=[dict(geoms=[geometry[item["geom1"]], geometry[item["geom2"]]], distance_m=item["dist"])
                               for item in row["contacts"]])
                    break
    return dict(id=case.name, native_result_sha256=sha(terminal), requests=result["physical_requests"],
                ticks=result["actual_native_integrations"], shutdown_clean=result["shutdown_clean"],
                technical_failure=result["technical_failure"], unsafe=result["unsafe"], failure=result["failure"],
                qualification_pass=detail.get("container_qualification_success", False),
                terminal_reason=detail.get("terminal_reason"), terminal_score=score, stages=detail.get("stages"),
                native_task_atoms=len(trajectory), first_held_atom=held[0] if held else None,
                best_center_while_held=min(held, key=lambda row:row["center_distance_mm"]) if held else None,
                last_completed_atom=trajectory[-1] if trajectory else None,
                first_native_contact_stop=contact, stops=result.get("physics_stops", []),
                post_STOP_ticks=(result.get("trace_audit") or {}).get("integrations_after_first_STOP"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layouts", type=Path)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    layout, = (item for item in read(args.layouts)["rows"] if item["id"] == args.case.name)
    result = review(args.case, layout)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(sha(args.output))
