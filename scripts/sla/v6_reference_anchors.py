"""Trace both Teacher H40 anchors through the original accepted receiver boundary."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from m38_gpu_preflight_v1 import pin, read, require
from scripts.pi0291_arx5_data.native_dataset import matrix_from10


def metric(p, q):
    angle = Rotation.from_matrix(p[..., :3, :3] @ q[..., :3, :3].swapaxes(-1, -2)).magnitude()
    distance = np.linalg.norm(p[..., :3, 3] - q[..., :3, 3], axis=-1)
    return np.stack((1000 * distance, np.degrees(angle)), axis=-1)


def receiver(pose):
    observed = np.eye(4)
    observed[:3, 3] = pose[:3]
    observed[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    return observed


def transport(poses, actual, accepted):
    result = poses.copy()
    result[..., :3, 3] += accepted[:3, 3] - actual[:3, 3]
    result[..., :3, :3] = accepted[:3, :3] @ actual[:3, :3].T @ poses[..., :3, :3]
    return result


def original_cases(native_root, commands):
    case_list = read(commands / "MANIFEST.json")["episodes"]
    require(len(case_list) == 52, "Use the fixed original V4 Teacher population")
    for row in case_list:
        matching = [p for p in native_root.glob(f"*/cases/{row['episode']}/EXECUTION_SPEC.json")
                    if hashlib.sha256(p.read_bytes()).hexdigest() == row["original_markers"]]
        require(len(matching) == 1, "Expected exactly one original execution specification")
        terminal = read(pin(matching[0].with_name("RESULT.private.json"), row["result"]))
        cap = terminal["detail"]["capture"]
        export = read(pin(Path(cap["path"]), cap["sha256"]))
        p = export["labels"]
        queries = read(pin(Path(p["path"]), p["sha256"]))
        require(len(queries) == row["windows"], "Original query number changed")
        yield row, read(matching[0]), queries, p["sha256"]


def run(native_root, commands, output):
    import inspect
    from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal

    exact_program = inspect.getsource(compile_proposal.__globals__["task"].FiftyMsProgram)
    require("reference[:3,3]+raw[:,:3]-actual[:3,3]" in exact_program
            and "reference[:3,:3]@actual[:3,:3].T@rotations" in exact_program,
            "Bind the runtime that applies the original left rotation and displacement")
    metrics = defaultdict(list)
    proofs = []
    for row, _, queries, query_sha in original_cases(native_root, commands):
        for index, query in enumerate(queries):
            latest = query["recent_reference"]
            tick = 5703 + 250 * index
            require(query["native_step"] == tick == latest["native_step"] and latest["exact_receiver_knot"],
                    "The reported receiver reference must be at the same physical boundary")
            nominal = matrix_from10(np.asarray(query["nominal_world_H40"]))
            installed = matrix_from10(np.asarray(query["installed_reference_H40"]))
            actual = np.asarray(query["actual_state"]["tcp"])
            accepted = receiver(latest["accepted_pose7"])
            executed = installed[:10]
            shifted = transport(nominal[:10], actual, accepted)
            second = transport(executed, actual, accepted)
            inverse = transport(executed, accepted, actual)
            calc = {"nominal_first_to_actual": metric(nominal[0], actual),
                    "installed_first_to_receiver": metric(installed[0], accepted),
                    "accepted_reference_to_actual": metric(accepted, actual),
                    "world_reanchor_P10_to_installed": metric(shifted, executed),
                    "reanchor_consumed_label_to_installed": metric(second, executed),
                    "inverse_then_reanchor_to_installed": metric(transport(inverse, actual, accepted), executed)}
            for name, result in calc.items():
                metrics[query["phase"], name].append(result.reshape(-1, 2))
        proofs.append(dict(episode=row["episode"], query_sha256=query_sha, native_windows=len(queries)))
    measures = []
    for (phase, relation), values in sorted(metrics.items()):
        amount = np.concatenate(values)
        measures.append(dict(phase=phase, relation=relation, matched_atoms=len(amount),
                             p50_mm_deg=np.median(amount, axis=0).tolist(),
                             p95_mm_deg=np.percentile(amount, 95, axis=0).tolist(),
                             max_mm_deg=amount.max(axis=0).tolist()))
    with output.open("x") as stream:
        json.dump(dict(queries=sum(row["native_windows"] for row in proofs), cases=proofs, metrics=measures,
                       original_program_source_sha256=hashlib.sha256(exact_program.encode()).hexdigest(),
                       native_integrations=0, model_forwards=0, optimizer_updates=0),
                  stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native_root", "commands", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
