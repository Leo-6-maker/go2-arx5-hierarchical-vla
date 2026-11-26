"""Seal a phase-matched 24-window retention diagnosis before any new forward."""

import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_bytes())


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def selected(case):
    root = case / "native-01/g7-queries"
    queries = []
    for folder in sorted(root.iterdir()):
        request = folder / "MODEL_REQUEST.private.json"
        returned = folder / "MODEL_RETURN.private.json"
        if request.is_file() and returned.is_file():
            queries.append((load(request)["native_step"], folder))
    return queries


def row(case, folder, phase, event):
    terminal = case / "RESULT.private.json"
    request = folder / "MODEL_REQUEST.private.json"
    returned = folder / "MODEL_RETURN.private.json"
    meta = load(folder / "INPUT_META.private.json")
    asked = load(request)
    response = load(returned)["response"]
    if not (load(case / "control/LEASE.json")["status"] == "CLOSED"
            and not load(terminal)["technical_failure"] and asked["input"] == meta["input"]
            and asked["previous_reference"] == meta["previous_reference"]
            and not meta["teacher_or_object_truth_in_model_input"]
            and len(response["comparison_noise_sha256"]) == 64):
        raise ValueError("Diagnosis needs the exact closed feedback input and actual noise")
    for key in (meta["input"], response["action"]):
        if ref(Path(key["path"])) != key:
            raise ValueError("Real archived sample was altered")
    return dict(id=f"{case.name}_Q{int(folder.name):04d}", phase=phase, query=int(folder.name),
        tick=asked["native_step"], grasp_tick=event, input=meta["input"], request=ref(request),
        terminal=ref(terminal), returned=ref(returned), noise_sha256=response["comparison_noise_sha256"],
        noise_id=asked["noise_id"])


def prepare(root, out):
    prior = load(root / "v4-long-watch-01/ROSTER.json")
    rows = list(prior["rows"])
    for task in ("PLACE", "HOLD"):
        package = root / f"V4-LONG_{task}-M1000-NATIVE-01-RUN"
        for case in sorted(package.glob("LONG_M1000_*")):
            result = case / "RESULT.private.json"
            if not result.is_file():
                continue
            queries = selected(case)
            for i, phase in ((-3, "PRECONTACT_EARLIER"), (-1, "PRECONTACT_LAST")):
                rows.append(row(case, queries[i][1], phase, None))
    targets = (("PLACE", "P02_701"), ("HOLD", "N02_A"), ("HOLD", "N02_B"))
    for task, suffix in targets:
        case = root / f"V4-LONG_{task}-PARENT-NATIVE-01" / f"LONG_PARENT_{suffix}"
        events = load(case / "RESULT.private.json")["detail"]["score"]["events"]
        grasp = events["target_grasp"]["native_step"]
        lift = events["target_lift20mm"]["native_step"]
        queries = selected(case)
        before = max((x for x in queries if x[0] < grasp), key=lambda x: x[0])
        lifted = min((x for x in queries if x[0] >= lift), key=lambda x: x[0])
        rows.append(row(case, before[1], "PARENT_BEFORE_GRASP", grasp))
        rows.append(row(case, lifted[1], "PARENT_AFTER_LIFT20", grasp))
    if len(rows) != 24 or len({(r["request"]["sha256"], r["noise_sha256"]) for r in rows}) != 24:
        raise ValueError("Expected 24 distinct archived inputs")
    with out.open("x") as sink:
        json.dump(dict(prior, rows=rows, prior_roster=ref(root / "v4-long-watch-01/ROSTER.json"),
                       scope="ARCHIVED_DEV_DIAGNOSIS", source_updates=0), sink, indent=2)
        sink.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    prepare(**vars(parser.parse_args()))
