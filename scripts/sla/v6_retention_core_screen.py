"""Screen a bounded retention checkpoint on the registered Corelab DEV states."""

import argparse
import json
from pathlib import Path

from v4_long_watch import watch
from v6_retention_compare import aggregate, feedback, key, read


def screen(base, root, checkpoint, provider_audit, output):
    proof, _ = read(provider_audit)
    step = proof["step"]
    roster = root / "ROSTER24.json"
    _, roster_digest = read(roster)
    watch(base, base / "V2-M-PLACE-01/worker/EXECUTION_SPEC.json", roster, checkpoint, step,
          provider_audit, output, execute=True, retained_roster_sha=roster_digest)
    generated, result_digest = read(output / "RESULT.json")
    original, reference_digest = read(root / "STEP0000_RETENTION/RESULT.json")
    full_course, course_digest = read(root / ("STEP0100_RETENTION" if step == 100 else
                              "STEP0250_RETENTION") / "RESULT.json")
    parent = {key(row): row for row in original["rows"]}
    intensive = {key(row): row for row in full_course["rows"]}
    rows = generated["rows"]
    if (len(parent) != 24 or len(rows) != 24 or generated["model_forwards"] != 24 or
            not {key(row) for row in rows} == set(parent) == set(intensive) or
            any(feedback(row) != feedback(parent[key(row)]) or feedback(row) != feedback(intensive[key(row)])
                for row in rows)):
        raise ValueError("Screen did not preserve the identical historical comparison")
    phases = sorted({row["phase"] for row in rows})
    by_phase = {name: aggregate([row for row in rows if row["phase"] == name], parent) for name in phases}
    diagnostic = dict(step=step, role=proof["role"], new_forwards=24, training_updates=0, physics_ticks=0,
                      original_course_comparator=full_course["step"], result_sha256=result_digest,
                      parent_sha256=reference_digest, original_course_sha256=course_digest,
                      roster_sha256=roster_digest, all=aggregate(rows, parent), phases=by_phase,
                      deltas_to_original_course={name: aggregate([r for r in rows if r["phase"] == name], intensive)
                                                 for name in phases})
    with (output / "SCREEN.json").open("x") as stream:
        json.dump(diagnostic, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "root", "checkpoint", "provider_audit", "output"):
        parser.add_argument(name, type=Path)
    screen(**vars(parser.parse_args()))
