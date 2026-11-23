"""Verify an independent deterministic CPU replay against sealed numerical outputs."""

from pathlib import Path
import sys
from audit import read, record, sha


def main(first, again):
    checks = {}
    for i in range(4):
        name = f"S{i}_ACTUAL.json"
        original = read(first / name)
        rerun = read(again / name)
        for d in (original, rerun):
            d.pop("elapsed_wall_s", None)
        if original != rerun:
            raise ValueError("Action or evidence difference: " + name)
        checks[name] = "equal excluding elapsed wall time"
    seen = sorted(p.name for p in (first / "arrays").glob("*.npz"))
    assert seen == sorted(p.name for p in (again / "arrays").glob("*.npz"))
    for name in seen:
        assert sha(first / "arrays" / name) == sha(again / "arrays" / name)
    record(
        again / "REPRODUCED.json",
        dict(
            reports=checks,
            array_pairs=len(seen),
            identical_hashes=True,
            extra_scientific_windows=0,
            native_ticks=0,
            new_model_forwards=0,
        ),
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
