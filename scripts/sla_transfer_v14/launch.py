"""Run CPU diagnostics with the recorded Python dependencies. No process from a teacher command is launched."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "shadow",
            "regression",
            "clock_replay",
            "historical_tracking",
            "technical_registration",
            "technical_shadow",
        ],
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--registration", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    prior = json.loads((root / "v13/run01/S0_ACTUAL.json").read_text())
    card = next(Path(k) for k in prior["inputs"] if k.endswith("/EXECUTION_SPEC.json"))
    spec = json.loads(card.read_text())
    env = {**os.environ, **spec["native_environment"]}
    env.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(root / "v13/release"),
            str(root / "v12"),
            str(root / "v12/CPU_LIB"),
            spec["source_root"],
            env["PYTHONPATH"],
        ]
    )
    script = Path(__file__).with_name(args.stage + ".py")
    if args.stage == "shadow" and args.archive is None:
        parser.error("--archive is required for the source hash comparison")
    params = [str(root), str(output)]
    if args.stage == "shadow":
        params = [str(output), str(args.archive.resolve())]
    elif args.stage == "clock_replay":
        params = [str(output)]
    elif args.stage == "technical_shadow":
        if args.registration is None:
            parser.error("--registration must identify the proposed card file")
        params.append(str(args.registration.resolve()))
    output.mkdir(exist_ok=False, parents=True, mode=0o700)
    os.execve(spec["python"], [spec["python"], "-B", str(script), *params], env)


if __name__ == "__main__":
    main()
