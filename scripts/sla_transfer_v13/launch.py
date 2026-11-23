"""Launch the offline analysis in the frozen native Python environment; no solver or simulator launch."""

import argparse, json, os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--stage", choices=["run", "canonical_gate"], default="run")
a = p.parse_args()
root = a.root.resolve()
out = a.output.resolve()
r = json.loads((root / "evidence/P1_REGISTRATION.json").read_text())["target_windows"][0]
spec = json.loads(
    (
        Path(r["episode_pin"]["path"]).parents[3] / "cases" / r["episode"] / "EXECUTION_SPEC.json"
    ).read_text()
)
env = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONPATH": os.pathsep.join(
        [str(root / "v12"), str(root / "v12/CPU_LIB"), spec["native_environment"]["PYTHONPATH"]]
    ),
}
os.execve(
    spec["python"],
    [spec["python"], str(Path(__file__).with_name(a.stage + ".py")), str(root), str(out)],
    env,
)
