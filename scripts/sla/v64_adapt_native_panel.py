"""Bind a closed A250 to the original Corelab HOLD and PLACE executor."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from m38_gpu_preflight_v1 import read, require, sha
from v62_source_native import V65, source_role
from v64_adapt_endpoint import END, PLAN


def dispatch(base, overlay, original, root, phase, *, execute, recovery=False, step=None, matched_entry=None):
    require(recovery or phase != "INDEPENDENT_HOLD", "Independent HOLD is reserved for V6.5")
    hold = phase in ("HOLD", "INDEPENDENT_HOLD")
    work = overlay.parent
    plan, end, kind = (V65, 1000, "RECOVERY") if recovery else (PLAN, END, "ADAPT")
    if step is not None:
        require(recovery and step in (1000, 3000), "Unknown V6.5 native milestone")
        end = step
    prefix = source_role(end, plan)
    invocation = f"{prefix if recovery else 'A250'}_PANEL_INVOCATION.json"
    expected = f"{prefix}-{'NATIVE' if execute else 'PREFLIGHT'}-{phase}-01"
    require(root.parent == work and root.name == expected and not root.exists(), "Use the unique fixed adaptation panel root")
    archived = work / "SOURCE0250-PLACE-01"
    old = read(archived / "SOURCE0250_P01_701/EXECUTION_SPEC.json")
    registration = read(archived / "COMPARISON.json")
    require(read(archived / "control/LEASE.json")["status"] == "CLOSED"
            and read(archived / "worker/control/LEASE.json")["status"] == "CLOSED",
            "Close the original archived execution owners before using their pinned transport")
    evidence = overlay / "reports/SLA_WBC_INTERFACE_COMPARISON_V1"
    study = overlay / "reports/SLA_SUCCESS_EXECUTION_AUDIT_V2"
    v2 = overlay / "docs/protocols/SLA_SUCCESS_EXECUTION_AUDIT_V2_REGISTRY_20261126.json"
    panel = overlay / f"{kind}_{phase}_PREFLIGHT.json"
    binding = read(panel)
    require(binding["role"] == prefix and binding["v62_plan"] == plan
            and binding["source_phase"] == ("LONG_HOLD" if hold else "LONG_PLACE"),
            "Wrong matched A250 panel artifact")
    if execute:
        for member in (f"{prefix}-DEV-01/RESULT.json", f"{prefix}-FEEDBACK-01/RESULT.json"):
            require(read(work / member)["checkpoint"] == binding["checkpoint_sha256"],
                    "Offline comparison belongs to another model")
        prior = work / f"{prefix}-PREFLIGHT-{phase}-01"
        cpu = read(prior / "COMPARISON.json")
        finished = read(prior / invocation)
        require(cpu["v4_panel"]["sha256"] == sha(panel) and cpu["v4_model_role"] == binding["role"]
                and not finished["executed"] and finished["phase"] == phase
                and finished["comparison_sha256"] == sha(prior / "COMPARISON.json"),
                "Require the completed pinned CPU phase first")
        if end == 3000:
            require(matched_entry is not None, "Require the full 3K matched native entry before dispatch")
            entry = read(matched_entry)
            group = entry["groups"][phase]
            require(entry["role"] == prefix and entry["plan"] == plan
                    and entry["stage"] == "NATIVE_ENTRY_MATCHED"
                    and entry["caps"] == dict(physical_requests=10, integrations=397_030, model_queries=1360)
                    and entry["launched_requests"] == 0 and entry["native_integrations"] == 0
                    and entry["checkpoint"] == binding["checkpoint_sha256"]
                    and group["preflight"]["sha256"] == finished["comparison_sha256"]
                    and group["source_audit"] == binding["source_audit"],
                    "The original ten-cell preflight must bind the executed phase")
    worker = read(base / "V2-M-PLACE-01/worker/EXECUTION_SPEC.json")
    env = dict(os.environ, **worker["native_environment"])
    env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(map(str, (overlay / "scripts/sla", overlay, original / "source",
                                                   original / "source/src", env["PYTHONPATH"]))))
    noise = binding["noise"]["hold" if hold else "place"]["path"]
    command = [worker["python"], "-B", str(overlay / "scripts/sla/m38_a_native_v1.py"),
               str(root), str(original), str(base / "V2-M-PLACE-01/worker/cache"), str(evidence), noise,
               old["terminal_step_library"], registration["qualified_transport_spec"]["path"],
               "--phase", binding["source_phase"], "--study-evidence", str(study),
               "--v4-registry", registration["v4_registry"]["path"], "--v4-panel", str(panel),
               "--v2-registry", str(v2)]
    if execute:
        command.append("--execute")
    subprocess.run(command, env=env, check=True)
    result = read(root / "COMPARISON.json")
    summary = dict(role=result["v4_model_role"], phase=phase, executed=execute,
                   conditions=[row["id"] for row in result["schedule"]], comparison_sha256=sha(root / "COMPARISON.json"))
    with (root / invocation).open("x") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "overlay", "original", "root"):
        parser.add_argument(name, type=Path)
    parser.add_argument("phase", choices=("HOLD", "PLACE", "INDEPENDENT_HOLD"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--step", type=int)
    parser.add_argument("--matched-entry", type=Path)
    dispatch(**vars(parser.parse_args()))
