"""Read-only native identities/contracts. No simulator or model is imported."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from go2arx5_vla.target_mujoco.reference import interface_contract


def file_identity(path):
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "status": "MISSING"}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _scalar(text, name):
    match = re.search(r"(?m)^\s*" + re.escape(name) + r"\s*:?\s+([^#;\n]+)", text)
    return match.group(1).strip().strip('"') if match else None


def check_clock(receipt):
    """Check recorded affine mapping; no clock fields are fabricated by doctor."""
    if not receipt:
        return {"status": "BLOCKED_LIVE_CLOCK_RECEIPT", "missing": ["real sim/ROS/monotonic pairs", "reset isolation", "expired-reference negative case"]}
    rows = receipt.get("samples", [])
    if len(rows) < 3:
        return {"status": "BLOCKED_LIVE_CLOCK_RECEIPT", "missing": ["at least3 actual clock samples"]}
    required = ("sim_time_s", "ros_time_s", "wall_monotonic_s", "mpc_time_s")
    if any(any(k not in x or isinstance(x[k], bool) or not isinstance(x[k], (int, float)) or not math.isfinite(x[k]) for k in required) for x in rows):
        raise ValueError("clock receipt has missing/nonfinite source time")
    monotonic = all(all(b[k] > a[k] for a, b in zip(rows, rows[1:])) for k in required)
    mapping = receipt.get("sim_to_mpc", {})
    slope, offset = mapping.get("slope"), mapping.get("offset_s")
    if not isinstance(slope, (int, float)) or not isinstance(offset, (int, float)) or not math.isfinite(slope) or not math.isfinite(offset) or slope <= 0:
        return {"status": "BLOCKED_LIVE_CLOCK_RECEIPT", "missing": ["finite positive sim_to_mpc affine map"]}
    residual = max(abs(x["mpc_time_s"] - (slope * x["sim_time_s"] + offset)) for x in rows)
    checks = {"monotonic": monotonic, "mapping_max_residual_s": residual,
              "real_output": receipt.get("real_output") is True,
              "reset_isolation": receipt.get("reset_isolation_checked") is True,
              "expired_reference_hold": receipt.get("expired_reference_hold_checked") is True,
              "episode_id": bool(receipt.get("episode_id"))}
    passed = monotonic and residual <= .025 and all(checks[k] for k in ("real_output", "reset_isolation", "expired_reference_hold", "episode_id"))
    return {"status": "CLOCK_CONTRACT_BOUND" if passed else "BLOCKED_CLOCK_OR_RESET", "checks": checks,
            "sim_to_mpc": mapping, "samples": rows}


def inspect(config):
    root = Path(config["private_root"]).resolve()
    source_paths = ["env.sh", "run.sh",
        "src/planning/go2_x5_planning/go2_x5_planning/ee_trajectory.py",
        "src/planning/go2_x5_planning/go2_x5_planning/ee_trajectory_node.py",
        "src/ctrl/go2_x5_ocs2/src/go2_x5_ocs2_node.cpp",
        "src/ctrl/go2_x5_ocs2/src/WbcBridgeCore.cpp",
        "src/ctrl/go2_x5_ocs2/include/wbc_bridge/wbc_bridge_core.hpp",
        "src/ctrl/go2_x5_ocs2/launch/go2_x5_ros_mpc.launch.py",
        "src/ctrl/go2_x5_ocs2/config/task_floating.info",
        "src/ctrl/rl_sar/src/rl_sar/src/rl_sim_mujoco.cpp",
        "src/ctrl/rl_sar/policy/go2_x5/base.yaml",
        "src/ctrl/rl_sar/policy/go2_x5/N1_s29_175632/config.yaml",
        "src/ctrl/rl_sar/policy/go2_x5/N1_s29_175632/policy.pt"]
    source_paths += [str(p.relative_to(root)) for p in root.glob("src/**/arx5_ac1_floating.urdf")]
    source_paths += [str(p.relative_to(root)) for p in root.glob("src/**/mjcf/go2_x5.xml")]
    binaries = config.get("binaries", ["build/mujoco/bin/rl_sim_mujoco", "install/lib/go2_x5_ocs2/go2_x5_ocs2_node", "install/lib/ocs2_mobile_manipulator_ros/mobile_manipulator_mpc_node", "install/lib/ocs2_mobile_manipulator_ros/mobile_manipulator_sqp_mpc_node", "install/lib/go2_x5_planning/ee_trajectory_node"])
    # ponytail: hash existing private files; no copy, model load or dependency probe.
    source_paths += [str(p.relative_to(root)) for p in root.glob("src/**/*") if p.is_file()]
    files = [file_identity(root / x) for x in dict.fromkeys(source_paths + binaries)]
    missing = [x["path"] for x in files if x.get("status") == "MISSING"]
    if not any(x.endswith("arx5_ac1_floating.urdf") for x in source_paths):
        missing.append("native planning URDF")
    if not any(x.endswith("mjcf/go2_x5.xml") for x in source_paths):
        missing.append("native robot MJCF")
    def read(relative):
        path = root / relative
        return path.read_text() if path.is_file() else ""
    base = read("src/ctrl/rl_sar/policy/go2_x5/base.yaml")
    task = read("src/ctrl/go2_x5_ocs2/config/task_floating.info")
    dt, decimation = _scalar(base, "dt"), _scalar(base, "decimation")
    rates = {"robot_control_dt_s": float(dt) if dt else None,
             "leg_RL_dt_s": float(dt) * int(decimation) if dt and decimation else None,
             "OCS2_bridge_dispatch_Hz_source_default": 200,
             "MPC_solver_Hz_config": _scalar(task, "mpcDesiredFrequency"),
             "MRT_Hz_config": _scalar(task, "mrtDesiredFrequency"),
             "MPC_horizon_s_config": _scalar(task, "timeHorizon"),
             "manager_republish_Hz_source_default": 10, "teacher_plan_period_s": .4,
             "data_sample_Hz": 20, "physics_dt_s": None,
             "physics_dt_status": "MUST_READ_ACTUAL_LOADED_MODEL_OPT_TIMESTEP"}
    clock_path = config.get("clock_receipt")
    clock_receipt = json.loads(Path(clock_path).read_text()) if clock_path else None
    clock = check_clock(clock_receipt)
    if clock_path:
        clock["receipt_identity"] = file_identity(clock_path)
    versions = {}
    for package in ("mujoco", "rclpy", "numpy", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    contract = interface_contract()
    contract["TCP_frame_config"] = _scalar(task, "eeFrame")
    contract["MPC_state_12"] = ["mount_world_x", "mount_world_y", "mount_world_z", "yaw", "pitch", "roll", "arm_q1", "arm_q2", "arm_q3", "arm_q4", "arm_q5", "arm_q6"]
    contract["MPC_input_12"] = ["world_vx", "world_vy", "world_vz", "yaw_rate", "pitch_rate", "roll_rate", "arm_dq1", "arm_dq2", "arm_dq3", "arm_dq4", "arm_dq5", "arm_dq6"]
    safety = {"odometry_timeout_s_source_default": 1.0, "arm_command_watchdog_s_source": .5,
              "base_command_timeout_s_source": .2, "gait_phase_timeout_s_source_default": .2,
              "arm_watchdog_initial_wait": "only armed after first valid target",
              "arm_timeout_behavior": "switch HOLD", "base_timeout_behavior": "CoastExternalBaseCommand",
              "policy_invalid_or_out_of_horizon": "no dispatch; arm watchdog preserves native stop",
              "reference_TTL": "none; trajectory end does not force HOLD",
              "hinge_epsilon_rad": 1e-4, "slide_epsilon_m": 1e-6,
              "hinge_immediate_stop_rad": 1e-3, "slide_immediate_stop_m": 1e-4,
              "epsilon_sustained_stop_s": .020, "formal_gate_no_numerical_warning": True,
              "native_limits": "source/config identities bound; actual loaded model limits recorded by owner"}
    domain = os.environ.get("ROS_DOMAIN_ID")
    localhost = os.environ.get("ROS_LOCALHOST_ONLY") == "1" or os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") == "LOCALHOST"
    access = config.get("access_receipt", {})
    live_access = access.get("domain_228_unused_before_launch") is True and access.get("single_owner_lease") is True and access.get("hardware_absent") is True and bool(access.get("process_topic_receipt"))
    if domain != "228" or not localhost:
        missing.append("localhost-only ROS domain228 environment")
    if not live_access:
        missing.append("actual domain/PID/lease/topic/hardware access receipt")
    loaded = config.get("loaded_runtime_receipt", {})
    runtime_keys = ("mujoco_version", "physics_timestep_s", "named_joint_actuator_map",
                    "actual_native_parameters", "core_pin_manifest_sha256", "safety_clock_source")
    if any(not loaded.get(k) for k in runtime_keys):
        missing.append("actual loaded model/version/mapping/core-pin/safety-clock receipt")
    if loaded.get("physics_timestep_s"):
        step = loaded["physics_timestep_s"]
        if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or step <= 0:
            raise ValueError("loaded physics timestep must be positive finite")
        rates["physics_dt_s"] = step
        rates["physics_dt_status"] = "ACTUAL_LOADED_MODEL_RECEIPT"
    if clock["status"] != "CLOCK_CONTRACT_BOUND":
        missing.append("actual clock/reset/expiry contract")
    return {"status": "G0_CONTRACT_BOUND" if not missing else "ENV_READY_INTERFACE_BLOCKED",
            "missing": missing, "SOURCE_MANIFEST": files, "INTERFACE_CONTRACT": contract,
            "CLOCK_CONTRACT": clock, "SAFETY_LIMITS": safety, "rates": rates,
            "runtime": {"python": sys.version, "python_executable": sys.executable, "package_versions": versions,
                        "ROS_DOMAIN_ID": domain, "localhost_only": localhost, "access_receipt": access,
                        "loaded_runtime_receipt": loaded},
            "GR00T_forwards": 0, "WBC_forwards": 0, "physics_integrations": 0,
            "selected": "NONE", "deployment": "NONE"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="local-only private_root/binaries/clock/access config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = inspect(json.loads(Path(args.config).read_text()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "missing": result["missing"]}))


if __name__ == "__main__":
    main()
