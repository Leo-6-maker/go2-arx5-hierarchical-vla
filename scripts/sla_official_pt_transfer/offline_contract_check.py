"""CPU-only public-contract audit for released post-trained SLA_Memory vs Go2 sensors.

This script performs NO model forward, downloads, physical simulation, or deployment.
It reconstructs the official RobodojoMemoryBuffer's frame-offset behavior
from its pinned source code and emits a transparent compatibility classification.

It does not attest checkpoint file hashes or prove student task success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


UPSTREAM = {
    "official_source_repository": "XPolicyLab/XPolicyLab",
    "official_source_sha": "c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454",
    "official_checkpoint_repository": "haofuly/SLA_benchmark_ckpts",
    "official_checkpoint_subdir": "SLA_Memory_robodojo_real_ckpt",
    "official_checkpoint_revision": "7f6880569c3be777291e646fad1b4951245e8553",
    "train_config_name": "train_robodojo_real",
    "extra_frames_first": False,
    "required_current_views": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
    "required_arm_state_layout": "left8+right8, each [xyz,qw,qx,qy,qz,gripper]",
    "action_horizon": 50,
    "action_width": 16,
    "native_arx_x5_fps": 30,
}


def memory_seconds(policy_fps: int, actual_fps: int, count: int = 8) -> list[float]:
    if policy_fps <= 0 or actual_fps <= 0 or count <= 0:
        raise ValueError("fps and history length must be positive")
    # The official RobodojoMemoryBuffer selects -seconds_ago*model_fps frames.
    # Each Go2 frame is observed at 1/actual_fps actual seconds.
    return [round(i * policy_fps / actual_fps, 6) for i in range(count, 0, -1)]


def audit(go2_fps: int, current_views: list[str], *, real_left_arm: bool) -> dict:
    seen = list(current_views)
    missing = [name for name in UPSTREAM["required_current_views"] if name not in seen]
    observed_history_seconds = memory_seconds(UPSTREAM["native_arx_x5_fps"], go2_fps)
    desired_history_seconds = list(range(8, 0, -1))
    mismatch = observed_history_seconds != desired_history_seconds
    return {
        "scope": "static_official_source_contract_audit_NO_NEW_MODEL_FORWARD",
        "official_config_source": UPSTREAM,
        "go2": {
            "native_fps": go2_fps,
            "available_real_views": seen,
            "real_left_arm_present": real_left_arm,
        },
        "direct_observation_contract_satisfied": not missing and real_left_arm and not mismatch,
        "missing_current_views": missing,
        "head_history_if_official_fps_is_used_on_raw_go2_frames_seconds_ago": observed_history_seconds,
        "required_head_history_seconds_ago": desired_history_seconds,
        "history_temporal_mismatch": mismatch,
        "not_a_model_weight_authentication": True,
        "not_a_model_forward_or_task_success_test": True,
        "verdict": "NOT_PLUG_AND_PLAY" if missing or mismatch or not real_left_arm else "STATIC_ABI_ONLY",
        "safe_next_step": "use source-pinned checkpoint+stats, honest OOD camera/arm placeholders, timestamp-correct history, action/world-frame adapter, and separate safety/gripper ownership before native trial",
    }


def self_test() -> None:
    assert memory_seconds(30, 30) == list(range(8, 0, -1))
    assert memory_seconds(30, 20) == [12.0, 10.5, 9.0, 7.5, 6.0, 4.5, 3.0, 1.5]
    result = audit(20, ["cam_head", "cam_right_wrist"], real_left_arm=False)
    assert result["missing_current_views"] == ["cam_left_wrist"]
    assert result["history_temporal_mismatch"]
    assert result["verdict"] == "NOT_PLUG_AND_PLAY"
    assert not audit(
        30,
        ["cam_head", "cam_left_wrist", "cam_right_wrist"],
        real_left_arm=True,
    )["history_temporal_mismatch"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--go2-fps", type=int, default=20)
    p.add_argument("--views", default="cam_head,cam_right_wrist")
    p.add_argument("--real-left-arm", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    self_test()
    result = audit(
        args.go2_fps,
        [s for s in args.views.split(",") if s],
        real_left_arm=args.real_left_arm,
    )
    payload = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
