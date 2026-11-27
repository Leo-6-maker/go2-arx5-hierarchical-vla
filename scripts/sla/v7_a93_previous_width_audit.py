"""Check A93's original TRAIN-069 windows against native consumed jaw commands."""

import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write


PUBLIC_INDEX = "9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56"
R4_CORRECTED = "1b57d74fd133a3acbfb57bdce10a7ac4b8d8b18fd7fad8ca72376530edcb2a21"
A74_ADDED = "e3487b9322a6ce7e795a69a596c762a775e7475af3752d6d0142ae5f37d3f0fd"
ORIGINS = {
    "CHV1": ("sla-core-hopper-v1-expert-chain-20261121-02", "sla-core-hopper-v1-red-place-pair-20261121-01",
             "sla-core-hopper-v1-red-hold-pilot-20261121-01", "sla-core-hopper-v1-blue-right-pilot-20261121-01"),
    "MEV1": ("sla-core-multienv-wave-a-20261122-02", "sla-core-multienv-wave-a-cold-20261122-03"),
}


def audit(archive, native_root, legacy_latest_case, r4_corrected, a74_added):
    index = read(pin(archive / "index.json", PUBLIC_INDEX))
    original = read(pin(r4_corrected, R4_CORRECTED))["episodes"]
    added = read(pin(a74_added, A74_ADDED))["episodes"]
    records = {row["episode"]: row["result"] for row in original}
    records.update({row["episode"]["id"]: row["native_result_sha256"] for row in added
                    if not row["episode"]["id"].startswith("EF5_")})
    episodes = [row for row in index["episodes"] if row["episode"] in records]
    require(len(episodes) == len(records) == 69
            and len({episode["episode"] for episode in episodes}) == 69,
            "The actual 52 R4 and 17 legacy A74 cases must cover the fixed A93 population")
    raw_receipts = []
    for episode in episodes:
        name = episode["episode"]
        terminal_sha = records[name]
        roots = [native_root / origin / "cases" / name for origin in ORIGINS[name.split("-", 1)[0]]]
        if name == legacy_latest_case.name:
            roots.append(legacy_latest_case)
        matches = [case for case in roots if (case / "RESULT.private.json").is_file()
                   and sha(case / "RESULT.private.json") == terminal_sha]
        require(len(matches) == 1, f"{name} actual TRAIN terminal must have a unique source")
        case, = matches
        lease = read(case / "control/LEASE.json")
        spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
        result = read(case / "RESULT.private.json")
        require(episode["split"] == spec["collection_split"] == "TRAIN"
                and lease["status"] == "CLOSED" and result["shutdown_clean"] and not result["unsafe"]
                and not result["technical_failure"] and result["completed"]
                and episode["scene_group_sha256"] == spec["frozen_files"][spec["model_path"]],
                "Expected the closed original TRAIN condition")
        portable = archive / episode["path"]
        require(portable.stat().st_size == episode["archive_bytes"], "Go2 portable payload size changed")
        with tarfile.open(portable, "r") as tar, tar.extractfile("labels.npz") as member:
            with np.load(io.BytesIO(member.read()), allow_pickle=False) as labels:
                ticks = labels["native_step"]
                widths = labels["prior_actual_width_m"]
        require(ticks.shape == widths.shape == (episode["windows"],)
                and (np.diff(ticks) == 250).all()
                and np.isfinite(widths).all(), "Invalid recorded action-query clock or width")
        remaining = dict(zip(ticks.tolist(), widths.tolist(), strict=True))
        matched = set()
        raw = case / "native-01/MOBILE_SAMPLES.private.jsonl"
        with raw.open() as stream:
            for row in map(json.loads, stream):
                tick = row["native_end_step"]
                if tick not in remaining:
                    continue
                measured = row["installed_EEF10"][9] * .044
                source_time = row["gripper_command"]["source_native_s"]
                require(tick not in matched and row["native_start_step"] == tick - 25 and row["ticks"] == 25
                        and row["safety_valid"] and abs(source_time - (tick - 25) * .002) < 1e-8
                        and abs(measured - remaining[tick]) < 1e-12,
                        f"{name} first or later width does not match the actual previous consumption")
                matched.add(tick)
        require(len(matched) == len(ticks), f"Incomplete native consumption coverage for {name}")
        raw_receipts.append(dict(episode=name, windows=len(ticks), first_tick=int(ticks[0]),
                                 first_prior_width_m=float(widths[0]), native_samples_sha256=sha(raw),
                                 native_terminal_sha256=terminal_sha))
    return dict(r4_corrected_sha256=R4_CORRECTED, a74_added_sha256=A74_ADDED, public_index_sha256=PUBLIC_INDEX,
                confirmed_windows=sum(row["windows"] for row in raw_receipts), confirmed_first_queries=len(raw_receipts),
                episodes=raw_receipts, model_forwards=0, optimizer_updates=0, native_integrations=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("archive", "native_root", "legacy_latest_case", "r4_corrected", "a74_added", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve the first raw TRAIN-069 audit")
    write(args.output, audit(args.archive, args.native_root, args.legacy_latest_case, args.r4_corrected, args.a74_added))
