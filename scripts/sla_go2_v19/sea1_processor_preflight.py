"""Read the actual cross-painted archive through the frozen SLA input transforms."""

import argparse
import os
from pathlib import Path

import numpy as np

from source_custody import BASE, pin, read, require, target_case, target_obs, write_new
from checkpoint_inputs import inspection_tools
from policy_query import input_identity


def changed(a, b):
    require(a.keys() == b.keys(), "processor current camera keys changed")
    return sorted(k for k in a if a[k] != b[k])


def run(root, bridge, output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("JAX_PLATFORMS") == "cpu",
            "CPU preprocessing only")
    from jax import default_backend

    require(default_backend() == "cpu", "CPU backend required")
    evidence = read(bridge / "TARGET_BRIDGE_ACTUAL.json")
    seals = read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")
    members = read(pin(root / "v18/sealed-Z1/members.json", seals["member_manifest_sha256"]))
    inspect, _, meta = inspection_tools(root)
    rows = []
    for row in evidence["slots"]:
        layout, index = row["id"].split("-")
        original = target_case(root, layout) / f"query-{int(index):02d}/INPUT.npz"
        raw = target_obs(pin(original, members[str(original.relative_to(root / "v18"))]))
        painted = pin(bridge / (row["id"] + ".private.npz"), row["archive_sha256"])
        with np.load(painted, allow_pickle=False) as z:
            candidate = dict(raw, images=dict(raw["images"], cam_high=z["head"], cam_right_wrist=z["wrist"]))
        require(input_identity(raw)[0] == row["original_input_sha256"]
                and input_identity(candidate)[0] == row["diagnostic_input_sha256"],
                "CPU processor input identity changed")
        base, new = inspect(raw), inspect(candidate)
        equal = ("state", "normalized", "tokens_sha256", "tokens_used", "tokens_full", "prompt")
        require(all(base[k] == new[k] for k in equal) and not new["prompt_truncated"],
                "same-view appearance change affected state or instruction")
        current = changed(base["network_image_sha256"], new["network_image_sha256"])
        require(current == ["base_0_rgb", "right_wrist_0_rgb"], "unexpected edited processor views")
        memory = [i for i, (a,b) in enumerate(zip(base["history_images_sha256"],
                                                new["history_images_sha256"], strict=True)) if a != b]
        require(memory == list(range(9)), "expected an independently rendered edit at all nine history slots")
        rows.append(dict(
            id=row["id"], tokens_used=new["tokens_used"], identical_state_prompt_tokens=True,
            changed_current_views=current, changed_causal_memory_indices=memory,
            original_left_rgb_placeholder_unchanged=True,
            original_processor_image_mask={"base_0_rgb": True, "right_wrist_0_rgb": True, "left_wrist_0_rgb": True},
            left_mask_only_treatment_run=False,
        ))
    require(len(rows) == 4, "expected fixed Go2 developer roster")
    write_new(output, dict(
        status="CPU_OFFICIAL_PROCESSOR_DIAGNOSTIC", token_capacity=meta["token_capacity"], rows=rows,
        new_model_forwards=0, physics_ticks=0, updates=0, scientific_observation_pairs_qualified=0,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "bridge", "output"):
        cli.add_argument(k, type=Path)
    args = cli.parse_args()
    run(args.root.resolve(), args.bridge.resolve(), args.output.resolve())
