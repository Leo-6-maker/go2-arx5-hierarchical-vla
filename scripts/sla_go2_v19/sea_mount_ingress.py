"""Verify that the two archived V2 captures change only the official auxiliary slot."""

import argparse
import os
from pathlib import Path

import numpy as np

from sea_mount_capture import PANEL, SCREEN_SHA, SLOTS
from sea_visual_support import EVIDENCE, T2_SHA
from sea_target_t2 import CONDITION, anchors_and_payloads
from sea1b_panel import fields_at_device, registration, verified_processor
from source_custody import digest, pin, read, require, write_new
from policy_query import input_identity, tensor_sha


def run(root, capture, output):
    import jax

    require(jax.default_backend() == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "CPU processor only")
    close_path = capture / "CAPTURE_CLOSE.json"
    result = read(close_path)
    require(result["panel"] == PANEL and result["screen_sha256"] == SCREEN_SHA
            and result["rendered"] == result["entered"] == 2
            and tuple(result["results"]) == SLOTS
            and result["new_physics_integrations"] == result["model_forwards"] == result["optimizer_updates"] == 0,
            "incomplete closed two-frame capture")
    old = read(pin(EVIDENCE / "TARGET_T2/COLD.json", T2_SHA))
    frozen, _ = registration()
    prepare, _, processor = verified_processor(root, frozen)
    require(processor == old["processor"], "fixed model processor changed")
    anchors, payloads, prior = anchors_and_payloads(root, prepare)
    rows = {}
    for name in SLOTS:
        rec = result["results"][name]
        path = pin(capture / (name+".aux.private.npy"), rec["rgb_file_sha256"])
        aux = np.load(path, allow_pickle=False)
        require(aux.shape == (480, 640, 3) and aux.dtype == np.uint8 and aux.any()
                and tensor_sha(aux) == rec["rgb_sha256"]
                and rec["old_anchor_sha256"] == anchors[name]["archive_sha256"]
                and rec["input_native_state_sha256"] == anchors[name]["native_packet_sha256"],
                "not the sealed same-state V2 RGB")
        current = payloads[name, CONDITION]
        v2 = dict(images={**current["images"], "cam_left_wrist": aux}, state=current["state"].copy(),
                  prompt=current["prompt"])
        previous_id, previous = input_identity(current)
        candidate_id, candidate = input_identity(v2)
        require(previous_id == anchors[name]["input_sha256"]
                and previous["cam_left_wrist"] == rec["old_rgb_sha256"]
                and {k for k, v in previous.items() if candidate[k] != v} == {"cam_left_wrist"},
                "V2 capture changed another raw channel")
        old_fields, new_fields = fields_at_device(prior[name]), fields_at_device(prepare(v2))
        require(old_fields == old["fields"][name][CONDITION]
                and all(old_fields[k] == new_fields[k] for k in ("memory", "state", "token_ids", "token_mask", "current_masks"))
                and all(new_fields["current_masks"].values())
                and {k for k, v in old_fields["current"].items() if new_fields["current"][k] != v}
                == {"left_wrist_0_rgb"}, "official preprocessor changed fields outside the auxiliary slot")
        rows[name] = dict(v1_input_sha256=previous_id, v2_input_sha256=candidate_id,
                          v1_fields=old_fields, v2_fields=new_fields,
                          new_aux_rgb_sha256=rec["rgb_sha256"], native_packet_sha256=rec["input_native_state_sha256"])
    write_new(output, dict(
        status="V2_TWO_ACTUAL_RGB_OFFICIAL_AUX_ONLY_VERIFIED", panel=PANEL, capture_sha256=digest(close_path),
        processor=processor, t2_cold_sha256=T2_SHA, rows=rows,
        model_forwards=0, physics_ticks=0, optimizer_updates=0,
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "capture", "output"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    os.umask(0o077)
    run(a.root.resolve(), a.capture, a.output)
