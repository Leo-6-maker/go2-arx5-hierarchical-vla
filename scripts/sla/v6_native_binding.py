"""Prepare the registered V6 SOURCE1K endpoint on the inherited paired native panel."""

import argparse
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from v4_long_native_binding import MAIN, REGISTRY_SHA256, schedule


PLAN = "d7c74953109374b69906036a0e2c6f39a32b3aa7"


def bind(registry, receipt, place_noise, hold_noise, study, v2_registry, checkpoint, original, audit, output):
    from scripts.pi0292.durable import write

    proof = read(audit)
    base = read(original / "CHECKPOINT.json")
    current_pin = dict(path=str(checkpoint), sha256=proof["provider_manifest_sha256"])
    model = read(pin(checkpoint / "CHECKPOINT.json", current_pin["sha256"]))
    require(base["checkpoint_sha256"] == model["previous_checkpoint_sha256"] == proof["parent_sha256"] == MAIN
            and model["schema"] == "SLA_V6_SOURCE_INFERENCE" and model["role"] == "SOURCE1K_V6"
            and model["phase"] == proof["step"] == 1000 and proof["formal"] and model["formal_inference_eligible"]
            and model["training_receipt_sha256"] == proof["receipt_sha256"]
            and model["training_contract_sha256"] == proof["contract_sha256"] and model["frozen_wbc"]
            and model["original_l1"] and not model["hardware_allowed"], "Expected the independently audited V6 source endpoint")
    inherited = ("foundation_sha256", "frozen_model", "action_sha256", "state_sha256",
                 "tokenizer_sha256", "partition_sha256")
    require(all(model[key] == base[key] for key in inherited), "Original Go2 observation and G7 ABI changed")
    fields = ("file", "key", "shape", "dtype")
    for original_row, learned_row in zip(base["arrays"], model["arrays"], strict=True):
        require(all(original_row[key] == learned_row[key] for key in fields), "Go2 provider leaf ABI changed")
        pin(checkpoint / learned_row["file"], learned_row["sha256"])
    qualified = study / "Q_CONFIRM_ACTUAL.json"
    place, hold = schedule(registry, read(receipt), place_noise, hold_noise, study, v2_registry,
                           dict(path=str(qualified), sha256=sha(qualified)), "M1000")
    for group, tag in ((place, "PLACE"), (hold, "HOLD")):
        for i, row in enumerate(group, start=1):
            row["id"] = f"V6_SOURCE1K_{tag}_{i:02d}"
            row["pair_id"] = f"V6_SOURCE1K_{tag}"
    write(output, dict(plan=PLAN, registry_sha256=REGISTRY_SHA256, role="SOURCE1K_V6",
          candidate=current_pin, checkpoint_sha256=model["checkpoint_sha256"],
          source_endpoint_readback_sha256=sha(audit), place=place, hold=hold,
          noise=dict(place=dict(path=str(place_noise), sha256=sha(place_noise)),
                     hold=dict(path=str(hold_noise), sha256=sha(hold_noise))),
          optimizer_updates=0, model_forwards=0, native_integrations=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "receipt", "place_noise", "hold_noise", "study", "v2_registry",
                 "checkpoint", "original", "audit", "output"):
        parser.add_argument(name, type=Path)
    bind(**vars(parser.parse_args()))
