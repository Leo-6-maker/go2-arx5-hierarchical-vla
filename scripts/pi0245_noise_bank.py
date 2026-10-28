#!/usr/bin/env python3
"""Freeze the four preregistered full-132D PI0245 sampler noise tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    tensors = {}
    for index, seed in enumerate((22301, 22302, 22303)):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        tensors[f"N{index}"] = torch.randn((1, 40, 132), generator=generator, dtype=torch.float32, device="cpu")
    tensors["N0_roll11"] = torch.roll(tensors["N0"], shifts=11, dims=1)
    rows = []
    for name, tensor in tensors.items():
        f32 = tensor.numpy().copy()
        bf16_bits = tensor.to(torch.bfloat16).view(torch.uint16).numpy().copy()
        path = args.output_root / f"{name}.npz"
        np.savez_compressed(path, float32=f32, bfloat16_bits=bf16_bits)
        with np.load(path, allow_pickle=False) as data:
            assert np.array_equal(data["float32"], f32) and np.array_equal(data["bfloat16_bits"], bf16_bits)
        rows.append({"name": name, "seed": {"N0": 22301, "N1": 22302, "N2": 22303}.get(name), "dependent_roll_of": "N0" if name == "N0_roll11" else None, "roll_horizon": 11 if name == "N0_roll11" else 0, "path": str(path), "file_sha256": sha(path.read_bytes()), "float32_bytes_sha256": sha(f32.tobytes()), "bfloat16_bytes_sha256": sha(bf16_bits.tobytes()), "shape": [1, 40, 132]})
    assert torch.equal(tensors["N0_roll11"], torch.roll(tensors["N0"], shifts=11, dims=1))
    manifest = {"schema": "PI0245_P1_FIXED_NOISE_BANK_V1", "generator": "torch.Generator(device=cpu)", "independent_seeds": [22301, 22302, 22303], "actual_model_dtype": "torch.bfloat16", "rows": rows, "model_queries": 0}
    target = args.output_root / "manifest.json"
    target.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"status": "PASS_FIXED_NOISE_BANK", "manifest": str(target), "manifest_sha256": sha(target.read_bytes()), "count": len(rows)}))


if __name__ == "__main__":
    main()
