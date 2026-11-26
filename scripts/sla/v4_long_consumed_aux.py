"""Recover unrounded jaw label and seam widths from the 2 ms native ledger."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter.gripper_latent_v2 import label_latent
from m38_gpu_preflight_v1 import pin, read, require, sha


def capture(commands, cases, output):
    manifest = read(commands / "MANIFEST.json")
    require(manifest["schema"] == "V3_MAIN_CONSUMED_G7", "Use original signed actual command labels")
    output.mkdir(mode=0o700)
    records = []
    total_valid = 0
    complete = 0
    for row in manifest["episodes"]:
        episode = row["episode"]
        case = cases / episode
        result = read(pin(case / "RESULT.private.json", row["result"]))
        require(read(case / "control/LEASE.json")["status"] == "CLOSED"
                and result["detail"]["score"]["task_success"] and result["shutdown_clean"],
                "Auxiliary inputs require a sealed successful Teacher episode")
        source = case / "native-01/MOBILE_SAMPLES.private.jsonl"
        raw = source.read_bytes()
        atoms = {}
        for line in raw.splitlines():
            atom = json.loads(line)
            tick = atom["native_start_step"]
            require(tick not in atoms and atom["native_end_step"] == tick + 25,
                    "Auxiliary input lost native 50 ms command identity")
            atoms[tick] = (atom["installed_EEF10"][9] * .044, atom["safety_valid"])

        with np.load(pin(commands / row["labels"], row["sha256"]), allow_pickle=False) as labels:
            ticks, mask, latent = (labels[k] for k in ("native_step", "valid", "action"))
        width = np.zeros(mask.shape, dtype="<f8")
        previous = np.empty(len(ticks), dtype="<f8")
        for i, anchor in enumerate(ticks):
            previous[i], safe = atoms[int(anchor) - 25]
            require(safe, "An auxiliary window lacks a safe last actual command")
            for j in np.flatnonzero(mask[i]):
                value, safe = atoms[int(anchor) + 25 * int(j)]
                require(safe, "An auxiliary label did not have safe actual consumption")
                width[i, j] = value
        g = (width[mask] - .002) / .040
        require(np.all((g >= -1e-10) & (g <= 1. + 1e-10))
                and np.all((previous >= .002 - 1e-10) & (previous <= .042 + 1e-10))
                and np.array_equal(label_latent(np.clip(g, 0., 1.)).astype(np.float32), latent[..., 6][mask]),
                "Native physical widths do not reproduce the original signed G7 labels")
        path = output / (episode + ".npz")
        np.savez_compressed(path, actual_width_m=width, prior_actual_width_m=previous,
                            source_tick=ticks, valid=mask)
        valid = int(mask.sum())
        full = int(mask.all(axis=1).sum())
        total_valid += valid
        complete += full
        records.append(dict(episode=episode, file=path.name, sha256=sha(path), complete_H40=full,
                            supported_atoms=valid, native_log_sha256=hashlib.sha256(raw).hexdigest(),
                            native_result_sha256=row["result"]))
    receipt = dict(schema="V4_LONG_NATIVE_AUX_WIDTHS_V1", episodes=records,
                   actual_supported_atoms=total_valid, complete_H40=complete,
                   command_manifest_sha256=sha(commands / "MANIFEST.json"),
                   new_optimizer_updates=0, new_native_integrations=0)
    (output / "MANIFEST.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("commands", "cases", "output"):
        parser.add_argument(name, type=Path)
    result = capture(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}))
