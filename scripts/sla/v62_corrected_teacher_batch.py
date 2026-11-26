"""Apply the R4 command convention to closed, actually consumed Factory TRAIN data."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_consumed_data import consume
from v6_reanchored_teacher_labels import SCHEMA, reexpress
from v62_source_admission import V62


def reference(path):
    return dict(path=str(path), sha256=sha(path))


def reader_witness(entry, archive, recorded, statistics):
    from go2arx5_vla.sla_adapter.codec_v1 import Quantiles, pad32
    from go2arx5_vla.sla_adapter.multitask_data_v6 import TeacherDataset
    from openpi_client.image_tools import resize_with_pad

    class CorrectedReader(TeacherDataset):
        def __init__(self):
            self.root = archive
            self.manifest = dict(episodes=[entry], windows=entry["windows"])
            self.statistics = {k: Quantiles(statistics[k]["q01"], statistics[k]["q99"]) for k in ("state", "actions")}
            self._load_episodes(resize_with_pad, 32 * 1024**2)
            native = self.labels[entry["id"]]
            require(np.array_equal(recorded["native_step"], native["native_step"])
                    and np.array_equal(recorded["phase"], native["phase"]), "Corrected labels lost their real camera anchor")

        def __getitem__(self, index):
            result = super().__getitem__(index)
            mask = recorded["valid"][index]
            label = pad32(self.statistics["actions"].normalize(recorded["action"][index])).astype(np.float32)
            result["actions"] = np.where(mask[:, None], label, 0)
            return result

    dataset = CorrectedReader()
    boundaries = []
    for index in (0, len(dataset) - 1):
        item = dataset[index]
        mask = recorded["valid"][index]
        require(item["actions"].shape == (40, 32) and not item["actions"][~mask].any()
                and np.isfinite(item["actions"]).all(), "Reader exposed unsupported commands")
        views = {key: hashlib.sha256(array.tobytes()).hexdigest() for key, array in item["image"].items()
                 if item["image_mask"][key]}
        require(len(views) == 2, "Fresh Go2 body and wrist frames are required")
        boundaries.append(dict(index=index, rgb=views, actions_sha256=hashlib.sha256(item["actions"].tobytes()).hexdigest()))
    return boundaries


def build(wave, state_file, action_file, template, out):
    from scripts.pi0292.durable import write

    base = read(template)
    state = read(pin(state_file, base["state_sha256"]))
    actions = read(pin(action_file, base["action_sha256"]))
    stats = dict(state=state["state"], actions=actions["actions"])
    frozen = wave / "RESULT.private.json"
    terminal = read(frozen)
    records = read(wave / "RECORDS.private.json")
    require(read(wave / "control/LEASE.json")["status"] == "CLOSED" and terminal["records"] == records
            and terminal["unreconciled"] is None and len({r["condition"] for r in records}) == len(records),
            "A native Teacher wave must close with reconciled cases")
    admitted = []
    out.mkdir(mode=0o700)
    for row in records:
        if not row["admitted"]:
            continue
        case = wave / "cases" / row["condition"]
        result = read(pin(case / "RESULT.private.json", row["result"]["sha256"]))
        lease = read(case / "control/LEASE.json")
        spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
        require(row["requests"] == 1 and row["split"] == "TRAIN" and spec["expert_training_eligible"],
                "Only independent TRAIN data can enter the batch")
        episode_pin = row["episode"]
        entry_file = pin(episode_pin["path"], episode_pin["sha256"])
        entry = read(entry_file)
        require(entry["split"] == episode_pin["split"] == "TRAIN" and entry["individual_pass"]
                and entry["id"] == row["condition"], "Wrong signed Teacher episode")
        archive = entry_file.parent.parent
        raw, provenance = consume(case, entry, dict(native_result_sha256=row["result"]["sha256"],
                                   confirmed_P10_atoms=episode_pin["confirmed_P10_atoms"]), archive)
        capture = result["detail"]["capture"]
        exported = read(pin(capture["path"], capture["sha256"]))["labels"]
        queries = read(pin(exported["path"], exported["sha256"]))
        corrected, error, p10 = reexpress(spec, queries, raw)
        require(max(error) < .001 and max(p10) < .001, "The unchanged physical receiver must replay the consumed poses")
        labels = dict(raw, action=corrected)
        boundaries = reader_witness(entry, archive, labels, stats)
        path = out / f"{entry['id']}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **labels)
        admitted.append(dict(provenance, archive=str(archive), original_episode=reference(entry_file),
                             corrected=reference(path), supported_atoms=int(raw["valid"].sum()),
                             complete_H40=int(raw["valid"].all(axis=1).sum()), max_roundtrip_mm_deg=error.tolist(),
                             max_p10_roundtrip_mm_deg=p10.tolist(), actual_reader_boundaries=boundaries))
    require(len({item["scene"] for item in admitted}) == len(admitted), "Each TRAIN episode must use a distinct scene")
    output = out / "MANIFEST.json"
    write(output, dict(plan=V62, schema=SCHEMA, wave=reference(frozen), raw_records=reference(wave / "RECORDS.private.json"),
                       episodes=admitted, train=len(admitted), windows=sum(row["windows"] for row in admitted),
                       complete_H40=sum(row["complete_H40"] for row in admitted), original_transport=True,
                       mask_preserved=True, gripper_preserved=True, actual_boundary_reads=2*len(admitted),
                       state_statistics=reference(state_file), action_statistics=reference(action_file),
                       new_native_integrations=0, model_forwards=0, optimizer_visits=0))
    print(sha(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wave", "state_file", "action_file", "template", "out"):
        parser.add_argument(name, type=Path)
    build(**vars(parser.parse_args()))
