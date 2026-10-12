"""Old PI0290 processor plumbing diagnostic, explicitly NOT ARX5 qualification."""
import argparse
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

from export_dataset import statistics
from native_dataset import TAG, build_processor, pose10, read, sha, write
from qualify_loader import check_batch, physical_errors


class ContractDataset:
    def __init__(self, rows, source, cosmos, statistics):
        self.rows, self.source, self.cosmos, self.statistics = rows, source, cosmos, statistics
        self.processor = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.processor is None:
            self.processor = build_processor(self.source, self.cosmos, self.statistics)
        from gr00t.data.types import VLAStepData
        from gr00t.data.embodiment_tags import EmbodimentTag
        row = self.rows[index]
        state, action = row["state"], row["commands"]
        content = VLAStepData(images=row["images"], states=dict(eef_9d=state[None, :9], gripper_position=state[None, 9:]),
                              actions=dict(eef_9d=action[:, :9], gripper_position=action[:, 9:]), text=row["instruction"], embodiment=EmbodimentTag.NEW_EMBODIMENT)
        return self.processor([dict(content=content)])


def collate_diagnostic(rows):
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cosmos", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    begin = time.monotonic()
    raw = []
    for identifier in ("P2900", "P2901"):
        folder = Path(args.raw_root) / "episodes" / identifier
        rows = read(folder / "samples.json")
        for index in (0, 80, 160, 320, 480, 600):
            state = pose10(rows[index]["policy_obs"]["actual_tcp"], rows[index]["policy_obs"]["actual_gripper"])
            commands = pose10(np.asarray([r["supervision"]["committed"] for r in rows[index:index + 40]]),
                              np.asarray([r["supervision"]["committed_gripper"] for r in rows[index:index + 40]]))
            relative = np.linalg.inv(np.asarray(rows[index]["policy_obs"]["actual_tcp"])) @ np.asarray([r["supervision"]["committed"] for r in rows[index:index + 40]])
            relative9 = pose10(relative, commands[:, 9])[:, :9]
            images, hashes = {}, {}
            for key, name in (("external_rgb", "body_rgb"), ("wrist_rgb", "wrist_rgb")):
                frame = rows[index]["policy_obs"][name]
                path = folder / frame["path"]
                if sha(path) != frame["sha256"]:
                    raise ValueError("Original diagnostic RGB changed")
                with Image.open(path) as image:
                    images[key] = [np.asarray(image).copy()]
                hashes[name] = frame["sha256"]
            raw.append(dict(id=identifier, anchor=index, state=state, commands=commands, relative=relative9,
                            images=images, image_hashes=hashes, instruction=rows[index]["policy_obs"]["instruction"]))
    states = np.asarray([r["state"] for r in raw])
    commands = np.asarray([r["commands"] for r in raw])
    relatives = np.asarray([r["relative"] for r in raw])
    stats = {TAG: dict(state=dict(eef_9d=statistics(states[:, :9], 0), gripper_position=statistics(states[:, 9:], 0)),
                      action=dict(eef_9d=statistics(commands[..., :9], (0, 1)), gripper_position=statistics(commands[..., 9:], (0, 1))),
                      relative_action=dict(eef_9d=statistics(relatives, (0, 1))))}
    result = dict(domain="PI0290_CONTRACT_DIAGNOSTIC_NOT_ARX5", status="NOT_RUN", model_loads=0, neural_forward=0,
                  physics_integrations=0, statistical_fit="Old fixture only; NEVER ARX5 TRAIN statistics", official_collator="NOT_RUN")
    try:
        processor = build_processor(args.source, args.cosmos, stats)
        from gr00t.data.types import VLAStepData
        from gr00t.data.embodiment_tags import EmbodimentTag
        processed = []
        errors = []
        for row in raw:
            state, action = row["state"], row["commands"]
            state_dict = dict(eef_9d=state[None, :9], gripper_position=state[None, 9:])
            action_dict = dict(eef_9d=action[:, :9], gripper_position=action[:, 9:])
            content = VLAStepData(images=row["images"], states=state_dict, actions=action_dict, text=row["instruction"], embodiment=EmbodimentTag.NEW_EMBODIMENT)
            value = processor([dict(content=content)])
            processed.append(value)
            decoded = processor.decode_action(value["action"][None], EmbodimentTag.NEW_EMBODIMENT,
                                              {key: value[None] for key, value in state_dict.items()})
            physical = np.concatenate([decoded[key] for key in ("eef_9d", "gripper_position")], axis=-1)[0]
            errors.append(dict(id=row["id"], anchor=row["anchor"], error=physical_errors(action, physical), RGB_hashes=row["image_hashes"]))
        result.update(status="PASS_PROCESSOR_PLUMBING_ONLY_NOT_ARX5", official_processor=type(processor).__name__,
                      official_collator=type(processor.collator).__name__, single=check_batch(processor.collator(processed[:1]), 1),
                      mixed=check_batch(processor.collator([processed[1], processed[7]]), 2), codec_rows=errors)
        from torch.utils.data import DataLoader
        from copy import deepcopy
        dataset = ContractDataset(raw, args.source, args.cosmos, stats)
        worker_checks = {}
        for workers in (0, 2):
            dataset.processor = None
            batches = DataLoader(dataset, batch_size=2, num_workers=workers, shuffle=False, collate_fn=collate_diagnostic,
                                 multiprocessing_context="spawn" if workers else None, persistent_workers=False)
            worker_checks[str(workers)] = [check_batch(processor.collator(batch), len(batch)) for batch in batches]
        if worker_checks["0"] != worker_checks["2"]:
            raise ValueError("CPU diagnostic two workers did not reproduce exact batches")
        short, long = deepcopy(processed[0]), deepcopy(processed[0])
        short["vlm_content"]["text"] += " Short."
        long["vlm_content"]["text"] += " Longer CPU-only instruction padding diagnostic, not rewritten task language."
        result.update(two_worker_actual_batches=worker_checks, two_workers_equal=True,
                      variable_instruction_diagnostic=check_batch(processor.collator([short, long]), 2), original_language_unchanged=True)
    except Exception as error:
        result.update(status="PROCESSOR_CONTRACT_DIAGNOSTIC_BLOCKED", error_type=type(error).__name__, error=str(error))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(CPU_s=usage.ru_utime + usage.ru_stime + children.ru_utime + children.ru_stime,
                  CPU_self_s=usage.ru_utime + usage.ru_stime, CPU_children_s=children.ru_utime + children.ru_stime,
                  elapsed_s=time.monotonic() - begin)
    write(args.output, result)
    print(result["status"], result["CPU_s"], result.get("error", ""))
    if "BLOCKED" in result["status"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
