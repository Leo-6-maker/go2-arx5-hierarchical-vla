"""Actual ARX5 commissioning loader diagnostic; no formal data admission."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import resource
import time

import numpy as np

from native_dataset import DOMAIN, OFFICIAL_SHA, TAG, DatasetLoader, build_processor, collate_project, pose10, read, sha, validate_samples, write
from export_dataset import statistics
from qualify_loader import check_batch, codec_all_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--cosmos", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture-directory", required=True)
    args = parser.parse_args()
    begin = time.monotonic()
    folder = Path(args.root) / "episodes/C29100"
    output = Path(args.fixture_directory)
    if output.exists():
        raise FileExistsError("Do not overwrite a previous commissioning diagnostic")
    output.mkdir(parents=True)
    samples = read(folder / "samples.json")
    validate_samples(samples, folder)
    result = read(folder / "RESULT.json")
    if result["domain"] != DOMAIN or result["status"] != "VALID_COMPLETE":
        raise ValueError("Actual ARX5 complete commissioning source required")
    actual = np.asarray([row["policy_obs"]["actual_tcp"] for row in samples[:601]])
    desired = np.asarray([row["supervision"]["committed"] for row in samples[:640]])
    grip = np.asarray([row["supervision"]["committed_gripper"] for row in samples[:640]])
    states = pose10(actual, np.asarray([row["policy_obs"]["actual_gripper"] for row in samples[:601]]))
    commands = pose10(desired, grip)
    future = np.arange(601)[:, None] + np.arange(40)
    relative = pose10(np.linalg.inv(actual)[:, None] @ desired[future], grip[future])[..., :9]
    paths = np.asarray([[row["policy_obs"][camera]["path"] for camera in ("body_rgb", "wrist_rgb")] for row in samples[:601]])
    hashes = np.asarray([[row["policy_obs"][camera]["sha256"] for camera in ("body_rgb", "wrist_rgb")] for row in samples[:601]])
    np.savez_compressed(output / "C29100.npz", actual_state10=states, committed_absolute10=commands, rgb_paths=paths, rgb_sha256=hashes,
                        valid_window=np.ones(601, dtype=bool), instruction=np.asarray(samples[0]["policy_obs"]["instruction"]),
                        label_representation=np.asarray("ABSOLUTE_COMMITTED_EEF10"))
    stats = {TAG: dict(state=dict(eef_9d=statistics(states[:, :9], 0), gripper_position=statistics(states[:, 9:], 0)),
                      action=dict(eef_9d=statistics(commands[future][..., :9], (0, 1)), gripper_position=statistics(commands[future][..., 9:], (0, 1))),
                      relative_action=dict(eef_9d=statistics(relative, (0, 1))))}
    write(output / "statistics.json", stats)
    manifest = dict(domain=DOMAIN, purpose="NOT_ADMITTED_COMMISSIONING_DIAGNOSTIC", official_source_sha=OFFICIAL_SHA,
                    statistics_sha256=sha(output / "statistics.json"), TRAIN_ids=[], DEV_ids=[],
                    episodes=[dict(id="C29100", split="DIAGNOSTIC", raw_root=str(folder.resolve()), file="C29100.npz",
                                   sha256=sha(output / "C29100.npz"), windows=601, source_sha=result["source_sha"],
                                   samples_sha256=sha(folder / "samples.json"), physics_sha256=sha(folder / "physics.npz"))])
    write(output / "DATASET_MANIFEST.json", manifest)
    proof = dict(status="NOT_RUN_ACTUAL_ARX5_PROCESSOR", purpose="NOT_ADMITTED_COMMISSIONING_DIAGNOSTIC", actual_source_id="C29100",
                 actual_physics_sha256=manifest["episodes"][0]["physics_sha256"], physical_source_sha=result["source_sha"],
                 diagnostic_statistics_sha256=sha(output / "statistics.json"), statistic_population="C29100 commissioning only, NEVER formal TRAIN",
                 TRAIN_episodes_admitted=0, DEV_episodes_admitted=0, formal_NATIVE_DATALOADER_PASS="NOT_RUN", model_loads=0, neural_forward=0, physics_integrations=0)
    try:
        processor = build_processor(args.source, args.cosmos, stats)
        offsets = [0, 40, 80, 120, 160, 200, 240, 320, 480, 600]
        dataset = DatasetLoader(output, args.source, args.cosmos, "DIAGNOSTIC", offsets)
        dataset.processor = processor
        single = check_batch(processor.collator([dataset[0]["processed"]]), 1)
        mixed = check_batch(processor.collator([dataset[1]["processed"], dataset[5]["processed"]]), 2)
        short, long = copy.deepcopy(dataset[0]["processed"]), copy.deepcopy(dataset[0]["processed"])
        short["vlm_content"]["text"] += " Short."
        long["vlm_content"]["text"] += " Longer CPU-only instruction padding diagnostic, not new ARX5 task language."
        variable = check_batch(processor.collator([short, long]), 2)
        from torch.utils.data import DataLoader
        worker_proofs = {}
        for workers in (0, 2):
            loader = DataLoader(dataset, batch_size=2, num_workers=workers, shuffle=False, collate_fn=collate_project,
                                multiprocessing_context="spawn" if workers else None, persistent_workers=False)
            worker_proofs[str(workers)] = [dict(identities=[item["identity"] for item in rows],
                  batch=check_batch(processor.collator([item["processed"] for item in rows]), len(rows))) for rows in loader]
        if worker_proofs["0"] != worker_proofs["2"]:
            raise ValueError("Actual ARX5 CPU two-worker batch differs")
        codec = codec_all_windows(output, processor, ("DIAGNOSTIC",))
        proof.update(status="PASS_ACTUAL_ARX5_NATIVE_PROCESSOR_DIAGNOSTIC_NOT_ADMITTED", official_processor=type(processor).__name__,
                     official_collator=type(processor.collator).__name__, project_loader="native_dataset.DatasetLoader split=DIAGNOSTIC", single_actual_sample=single,
                     mixed_actual_phase_batch=mixed, variable_text_padding_diagnostic=variable, original_language_modified=False,
                     two_worker_readbacks=worker_proofs, two_workers_equal=True, codec=codec, checked_anchors=offsets,
                     multiple_episode_requirement="UNMET: only one commissioning episode used; formal dataset still NOT_RUN")
    except Exception as error:
        proof.update(status="ACTUAL_ARX5_PROCESSOR_DIAGNOSTIC_BLOCKED", error_type=type(error).__name__, error=str(error))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    proof.update(CPU_s=usage.ru_utime + usage.ru_stime + children.ru_utime + children.ru_stime,
                 elapsed_s=time.monotonic() - begin, official_source_sha=OFFICIAL_SHA, source_script_sha256=sha(__file__))
    write(args.output, proof)
    print(proof["status"], proof["CPU_s"], proof.get("error", ""))
    if "BLOCKED" in proof["status"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
