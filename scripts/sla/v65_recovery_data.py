"""Read the three registered, actually executed R4 Recovery teacher suffixes."""

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write


ROSTER = (
    ("02", 45, "adc0c371c431017700462ab352532b5cd1df51c776d528f5afc4704f0ec118dc"),
    ("04", 35, "89ee4dcefa4d099aac2acadf5830ac90d7d129a3df1514f1486722d0e6e87ca2"),
    ("06", 35, "9a788dc687a0f44e3b19e447c1d9ee786176e84d885ef382ff4845cf56130698"),
)
A93 = "1069a89550ea667686a5670249f7b41aa950eb3e687b835fc5e600ca61c08773"
POPULATIONS = (
    "ea8b99355b0a8fc2367d17b5e2f18e8a4d583c73ddb39b3848b1add25e143f96",
    "e3487b9322a6ce7e795a69a596c762a775e7475af3752d6d0142ae5f37d3f0fd",
    "b16f0993e88a4fa7058ed54fdac163021b45d95e46777166b88d2a245058cc1b",
    "91b2c0c2c4b4f8df887474bb74e1deabcc0559414b8220e2025220e240fff91f",
    "12af4e2db606688074e056ef0ebca5e89f3b13d0649cf1c472bcafbce627a737",
)
SCHEMA = "V65_R4_RECOVERY3_REAL_TEACHER_SUFFIX"


def checked(reference):
    return pin(Path(reference["path"]), reference["sha256"])


def transfer(root, output, reference):
    source = checked(reference)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with source.open("rb") as original, output.open("xb") as portable:
        shutil.copyfileobj(original, portable, length=4*1024**2)
    pin(output, reference["sha256"])
    return {"file": str(output.relative_to(root)), "sha256": sha(output)}


def source_episode(work, output, wave, suffix, windows, digest):
    name = f"V64_RECOVERY_TRAIN_{suffix}"
    seal = read(pin(work / f"RECOVERY-TRAIN-SEALED-{wave:02d}" / name / "ACTUAL.json", digest))
    terminal = checked(seal["native_result"])
    case = terminal.parent
    lease = read(case / "control/LEASE.json")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    result = read(terminal)
    capture = read(checked(seal["capture"]))
    queries = read(checked(capture["labels"]))
    require(lease["status"] == "CLOSED" and case.name == name and result["completed"]
            and result["shutdown_clean"] and not result["unsafe"] and not result["technical_failure"]
            and result["detail"]["teacher_takeover"] and result["detail"]["recovery_task_success"]
            and spec["collection_split"] == "TRAIN" and spec["evaluation_split"] == "TRAIN_RECOVERY"
            and seal["actual_recovery"] and seal["training_windows"] == len(queries) == windows
            and seal["actual_teacher_atoms"] == 10 * windows
            and capture["frame_shards"] == seal["rgb"]["shards"]
            and capture["language"] == spec["language"] and capture["frozen_scene_sha256"] == seal["scene"],
            "The original closed Recovery rollout or observation identity changed")
    samples = {}
    with checked(seal["actual_native_samples"]).open() as stream:
        for line in stream:
            atom = json.loads(line)
            tick = atom["native_start_step"]
            require(tick not in samples, "Duplicate integrator receipt")
            samples[tick] = atom
    starts = seal["native_teacher_start_tick"] + 250 * np.arange(windows)
    require(result["actual_native_integrations"] == starts[-1] + 250, "Unexpected teacher suffix end")
    prior, physical = [], []
    frame = queries[0]["episode_fixed_T_policy_from_mount"]
    for i, (query, tick) in enumerate(zip(queries, starts, strict=True)):
        context = query["previous_confirmed_command"]
        previous = samples[tick - 25]
        width = previous["installed_EEF10"][9] * .044
        require(query["native_step"] == tick and query["query"] == queries[0]["query"] + i
                and query["episode_fixed_T_policy_from_mount"] == frame
                and query["supervision_role"] == "RECOVERY_TEACHER"
                and query["rgb_provenance"] == "CURRENT_OWNED_PAUSED_NATIVE_PACKET"
                and query["consumed_mask"] == [True]*10 + [False]*30
                and len(query["intervals"]) == 10 and query["partial_interval_ticks"] == 0
                and query["object_phase_memory_in_student_inputs"] is False
                and previous["safety_valid"] and previous["native_end_step"] == tick
                and abs(width - context["width_m"]) < 1e-12 and .002 <= width <= .042
                and previous["gripper_command"]["source_native_s"] == context["source_native_s"]
                and abs(context["query_native_s"] - tick * .002) < 1e-8,
                "A Recovery query lost its actual preceding command or observation")
        prior.append(width)
        for j, interval in enumerate(query["intervals"]):
            start = tick + 25 * j
            atom, boundary = samples[start], interval["boundary"]
            require(atom["safety_valid"] and atom["ticks"] == interval["confirmed_ticks"] == 25
                    and atom["native_end_step"] == interval["end_native_step"] == boundary["native_step"] == start+25
                    and boundary["exact_receiver_knot"] and boundary["accepted_wire_sha256"]
                    == interval["installed_reference_sha256"] == atom["installed_reference_sha256"],
                    "A training atom lacks an exact, real consumption receipt")
            physical.append(atom["installed_EEF10"][9] * .044)
    with np.load(checked(seal["labels"]), allow_pickle=False) as data:
        require(np.array_equal(data["native_step"], starts)
                and np.array_equal(data["state_eef10"], [q["state_eef10"] for q in queries])
                and data["phase"].tolist() == [q["phase"] for q in queries], "Wrong actual Recovery label anchors")
        phase = data["phase"].tolist()
        require(dict(Counter(phase)) == seal["phase_windows"], "Wrong intervention phase inventory")
    require(max(seal["maximum_roundtrip_mm_deg"]) < .001
            and max(seal["maximum_P10_roundtrip_mm_deg"]) < .001, "Original R4 inverse transport failed")
    folder = output / name
    labels = transfer(output, folder / "LABELS.npz", seal["labels"])
    shards = [dict(transfer(output, folder / Path(r["path"]).name, r), first_frame=r["first_query"], frames=r["count"])
              for r in seal["rgb"]["shards"]]
    widths = folder / "ACTUAL_GRIPPER.npz"
    with widths.open("xb") as stream:
        np.savez_compressed(stream, prior=np.asarray(prior), issued=np.asarray(physical))
    return dict(id=name, scene_sha256=seal["scene"], reset=seal["reset"], split="TRAIN_RECOVERY", windows=windows,
                task=seal["task"], language=spec["language"], expert_stage_audit_only=phase,
                T_policy_from_mount=frame, labels=labels, frame_shards=shards,
                gripper=dict(file=str(widths.relative_to(output)), sha256=sha(widths)),
                source_receipt_sha256=digest, native_result_sha256=sha(terminal),
                actual_sample_log_sha256=seal["actual_native_samples"]["sha256"])


def build(work, output, protected, a93, populations):
    latest = read(pin(a93, A93))
    require(latest["cumulative_target_episodes"] == 93 and latest["cumulative_target_windows"] == 7916,
            "Bind the original A93 curriculum")
    clusters = read(protected)["clusters"]
    excluded = {row["scene"]["sha256"] for row in clusters
                if row.get("split") in ("QUAL", "DEV", "VAL", "CONF", "TEST")}
    legacy, *increments = [read(pin(path, expected)) for path, expected in zip(populations, POPULATIONS, strict=True)]
    originals = {row["episode"]: row["scene"] for row in legacy["episodes"]}
    for data in [*increments, latest]:
        for row in data["episodes"]:
            episode = row["episode"]
            require(episode["id"] not in originals, "A93 episode duplicated across versions")
            originals[episode["id"]] = episode["scene_sha256"]
    require(len(originals) == 93, "Compare the full preexisting scene population")
    output.mkdir(mode=0o700)
    episodes = [source_episode(work, output, wave, *entry) for wave, entry in enumerate(ROSTER, start=1)]
    scenes = {e["scene_sha256"] for e in episodes}
    require(len(scenes) == 3 and not scenes & excluded, "Recovery overlaps a protected evaluation scene group")
    shared_train = {}
    for ep in episodes:
        require(ep["id"] not in originals, "Recovery was counted as a complete demonstration")
        matches = [name for name, scene in originals.items() if scene == ep["scene_sha256"]]
        for name in matches:
            sources = [r for r in clusters if r["id"] == name and r.get("split") == "TRAIN"]
            require(len(sources) == 1 and sources[0]["reset"] != ep["reset"],
                    "Shared TRAIN geometry must use a separately registered reset and rollout")
        shared_train[ep["id"]] = matches
    manifest = dict(schema=SCHEMA, windows=115, episodes=episodes, a93_manifest_sha256=A93,
                    statistics_sha256=latest["statistics_sha256"], protected_sha256=sha(protected),
                    compared_scene_count=len(excluded | set(originals.values())), shared_train_scenes=shared_train,
                    optimizer_visits=0)
    write(output / "MANIFEST.json", manifest)


def dataset(root, manifest, stats, resize_with_pad):
    from go2arx5_vla.sla_adapter.codec_v1 import Codec, Quantiles, pad32
    from go2arx5_vla.sla_adapter.dataset_v1 import images224
    from go2arx5_vla.sla_adapter.dense_dataset_v4 import DensePilotDataset, member
    from go2arx5_vla.sla_adapter.gripper_latent_v2 import label_latent

    class Recovery(DensePilotDataset):
        def __init__(self):
            self.root, self.resize = root, resize_with_pad
            self.cache, self.cache_bytes, self.cache_limit = OrderedDict(), 0, 64*1024**2
            self.rows, self.labels = [], {}
            self.statistics = {role: Quantiles(value["q01"], value["q99"]) for role, value in stats.items()}
            for ep in manifest["episodes"]:
                n = ep["windows"]
                with np.load(pin(member(root, ep["labels"]["file"]), ep["labels"]["sha256"]), allow_pickle=False) as f:
                    values = {key: f[key] for key in f.files}
                with np.load(pin(member(root, ep["gripper"]["file"]), ep["gripper"]["sha256"]), allow_pickle=False) as f:
                    prior, issued = f["prior"], f["issued"]
                mask = values["valid"]
                atom = 10 * np.arange(n)[:, None] + np.arange(40)
                require(mask.dtype == bool and mask.shape == (n, 40) and np.array_equal(mask, atom < 10*n)
                        and values["action"].shape == (n, 40, 7) and np.isfinite(values["action"]).all()
                        and not values["action"][~mask].any() and prior.shape == (n,) and issued.shape == (10*n,)
                        and np.isfinite(prior).all() and np.array_equal(prior[1:], issued[9:-1:10])
                        and np.array_equal(values["action"][..., 6][mask],
                                           label_latent(np.clip((issued[atom[mask]]-.002)/.040, 0., 1.)).astype(np.float32)),
                        "Invalid real P10 support or gripper context")
                codec = Codec(ep["T_policy_from_mount"])
                values["state7"] = np.asarray([codec.encode(state, np.tile(state, (40, 1)))[0]
                                               for state in values["state_eef10"]])
                offset = 0
                for shard in ep["frame_shards"]:
                    require(shard["first_frame"] == offset and 1 <= shard["frames"] <= 16, "RGB shard discontinuity")
                    pin(member(root, shard["file"]), shard["sha256"])
                    offset += shard["frames"]
                require(offset == n and ep["split"] == "TRAIN_RECOVERY", "Incomplete Recovery RGB coverage")
                self.labels[ep["id"]] = values
                self.rows.extend((ep, anchor) for anchor in range(n))

        def __getitem__(self, index):
            ep, anchor = self.rows[index]
            values = self.labels[ep["id"]]
            shard, offset = self._frame(ep, anchor)
            require(shard["native_step"][offset] == values["native_step"][anchor]
                    and np.array_equal(shard["state_eef10"][offset], values["state_eef10"][anchor]),
                    "RGB and corrected actions must refer to the same actual native state")
            action = pad32(self.statistics["actions"].normalize(values["action"][anchor])).astype(np.float32)
            item = images224(shard["rgb"][offset], self.resize)
            return dict(item, state=pad32(self.statistics["state"].normalize(values["state7"][anchor])).astype(np.float32),
                        actions=np.where(values["valid"][anchor, :, None], action, 0.), prompt=ep["language"])

        def support(self, index):
            ep, anchor = self.rows[index]
            return self.labels[ep["id"]]["valid"][anchor]

    return Recovery()


def readback(root, expected, state, actions, output):
    from openpi_client.image_tools import resize_with_pad
    from v64_a78_delta import statistics

    manifest = read(pin(root / "MANIFEST.json", expected))
    identity = [(r["id"][-2:], r["windows"], r["source_receipt_sha256"]) for r in manifest["episodes"]]
    require(manifest["schema"] == SCHEMA and manifest["a93_manifest_sha256"] == A93
            and identity == list(ROSTER), "Unexpected Recovery scientific cohort")
    reader = dataset(root, manifest, statistics(manifest, state, actions), resize_with_pad)
    boundaries, phase, supported = [], Counter(), 0
    digest = hashlib.sha256()
    views = dict(base_0_rgb=True, left_wrist_0_rgb=True, right_wrist_0_rgb=False)
    for i, (episode, anchor) in enumerate(reader.rows):
        sample, mask = reader[i], reader.support(i)
        require(sample["image_mask"] == views and sample["actions"].shape == (40, 32)
                and sample["state"].shape == (32,) and not sample["actions"][~mask].any(),
                "Real state, action, RGB or mask differs from the Go2 training contract")
        phase[episode["expert_stage_audit_only"][anchor]] += 1
        supported += int(mask.sum())
        fields = [sample[key] for key in ("actions", "state")] + [sample["image"][key] for key, valid in views.items() if valid]
        row = hashlib.sha256()
        for value in fields:
            row.update(value.tobytes())
        row.update(sample["prompt"].encode())
        row.update(mask.tobytes())
        digest.update(row.digest())
        if anchor in (0, episode["windows"]-1):
            boundaries.append(dict(episode=episode["id"], anchor=anchor, row_sha256=row.hexdigest()))
    require(len(reader) == manifest["windows"] == 115, "Recovery training denominator changed")
    write(output, dict(manifest_sha256=expected, windows=len(reader), supported_action_points=supported,
                       phase=dict(phase), boundary_readbacks=boundaries, all_training_rows_sha256=digest.hexdigest(),
                       optimizer_visits=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("build")
    for key in ("work", "output", "protected", "a93"):
        prepare.add_argument(key, type=Path)
    prepare.add_argument("populations", nargs="+", type=Path)
    verify = modes.add_parser("readback")
    for key in ("root", "expected", "state", "actions", "output"):
        verify.add_argument(key, type=str if key == "expected" else Path)
    arguments = vars(cli.parse_args())
    {"build": build, "readback": readback}[arguments.pop("mode")](**arguments)
