"""Build an inspection copy of the exact V6.5 Go2/R3 dataset for a private Hub repo."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha


LEGACY = "80876f04260377da07c20b547ac6c923751f5a84dbba8db6dbdb361ded164639"
R4 = "1b57d74fd133a3acbfb57bdce10a7ac4b8d8b18fd7fad8ca72376530edcb2a21"
AUX = "584fe14655538cbe22509cd27f5fc432026808d5c577c84b6ab9ed81d2e7838b"
INCREMENTS = (
    "e3487b9322a6ce7e795a69a596c762a775e7475af3752d6d0142ae5f37d3f0fd",
    "b16f0993e88a4fa7058ed54fdac163021b45d95e46777166b88d2a245058cc1b",
    "91b2c0c2c4b4f8df887474bb74e1deabcc0559414b8220e2025220e240fff91f",
    "12af4e2db606688074e056ef0ebca5e89f3b13d0649cf1c472bcafbce627a737",
    "1069a89550ea667686a5670249f7b41aa950eb3e687b835fc5e600ca61c08773",
)
RECOVERY = "362c8756c7b7ea62a539a441dcfa5a6bdf175f4eb92f79088ed7a7ec7a884991"


def member(root, ref):
    target = root / ref["file"]
    require(target.resolve(strict=True).is_relative_to(root.resolve(strict=True)), "Dataset member outside its approved archive")
    return pin(target, ref["sha256"])


def packed_npz(**arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def add(tar, name, contents):
    record = tarfile.TarInfo(name)
    record.mode = 0o644
    if isinstance(contents, Path):
        record.size = contents.stat().st_size
        with contents.open("rb") as stream:
            tar.addfile(record, stream)
    else:
        record.size = len(contents)
        tar.addfile(record, io.BytesIO(contents))


def write_episode(output, root, ep, corrected, gripper, skill, *, recovery=False):
    from PIL import Image

    name, n = ep["id"], ep["windows"]
    require(name == Path(name).name and n > 0 and ep["split"] == ("TRAIN_RECOVERY" if recovery else "TRAIN"),
            "Export only the admitted Go2 training population")
    with np.load(corrected, allow_pickle=False) as f:
        values = {key: f[key] for key in ("action", "valid", "phase", "native_step")}
    original = member(root, ep["labels"])
    with np.load(original, allow_pickle=False) as f:
        state_eef10 = f["state_eef10"]
        require(np.array_equal(values["native_step"], f["native_step"]) and np.array_equal(values["phase"], f["phase"]),
                "State and corrected physical commands use different anchors")
    mask = values["valid"]
    require(mask.shape == (n, 40) and mask.dtype == bool and values["action"].shape == (n,40,7)
            and np.isfinite(values["action"]).all() and mask[:, :10].all() and not values["action"][~mask].any()
            and state_eef10.shape == (n,10) and np.isfinite(state_eef10).all()
            and (np.diff(values["native_step"]) == 250).all(), "Actual R4 data layout changed")
    with np.load(gripper, allow_pickle=False) as f:
        if recovery:
            atom = 10*np.arange(n)[:,None] + np.arange(40)
            prior, issued = f["prior"], f["issued"]
            width = np.zeros(mask.shape, dtype=issued.dtype)
            width[mask] = issued[atom[mask]]
        else:
            prior, width = f["prior_actual_width_m"], f["actual_width_m"]
            require(np.array_equal(f["valid"], mask), "Actual gripper support differs from training labels")
    require(prior.shape == (n,) and width.shape == (n,40) and np.isfinite(prior).all() and np.isfinite(width[mask]).all(),
            "Expected actual prior and per-atom gripper command")
    info = dict(episode=name, split=ep["split"], skill=skill, task=ep["task"], language=ep["language"], windows=n,
                full_h40=int(mask.all(axis=1).sum()), supported_action_rows=int(mask.sum()), scene_group_sha256=ep["scene_sha256"],
                phases=dict(Counter(values["phase"].tolist())), label_source_sha256=sha(corrected),
                T_policy_from_mount=ep["T_policy_from_mount"] if recovery else ep["mount"]["T_policy_from_mount"],
                camera_order=["body", "wrist"], physics_tick_s=.002, action_sample_s=.05,
                observation_query_s=.5, state_eef10="xyz, rotation6d, measured jaw fraction")
    relative = Path("episodes") / ep["split"].lower() / f"{name}.tar"
    archive = output / relative
    archive.parent.mkdir(exist_ok=True, parents=True)
    previews = {}
    if not (output / "previews" / f"{skill}.jpg").exists():
        previews = {0: None, n//2: None, n-1: None}
    frames = 0
    with tarfile.open(archive, mode="x") as tar:
        for index, shard in enumerate(ep["frame_shards"]):
            path = member(root, shard)
            size = shard["frames"]
            with np.load(path, allow_pickle=False) as f:
                require(set(f.files) == {"rgb", "state_eef10", "native_step"} and shard["first_frame"] == frames
                        and np.array_equal(f["native_step"], values["native_step"][frames:frames+size])
                        and np.array_equal(f["state_eef10"], state_eef10[frames:frames+size]),
                        "RGB data differs from its actual label anchor")
                positions = [i for i in previews if frames <= i < frames+size]
                if positions:
                    pixels = f["rgb"]
                    require(pixels.shape == (size, 2, 256, 256, 3) and pixels.dtype == np.uint8, "RGB ABI changed")
                    for i in positions:
                        previews[i] = pixels[i-frames]
            add(tar, f"frames-{index:03d}.npz", path)
            frames += size
        require(frames == n, "Incomplete actual dual-camera observation coverage")
        add(tar, "episode.json", (json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode())
        add(tar, "labels.npz", packed_npz(**values, state_eef10=state_eef10, prior_actual_width_m=prior,
                                          actual_width_m=width))
    if previews:
        panel = Image.new("RGB", (256*len(previews),512))
        for k, pixels in enumerate(previews.values()):
            for view in (0, 1):
                panel.paste(Image.fromarray(pixels[view]), (256*k, 256*view))
        panel.save(output / "previews" / f"{skill}.jpg", quality=94)
    return dict(info, path=relative.as_posix(), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size)


def kind(task):
    return task.partition("-")[0].lower()


def build(archive, corrected, auxiliary, recovery, output, increments, state, actions):
    raw = read(pin(archive / "MANIFEST.private.json", LEGACY))
    target = read(pin(corrected / "MANIFEST.json", R4))
    physical = read(pin(auxiliary / "MANIFEST.json", AUX))
    old = {row["episode"]: row for row in target["episodes"]}
    widths = {row["episode"]: row for row in physical["episodes"]}
    train = [row for row in raw["episodes"] if row["id"] in old]
    require(len(train) == len(old) == len(widths) == 52 and sum(row["windows"] for row in train) == 4450,
            "Expected only the 52 actual legacy training demonstrations")
    output.mkdir(mode=0o700)
    (output / "previews").mkdir()
    records = []
    for ep in train:
        name = ep["id"]
        row = old[name]
        require(ep["individual_pass"] and row["scene"] == ep["scene_sha256"], "Original training identity changed")
        records.append(write_episode(output, archive, ep, member(corrected, dict(file=row["labels"], sha256=row["sha256"])),
                                     member(auxiliary, widths[name]), kind(ep["task"])))
        if len(records) % 16 == 0:
            print(f"Sealed {len(records)} episodes", file=sys.stderr, flush=True)
    for root, expected in zip(increments, INCREMENTS, strict=True):
        batch = read(pin(root / "MANIFEST.json", expected))
        for row in batch["episodes"]:
            ep = row["episode"]
            require(ep["individual_pass"], "Only individually qualified Teacher demonstrations may leave Core")
            records.append(write_episode(output, root, ep, member(root, row["corrected"]), member(root, row["aux"]), kind(ep["task"])))
    require(len(records) == 93 and sum(r["windows"] for r in records) == 7916, "Published A93 differs from training A93")
    suffix = read(pin(recovery / "MANIFEST.json", RECOVERY))
    for ep in suffix["episodes"]:
        records.append(write_episode(output, recovery, ep, member(recovery, ep["labels"]), member(recovery, ep["gripper"]),
                                     "recovery", recovery=True))
    require(len({r["episode"] for r in records}) == len(records) == 96
            and sum(r["windows"] for r in records[-3:]) == 115, "Unexpected Recovery window set")
    stats = {}
    for role, path, key in (("state", state, "state_statistics"), ("actions", actions, "action_statistics")):
        digest = suffix["statistics_sha256"][key]
        values = read(pin(path, digest))[role]
        stats[role] = dict(q01=values["q01"], q99=values["q99"], original_statistics_sha256=digest)
    (output / "r4_statistics.json").write_text(json.dumps(stats, indent=2, allow_nan=False)+"\n")
    counts = Counter(row["skill"] for row in records)
    payload = dict(schema="GO2_ARX5_A93_R3_INSPECTION_V1", training_episodes=93, training_windows=7916,
                   recovery_teacher_suffixes=3, recovery_windows=115, category_counts=dict(counts), episodes=records,
                   training_a93_sha256=INCREMENTS[-1], training_r3_sha256=RECOVERY)
    (output / "index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    with (output / "episodes.csv").open("w", newline="") as stream:
        fields = ("episode", "split", "skill", "task", "language", "windows", "full_h40", "supported_action_rows", "path", "archive_bytes", "archive_sha256")
        table = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        table.writeheader()
        table.writerows(records)
    previews = "\n\n".join(f"**{skill.upper()}**\n\n![{skill} snapshots](previews/{skill}.jpg)"
                             for skill in sorted(counts))
    (output / "README.md").write_text(f"""---
tags:
- robotics
- mujoco
- go2
- arx5
pretty_name: Go2 + ARX5 Embodiment A93 and Recovery3
---

# Go2＋ARX5 本体示范 A93 / Recovery3

这份私有快照来自 Corelab 的 MuJoCo 采集。A93 有 **93 条完整示范 / 7,916 窗**，Recovery3 有 **3 条独立教师接管后缀 / 115 窗**。Recovery 与 A93 的三个 TRAIN 布局共用场景几何，各自使用独立 reset。快照绑定 [SLA V6.5](https://github.com/Leo-6-maker/go2-arx5-hierarchical-vla/pull/528)。

- [episodes.csv](episodes.csv) 提供可排序的回合索引、任务指令、窗口数和归档路径。
- [index.json](index.json) 包含每个回合的阶段计数、物理时钟、坐标和内容指纹。
- [r4_statistics.json](r4_statistics.json) 保留原训练归一化分位值。
- 每个 `.tar` 含 `episode.json`、`labels.npz` 和原始双相机 RGB 分片。`labels.npz` 包含 R4 修订动作 `[窗,40,7]`、真实消费 `valid` 掩码、同刻 EEF10 和前一实际夹爪命令。支持内的夹爪宽度单位为米。默认不消费掩码之外的行。
- 场景用 MuJoCo 2 ms 时钟，动作按 50 ms 消费，常规查询间隔 0.5 s。Recovery 只包含真实教师接管之后的后缀。

## 实际相机预览

每张图左至右为同一回合的起始、中段与末段。上排为机身 RGB，下排为腕部 RGB。

{previews}
""")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    checksums = [f"{sha(path)}  {path.relative_to(output).as_posix()}\n" for path in files]
    (output / "SHA256SUMS.txt").write_text("".join(checksums))
    print(json.dumps(dict(files=len(files)+1, total_bytes=sum(path.stat().st_size for path in files),
                          counts=dict(counts), manifest_sha256=sha(output / "index.json")), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for item in ("archive", "corrected", "auxiliary", "recovery", "output"):
        cli.add_argument(item, type=Path)
    cli.add_argument("increments", nargs=5, type=Path)
    cli.add_argument("--state", type=Path, required=True)
    cli.add_argument("--actions", type=Path, required=True)
    build(**vars(cli.parse_args()))
