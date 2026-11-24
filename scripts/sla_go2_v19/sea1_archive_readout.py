"""Score immutable SEA1 anchors and prove exact archival reuse without a sampler call."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from source_custody import (
    BASE, O0, SEEDS, digest, pin, read, replay_real_input, require, target_case, target_obs,
)
from checkpoint_inputs import configured_transforms
from demonstrated_grip import events, rows
from policy_query import input_identity, tensor_sha
from visual_lift_panel import write_new

SEA = BASE / "reports/SLA_GO2_V19"
ATTRIBUTION = BASE / "reports/SLA_GO2_V18/Z1_ATTRIBUTION"
TIMES = np.arange(50) / 30.
S0_PIN = "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9"
P10 = 10
FULL = 50
CURRENT_VIEWS = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
OFFICIAL_NORM_SHA256 = "cf6ddf4f89180e762e5e6210881a763f8e29d3499fb1cc69a440de7aa585d276"


def pose_rows(arm):
    p = np.repeat(np.eye(4)[None], len(arm), axis=0)
    p[:, :3, 3] = arm[:, :3]
    p[:, :3, :3] = Rotation.from_quat(arm[:, [4, 5, 6, 3]]).as_matrix()
    return p


def rms(values):
    return float(np.sqrt(np.mean(values**2)))


def alignment(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if np.linalg.norm(a) < .001 or np.linalg.norm(b) < .001:
        return None
    return float(np.clip(a @ b / denom, -1, 1))


def pose_error(a, b):
    dp = np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1)
    dr = np.rad2deg(Rotation.from_matrix(a[:, :3, :3] @ b[:, :3, :3].transpose(0,2,1)).magnitude())
    return dict(xyz_rms_mm=1000*rms(dp), xyz_peak_mm=1000*float(dp.max()),
                so3_rms_deg=rms(dr), so3_peak_deg=float(dr.max()))


def source_score(pred, expected, measured):
    result = {}
    for label, n in (("P10", P10), ("FULL50", FULL)):
        h = np.repeat(measured[None], n, axis=0)
        p, a = pred[:n], expected[:n]
        result[label] = dict(
            model=pose_error(p, a), hold=pose_error(h, a),
            measured_to_final_direction_cos=alignment(p[-1, :3, 3]-measured[:3, 3],
                                                       a[-1, :3, 3]-measured[:3, 3]),
        )
    return result


def source_rows(root, anchor_rows):
    o0 = {(x["episode"], x["frame"], x["call"]): x for x in read(O0)["source"]}
    path = root / "evidence/O0_PANEL_01"
    results = []
    for a in anchor_rows:
        k, frame = a["o0_call"], a["frame"]
        original = o0[a["episode"], frame, k]
        meta = read(pin(path / f"call_{k:02d}_input.json", a["o0_input_receipt_sha256"]))
        with h5py.File(pin(root / "data" / a["episode"], a["source_h5_sha256"])) as h:
            require(replay_real_input(h, original, meta) == a["input_sha256"], "source input changed")
            actions = np.column_stack([
                h[f"action/{side}_{kind}"][frame:frame+50]
                for side in ("left", "right") for kind in ("ee_poses", "ee_joint_states")
            ])
        with np.load(pin(path / f"call_{k:02d}_actions.npz", a["o0_archive_sha256"]), allow_pickle=False) as z:
            require(np.array_equal(actions, z["expert_absolute_arms"])
                    and np.array_equal(z["source_indices"], frame+np.arange(50))
                    and np.allclose(z["source_offset_s"], TIMES, atol=1e-12, rtol=0),
                    "source action labels or clock changed")
            expected, pred, state = (z[n].copy() for n in ("expert_absolute_arms", "native_absolute_arms", "state"))
            require(not {"actions", "sampler_state"} <= set(z.files),
                    "source raw archive availability changed")
        require(all(np.isfinite(v).all() for v in (actions, pred, state))
                and meta["prng_key_before_split"] == a["record_seed"]
                and np.array_equal(state.astype(np.float32), np.asarray(meta["state"], dtype=np.float32)),
                "source return or seed custody")
        metrics = source_score(pose_rows(pred[:, 8:]), pose_rows(expected[:, 8:]), pose_rows(state[None, 8:])[0])
        require(abs(metrics["FULL50"]["model"]["xyz_rms_mm"] - a["source_rmse_mm"]) < 1e-5
                and abs(metrics["FULL50"]["hold"]["xyz_rms_mm"] - a["original_hold_rmse_mm"]) < 1e-5,
                "historical source metric changed")
        results.append(dict(
            id=a["id"], frame=frame, episode=a["episode"], seed=a["record_seed"],
            on_pilot_seed_grid=a["record_seed"] in SEEDS, raw_sampler_available=False,
            scope="SOURCE_LINK6_RECORDED_NOMINAL_30HZ", metrics=metrics,
            gripper=dict(predicted=events(TIMES, pred[:, 15], step=False),
                         recorded=events(TIMES, expected[:, 15], step=True),
                         fraction_rms=rms(pred[:, 15]-expected[:, 15])),
        ))
    return results


def processor(root):
    from openpi import transforms

    cfg, data, stats = configured_transforms(root)
    prepare = transforms.compose([
        transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs,
        transforms.Normalize(stats, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs,
    ])
    inverse = transforms.compose([
        *data.model_transforms.outputs, transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
        *data.data_transforms.outputs,
    ])
    require(cfg.name == "train_robodojo_real" and cfg.model.action_dim == 32
            and cfg.model.action_horizon == 50 and not cfg.model.extra_frames_first,
            "unexpected official model contract")
    path = root / "evidence/SOURCE_AND_WEIGHT_MANIFEST.actual.json"
    manifest = read(path)
    original = read(O0)["checkpoint"]
    require(manifest["model_revision"] == original["revision"]
            and manifest["source_commit"] == original["source_commit"],
            "official identity different from sealed O0")
    witness = dict(
        model_revision=manifest["model_revision"], source_commit=manifest["source_commit"],
        effective_config_sha256=manifest["effective_config_sha256"],
        norm_sha256=OFFICIAL_NORM_SHA256, tokenizer_sha256=manifest["tokenizer"]["sha256"],
        manifest_sha256=digest(path), action_horizon=cfg.model.action_horizon,
        action_dim=cfg.model.action_dim, attention_impl=cfg.model.attention_impl,
        extra_frames_first=cfg.model.extra_frames_first,
    )
    return prepare, inverse, witness


def mask_and_output(root, sealed, anchor, cell):
    case = target_case(root, anchor["layout"])
    i = int(anchor["id"][-2:])
    query = case / f"query-{i:02}"
    request = read(sealed(query / "REQUEST.json"))
    source = 483 if anchor["original_return_present"] and cell["arm"] == "RAW" and request["seed"] == cell["seed"] else (
        489 if cell["arm"] == "LEFT_MASK_ONLY" else 486
    )
    if source == 483:
        rc = sealed(query / "RETURN.json")
        output = sealed(query / "OUTPUT.npz")
        journal = rows(sealed(query / "events.jsonl"))
        start = next(z for z in journal if z["event"] == "FORWARD_ENTERED")
        require(start["seed"] == cell["seed"] and start["digest"] == cell["input_sha256"]
                and sum(z["event"] == "FORWARD_RETURNED" for z in journal) == 1
                and read(sealed(query / "GPU_EXIT.json"))["code"] == 0,
                "original worker did not return cleanly")
        mask = {k: True for k in CURRENT_VIEWS}
    else:
        name = "CROSSED" if source == 486 else "WRIST_MASK"
        receipt = read(ATTRIBUTION / (name + "_CUSTODY.json"))
        hashes = receipt["members" if source == 486 else "hashes"]
        folder = root / "v18" / ("Z1_CROSSED_486" if source == 486 else "Z1_MASK_489")

        def file(name):
            return pin(folder / name, hashes[name])

        require(read(file("EXIT.json"))["returned"] == 2, "incomplete archival ablation")
        start = read(file(f"entered-{i}.json"))
        require(start["seed"] == cell["seed"] and start["input_sha256"] == cell["input_sha256"],
                "archival seed or observation mismatch")
        rc = file(f"returned-{i}.json")
        output = file(f"cross-{i}.npz")
        if source == 489:
            mask = read(pin(file(f"mask-{i}.json"), cell["sampler_mask_sha256"]))
            require(mask == {k: k != "left_wrist_0_rgb" for k in CURRENT_VIEWS}, "left mask proof")
        else:
            mask = {k: True for k in CURRENT_VIEWS}
    reply = read(pin(rc, cell["receipt_sha256"]))
    require(reply["input_sha256"] == anchor["input_sha256"] == cell["input_sha256"]
            and reply["output_sha256"] == cell["output_sha256"]
            and reply.get("seed", cell["seed"]) == cell["seed"], "output cache receipt changed")
    return source, mask, pin(output, cell["output_sha256"])


def target_context(root, sealed, item, prep, geometry, calibration):
    case = target_case(root, item["layout"])
    i = int(item["id"][-2:])
    q = case / f"query-{i:02}"
    expected = target_obs(pin(sealed(q / "INPUT.npz"), item["input_file_sha256"]))
    require(input_identity(expected)[0] == item["input_sha256"], "input bytes changed")
    measured_state = expected["state"].copy()
    data = prep(expected)
    require(set(data["image"]) == set(data["image_mask"]) == CURRENT_VIEWS
            and all(data["image_mask"].values()) and data["extra_images"].shape == (9,224,224,3)
            and all(v.shape == (224,224,3) for v in data["image"].values())
            and data["state"].shape == (32,), "official effective input contract")
    image_hash = dict(
        current={k: tensor_sha(v) for k, v in data["image"].items()},
        head_history=[tensor_sha(x) for x in data["extra_images"]],
        tokens=tensor_sha(data["tokenized_prompt"]),
        token_mask=tensor_sha(data["tokenized_prompt_mask"]),
        state=tensor_sha(data["state"]),
    )
    episode = read(sealed(case / "NATIVE_REQUEST.json"))["episode"]
    atom = next(r for r in rows(sealed(case / episode / "MOBILE_SAMPLES.private.jsonl"))
                if r["native_end_step"] == item["tick"])
    tool = np.array(calibration["T_link6_TCP"])
    relative = np.eye(4)
    relative[:3, 3] = calibration["source_origin"]
    origin = np.array(atom["actual_mount"]) @ relative
    actual = np.array(atom["actual_TCP"])
    decoded_tcp = origin @ pose_rows(measured_state[None, 8:])[0] @ tool
    require(np.allclose(decoded_tcp, actual, atol=1e-7, rtol=0), "Go2 state to actual TCP changed")
    reference = geometry[case.name, item["tick"]]
    require(reference["input_sha256"] == item["input_sha256"] and reference["current_rgb_exact"],
            "object geometry belongs to another capture")
    centers = {color: p["center_world_m"] for color, p in reference["image_objects"].items()}
    centers["red_native_audit"] = np.asarray(atom["object_pose"])[:3, 3]
    return data["state"], image_hash, origin, tool, actual, centers


def read_cell(root, sealed, cell, item, context, inverse):
    state, fields, origin, tool, actual, centers = context
    issue, mask, path = mask_and_output(root, sealed, item, cell)
    with np.load(path, allow_pickle=False) as z:
        raw = z["actions"].astype(str(z["actions_dtype"]))
        sampled_state = z["sampler_state"].astype(str(z["sampler_state_dtype"]))
        decoded = z["absolute"].copy()
    require(raw.shape == (50, 32) and sampled_state.shape == (32,)
            and np.isfinite(raw).all() and np.isfinite(decoded).all(), "invalid archived sampler output")
    state_hash = tensor_sha(sampled_state)
    decoded_hash = tensor_sha(decoded)
    restored = inverse(dict(state=sampled_state.copy(), actions=raw.copy()))["actions"].astype(np.float32)
    require(state_hash == tensor_sha(state.astype(sampled_state.dtype)) and decoded_hash == tensor_sha(restored),
            "official processor or inverse does not reproduce the returned bytes")
    poses = origin @ pose_rows(decoded[:, 8:]) @ tool
    outcomes = {}
    for label, n in (("P10", P10), ("FULL50", FULL)):
        start, end = poses[0,:3,3], poses[n-1,:3,3]
        desired = {}
        for key, center in centers.items():
            if center is None:
                desired[key] = None
                continue
            center = np.asarray(center)
            desired[key] = dict(
                first_to_last_gain_mm=1000*float(np.linalg.norm(start-center)-np.linalg.norm(end-center)),
                measured_to_last_gain_mm=1000*float(np.linalg.norm(actual[:3,3]-center)-np.linalg.norm(end-center)),
                direction_cos=alignment(end-start, center-start),
            )
        outcomes[label] = dict(next_actual_distance_mm=1000*float(np.linalg.norm(start-actual[:3,3])),
                               source_travel_mm=1000*float(np.linalg.norm(end-start)), object_forecast=desired)
    grip = dict(events=events(TIMES, decoded[:,15], step=False),
                first=float(decoded[0,15]), last=float(decoded[-1,15]))
    receipt = dict(
        id=cell["id"], condition=cell["arm"], seed=cell["seed"], issue=issue,
        input_sha256=cell["input_sha256"], output_sha256=cell["output_sha256"],
        recorded_seed_and_inverse_verified=True,
        actual_noise_tensor_archived=False, sampler_state_sha256=state_hash,
        raw_actions_sha256=tensor_sha(raw), decoded_absolute_sha256=decoded_hash,
        reconstructed_processor_fields=fields, image_masks=mask,
        mask_evidence="AT_SAMPLER" if issue == 489 else "FROZEN_DEFAULT_PROCESSOR",
        per_horizon=outcomes, right_gripper_fraction=grip,
    )
    return receipt, (raw, poses, decoded[:,15])


def paired(available):
    tests = [
        ("EQUAL_OBSERVATION_SEED_CHANGE", row, "RAW", 0, row, "RAW", 1) for row in ("G01-00", "G01-01")
    ] + [
        ("EQUAL_OBSERVATION_LEFT_MASK", row, "RAW", 0, row, "LEFT_MASK_ONLY", 0)
        for row in ("G01-00", "G01-01")
    ] + [
        ("EQUAL_SEED_JOINT_OBSERVATION_CHANGE", "G01-00", "RAW", seed, "G01-01", "RAW", seed) for seed in (0,1)
    ]
    out = []
    for name, a, ca, sa, b, cb, sb in tests:
        x, y = available[a,ca,sa], available[b,cb,sb]
        entry = dict(kind=name, first=[a,ca,[0,sa]], second=[b,cb,[0,sb]])
        for key,n in (("P10",10),("FULL50",50)):
            entry[key] = dict(world_tcp=pose_error(x[1][:n], y[1][:n]),
                              raw_sampler_rms=rms(x[0][:n].astype(float)-y[0][:n].astype(float)),
                              gripper_fraction_rms=rms(x[2][:n]-y[2][:n]))
        out.append(entry)
    return out


def run(root, output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("JAX_PLATFORMS") == "cpu",
            "CPU processor and inverse only")
    from jax import default_backend

    require(default_backend() == "cpu", "wrong processing backend")
    roster_path = SEA / "SEA0/SEA0_PAIRED_ROSTER.json"
    roster = read(roster_path)
    seals = read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")
    members = read(pin(root / "v18/sealed-Z1/members.json", seals["member_manifest_sha256"]))

    def sealed(path):
        return pin(path, members[str(path.relative_to(root / "v18"))])

    geom = read(ATTRIBUTION / "IMAGE_GEOMETRY.json")
    require(geom["source_archive_sha256"] == seals["archive_sha256"], "geometry archive identity")
    geometry = {(x["slot"], x["tick"]): x for x in geom["samples"]}
    inputs = read(pin(root / "v13/run01/S0_ACTUAL.json", S0_PIN))["inputs"]
    cal_path = root / "v12/TARGET_REGISTRATION.json"
    calibration = read(pin(cal_path, inputs[str(cal_path)]))
    prepare, inverse, witness = processor(root)
    source = source_rows(root, roster["source"])
    anchors = {r["id"]: r for r in roster["go2"]}
    contexts = {k: target_context(root, sealed, a, prepare, geometry, calibration) for k,a in anchors.items()}
    results, available = [], {}
    for cell in roster["exact_archive_candidates"]:
        item = anchors[cell["id"]]
        row, vectors = read_cell(root, sealed, cell, item, contexts[item["id"]], inverse)
        k = (item["id"], cell["arm"], cell["seed"][1])
        require(k not in available, "repeated archival cell")
        available[k] = vectors
        results.append(row)
    require(len(available) == 7 and len(roster["conditions"]) == 64
            and sum(bool(a["on_pilot_seed_grid"]) for a in source) == 1, "unexpected frozen denominator")
    cells, registered = [], set()
    for row in roster["conditions"]:
        key = (row["id"], row["arm"], row["seed"][1])
        require(row["seed"] in SEEDS and key not in registered, "invalid fixed observation/seed roster")
        registered.add(key)
        state = ("ARCHIVE_CPU_VERIFIED_CONDITIONAL_REUSE" if key in available else row["planned_status"])
        cells.append(dict(id=row["id"], condition=row["arm"], seed=row["seed"], status=state))
    require(available.keys() <= registered, "unregistered archival outputs")
    write_new(output, dict(
        scope="IMMUTABLE_HISTORICAL_RETURNS_CPU_ONLY", roster_sha256=digest(roster_path),
        processor_identity=witness, historical_raw_retained_and_cpu_verified=len(available),
        source_decoded_only=source,
        archived_target_cells=results, paired_controls=paired(available), fixed_panel_cells=cells,
        science_panel_authorized=False, new_comparable_rendered_cells=0,
        new_gpu_forwards=0, native_ticks=0, optimizer_updates=0,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    run(args.root.resolve(), args.output.resolve())
