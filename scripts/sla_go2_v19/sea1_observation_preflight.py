"""Verify the SEA sensor ingress on the four source and four frozen Go2 captures."""

import argparse
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from sea_observation import AppearanceEdit, Frame, LEFT_MASK, RasterEdit, SensorObservation, prepare_sensor
from sea1_archive_readout import processor
from source_custody import BASE, O0, digest, pin, read, real_source_observation, require, target_case, target_obs, write_new
from policy_query import input_identity, tensor_sha

SEA = BASE / "reports/SLA_GO2_V19"


def packet(legacy, episode, query_id, tick, clock, history, *, left_present):
    views = legacy["images"]
    head = tuple(Frame(episode, "cam_high", t, im) for t, im in zip(history, views["cam_high"], strict=True))
    right = Frame(episode, "cam_right_wrist", tick, views["cam_right_wrist"])
    left = Frame(episode, "cam_left_wrist", tick, views["cam_left_wrist"]) if left_present else None
    return SensorObservation(episode, query_id, tick, clock, head, right, left, legacy["state"], legacy["prompt"])


def sampler_fields(prepared):
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation, preprocess_observation

    batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], prepared)
    sampler = preprocess_observation(None, Observation.from_dict(batch), train=False)
    require(sampler.state.shape == (1, 32) and sampler.extra_images.shape == (1, 9, 224, 224, 3)
            and all(im.shape == (1, 224, 224, 3) for im in sampler.images.values()),
            "unexpected sampler image or state dimensions")
    return dict(
        current={k: tensor_sha(v) for k, v in sampler.images.items()},
        current_masks={k: bool(v[0]) for k, v in sampler.image_masks.items()},
        memory=[tensor_sha(im) for im in sampler.extra_images[0]],
        token_ids=tensor_sha(sampler.tokenized_prompt), token_mask=tensor_sha(sampler.tokenized_prompt_mask),
        state=tensor_sha(sampler.state),
    )


def source_replay(root, anchor, old_row, official):
    call = anchor["o0_call"]
    receipt = read(pin(root / f"evidence/O0_PANEL_01/call_{call:02d}_input.json", anchor["o0_input_receipt_sha256"]))
    name = anchor["episode"]
    with h5py.File(pin(root / "data" / name, anchor["source_h5_sha256"])) as h:
        raw = real_source_observation(h, old_row, receipt)
    wrapped = packet(raw, name, anchor["id"], anchor["frame"], "ROBODOJO_RECORDED_30HZ",
                     receipt["head_frames"], left_present=True)
    ready, received_id = prepare_sensor(wrapped, official)
    fields = sampler_fields(ready)
    require(received_id == input_identity(raw)[0] == anchor["input_sha256"]
            and fields == sampler_fields(official(raw)) and all(fields["current_masks"].values()),
            "real source bypass diverged from official input")
    conflicting_reset = None
    if wrapped.head[1].tick == 0:
        rgba = wrapped.head[1].rgb.copy()
        rgba[0, 0, 0] ^= np.uint8(1)
        conflicting = replace(wrapped.head[1], rgb=rgba)
        conflicting_reset = expect_refusal(lambda: replace(wrapped, head=(
            wrapped.head[0], conflicting, *wrapped.head[2:],
        )))
    return dict(id=anchor["id"], tick=anchor["frame"], input_sha256=received_id,
                clock=wrapped.clock, history_ticks=receipt["head_frames"], tensors=fields,
                conflicting_reset_control=conflicting_reset)


def edit_for(frame, rgb, mask):
    return RasterEdit(frame.episode, frame.camera, frame.tick, tensor_sha(frame.rgb), rgb, mask)


def target_replay(anchor, row, episode, query, raw, archive, official):
    expected = anchor["input_sha256"]
    require(query["tick"] == anchor["tick"] == row["tick"] and query["episode"] == episode
            and query["input_sha256"] == expected == row["original_input_sha256"] == input_identity(raw)[0]
            and [z["tick"] for z in row["head"]] == anchor["history_ticks"]
            and row["wrist"]["tick"] == anchor["tick"], "unexpected archive capture identity")
    wrapped = packet(raw, episode, anchor["id"], anchor["tick"], "GO2_NATIVE_2MS",
                     anchor["history_ticks"], left_present=False)
    with np.load(archive, allow_pickle=False) as z:
        head = tuple(edit_for(f, im, claim) for f, im, claim in
                     zip(wrapped.head, z["head"], z["head_paint"], strict=True))
        edit = AppearanceEdit(head, edit_for(wrapped.right, z["wrist"], z["wrist_paint"]))
    before, raw_id = prepare_sensor(wrapped, official)
    mask, mask_id = prepare_sensor(wrapped, official, left_mask_only=True)
    coated, coated_id = prepare_sensor(wrapped, official, appearance=edit)
    base, masked, appearance = map(sampler_fields, (before, mask, coated))
    require(raw_id == mask_id == expected and base == sampler_fields(official(raw))
            and all(base["current_masks"].values()), "historical sensor bypass changed")
    require(masked == dict(base, current_masks=dict(base["current_masks"], **{LEFT_MASK: False})),
            "left mask changed a second effective sampler field")
    require(coated_id == row["diagnostic_input_sha256"], "appearance candidate changed since sealed CPU replay")
    changed = [key for key in base["current"] if base["current"][key] != appearance["current"][key]]
    require(set(changed) == {"base_0_rgb", "right_wrist_0_rgb"}
            and all(x != y for x, y in zip(base["memory"], appearance["memory"], strict=True))
            and all(base[key] == appearance[key] for key in ("state", "token_ids", "token_mask", "current_masks")),
            "appearance edit changed a nonvisual sampler field or omitted a history slot")
    return dict(id=anchor["id"], tick=anchor["tick"], clock=wrapped.clock, source_left_camera_observed=False,
                history_ticks=anchor["history_ticks"], raw_input_sha256=raw_id, raw_fields=base,
                mask_only=masked["current_masks"], diagnostic_input_sha256=coated_id,
                diagnostic_fields=appearance, original_sampler_return_present=anchor["original_return_present"]), wrapped, edit


def expect_refusal(operation):
    try:
        operation()
    except ValueError as exc:
        return str(exc)
    raise ValueError("invalid observation passed the sensor ingress")


def negative_controls(obs, edit):
    past = obs.head[1]
    leaked = SimpleNamespace(**vars(obs), hidden_object_position=(999., 999., 999.))
    rgb = edit.right.rgb.copy()
    y, x = np.argwhere(~edit.right.support)[0]
    rgb[y, x, 0] ^= np.uint8(1)
    edited = replace(edit, right=replace(edit.right, rgb=rgb))

    def unexpected(_):
        raise AssertionError("invalid observation reached the official processor")

    return dict(
        future=expect_refusal(lambda: replace(obs, head=(obs.head[0], replace(past, tick=obs.tick + 1), *obs.head[2:]))),
        cross_episode=expect_refusal(lambda: replace(obs, right=replace(obs.right, episode="another-reset"))),
        reordered=expect_refusal(lambda: replace(obs, head=(obs.head[1], obs.head[0], *obs.head[2:]))),
        wrong_camera=expect_refusal(lambda: replace(obs, right=replace(obs.right, camera="cam_high"))),
        untyped_oracle=expect_refusal(lambda: prepare_sensor(leaked, unexpected)),
        cross_frame_edit=expect_refusal(lambda: prepare_sensor(obs, unexpected, appearance=replace(
            edit, right=replace(edit.right, tick=obs.tick - 1)))),
        unsupported_edit=expect_refusal(lambda: prepare_sensor(obs, unexpected, appearance=edited)),
        combined_left_mask_and_appearance=expect_refusal(lambda: prepare_sensor(
            obs, unexpected, appearance=edit, left_mask_only=True)),
    )


def run(root, bridge, output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("JAX_PLATFORMS") == "cpu",
            "CPU preprocessing only")
    from jax import default_backend

    require(default_backend() == "cpu", "unexpected backend")
    official, _, identity = processor(root)
    roster = read(SEA / "SEA0/SEA0_PAIRED_ROSTER.json")
    sealed = read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")
    members = read(pin(root / "v18/sealed-Z1/members.json", sealed["member_manifest_sha256"]))

    def checked(path):
        return pin(path, members[str(path.relative_to(root / "v18"))])

    custody = {r["id"]: r for r in roster["go2"]}
    source_rows = {(r["episode"], r["call"]): r for r in read(O0)["source"]}
    source = [source_replay(root, a, source_rows[a["episode"], a["o0_call"]], official) for a in roster["source"]]
    evidence = read(pin(bridge / "TARGET_BRIDGE_ACTUAL.json", digest(SEA / "SEA1/TARGET_BRIDGE_ACTUAL.json")))
    go2, controls = [], {}
    for row in evidence["slots"]:
        a = custody[row["id"]]
        case = target_case(root, a["layout"])
        folder = case / f"query-{int(a['id'][-2:]):02d}"
        req = read(checked(folder / "REQUEST.json"))
        ep = read(checked(case / "NATIVE_REQUEST.json"))["episode"]
        src = target_obs(pin(checked(folder / "INPUT.npz"), a["input_file_sha256"]))
        archive = pin(bridge / (row["id"] + ".private.npz"), row["archive_sha256"])
        summary, obs, edit = target_replay(a, row, ep, req, src, archive, official)
        go2.append(summary)
        controls[row["id"]] = negative_controls(obs, edit)
    require(len(source) == len(go2) == 4, "fixed preflight population changed")
    write_new(output, dict(
        scope="ARCHIVAL_SENSOR_INPUT_CPU_PREFLIGHT", processor_identity=identity,
        source=source, go2=go2, rejected_ingress_controls=controls,
        formal_pose_and_camera_pairs_qualified=0, model_forwards=0, physics_ticks=0, updates=0,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "bridge", "output"):
        cli.add_argument(k, type=Path)
    args = cli.parse_args()
    run(args.root.resolve(), args.bridge.resolve(), args.output.resolve())
