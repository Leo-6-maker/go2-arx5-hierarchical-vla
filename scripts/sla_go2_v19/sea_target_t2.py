"""Read the four frozen compact H0 anchors through the official SLA ingress."""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from source_custody import BASE, SEEDS, digest, pin, read, require, write_new
from policy_query import input_identity, tensor_sha
from sea1_archive_readout import S0_PIN
from sea1a_panel import infer
from sea1b_panel import discrete_token_reader, fields_at_device, registration, source_inputs, verified_processor, verify_discrete_prompt
from sea_observation import AUXILIARY_CAMERA, THREE_VIEW_V1, Frame, SensorObservation, prepare_sensor
from sea_task_193 import LOCAL, PLAN, PLAN_SHA
from sea_three_view import calibration, rigid

PANEL = "SEA-TASK-T2-193-3V-01"
H0 = "V193-COMPACT-FRUIT-H0-01"
H0_SHA = "d2d7b94ea4ac4d33fdfa15082d12edd0c87276710ffc380c3e2ecd83a81fe443"
TL_SHA = "ec294a686c0fc95cb2fb89a0cfa9fbbba845703e219dda814502a4b8046eb8a6"
EVIDENCE = BASE / "reports/SLA_GO2_V193"
LAYOUTS = ("G00-H0", "G01-H0")
EVENTS = ("before_close", "first_query_after_grasp")
STATE_STATUS = "HISTORICAL_A1_VIRTUAL_LEFT_AND_GO2_MECHANICAL_RIGHT_PROXY"
NO_CODEC = "UNAVAILABLE_MATCHED_SOURCE_TOOL_AND_PAD_GAP"
CONDITION = "THREE_VIEW_TL"


def code_binding():
    scripts = Path(__file__).parent
    return {name: digest(scripts / name) for name in (
        "sea_target_t2.py", "sea_target_t2_execute.py", "sea_target_t2_score.py",
        "sea1a_panel.py", "sea1b_panel.py", "sea_task_193.py", "sea_observation.py", "sea_three_view.py",
    )}


def target_registration(root):
    original = read(pin(root / "v13/run01/S0_ACTUAL.json", S0_PIN))
    path = root / "v12/TARGET_REGISTRATION.json"
    return read(pin(path, original["inputs"][str(path)])), digest(path)


def target_state(z, setup, virtual):
    arm, flange, tcp = (rigid(z[k]) for k in ("armbase", "link6", "tcp"))
    tool = rigid(setup["T_link6_TCP"])
    require(np.allclose(flange @ tool, tcp, rtol=0, atol=2e-9), "measured target TCP binding changed")
    offset = np.eye(4)
    offset[:3, 3] = setup["source_origin"]
    origin = arm @ offset
    local = np.linalg.inv(origin) @ tcp @ np.linalg.inv(tool)
    wxyz = Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]]
    if wxyz[0] < 0:
        wxyz = -wxyz
    fingers = z["finger_slide_m"]
    require(fingers.shape == (2,) and np.isfinite(fingers).all(), "invalid actual target gripper state")
    aperture = fingers.mean() / .044
    require(0 <= aperture <= 1, "target observation gripper outside normalized input range")
    return np.r_[virtual, local[:3, 3], wxyz, aperture].astype(np.float32), origin


def anchors_and_payloads(root, prepare):
    cap = next(s["cap"] for s in read(pin(PLAN, PLAN_SHA))["stages"] if s["id"] == "TARGET_T2_3V")
    require(cap == dict(layouts=2, anchors_per_layout=2, seeds=2,
                        new_sampler_enters=8, dual_camera_new_forwards=0), "unexpected T2 limit")
    h0 = read(pin(EVIDENCE / "GRIP_H0/COMPACT_H0_ACTUAL.json", H0_SHA))
    panel = root / "v19" / H0
    for name, sha in h0["campaign_sha256"].items():
        pin(panel / name, sha)
    require(h0["started"] == h0["qualified"] == 2 and not h0["unstarted"]
            and set(h0["rows"]) == set(LAYOUTS), "incomplete compact H0 controls")
    _, source = registration()
    inputs, _, virtual_sha = source_inputs(root, source)
    virtual = inputs["S00"][1]["state"][:8]
    require(tensor_sha(virtual) == virtual_sha, "registered virtual left arm changed")
    setup, setup_sha = target_registration(root)
    rows, payloads, effective = {}, {}, {}
    for layout in LAYOUTS:
        row = h0["rows"][layout]
        case = panel / layout
        require(row["qualified"] and row["lease_closed"], "unqualified H0")
        require(read(pin(case / "control/LEASE.json", row["evidence_sha256"]["LEASE.json"]))["status"] == "CLOSED",
                "teacher simulation lease not closed")
        folder = case / "THREE_VIEW_H0.private"
        manifest = read(pin(folder / "MANIFEST.json", row["capture"]["manifest_sha256"]))
        for event in EVENTS:
            mark = row["capture"]["events"][event]
            meta = manifest[event]
            require(meta["status"] == "SEALED" and meta == {k: mark[k] for k in meta}
                    and meta["aux_calibration"] == calibration(), "unexpected archived event or auxiliary mount")
            name = layout[:3] + "-" + event
            tick = meta["tick"]
            path = pin(folder / (event + ".npz"), mark["artifact_sha256"])
            # Event labels and contact scoring remain in custody, outside model input.
            with np.load(path, allow_pickle=False) as z:
                state, origin = target_state(z, setup, virtual)
                head, wrist, aux = z["head"], z["wrist"], z["aux"]
                history = z["history_ticks"].tolist()
            require(history == meta["history_ticks"] and [tensor_sha(x) for x in head] == meta["history_frame_sha256"]
                    and tensor_sha(wrist) == meta["wrist_sha256"] and tensor_sha(aux) == meta["aux_sha256"]
                    and head[-1].any() and wrist.any(), "bad archived RGB custody")
            sensor = SensorObservation(
                layout, f"tick-{tick}", tick, "GO2_NATIVE_2MS",
                tuple(Frame(layout, "cam_high", t, rgb) for t, rgb in zip(history, head, strict=True)),
                Frame(layout, "cam_right_wrist", tick, wrist), Frame(layout, AUXILIARY_CAMERA, tick, aux),
                state, LOCAL, THREE_VIEW_V1,
            )
            result, sha = prepare_sensor(sensor, prepare)
            payload = sensor.legacy_input()
            require(set(payload) == {"images", "state", "prompt"}
                    and input_identity(payload)[0] == sha, "sensor-only payload changed")
            rows[name] = dict(
                layout=layout, event=event, tick=tick, time_s=meta["time_s"],
                scene_sha256=meta["scene_sha256"], archive_sha256=mark["artifact_sha256"],
                native_packet_sha256=meta["current_packet_sha256"], history_ticks=history,
                historical_target_registration_sha256=setup_sha, virtual_left_sha256=virtual_sha,
                right_observed_channel=state[8:].tolist(), go2_inner_pad_gap_mm=mark["measured_go2_inner_pad_gap_mm"],
                world_from_diagnostic_flange_frame=origin.tolist(), input_sha256=sha,
            )
            payloads[name, CONDITION], effective[name] = payload, result
    return rows, payloads, effective


def exact_noise(root):
    import jax
    import jax.numpy as jnp

    tl = read(pin(EVIDENCE / "SOURCE_TASK/TL_ACTUAL.json", TL_SHA))
    require(tl["plan_sha256"] == PLAN_SHA and tl["owner_receipt"]["project_lock_released"]
            and tl["owner_receipt"]["entered"] == tl["owner_receipt"]["returned"] == 8, "unclosed T-L source panel")
    source_hashes = {n: {r["actual_noise_sha256"] for r in tl["results"] if r["seed"] == seed}
                     for n, seed in enumerate(SEEDS)}
    noises = {}
    for n, seed in enumerate(SEEDS):
        rng = jax.random.wrap_key_data(jnp.asarray(seed, dtype=jnp.uint32))
        key = np.asarray(jax.random.key_data(jax.random.split(rng)[1]))
        with np.load(root / "v19" / tl["run"] / f"noise-{n}.npz", allow_pickle=False) as saved:
            noise = saved["noise"]
            require(np.array_equal(key, saved["sample_key"]) and noise.shape == (1, 50, 32)
                    and noise.dtype == np.float32 and np.isfinite(noise).all()
                    and source_hashes[n] == {tensor_sha(noise)}, "T-L initial noise differs")
            noises[n] = rng, key, noise.copy()
    return noises


def cell_list(rows, noises):
    return [dict(id=f"{name}-seed{n}", anchor=name, condition=CONDITION, seed=seed,
                 input_sha256=mark["input_sha256"], initial_noise_sha256=tensor_sha(noises[n][2]))
            for n, seed in enumerate(SEEDS) for name, mark in rows.items()]


def cold(root, output):
    import jax
    import jax.numpy as jnp
    from openpi.policies.policy import Policy
    from types import SimpleNamespace

    require(jax.default_backend() == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preflight only")
    frozen, _ = registration()
    prepare, inverse, processor = verified_processor(root, frozen)
    rows, payloads, effective = anchors_and_payloads(root, prepare)
    decoder = discrete_token_reader(root)
    fields = {}
    for name, value in effective.items():
        verify_discrete_prompt(decoder, value, LOCAL)
        f = fields_at_device(value)
        require(len(f["memory"]) == 9 and all(f["current_masks"].values()), "three-view memory or masks changed")
        fields[name] = {CONDITION: f}
    noises = exact_noise(root)
    cells = cell_list(rows, noises)
    stub = SimpleNamespace(_input_transform=prepare, _output_transform=inverse, _sample_kwargs={},
                           _sample_actions=lambda _, __, **kw: jnp.zeros_like(kw["noise"]))
    stub.infer = Policy.infer.__get__(stub)
    calls = []
    for cell in cells:
        name = cell["anchor"]
        answer, seen = infer(stub, payloads[name, CONDITION], fields[name][CONDITION],
                             *noises[cell["seed"][1]], calls.append)
        require(answer["actions"].shape == (50, 16) and seen["actions"].shape == (50, 32),
                "official forward/inverse ABI differs")
    require(len(calls) == len(cells) == 8, "incomplete T2 surrogate sampler seam")
    write_new(output, dict(
        status="T2_3V_CPU_COLD_VERIFIED", panel=PANEL, source_plan_sha256=PLAN_SHA,
        compact_h0_receipt_sha256=H0_SHA, source_tl_receipt_sha256=TL_SHA, processor=processor,
        source_state_profile=STATE_STATUS, physical_action_admission=NO_CODEC, code=code_binding(),
        prompt=LOCAL, rows=rows, cells=cells, fields=fields,
        surrogate_calls=len(calls), frozen_model_forwards=0, new_native_starts=0,
        physics_ticks=0, optimizer_updates=0,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    os.umask(0o077)
    cold(a.site.resolve(), a.output)
