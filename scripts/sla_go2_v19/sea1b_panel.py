"""Bind the fixed V1.9.2 source mask/state factorial to exact official inputs."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from sea1a_panel import infer, observation_fields, rng_noise
from sea1_archive_readout import processor
from source_custody import BASE, O0, array_hash, digest, pin, read, real_source_observation, require, write_new
from policy_query import input_identity, tensor_sha
from demonstrated_grip import rows
from reach_wrist_mask import WRIST, mask_left

PLAN = BASE / "docs/protocols/SLA_GO2_V192_SEA_REGISTRY_20261124.json"
PLAN_SHA = "9cc9bffc1083218a8a5eb9709ed7b6a4996caf27ab45fd4d3b38b56bbe4c6003"
ANCHORS = ("S00", "S01", "S02", "S03")
CONDITIONS = ("S0", "S1", "S2", "S3")
CONDITION_SCOPE = [(True, "REAL"), (False, "REAL"), (True, "A1_FIXED_VIRTUAL"), (False, "A1_FIXED_VIRTUAL")]
SOURCE = BASE / "reports/SLA_GO2_V19/SEA0/SEA0_PAIRED_ROSTER.json"
HISTORICAL = BASE / "reports/SLA_HIERARCHICAL_V16/V4_SLA_SKILL_AND_LONG_HORIZON_DIAGNOSTIC.json"
A1_STATE = "O1_STATE_ONLY"
PANEL = "SEA1B-SOURCE-FACTORIAL-01"


def registration():
    plan = read(pin(PLAN, PLAN_SHA))
    panel = next(s for s in plan["stages"] if s["id"] == "SEA1B")
    cells = [(c["id"], c["anchor"], c["condition"], c["seed"]) for c in panel["cells"]]
    order = [(f"SEA1B-{a}-{c}-seed{n}", a, c, [0, n])
             for n in range(2) for a in ANCHORS for c in CONDITIONS]
    require(cells == order and panel["cap"] == dict(candidate_cells=32, new_sampler_enters=32,
                                                     physics_ticks=0, updates=0), "registered population changed")
    require([(c["id"], c["left_wrist_mask"], c["left_arm"]) for c in panel["conditions"]]
            == [(c, *shape) for c, shape in zip(CONDITIONS, CONDITION_SCOPE, strict=True)],
            "registered condition changed")
    return plan["frozen"], panel


def recorded(h, field, selection):
    return np.column_stack([h[f"{field}/{side}_{kind}"][selection]
                            for side in ("left", "right") for kind in ("ee_poses", "ee_joint_states")])


def source_inputs(root, panel):
    anchors = {a["id"]: a for a in read(SOURCE)["source"]}
    historical = read(HISTORICAL)["source_factorial"]["source_sha256"]
    report = read(pin(root / "v12/A1_SOURCE_FACTORIAL.json", historical))
    archive = root / "v12/A1_SOURCE_01"
    journal = rows(pin(archive / "calls.jsonl", report["journal_sha256"]))
    a1_registration = read(pin(root / "v12/A1_REGISTRATION.json", report["registration_sha256"]))
    require(a1_registration["inactive_arm"] == "same source episode 0 frame 0 left-state vector as P1"
            and report["returned"] == 12, "historical surrogate origin changed")
    calls = {(r["episode"], r["frame"]): r["call"] for r in journal
             if r["event"] == "FORWARD_RETURNED" and r["profile"] == A1_STATE}
    a1_rows = {(r["episode"], r["frame"]): r for r in report["windows"]}
    o0 = {r["call"]: r for r in read(O0)["source"]}
    source_files = {}
    for anchor in anchors.values():
        path = root / "data" / anchor["episode"]
        if path not in source_files:
            source_files[path] = pin(path, anchor["source_h5_sha256"])
    path = root / "data/episode_0000000.hdf5"
    with h5py.File(source_files[path]) as h:
        virtual = recorded(h, "state", slice(0, 1))[0, :8].astype(np.float32)
    inputs, fingerprints = {}, []
    for p in panel["anchors"]:
        name, episode, frame = p["id"], p["episode"], p["frame"]
        a = anchors[name]
        require((episode, frame, p["input_sha256"]) == (a["episode"], a["frame"], a["input_sha256"]),
                "fixed anchor changed")
        meta = read(pin(root / f"evidence/O0_PANEL_01/call_{a['o0_call']:02d}_input.json",
                        a["o0_input_receipt_sha256"]))
        with h5py.File(source_files[root / "data" / episode]) as h:
            require(int(h["additional_info/frequency"][()]) == 30
                    and len(h["state/right_ee_poses"]) > frame+50, "source window clock or reach changed")
            raw = real_source_observation(h, o0[a["o0_call"]], meta)
        old = a1_rows[episode, frame]["assets"][A1_STATE]
        original = read(pin(archive / f"call_{calls[episode, frame]:02d}_input.json", old["input"]))
        altered = {**raw, "state": raw["state"].copy()}
        altered["state"][:8] = virtual
        require(input_identity(raw)[0] == p["input_sha256"]
                and np.array_equal(altered["state"], np.asarray(original["state"], np.float32))
                and original["prompt"] == raw["prompt"]
                and original["head_frames"] == a["history_frames"]
                and original["head_sha256"] == [array_hash(x) for x in raw["images"]["cam_high"]]
                and original["wrists_sha256"] == {k: array_hash(raw["images"][k])
                                                   for k in ("cam_left_wrist", "cam_right_wrist")},
                "virtual state or A1 state-only custody changed")
        inputs[name] = raw, altered
        fingerprints.append(dict(anchor=name, episode=episode, frame=frame, source_sha256=a["source_h5_sha256"],
                                 original_input_sha256=p["input_sha256"],
                                 state_only_input_sha256=input_identity(altered)[0],
                                 virtual_state_a1_receipt_sha256=old["input"], history_frames=a["history_frames"]))
    return inputs, fingerprints, tensor_sha(virtual)


def verified_processor(root, frozen):
    prepare, inverse, identity = processor(root)
    contract = dict(
        model_revision=frozen["model_revision"], source_commit=frozen["upstream_commit"],
        effective_config_sha256=frozen["effective_config_sha256"], norm_sha256=frozen["norm_sha256"],
        tokenizer_sha256=frozen["tokenizer_sha256"], manifest_sha256=frozen["manifest_sha256"],
        action_horizon=frozen["source_horizon"], action_dim=frozen["raw_action_dim"],
        attention_impl=frozen["attention_impl"], extra_frames_first=frozen["extra_frames_first"],
    )
    require(identity == contract, "registered processor identity changed")
    return prepare, inverse, identity


def packed(prepare, obs, masking):
    value = {**obs, "images": dict(obs["images"]), "state": obs["state"].copy()}
    return mask_left(prepare, value) if masking else prepare(value)


def fields_at_device(value):
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation

    batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], value)
    return observation_fields(Observation.from_dict(batch))


def discrete_token_reader(root):
    from checkpoint_inputs import configured_transforms
    from openpi.transforms import TokenizePrompt

    _, data, _ = configured_transforms(root)
    transform = next(t for t in data.model_transforms.inputs if isinstance(t, TokenizePrompt))
    require(transform.discrete_state_input, "source codec changed its discrete state format")
    return transform.tokenizer._tokenizer


def verify_discrete_prompt(decoder, prepared, prompt):
    mask = prepared["tokenized_prompt_mask"]
    length = int(mask.sum())
    require(0 < length < len(mask) and np.array_equal(mask, np.arange(len(mask)) < length),
            "source state tokens were truncated or scattered")
    encoded = prepared["tokenized_prompt"][mask].astype(int).tolist()
    canonical_prompt = prompt.strip().replace("_", " ").replace("\n", " ")
    values = np.digitize(prepared["state"][:14], np.linspace(-1, 1, 257)[:-1]) - 1
    expected = f"Task: {canonical_prompt}, State: {' '.join(map(str, values))};\nAction: "
    require(decoder.decode(encoded) == expected, "unexpected source task or state token encoding")
    return length


def verify_factorial(prepare, observations, expect, decoder):
    """Verify every factor before running any sampler and attest the transformed tokens."""
    out, spans = {}, {}
    for name, (real, virtual) in observations.items():
        prepared = [packed(prepare, obs, mask) for obs in (real, virtual) for mask in (False, True)]
        a, b, c, d = map(fields_at_device, prepared)
        require(a == expect[name], "original preflight sampler input changed")
        for with_wrist, masked in ((a, b), (c, d)):
            require(all(with_wrist["current_masks"].values())
                    and masked == {**with_wrist, "current_masks": {**with_wrist["current_masks"], WRIST: False}},
                    "mask changed sensor values outside left wrist attention")
        require(all(a[k] == c[k] for k in ("current", "memory", "current_masks"))
                and np.array_equal(prepared[0]["state"][7:], prepared[2]["state"][7:])
                and not np.array_equal(prepared[0]["state"][:7], prepared[2]["state"][:7])
                and a["token_ids"] != c["token_ids"]
                and real["prompt"] == virtual["prompt"],
                "state-only factor changed a non-left-state input or did not reach state tokens")
        spans[name] = [verify_discrete_prompt(decoder, item, real["prompt"])
                       for item in (prepared[0], prepared[2])]
        out[name] = dict(zip(CONDITIONS, (a, b, c, d), strict=True))
    return out, spans


def predict_fields():
    old = BASE / "reports/SLA_GO2_V19/SEA1/OBSERVATION_INGRESS_ACTUAL.json"
    return {r["id"]: r["tensors"] for r in read(old)["source"]}


def code_fingerprints():
    paths = [
        Path(__file__), Path(__file__).with_name("sea1a_panel.py"), Path(__file__).with_name("sea1_archive_readout.py"),
        Path(__file__).with_name("source_custody.py"), BASE / "scripts/sla_go2_v18/reach_wrist_mask.py",
    ]
    return {str(p.relative_to(BASE)): digest(p) for p in paths}


def cold(root, output):
    import jax
    import jax.numpy as jnp
    from openpi.policies.policy import Policy

    require(jax.default_backend() == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU cold only")
    frozen, panel = registration()
    observations, identity, virtual = source_inputs(root, panel)
    prepare, inverse, processor_identity = verified_processor(root, frozen)
    fields, spans = verify_factorial(prepare, observations, predict_fields(), discrete_token_reader(root))
    samples = []

    def surrogate(_, obs, **kwargs):
        return jnp.zeros_like(kwargs["noise"])

    policy = SimpleNamespace(_output_transform=inverse, _sample_actions=surrogate, _sample_kwargs={})
    policy.infer = Policy.infer.__get__(policy)
    noises = {tuple(k): rng_noise(k) for k in panel["seeds"]}
    for cell in panel["cells"]:
        src, cond, seed = (cell[k] for k in ("anchor", "condition", "seed"))
        payload = observations[src][cond in ("S2", "S3")]
        policy._input_transform = lambda obs: packed(prepare, obs, cond in ("S1", "S3"))
        result, captured = infer(policy, payload, fields[src][cond], *noises[tuple(seed)], samples.append)
        require(result["actions"].shape == (50, 16) and captured["actions"].shape == (50, 32)
                and all(np.isfinite(a).all() for a in (result["actions"], captured["actions"])), "shape/inverse cold check")
    require(len(samples) == 32, "missing cold cell")
    write_new(output, dict(
        id=PANEL, status="COLD_32_INPUT_SLOTS_VERIFIED", registry_sha256=PLAN_SHA, processor=processor_identity,
        code=code_fingerprints(), inputs=identity, effective_sampler_fields=fields,
        real_and_virtual_valid_token_counts=spans, fixed_virtual_state_sha256=virtual,
        cells=[c["id"] for c in panel["cells"]], verified_exact_archival_reuse=0,
        historical_initial_noise_available=False, fixed_actual_gpu_noise_required=True,
        allocation=None, cold_surrogate_calls=len(samples), learned_model_forwards=0, physics_ticks=0, training_updates=0,
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    cold(args.root.resolve(), args.output.resolve())
