"""Verify the real official A/C policy load and replay a retained sampler output on CPU."""

import argparse
from contextlib import redirect_stdout
import gc
import os
from pathlib import Path
import sys
import tempfile

import jax
import numpy as np
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict

from r5_checkpoint import payload_digest, restore, save
from r5_policy_endpoint import (
    BASE, OPTIMIZER, adapter_template, endpoint_receipts, frozen_digests,
    official_policy, parameter_layout,
)
from demonstrated_grip import pin, read, require
from feedback_gpu import verify_files
from policy_query import frozen_predictor, input_identity
from r5_training_owner import checksum
from visual_lift_panel import write_new

INPUT_ARCHIVE = {
    "INPUT.npz": "50d497f7b25c7bb12ec3acad221bac15ef310664b86a092903cf49bbb2d0de6e",
    "OUTPUT.npz": "f278b7a2384f3c0198b4aa598f1b778e1e1e66b1507f3de2d53bb7cf932cb6d8",
    "REQUEST.json": "43f9f065925d3189382c10cd2fb58a1129a2458cd7373e26616d2364c98775e9",
}


def sealed_window(root):
    case = root / "v18/Z1_REACH_483/G00-A/query-00"
    paths = {name: pin(case / name, sha) for name, sha in INPUT_ARCHIVE.items()}
    meta = read(paths["REQUEST.json"])
    with np.load(paths["INPUT.npz"], allow_pickle=False) as n:
        observation = dict(
            images=dict(cam_high=n["head"], cam_right_wrist=n["wrist"],
                        cam_left_wrist=np.zeros((480, 640, 3), np.uint8)),
            state=n["state"], prompt=str(n["prompt"]),
        )
    require(input_identity(observation)[0] == meta["input_sha256"], "paused RGB input changed")
    with np.load(paths["OUTPUT.npz"], allow_pickle=False) as n:
        sampler = {
            "state": n["sampler_state"].astype(str(n["sampler_state_dtype"])),
            "actions": n["actions"].astype(str(n["actions_dtype"])),
        }
        expected = n["absolute"]
    require(sampler["actions"].shape == (50, 32) and expected.shape == (50, 16),
            "retained sampler and decoded action shape changed")
    return observation, sampler, expected, meta


def processor_receipt(model, observation, sampler, expected):
    transformed = model._policy._input_transform(jax.tree.map(lambda x: x, observation))
    # Match Policy.infer's device dtype conversion before comparing the
    # retained sampler state. The quantile transform can produce float64.
    inputs = jax.tree.map(lambda x: np.asarray(jax.numpy.asarray(x)), transformed)
    require(np.array_equal(inputs["state"], sampler["state"]),
            "official processor state differs from retained sampler input")

    # Feed the already archived sample to each policy's official decoder
    # and end-effector packer. No policy sampling method is invoked here.
    rebuilt = frozen_predictor(
        model, observation,
        infer=lambda _: model._policy._output_transform(jax.tree.map(np.copy, sampler)),
    )
    require(np.array_equal(rebuilt, expected), "official decoder or EEF packing drift")
    return dict(
        processor_sha256=payload_digest(inputs),
        decoded_eef_sha256=payload_digest(rebuilt),
        image_masks={key: bool(x) for key, x in inputs["image_mask"].items()},
        memory_shape=list(inputs["extra_images"].shape),
        action_shape=list(rebuilt.shape),
    )


def run(root):
    require(jax.default_backend() == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "isolated CPU load only")
    cfg = verify_files(root)
    observation, sampler, expected, meta = sealed_window(root)
    a = official_policy(cfg)
    ref = processor_receipt(a, observation, sampler, expected)
    a_digests = frozen_digests(nnx.state(a._policy._model).to_pure_dict())
    require(len(a_digests) == 51, "reference frozen parent layout changed")
    del a
    gc.collect()

    layout = read(OPTIMIZER)["trainable"]
    binding = dict(scope="R5_COLD_ZERO_ADAPTER_FIXTURE", max_steps=1)
    with tempfile.TemporaryDirectory(dir=root / "v18", prefix="r5-endpoint-cpu-") as scratch:
        store = Path(scratch)
        result = {}

        def adapter_supplier(config, checkpoint):
            template = adapter_template(config, checkpoint)
            save(store, completed=0, adapters=template["adapters"], optimizer=template["adamw"],
                 binding=binding, adapter_layout=layout)
            restored = restore(store, completed=0, template=template, binding=binding,
                               adapter_layout=layout)
            require(payload_digest(restored) == payload_digest(template), "initial adapter readback changed")
            result["initialized_checkpoint_sha256"] = payload_digest(restored)
            result["loaded_adapters_sha256"] = payload_digest(restored["adapters"])
            return restored["adapters"], a_digests

        c = official_policy(cfg, load_adapters=adapter_supplier)
        actual = processor_receipt(c, observation, sampler, expected)
        require(actual == ref, "A and C input or output pipeline differs")
        pure = nnx.state(c._policy._model).to_pure_dict()
        frozen, learned = {}, {}
        for key, value in flatten_dict(pure).items():
            dest = learned if "/".join(key) in layout else frozen
            dest[key] = value
        require(frozen_digests(unflatten_dict(frozen)) == a_digests,
                "frozen parent leaves drifted after official load")
        attached = unflatten_dict(learned)
        require(parameter_layout(attached) == layout
                and payload_digest(attached) == result["loaded_adapters_sha256"],
                "loaded adapter layout or bytes differ from the checkpoint")
        zero_b = ["/".join(k) for k, v in learned.items() if k[-1].endswith("lora_b") and not np.any(v)]
        require(len(zero_b) == 5, "initial LoRA B leaves perturbed")
        try:
            endpoint_receipts(root, dict(trial=str(root / "v18/r5-smoke-cold-endpoint"), completed_step=0))
        except ValueError as exc:
            require(str(exc) == "R5 committed training checkpoint required",
                    "untrained direct endpoint rejected for the wrong reason")
        else:
            raise AssertionError("an initial checkpoint cannot identify an adapted candidate")
        del c, pure, frozen, learned
        gc.collect()

    return dict(
        scope="OFFICIAL_C_POLICY_INJECTION_COLD", frozen_parent_leaves=len(a_digests),
        loaded_adapter_leaves=len(layout), zero_B_leaves=zero_b, official_model_loads=2,
        common_official_processor_and_decoder=ref,
        same_processor_bytes=True, historical_decoder_output_exact=True,
        closed_trained_endpoint_requires_positive_updates=True,
        input_receipts_sha256=INPUT_ARCHIVE, observation_sha256=meta["input_sha256"],
        retained_sampler_seed=meta["seed"], model_sampling_forwards=0,
        actual_optimizer_updates=0, resource_leases_acquired=0, new_mujoco_ticks=0,
        trained_C_evaluated=False, all_adapter_bytes_preserved=True,
        implementation_sha256={
            name: checksum(BASE / "scripts/sla_go2_v18" / name)
            for name in ("r5_policy_endpoint.py", "r5_policy_cold_check.py", "r5_checkpoint.py")
        },
        **result,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        output = run(args.root.resolve())
    write_new(args.output, output)
