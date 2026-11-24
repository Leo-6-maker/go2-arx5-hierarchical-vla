"""One-variable left RGB attention control on the input to the official sampler."""

import hashlib
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from reach_crossed_seeds import case_inputs
from demonstrated_grip import require

WRIST = "left_wrist_0_rgb"


def mask_left(transform, obs):
    result = transform(obs)
    masks = result["image_mask"]
    require(
        set(masks) == {"base_0_rgb", WRIST, "right_wrist_0_rgb"}
        and all(bool(v) for v in masks.values()),
        "unexpected baseline wrist attention contract",
    )
    return {**result, "image_mask": {**masks, WRIST: np.bool_(False)}}


def cpu_check(root):
    import jax
    from checkpoint_inputs import configured_transforms
    from openpi import transforms
    from openpi.models.model import Observation, preprocess_observation

    require(jax.default_backend() == "cpu", "CPU preflight only")
    cfg, data, stats = configured_transforms(root)
    original = transforms.compose(
        [
            *data.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ]
    )
    rows = []
    for i, (obs, _) in enumerate(case_inputs(root)):
        before = original(jax.tree.map(lambda x: x, obs))
        after = mask_left(original, jax.tree.map(lambda x: x, obs))
        require(before.keys() == after.keys(), "keys")
        for k in before:
            if k == "image_mask":
                continue
            a, tree = jax.tree.flatten(before[k])
            b, other = jax.tree.flatten(after[k])
            require(
                tree == other
                and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True)),
                "change outside image mask",
            )

        def sampler_observation(d):
            batch = jax.tree.map(lambda x: np.asarray(x)[None, ...], d)
            return preprocess_observation(
                None, Observation.from_dict(batch), train=False
            )

        a, b = map(sampler_observation, (before, after))
        require(
            all(np.array_equal(a.images[k], b.images[k]) for k in a.images)
            and np.array_equal(a.extra_images, b.extra_images)
            and np.array_equal(a.state, b.state)
            and np.array_equal(a.tokenized_prompt, b.tokenized_prompt),
            "unequal sampler inputs",
        )
        keys = sorted(a.image_masks)
        require(
            [k for k in keys if not np.array_equal(a.image_masks[k], b.image_masks[k])]
            == [WRIST],
            "mask missing before embedding",
        )
        rows.append(
            dict(
                query=i,
                mask_control={k: bool(a.image_masks[k][0]) for k in keys},
                mask_treatment={k: bool(b.image_masks[k][0]) for k in keys},
                unchanged_history_sha256=hashlib.sha256(
                    np.asarray(a.extra_images).tobytes()
                ).hexdigest(),
                identical_pixels_state_tokens=True,
            )
        )
    return dict(
        config=cfg.name, rows=rows, model_forwards=0, training_updates=0, integrations=0
    )


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        result = cpu_check(Path(sys.argv[1]).resolve())
    print(json.dumps(result, indent=2, allow_nan=False))
