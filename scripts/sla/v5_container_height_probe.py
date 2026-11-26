"""Run a fixed cold screen of carry height with the stock V5 corridor solver."""

import argparse
from itertools import product
from pathlib import Path

import mujoco as mj

from go2arx5_vla.world_model.native_replay import forbid_integration
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v5_mobile_container_corridor import COLD, SCORES, corridor, terminal_pose
import v5_mobile_container_corridor as existing


BASELINE = "fd815693dede6e41f4b53005364ccaa96103cf9de267618c97153491c9db3df9"
SOLVER = "9c1d0001da10405b404e8b2aa51f801067605fd9b3124252fc380d00e3cd88c1"
HEIGHTS_MM = (12, 15, 20, 25, 30)
LATERAL_MM = (-5, 0, 5)


def run(factory, hold_red, hold_blue, output, lateral_screen):
    solver = pin(existing.__file__, SOLVER)
    layout_root = factory / "SCENE_FACTORY_CONTAINER-PROJECT-01"
    layouts = read(pin(layout_root / "ACTUAL.json", COLD))["rows"]
    previous = read(pin(factory / "SCENE_FACTORY_CONTAINER-CORRIDOR-06.json", BASELINE))
    require(len(layouts) == len(previous["rows"]) == 2 and previous["carry_clearance_mm"] == 10,
            "Two frozen layouts and the original 10 mm screen are required")
    heights = (12, 15) if lateral_screen else HEIGHTS_MM
    offsets = list(product(LATERAL_MM, repeat=2)) if lateral_screen else [(0, 0)]
    output.mkdir(mode=0o700)
    write(output / "DESIGN.json", dict(carry_clearance_mm=heights, target_offset_xy_mm=offsets,
              historical_reference_mm=10, two_layouts=[row["id"] for row in layouts],
              solver_sha256=sha(solver), original_comparison_sha256=BASELINE, native_integrations=0,
              decision="Report matched carry and release across prescribed target offsets with fixed bowl geometry"))
    trials = []
    with forbid_integration(mj) as effects:
        for layout, historical, folder, score_sha in zip(layouts, previous["rows"],
                (hold_red, hold_blue), SCORES, strict=True):
            scene = pin(layout_root / (layout["id"] + ".mjb"), layout["compiled_sha256"])
            model = mj.MjModel.from_binary_path(str(scene))
            qpos, evidence = terminal_pose(folder, score_sha)
            require(layout["id"] == historical["id"] and layout["compiled_sha256"] == historical["scene_sha256"]
                    and evidence == historical["source_evidence"], "Original state or scene changed")
            for height, (dx, dy) in product(heights, offsets):
                x, y, z = layout["fixture_world_xyz"]
                aim = dict(layout, fixture_world_xyz=[x + dx / 1000, y + dy / 1000, z])
                result = corridor(mj, model, qpos, aim, height / 1000.0)
                trials.append(dict(id=layout["id"], command_carry_height_mm=height,
                                   target_offset_xy_mm=[dx, dy],
                                   scene_sha256=layout["compiled_sha256"], source_evidence=evidence, **result))
    require(not effects and len(trials) == len(heights)*len(offsets)*2, "Incomplete or integrated cold screen")
    write(output / "ACTUAL.json", dict(design_sha256=sha(output / "DESIGN.json"), baseline=previous["rows"],
              trials=trials, native_integrations=0, model_forwards=0, training_updates=0,
              native_dispatch_qualified=False))
    print(sha(output / "ACTUAL.json"), len(trials), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("factory", "hold_red", "hold_blue", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--lateral-screen", action="store_true")
    run(**vars(cli.parse_args()))
