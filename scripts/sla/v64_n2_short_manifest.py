"""Bind the registered N2 short pairs to the existing references and site roles."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest
from scripts.sla.core_multienv_geometry import load_population
from scripts.sla.core_multienv_n1 import fixed_cards
from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases


CAMERA_SHA = "9ab93080b0ccb982ef55aa633956644a833d42c799ebfd11615f6b28b9984798"
DDS_SHA = "18129d259af6bd0c3979e6df58b5ed68d310102f93548bf37208b802c84c1834"
OBSERVER_SHA = "a7630e674493356f2e06a36c40556250e04a742723e25a02a03141046c0a3a13"
N1_MISMATCH_SHA = "565aca43d220ec34e71ef5155af74e62471cafae7ed3b65cd0d8d1d3043c1d89"
CPUS = ((6, 7, 10, 11), (8, 9, 12, 13))


def read(path):
    return json.loads(path.read_bytes())


def pin(path, expected=None):
    observed = digest(path)
    if expected is not None and observed != expected:
        raise ValueError(f"Bound site evidence changed: {path.name}")
    return dict(path=str(path), sha256=observed)


def locks_in_project(owner):
    locks = [owner / name for name in ("ros-192.lock", "ros-196.lock",
                                       "sla-pt-gpu0-owner.lock", "sla-pt-gpu1-owner.lock")]
    if any(path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid() for path in locks):
        raise PermissionError("Existing project world and rendering lease files are required")
    if len({(path.stat().st_dev, path.stat().st_ino) for path in locks}) != len(locks):
        raise PermissionError("N2 owners resolve to a shared lease file")
    return locks


def gpu_slots(camera, domains, locks):
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                                     "--format=csv,noheader,nounits"], text=True, timeout=5)
    cards = {int(index): (uuid, int(free)) for index, uuid, free in
             (row.split(", ") for row in lines.splitlines())}
    if not set().union(*CPUS) <= os.sched_getaffinity(0) or set(CPUS[0]) & set(CPUS[1]):
        raise PermissionError("The proposed CPU roles must be available and disjoint")
    slots = []
    for i, (cpus, domain) in enumerate(zip(CPUS, domains, strict=True)):
        observed = camera["roles"][f"gpu{i}"]["identity"]
        uuid, free = cards[i]
        if observed["uuid"] != uuid:
            raise ValueError("GPU identity differs from the native camera replay")
        slots.append(dict(role=f"world-{i}", cpus=cpus, ROS_DOMAIN_ID=domain,
                          gpu_index=i, gpu_uuid=uuid, egl_device=observed["observed_egl_cuda"],
                          available_VRAM_MiB=free, world_lock=str(locks[i]), renderer_lock=str(locks[2+i])))
    if slots[0]["gpu_uuid"] == slots[1]["gpu_uuid"]:
        raise ValueError("Each world requires its assigned GPU")
    return slots


def compile_rows(authority, site_spec, mismatch):
    original = fixed_cards(authority)
    short = [card for card in original.values() if card["slot"]["group"] == "N1-SHORT"]
    logical = authority["logical"]
    slots = [row for row in load_population(Path(logical["path"]))["engineering"] if row["group"] == "N2-SHORT"]
    history = {row["id"]: row for row in mismatch["comparators"]}
    if len(slots) != len(short) or len(slots) != 8:
        raise ValueError("Expected the original eight N2 short slots and comparators")
    require_unstarted_cases(site_spec["registry"], {row["id"] for row in slots})
    cases = []
    for i, (slot, card) in enumerate(zip(slots, short, strict=True)):
        n1_id = card["slot"]["id"]
        template = read_pin(card["reference"])
        record = pin(Path(card["recording"]["path"]), card["recording"]["sha256"])
        cases.append(dict(slot=slot, world_role=f"world-{i % 2}", pair=i // 2,
                          family=template["mobile_condition"]["family"], motion=template["d2_probe_motion"],
                          reference=card["reference"], archived_comparator=record, n1_comparator=n1_id,
                          n1_warm_measurement=history[n1_id]["pretask_error"],
                          n1_frozen_verdict=history[n1_id]["frozen_verdict"]))
    pairs = list(zip(cases[::2], cases[1::2], strict=True))
    for a, b in pairs:
        if {a["family"], b["family"]} != {"F1", "F2"} or a["motion"] != b["motion"]:
            raise ValueError("Each registered pair must match motion across both families")
    for motion in ("STABLE", "SMALL_EEF"):
        for family in ("F1", "F2"):
            roles = [row["world_role"] for row in cases if row["family"] == family and row["motion"] == motion]
            if set(roles) != {"world-0", "world-1"} or len(roles) != 2:
                raise ValueError("Both registered rounds must exchange each family/motion between roles")
    return cases


def stage(n1, observer, camera, transport, mismatch, output):
    site_spec = read(n1 / "EXECUTION_SPEC.json")
    authority = read_pin(site_spec["sim_only_project_authorization"])
    refs = dict(observer=pin(observer, OBSERVER_SHA), camera=pin(camera, CAMERA_SHA),
                transport=pin(transport, DDS_SHA), mismatch=pin(mismatch, N1_MISMATCH_SHA))
    close, rgb, dds, history = (read_pin(ref) for ref in refs.values())
    if (close["exit_code"] or close["ticks"] != 10703 or not close["observer_shutdown"]["actual_signal_disarm"]
            or close["comparison"]["status"] != "WITHIN_FROZEN_STATE_TOLERANCE"
            or dds["status"] != "LOOPBACK_FIXTURE_PASS" or dds["scopes"] != [192, 196]
            or rgb["input_native_pairs"] != 20):
        raise ValueError("Require the real N1 observer, camera and project transport evidence")
    parent_lock = Path(authority["world_lock"])
    worlds = gpu_slots(rgb, dds["scopes"], locks_in_project(parent_lock.parent))
    cases = compile_rows(authority, site_spec, history)
    with output.open("x") as stream:
        json.dump(dict(status="SHORT_ROSTER_AND_ROLE_PREFLIGHT", source_evidence=refs, candidate_worlds=worlds,
                       original_global_lock=str(parent_lock), original_population=authority["logical"],
                       tolerances=authority["tolerance"], cases=cases, request_cap=sum(
                           row["slot"]["max_native_requests"] for row in cases), tick_cap=sum(
                           row["slot"]["max_native_ticks"] for row in cases), native_requests=0, physics_ticks=0), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("n1", "observer", "camera", "transport", "mismatch", "output"):
        parser.add_argument(name, type=Path)
    stage(**vars(parser.parse_args()))
