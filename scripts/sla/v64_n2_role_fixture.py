"""Exercise two N2 role children without starting a simulator or a controller."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sys
import time

from v64_n2_roles import check_descriptors, verify_fixture


ROSTER = "5896bbde4474aeccb11cf4653bbd48daef447a4eb1919f6ffd361b584df993e3"
PROFILE = "12135d6832dc6e833b52e15eb8ba196bed865b382480736d3294eb813cb94c06"
LIBRARY = "bf7bbeeb3760302c2d6b9300985f85d5bea140628c7e8e2f0ffa15cf2fe884f2"
TOPIC = "/sla_v64_n2_observed_role_fixture"
MESSAGES = 16


def actor(root, slot, world_fd, gpu_fd, *, camera_case):
    import rclpy
    from rclpy.utilities import get_rmw_implementation_identifier
    from std_msgs.msg import String

    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.durable import write
    from v64_n2_camera_replay import probe

    descriptors = check_descriptors(slot, (world_fd, gpu_fd))
    try:
        check_descriptors(slot, (gpu_fd, world_fd))
    except PermissionError:
        pass
    else:
        raise RuntimeError("Cross-wired descriptors were accepted")
    if (os.environ["ROS_DOMAIN_ID"] != str(slot["ROS_DOMAIN_ID"])
            or os.environ["SLA_MULTIWORLD_UST_REALM"] != str(root / "ust")):
        raise PermissionError("Native role has a foreign domain or tracing path")
    node_name = "sla_v64_n2_" + slot["role"].replace("-", "_")
    rclpy.init(args=[])
    node = rclpy.create_node(node_name)
    received = []
    subscription = node.create_subscription(String, TOPIC, lambda msg: received.append(msg.data), 64)
    publisher = node.create_publisher(String, TOPIC, 64)
    try:
        if get_rmw_implementation_identifier() != "rmw_fastrtps_cpp":
            raise RuntimeError("The pinned UDP transport is not loaded")
        write(root / "READY.json", dict(role=slot["role"], pid=os.getpid()))
        deadline = time.monotonic() + 9
        while len(list(root.parent.glob("world-*/READY.json"))) != 2 or not publisher.get_subscription_count():
            rclpy.spin_once(node, timeout_sec=.05)
            if time.monotonic() >= deadline:
                raise TimeoutError("Both bounded role children did not become ready")
        probe(camera_case, root / "camera", gpu_uuid=slot["gpu_uuid"])
        camera = root / "camera/RESULT.json"
        frames = json.loads(camera.read_bytes())["frames"]
        expected = [json.dumps(dict(role=slot["role"], tick=row["tick"], hashes=row["hashes"]), sort_keys=True)
                    for row in frames[:MESSAGES]]
        for value in expected:
            publisher.publish(String(data=value))
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.03)
        if received != expected or set(node.get_node_names()) != {node_name}:
            raise RuntimeError("Received unexpected native camera payload or peer node")
        mappings = Path("/proc/self/maps").read_text()
        pages = [root / "ust" / f"{key}.wait" for key in ("user", "global")]
        if any(str(path) not in mappings for path in pages) or "/dev/shm/fastrtps_" in mappings:
            raise RuntimeError("DDS or UST still uses a shared transport mapping")
        write(root / "RESULT.json", dict(role=slot["role"], pid=os.getpid(), descriptors=descriptors,
              domain=slot["ROS_DOMAIN_ID"], cpus=slot["cpus"], received_actual_frames=len(received),
              camera_receipt_sha256=digest(camera), trace_inodes=[p.stat().st_ino for p in pages],
              checked_swapped_descriptors=True, physical_requests=0, ticks=0, optimizer_updates=0))
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


def run(manifest, profile, library, case, output):
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.durable import write

    if [digest(path) for path in (manifest, profile, library)] != [ROSTER, PROFILE, LIBRARY]:
        raise ValueError("Require the sealed N2 roles and the known transport bytes")
    roster = json.loads(manifest.read_bytes())
    spec = json.loads((case / "EXECUTION_SPEC.json").read_bytes())
    source = spec["source_root"]
    base = dict(os.environ, **spec["native_environment"])
    base["PYTHONPATH"] = os.pathsep.join((source, source + "/src", base["PYTHONPATH"]))
    base.pop("CUDA_VISIBLE_DEVICES", None)
    base.update(LIBGL_ALWAYS_SOFTWARE="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    output.mkdir(mode=0o700)
    launch = output / "WORKER_CONFIG.private.json"
    write(launch, dict(camera_case=str(case)))
    verify_fixture(roster, profile, library, base, spec["python"], output, Path(__file__).resolve())
    results = [json.loads((output / slot["role"] / "RESULT.json").read_bytes())
               for slot in roster["candidate_worlds"]]
    if len({p for role in results for p in role["trace_inodes"]}) != 4:
        raise RuntimeError("The two child processes share trace pages")
    write(output / "RESULT.json", dict(roles=results, scope="N2_TWO_REAL_ROS_RENDER_CHILDREN",
          native_requests=0, new_integrations=0, training_updates=0))


if __name__ == "__main__":
    if len(sys.argv) == 5:
        root, slot = Path(sys.argv[1]), json.loads(sys.argv[2])
        args = json.loads((root.parent / "WORKER_CONFIG.private.json").read_bytes())
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(slot["egl_device"])
        actor(root, slot, sys.argv[3], sys.argv[4], camera_case=Path(args["camera_case"]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for key in ("manifest", "profile", "library", "case", "output"):
            parser.add_argument(key, type=Path)
        run(**vars(parser.parse_args()))
