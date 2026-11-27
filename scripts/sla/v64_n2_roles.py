"""Give each N2 child its own CPU, ROS, trace, and renderer lease."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
import ctypes
import json
import os
from pathlib import Path
import subprocess


def role_environment(base, slot, root, profile, realm, library):
    env = dict(base, ROS_DOMAIN_ID=str(slot["ROS_DOMAIN_ID"]), ROS_AUTOMATIC_DISCOVERY_RANGE="SYSTEM_DEFAULT",
               ROS_STATIC_PEERS="", RMW_IMPLEMENTATION="rmw_fastrtps_cpp", RMW_FASTRTPS_USE_QOS_FROM_XML="0",
               FASTDDS_DEFAULT_PROFILES_FILE=str(profile), FASTRTPS_DEFAULT_PROFILES_FILE=str(profile),
               LTTNG_HOME=str(realm), LTTNG_UST_REGISTER_TIMEOUT="0", SLA_MULTIWORLD_UST_REALM=str(realm),
               LD_PRELOAD=str(library), TMPDIR=str(root / "tmp"), ROS_LOG_DIR=str(root / "logs/ros"),
               XDG_CACHE_HOME=str(root / "cache"), XDG_RUNTIME_DIR=str(root / "runtime"),
               XDG_CONFIG_HOME=str(root / "config"))
    for key in ("ROS_LOCALHOST_ONLY", "ROS_DISCOVERY_SERVER", "DISPLAY"):
        env.pop(key, None)
    return env


def create_role_folders(root):
    root.mkdir(mode=0o700)
    for name in ("cache", "config", "tmp", "runtime", "logs", "ust"):
        (root / name).mkdir(mode=0o700)
    for name in ("user.wait", "global.wait"):
        with (root / "ust" / name).open("xb") as stream:
            stream.truncate(os.sysconf("SC_PAGE_SIZE"))


def acquire_role_locks(stack, slots):
    from scripts.pi0292.durable import OwnerLock

    owners = {}
    for slot in slots:
        key = slot["role"]
        if key in owners:
            raise PermissionError("Repeated child role")
        world = stack.enter_context(closing(OwnerLock(slot["world_lock"])))
        renderer = stack.enter_context(closing(OwnerLock(slot["renderer_lock"])))
        owners[key] = (world.handle.fileno(), renderer.handle.fileno())
    if len({fd for values in owners.values() for fd in values}) != 4:
        raise PermissionError("Expected four distinct live owner descriptors")
    return owners


def launch_pair(just_registered, owners):
    """Start both members and retain their separate bounded terminal results."""
    cards = dict(just_registered)
    if len(just_registered) != 2 or set(cards) != set(owners):
        raise PermissionError("Both assigned roles require exactly one registered child")
    with ThreadPoolExecutor(max_workers=2) as pool:
        work = {role: pool.submit(launch, owners[role]) for role, launch in cards.items()}
    return {role: future.result() for role, future in work.items()}


def check_descriptors(slot, descriptors):
    def identity(info):
        return info.st_dev, info.st_ino

    fds = [int(fd) for fd in descriptors]
    expected = [identity(Path(slot[key]).stat()) for key in ("world_lock", "renderer_lock")]
    observed = [identity(os.fstat(fd)) for fd in fds]
    if observed != expected or os.sched_getaffinity(0) != set(slot["cpus"]):
        raise PermissionError("Child received another role's CPU or owner descriptor")
    return observed


def verify_fixture(roster, profile, library, base, python, output, fixture_worker):
    from scripts.pi0292.durable import write

    slots = roster["candidate_worlds"]
    roots = {}
    for slot in slots:
        key = slot["role"]
        root = output / key
        create_role_folders(root)
        roots[key] = root

    def launch(slot, descriptors):
        path = roots[slot["role"]]
        env = role_environment(base, slot, path, profile, path / "ust", library)
        arguments = [python, "-B", str(fixture_worker), str(path), json.dumps(slot),
                     *(str(fd) for fd in descriptors)]
        with (path / "logs/role.log").open("xb") as log:
            child = subprocess.Popen(["taskset", "-c", ",".join(map(str, slot["cpus"])), *arguments],
                                     env=env, cwd=path, pass_fds=descriptors, stdout=log, stderr=subprocess.STDOUT)
            try:
                returncode = child.wait(timeout=35)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
                raise
        return dict(pid=child.pid, exit_code=returncode)

    from functools import partial

    with ExitStack() as stack:
        owners = acquire_role_locks(stack, slots)
        outcomes = launch_pair([(slot["role"], partial(launch, slot)) for slot in slots], owners)
    write(output / "CHILDREN.private.json", outcomes)
    if any(row["exit_code"] != 0 for row in outcomes.values()):
        raise RuntimeError("N2 role fixture failed. Preserve each child log.")
