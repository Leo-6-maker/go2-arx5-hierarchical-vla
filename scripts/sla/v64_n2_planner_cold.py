"""Reproduce the N2 Python receiver's cold startup without a simulator."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def run_receiver(case, launcher, library, folder, cpu, domain, early, seconds):
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.durable import write

    folder.mkdir(mode=0o700)
    realm = folder / "ust"
    realm.mkdir(mode=0o700)
    for name in ("user.wait", "global.wait"):
        fd = os.open(realm / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.ftruncate(fd, os.sysconf("SC_PAGE_SIZE"))
        finally:
            os.close(fd)
    env = dict(os.environ, **case["native_environment"])
    env.update(ROS_DOMAIN_ID=str(domain), PI0294_EPISODE=folder.name, SLA_MULTIWORLD_UST_REALM=str(realm),
               LTTNG_HOME=str(realm), TMPDIR=str(folder), ROS_LOG_DIR=str(folder / "ros"),
               XDG_RUNTIME_DIR=str(folder), XDG_CONFIG_HOME=str(folder),
               PI0294_PLANNER_TRACE=str(folder / "trace.jsonl"))
    for key in ("ROS_LOCALHOST_ONLY", "ROS_DISCOVERY_SERVER", "DISPLAY"):
        env.pop(key, None)
    if early:
        env["LD_PRELOAD"] += ":" + str(library)
    args = ["taskset", "-c", str(cpu), *launcher["argv"]]
    log = folder / "planner.log"
    start = time.monotonic()
    result = dict(early_ust_load=early, cpu=cpu, domain=domain, ready_s=None)
    with log.open("x") as stream:
        process = subprocess.Popen(args, env=env, cwd=case["native_root"], stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None and time.monotonic() - start < seconds:
            if "Trajectory JSON input:" in log.read_text(errors="replace"):
                result["ready_s"] = time.monotonic() - start
                break
            time.sleep(.025)
        maps = Path("/proc") / str(process.pid) / "maps"
        observed = maps.read_text() if maps.is_file() else ""
        result["private_pages"] = [str(realm / f"{name}.wait") in observed for name in ("user", "global")]
        result["fastdds_shared"] = "/dev/shm/fastrtps_" in observed
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        result["exit"] = process.returncode
    result["log_sha256"] = digest(log)
    write(folder / "RESULT.json", result)
    return result


def test(closed_case, library, output, rounds, seconds):
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.durable import read, write

    spec = closed_case / "EXECUTION_SPEC.json"
    case = read(spec)
    terminal_file = closed_case / "RESULT.private.json"
    terminal = read(terminal_file)
    if (read(closed_case / "control/LEASE.json")["status"] != "CLOSED" or terminal["unsafe"]
            or terminal["actual_native_integrations"] or terminal["physical_requests"] != 1):
        raise PermissionError("Use the sealed zero-integration receiver failure")
    commands_file = closed_case / "native-01/PROCESS_COMMANDS.private.json"
    launcher, = (row for row in read(commands_file) if row["name"] == "planner")
    receiver = Path(case["source_root"]) / "scripts/pi0294/native_receiver.py"
    if launcher["argv"][:3] != [case["python"], "-B", str(receiver)]:
        raise PermissionError("Only the exact frozen receiver command is tested")
    available = sorted(os.sched_getaffinity(0) - {6, 7, 8, 9, 10, 11, 12, 13})
    cpus = available[:2]
    if len(cpus) != 2 or rounds < 2 or rounds % 2:
        raise ValueError("The matched comparison needs two independent CPUs and balanced assignments")
    output.mkdir(mode=0o700)
    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i in range(rounds):
            future = []
            for early in (False, True):
                role = (i + int(early)) % 2
                folder = output / f"round-{i:02d}-{'early' if early else 'original'}"
                future.append(pool.submit(run_receiver, case, launcher, library, folder, cpus[role],
                                          181 + role, early, seconds))
            records.extend(run.result() for run in future)

    groups = {}
    for key, early in (("original", False), ("early", True)):
        rows = [row for row in records if row["early_ust_load"] == early]
        groups[key] = dict(starts=len(rows), ready=sum(row["ready_s"] is not None for row in rows),
                           private_pages=sum(row["private_pages"] == [True, True] for row in rows),
                           clean_exits=sum(row["exit"] == 0 for row in rows),
                           fastdds_shared=sum(row["fastdds_shared"] for row in rows))
    result = dict(scope="ZERO_PHYSICS_FROZEN_RECEIVER_COLD_START",
                  library_sha256=digest(library), original_case_result_sha256=digest(terminal_file),
                  original_command_sha256=digest(commands_file), groups=groups, records=records,
                  native_requests=0, native_integrations=0)
    path = output / "RESULT.json"
    write(path, result)
    print(json.dumps(dict(sha256=digest(path), groups=groups)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("closed_case", "library", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--rounds", default=8, type=int)
    parser.add_argument("--seconds", default=7, type=int)
    test(**vars(parser.parse_args()))
