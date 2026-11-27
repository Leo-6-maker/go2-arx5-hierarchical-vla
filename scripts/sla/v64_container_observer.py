"""Build and exercise the separately bound container contact observer on Corelab."""

import argparse
import ctypes
from pathlib import Path
import subprocess

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v64_container_preflight import verify_release
from v64_container_terminal import verify_bowl_surface


OBSERVER_RECEIPT = "6733baa0872d3e4076d4dd7c681863480d5058af1d5a3460d3ba7d43455a66f5"
BUILD_SPEC = "c13e240346520496afd3e9bfcb3ae586894ca76dd0efbe5011ad648972a6c848"
RELEASE_PROOF = "3654c07cd2e3663b2e670ce6b5d7c41ee3cbb7d1ca6975084f8dd9419777dc2e"
DECLARATION = ('extern "C" bool WM02ContainerFloorContactV64(const mjModel*, const mjData*, int, '
               'const char*, long, const char*, const char*, long);\n'
               'extern "C" bool WM02ContainerHasForeignContactV64(const mjModel*, const mjData*, '
               'const char*);\n\n')
LOOP = "  bool illegal_contact=false;"
RECONTACT = """  const char* container_case = std::getenv("WM02_V64_CONTAINER_CASE");
  static long last_container_contact = -1;
  if (container_case && WM02ContainerHasForeignContactV64(m,d,container_case)) {
    last_container_contact = count;
  }
  bool illegal_contact=false;"""
ORIGINAL = "    allowed=allowed || task_pair;"
VARIANT = """    const bool released_floor = container_case && WM02ContainerFloorContactV64(
      m,d,i,selected,count,container_case,std::getenv("WM02_V64_CONTAINER_RELEASE"),
      last_container_contact);
    allowed=allowed || task_pair || released_floor;"""


def compile_observer(original, output):
    prior = read(pin(original / "RESULT.private.json", OBSERVER_RECEIPT))
    original_spec = read(pin(original / "EXECUTION_SPEC.json", BUILD_SPEC))
    closed = read(original / "control/LEASE.json")
    require(closed["status"] == "CLOSED" and closed["spec_sha256"] == BUILD_SPEC
            and prior["exit_code"] == 0 and prior["new_integrations"] == 0,
            "The exact closed V6 observer build is required")
    binding = prior["observer_candidate"]
    old_source = pin(original / "observer.cpp", binding["source_sha256"])
    old_binary = pin(original / "observer.so", binding["library_sha256"])
    baseline = old_source.read_text()
    require(baseline.count(ORIGINAL) == baseline.count(LOOP) == baseline.count("namespace {") == 1,
            "The single authorized contact branch must exist")
    variant = (baseline.replace("namespace {", DECLARATION + "namespace {", 1)
               .replace(LOOP, RECONTACT, 1).replace(ORIGINAL, VARIANT, 1))
    recovered = variant.replace(DECLARATION, "", 1).replace(RECONTACT, LOOP, 1).replace(VARIANT, ORIGINAL, 1)
    require(recovered == baseline,
            "The original clock, STOP and safety source must recover byte-for-byte")
    output.mkdir(mode=0o700)
    candidate_source = output / "observer.cpp"
    candidate_source.write_text(variant)
    candidate_binary = output / "observer.so"
    implementation = Path(__file__).with_name("v64_container_contact.cpp")
    config = original_spec["observer_build"]
    for name in ("trace_header", "trace_library"):
        pin(config[name]["path"], config[name]["sha256"])
    args = list(config["observer_argv"])
    source_arg = args.index(config["source"]["path"])
    args[source_arg:source_arg+1] = [str(candidate_source), str(implementation)]
    args[args.index(config["library"]["path"])] = str(candidate_binary)
    with (output / "compile.log").open("x") as log:
        subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=60)
    return old_binary, candidate_binary, implementation


def check_observer(old, new, factory, holds, release_proof, receipt, output):
    import mujoco as mj

    old_lib, new_lib = (ctypes.CDLL(str(path)) for path in (old, new))
    for lib in (old_lib, new_lib):
        lib.mj_version.restype = ctypes.c_int
        require(lib.mj_version() == mj.mj_version(), "Cold observer and actual MuJoCo ABI differ")
        lib.WM02MultitaskTaskPairV6.argtypes = [ctypes.c_char_p] * 3
        lib.WM02MultitaskTaskPairV6.restype = ctypes.c_int
    native = new_lib.WM02ContainerFloorContactV64
    native.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_char_p, ctypes.c_long,
                    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_long]
    native.restype = ctypes.c_bool
    foreign = new_lib.WM02ContainerHasForeignContactV64
    foreign.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
    foreign.restype = ctypes.c_bool
    words = receipt.read_bytes().split()
    require(words[:2] == [b"V64_CONTAINER_RELEASE", b"EF5_Q_CONTAINER_01"] and len(words) == 8,
            "The F1 release must have been written by the actual state replay")
    pause = int(words[2])
    blue_fixture = output / "blue-fixture.release"
    blue_fixture.write_bytes(b" ".join([words[0], b"EF5_Q_CONTAINER_02", *words[2:]]) + b"\n")
    paths = {"EF5_Q_CONTAINER_01": bytes(receipt), "EF5_Q_CONTAINER_02": bytes(blue_fixture)}
    invalid = output / "invalid.release"
    invalid.write_bytes(b" ".join([*words[:-2], str(pause + 1).encode(), words[-1]]) + b"\n")
    proof = read(pin(release_proof, RELEASE_PROOF))
    cases = []

    def check_contact(layout, model, data, target, floor):
        identity = layout["id"]
        role = identity.encode()
        color = layout["target"].encode()
        floor_index, = (i for i, c in enumerate(data.contact) if {c.geom1, c.geom2} == {floor, target})

        def call(tick, path, *, supplied_role=role, supplied_color=color, previous_contact=-1):
            data.time = .002 * tick
            latest_contact = tick if foreign(model._address, data._address, supplied_role) else previous_contact
            return native(model._address, data._address, floor_index, supplied_color, tick, supplied_role,
                          path, latest_contact)

        path = paths[identity]
        require(call(pause + 1, path) and call(pause + 25, path), "The whole next actual interval must be covered")
        rejects = {
            "unconfigured": not call(pause + 1, None, supplied_role=None),
            "missing_release": not call(pause + 1, bytes(output / "not-published.release")),
            "unconsumed_ack": not call(pause + 1, bytes(invalid)),
            "other_condition": not call(pause + 1, paths[next(key for key in paths if key != identity)]),
            "other_color": not call(pause + 1, path, supplied_color=(b"blue" if color == b"red" else b"red")),
            "unread_current_tick": not call(pause, path),
            "expired_after_one_atom": not call(pause + 26, path),
            "recorded_recontact_after_publication": not call(pause + 2, path, previous_contact=pause + 1),
        }
        contact = data.contact[floor_index]
        original_distance = float(contact.dist)
        contact.dist = -.004001
        rejects["over_original_penetration_limit"] = not call(pause + 1, path)
        contact.dist = original_distance
        center = data.geom_xpos[target, 0]
        data.geom_xpos[target, 0] += .050
        rejects["footprint_outside_floor"] = not call(pause + 1, path)
        data.geom_xpos[target, 0] = center
        added = mj.MjContact()
        added.geom1, added.geom2 = target, model.geom("task_left_pad").id
        added.geom[:] = (added.geom1, added.geom2)
        added.dist = -.0001
        require(mj.mj_addContact(model, data, added) == 0, "Cold simultaneous contact fixture could not be added")
        rejects["robot_recontact"] = not call(pause + 1, path)
        cases.append(dict(case=identity, geometry_state_rejections=rejects))
        require(all(rejects.values()), f"An invalid contact state was admitted in {identity}: {rejects}")

    surface = verify_bowl_surface(factory, holds, proof, check_contact=check_contact)
    geoms = (b"floor", b"FL", b"task_table", b"task_left_pad", b"task_right_pad", b"task_cube_geom",
             b"task_blue_cube_geom", b"ef5_static_bowl_base", b"x5_link6")
    for a in geoms:
        for b in geoms:
            for selected in (b"red", b"blue"):
                require(old_lib.WM02MultitaskTaskPairV6(a, b, selected) ==
                        new_lib.WM02MultitaskTaskPairV6(a, b, selected), "The stock pair predicate changed")
    return dict(first_permitted_tick=pause+1, last_permitted_tick=pause+25,
                real_replay_handoff_sha256=sha(receipt), abi_version=mj.mj_version(),
                blue_timing_is_cpu_fixture=True, original_contact_pair_comparisons=2*len(geoms)**2,
                cases=cases, actual_compiled_support_states=surface)


def run(original, case, factory, hold_red, hold_blue, release_proof, output):
    old, candidate, implementation = compile_observer(original, output)
    handoff = output / "TABLETOP_NATIVE_RELEASE.private"
    replay = verify_release(case, handoff)
    verified = check_observer(old, candidate, factory, (hold_red, hold_blue), release_proof, handoff, output)
    result = output / "ACTUAL.json"
    write(result, dict(implementation_sha256=sha(implementation), library_sha256=sha(candidate),
              original_library_sha256=sha(old), release_replay=replay, compiled_guard=verified,
              new_native_requests=0, native_integrations=0, native_dispatch_qualified=False))
    print(sha(result), len(verified["cases"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "case", "factory", "hold_red", "hold_blue", "release_proof", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
