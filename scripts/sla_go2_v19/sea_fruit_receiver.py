"""Build and cold-replay the stationary fruit gripper and STOP interposer."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from sea_fruit_h0_cold import BASE, pin, read, require, sha
from build_student_receiver import replace_once
from receiver_fault_replay import replay_treatments

CAP = 50_000
PARENT_SPEC = "Z1_REACH_483/G00-A"
PARENT_HASHES = {
    "task_actuation_r1.cpp": "ebf690005a8338c8da14c30f20e1e521de85f8bf92348f4f3f8f5e0c1c79cb96",
    "init_trace.hpp": "0c196146a73052787fe2f10586eea48e004b8201e0198a48cd6b967c0c349ef0",
    "student_receiver.hpp": "0561759815c7e9d9126a8fb9872624c0fe076cc7b01507786c4ef40df513220f",
    "student_grip.so": "927f2179793916dff30590a82b0e5ce3d92811e21446459416613da59e9e3878",
}

CONTACT = """   if (!support(a,b) && !support(b,a)) {
     auto is_pad=[](const char* name) {
       return name && (!std::strcmp(name,"task_left_pad") || !std::strcmp(name,"task_right_pad"));
     };
     // Use the role in the frozen observer for the eligible object.
     if (!has_command || opening >= .042/.044 || !(is_pad(a) || is_pad(b))
         || task_role(a,b,selected)!=1 || !(d->contact[i].dist>=-.004)) contact_ok=false;
   }"""
CONTACT_STOP = """ if(!contact_ok){
   student.stop("PROHIBITED_CONTACT_STOP",steps,target,measured_after);
   const float source[] = {2.0f, static_cast<float>(student.last.sequence)};
   WM02InitEmit(wm02_init::STOP, d->time, steps, source, 2);
 }"""


def candidate_source(original):
    source = original
    for token in ('std::strcmp(cap,"20000")', 'std::getenv("V16_GRIP_EPOCH"),20000,target'):
        source = replace_once(source, token, token.replace("20000", str(CAP)))
    source = replace_once(source, "student_grip::Session student;", """student_grip::Session student;
using TaskRole = int(*)(const char*, const char*, const char*);
TaskRole task_role=nullptr;
const char* selected=nullptr;""")
    source = replace_once(source, " initialized=true;", """ selected=std::getenv("MFREE_V6_SELECTED_COLOR");
 task_role=reinterpret_cast<TaskRole>(dlsym(RTLD_DEFAULT,"WM02MultitaskTaskPairV6"));
 if(!task_role || !selected || std::strcmp(selected,"red"))std::abort();
 initialized=true;""")
    source = replace_once(source, "   if(!support(a,b) && !support(b,a))contact_ok=false;", CONTACT)
    source = replace_once(source, ' if(!contact_ok)student.stop("PROHIBITED_CONTACT_STOP",steps,target,measured_after);',
                          CONTACT_STOP)
    return replace_once(source, "V16_NO_CONTACT_VIOLATION", "SEA_H0_PROHIBITED_CONTACT")


def build(site, out, spec):
    native = read(site / "feedback/sla-feedback-f2-a1-20261123-01/F2-01/EXECUTION_SPEC.json")
    old = Path(native["outer_library"])
    argv = next(v["argv"] for v in read(old.parent / "COMMANDS.private.json")
                if v["label"] == "task_actuation_r1_compile")
    root = site / "v18/receiver-preview02"
    for name, digest in PARENT_HASHES.items():
        pin(root / name, digest)
    require(spec["frozen_files"][spec["outer_library"]] == PARENT_HASHES["student_grip.so"],
            "different loaded no-contact receiver predecessor")
    for name in ("task_actuation_r1.cpp", "init_trace.hpp", "student_receiver.hpp"):
        shutil.copyfile(root / name, out / name)
    source = out / "task_actuation_r1.cpp"
    receiver = out / "fruit_grip.so"
    command = [str(source) if x == str(old.with_suffix(".cpp")) else
               str(receiver) if x == str(old) else x for x in argv]
    require(str(source) in command and command[-2:] == ["-o", str(receiver)],
            "unrecognized receiver compile command")
    subprocess.run(command, check=True, timeout=120)
    pin(receiver, PARENT_HASHES["student_grip.so"])
    source.write_text(candidate_source(source.read_text()))
    subprocess.run(command, check=True, timeout=120)
    observer = pin(Path(spec["observer_library"]), spec["frozen_files"][spec["observer_library"]])
    writer = pin(Path(spec["trace_library"]), spec["frozen_files"][spec["trace_library"]])
    engine = Path(next(x for x in argv if x.endswith("/libmujoco.so")))
    inc = next(x for x in argv if x.startswith("-I") and "/library/mujoco/include" in x)
    return receiver, observer, writer, engine, ["g++", "-std=c++17", "-O2", "-DV18_TWO_OBJECT_ARCHIVE", inc]


def build_probe(out, common, receiver, observer, writer, engine, probe_source, step_source):
    out.mkdir(mode=0o700)
    recorded = out / "librecorded_step.so"
    probe = out / "fault_probe"
    rpath = "-Wl,-rpath," + str(engine.parent)
    subprocess.run([*common, "-fPIC", "-shared", str(step_source), str(engine), rpath, "-o", str(recorded)],
                   check=True, timeout=120)
    subprocess.run([*common, "-rdynamic", str(probe_source), "-Wl,--no-as-needed",
                    str(receiver), str(recorded), str(engine), str(writer), str(observer),
                    "-ldl", rpath, "-o", str(probe)], check=True, timeout=120)
    return probe, recorded


def verify_contact_traces(out, probe, receiver, recorded, writer, scene, reset, journal):
    from go2arx5_vla.world_model.init_trace_reader import read_trace

    cases = (
        ("SELECTED_WHILE_CLOSING", "task_left_pad", "task_cube_geom", "-.004", True, False),
        ("SELECTED_WHILE_OPEN", "task_left_pad", "task_cube_geom", "-.001", False, True),
        ("EXCESS_PENETRATION", "task_right_pad", "task_cube_geom", "-.004001", True, True),
        ("WRONG_OBJECT", "task_right_pad", "task_blue_cube_geom", "-.001", True, True),
        ("PAD_TABLE", "task_left_pad", "task_table", "-.001", True, True),
    )
    results = {}
    for name, a, b, penetration, closing, rejects in cases:
        folder = out / name
        folder.mkdir(mode=0o700)
        epoch = "896cd25f553410635bbc6398bcba2c580d1001107ebbde99b699109319edf459"
        (folder / "binding").write_text(f"GRTQUOTA2\n{name}\n{epoch}\n{name}\n6503\n5\n128\n")
        env = dict(os.environ, GR00T_TASK_GRIPPER_COMMAND=str(folder / "unused.bin"),
                   GR00T_TASK_STATE=str(folder / "record.bin"), GR00T_TASK_STEP_QUOTA=str(folder / "quota"),
                   GR00T_TASK_QUOTA_BINDING=str(folder / "binding"), GR00T_TASK_RUN_UUID=name,
                   GR00T_TASK_CARD_SHA256=epoch, GR00T_TASK_RESERVATION_ID=name,
                   GR00T_TASK_ERROR=str(folder / "reject.txt"), V16_GRIP_DIR=str(folder),
                   V16_GRIP_EPOCH=epoch,
                   V16_GRIP_STOP_TICK=str(CAP), V3_REPLAY_JOURNAL=str(journal),
                   V3_FROZEN_TRACE_WRITER=str(writer), LD_PRELOAD=str(writer),
                   WM02_INIT_TRACE=str(folder / "native.trace"), MFREE_V6_SELECTED_COLOR="red",
                   SEA_FIXTURE_A=a, SEA_FIXTURE_B=b, SEA_FIXTURE_DIST=penetration,
                   SEA_FIXTURE_CLOSE="1" if closing else "0")
        env.pop("WM02_INIT_SEAL_REQUEST", None)
        subprocess.run([str(probe), str(receiver), str(recorded), str(scene), str(reset),
                        "VALID", "RETURNING"], cwd=folder, env=env, check=True, timeout=120)
        routes = (folder / "route.txt").read_text().splitlines()
        require([Path(v).resolve() for v in routes] == [receiver, recorded, writer],
                "nonintegrating step routing not verified")
        delivered = np.fromfile(folder / "delivered_ticks.bin", dtype="<u8")
        terminal = 6439 if rejects else 6453
        require(np.array_equal(delivered, np.arange(1, terminal + 1)),
                "wrong post-contact replay horizon")
        with (folder / "receiver.csv").open() as stream:
            events = [row for row in csv.reader(stream) if "STOP" in row[0]]
        trace = read_trace(folder / "native.trace")
        require(len(events) == len(trace) == int(rejects), "unexpected contact STOP count")
        interruptions = int((folder / "signals.txt").read_text())
        if rejects:
            require(events[0][0] == "PROHIBITED_CONTACT_STOP" and int(events[0][2]) == terminal
                    and trace[0]["kind"] == 13 and trace[0]["token"] == terminal
                    and list(trace[0]["values"]) == [2., 157.]
                    and interruptions == 15, "contact STOP trace or latch failed")
        else:
            require(interruptions == 0, "valid selected-object contact stopped execution")
        results[name] = dict(prohibited=rejects, contact_tick=6439, last_delivered_tick=terminal,
                             post_stop_deliveries=0 if rejects else None,
                             returning_interrupts=interruptions, receiver_log_sha256=sha(folder / "receiver.csv"))
    return results


def contact_probe_sources(out, probe, step):
    """Inject one synthetic contact in an otherwise unchanged recorded trajectory."""
    source = replace_once(step.read_text(), "  mj_forward(model,data);", """  mj_forward(model,data);
  if (tick == 6439) {
    const int a=mj_name2id(model,mjOBJ_GEOM,std::getenv("SEA_FIXTURE_A"));
    const int b=mj_name2id(model,mjOBJ_GEOM,std::getenv("SEA_FIXTURE_B"));
    if(a<0 || b<0)std::abort();
    data->ncon=1;
    auto& c=data->contact[0];
    c=mjContact{};
    c.geom1=c.geom[0]=a;
    c.geom2=c.geom[1]=b;
    c.dist=std::atof(std::getenv("SEA_FIXTURE_DIST"));
    c.efc_address=-1;
  }""")
    target = out / "contact_recorded_step.cc"
    target.write_text(source)
    source = replace_once(probe.read_text(), "        write_command((tick-2503)/25,data->time,.042,expiry);",
                         """        const double opening=(tick>=6428 && !std::strcmp(std::getenv("SEA_FIXTURE_CLOSE"),"1")) ? .041 : .042;
        write_command((tick-2503)/25,data->time,opening,expiry);""")
    source = replace_once(source, 'if (returning && tick == 6439 && method != "VALID") {',
                          'if (returning && tick == 6438 && received_interrupts) {')
    harness = out / "contact_fault_probe.cc"
    harness.write_text(source)
    return harness, target


def run(site, out):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU build and replay only")
    sealed = read(pin(site / "v18/sealed-Z1/members.json",
                      read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")["member_manifest_sha256"]))
    key = f"{PARENT_SPEC}/EXECUTION_SPEC.json"
    spec = read(pin(site / "v18" / key, sealed[key]))
    inputs = {}
    for suffix in ("initialization.jsonl", "state.bin.journal"):
        key, = [k for k in sealed if k.startswith(PARENT_SPEC + "/") and k.endswith("/" + suffix)]
        inputs[suffix] = pin(site / "v18" / key, sealed[key])
    scene = pin(Path(spec["model_path"]), spec["frozen_files"][spec["model_path"]])
    out.mkdir(mode=0o700)
    receiver, observer, writer, engine, common = build(site, out, spec)
    with inputs["initialization.jsonl"].open() as stream:
        init = json.loads(next(stream))
    require(init["time"] == 0 and len(init["qpos"]) == 41 and len(init["qvel"]) == 38, "archived reset ABI")
    reset = out / "reset.bin"
    np.r_[init["qpos"], init["qvel"]].astype("<f8").tofile(reset)
    journal = inputs["state.bin.journal"]
    code = BASE / "scripts/sla_hierarchical_v16"
    probe, step = code / "receiver_fault_probe.cc", code / "recorded_step.cc"
    os.environ["MFREE_V6_SELECTED_COLOR"] = "red"
    domains = {}
    for name, binary, cap in (("NO_CONTACT", site / "v18/receiver-preview02/student_grip.so", 20_000),
                              ("FRUIT", receiver, CAP)):
        dest = out / name
        exe, lib = build_probe(dest, common, binary, observer, writer, engine, probe, step)
        domains[name] = replay_treatments(dest, exe, binary, lib, writer, scene, reset, journal,
                                         cap, probe, "BRIDGE")
    require(domains["NO_CONTACT"] == domains["FRUIT"], "legacy clock or STOP behavior changed")
    fixture_probe, fixture_step = contact_probe_sources(out, probe, step)
    contacts = out / "CONTACT"
    exe, lib = build_probe(contacts, common, receiver, observer, writer, engine, fixture_probe, fixture_step)
    layout = read(BASE / "reports/SLA_GO2_V193/GRIP_H0/FRUIT_SCENES_CPU.json")["layouts"][0]
    fruit_scene = pin(site / "v19/SEA193_FRUIT_H0_COLD_01" / layout["layout"] / "SCENE.private.mjb",
                      layout["scene_sha256"])
    contact_results = verify_contact_traces(contacts, exe, receiver, lib, writer, fruit_scene, reset, journal)
    return dict(status="CPU_RECEIVER_REPLAY_VERIFIED", receiver_sha256=sha(receiver),
                receiver_source_sha256=sha(out / "task_actuation_r1.cpp"), card_tick_cap=CAP,
                parent_receiver_rebuilt_byte_identical=True, frozen_observer_sha256=sha(observer),
                frozen_trace_sha256=sha(writer), unchanged_treatments=domains["FRUIT"],
                synthetic_contact_treatments=contact_results, contact_fixture_scene_sha256=sha(fruit_scene),
                new_native_starts=0, new_physics_ticks=0, new_model_forwards=0, new_training_updates=0,
                whole_system_qualified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    result = run(args.site.resolve(), args.out.resolve())
    (args.out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
