"""Build a cold candidate for the optional PD callback during warmup."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write


BASELINE_BINARY = "39e5136603f51c092b2f3e595245468dbb0c12dbca79818b0df08da8dd4af7dd"
BASELINE_MAIN = "1db61e5bb6ef838bc08616e7efda2739f80215acd324f17bfd1ff4277a35cfa0"
BASELINE_UTILS = "0d68c82215e8ada9d4c2a56ebd7bb666bd49fafe17750d4d20c50d8abb1b6a2c"
UI_SPEC = "2a9881c6e59e8f058bb2da8889eedc0c8cdd940ff66823cd198ff4846582804d"
UI_RECEIPT = "460214e35fed0c067dc78433b18c0b92866135c2ac4ec0f47148044de8263b9a"
FASTCDR_OLD = Path("/opt/ros/jazzy/lib/libfastcdr.so.2.2.7")
FASTCDR_SITE = Path("/opt/ros/jazzy/lib/libfastcdr.so.2.2.8")
FASTCDR_SITE_SHA = "805bc734c8dacbeb9383bea4539484c2cc89f3f6d33823251d4a8236e3f1f043"
ARCHIVED_TRACES = {
    "night1-decision3-panel-a-r1-pair3-slot-0": {"RESULT.json", "init.trace", "init.trace.summary"},
    "night1-c07-decision-p03-q01": {"RESULT.json", "init.trace", "physics.jsonl"},
}
INCLUDE = '#include "warmup_pd_phase_v64.hpp"\n'
HEADER = r"""#pragma once
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace wm02_warmup_pd_phase_v64 {
inline std::int64_t tick(double seconds) { return std::llround(seconds * 500.0); }

class Phase {
 public:
  explicit Phase(const char* mask) : enabled_(mask != nullptr) {
    if (!enabled_) return;
    std::string_view text(mask);
    if (text.size() != required_.size()) std::abort();
    for (std::size_t i = 0; i < text.size(); ++i) {
      if (text[i] != '0' && text[i] != '1') std::abort();
      required_[i] = text[i] == '1';
    }
  }

  bool allow_callback(std::int64_t t) const {
    return !optional(t) || required_[t / 25];
  }

  bool ready(std::int64_t t) const {
    return !optional(t) || !required_[t / 25] || completed_tick_.load() == t;
  }

  void complete(std::int64_t t) {
    if (optional(t)) completed_tick_.store(t);
  }

 private:
  bool optional(std::int64_t t) const {
    return enabled_ && t >= 0 && t < 2500 && t % 25 == 2;
  }

  const bool enabled_;
  std::array<bool, 100> required_{};
  std::atomic<std::int64_t> completed_tick_{-1};
};

inline Phase& phase() {
  static Phase instance(std::getenv("WM02_WARMUP_PD_PHASE_V64"));
  return instance;
}
}  // namespace wm02_warmup_pd_phase_v64
"""


def replace_once(source, old, new):
    require(source.count(old) == 1, "The exact frozen source seam is required")
    return source.replace(old, new, 1)


def overlay(main, utils):
    check = ("    if (!wm02_warmup_pd_phase_v64::phase().allow_callback("
             "wm02_warmup_pd_phase_v64::tick(this->mj_data->time))) return;\n")
    stamp = ("    wm02_warmup_pd_phase_v64::phase().complete("
             "wm02_warmup_pd_phase_v64::tick(this->mj_data->time));\n")
    controller = "    this->StateController(&this->robot_state, &this->robot_command);\n"
    control = '    wm02_applied_control_clock::clock().control(this->mj_data->time,std::llround(double(this->params.Get<float>("dt"))*1e9));\n'
    body_start = main.index("void RL_Sim::RobotControl()\n{")
    body_end = main.index("void RL_Sim::StartRosInterface()")
    body = replace_once(main[body_start:body_end], controller, check+controller)
    body = replace_once(body, control, control+stamp)
    main_candidate = INCLUDE+main[:body_start]+body+main[body_end:]
    old = "!wm02_applied_control_clock::clock().ready(d->time)"
    new = old+" || !wm02_warmup_pd_phase_v64::phase().ready(wm02_warmup_pd_phase_v64::tick(d->time))"
    require(utils.count(old) == 2, "Both stock physics readiness paths are required")
    utils_candidate = INCLUDE+utils.replace(old,new)
    require(main_candidate[len(INCLUDE):].replace(check,"",1).replace(stamp,"",1) == main
            and utils_candidate[len(INCLUDE):].replace(new,old) == utils,
            "Removing phase hooks must recover the original simulator source exactly")
    return main_candidate, utils_candidate


def check_header(output, fixtures):
    masks = {tuple(x["phase_fixture"]["require_callback"]) for x in fixtures["optional_phase_fixture"]}
    cases = "\n".join('    check("'+"".join(str(int(bit)) for bit in mask)+'");' for mask in sorted(masks))
    source = output / "phase_check.cpp"
    source.write_text(r"""#include <cassert>
#include "warmup_pd_phase_v64.hpp"

using wm02_warmup_pd_phase_v64::Phase;
void check(const char* mask) {
  Phase phase(mask);
  assert(phase.ready(-1) && phase.allow_callback(-1));
  for (int i = 0; i < 100; ++i) {
    const auto start = i * 25;
    const bool expected = mask[i] == '1';
    assert(phase.ready(start) && phase.allow_callback(start + 1));
    assert(phase.allow_callback(start + 2) == expected);
    phase.complete(start + 1);
    assert(phase.ready(start + 2) != expected);
    if (expected) phase.complete(start + 2);
    assert(phase.ready(start + 2) && phase.allow_callback(start + 3));
  }
  assert(phase.ready(2502) && phase.allow_callback(2502));
}
int main() {
    Phase stock(nullptr);
    assert(stock.ready(2) && stock.allow_callback(2));
"""+cases+"\n}\n")
    executable = output / "phase_check"
    subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(executable)], check=True, timeout=30)
    subprocess.run([executable], check=True, timeout=5)
    return len(masks), len(masks) * 100


def exact_command(path, label):
    row, = [r for r in read(path) if r["label"] == label and r["exit_code"] == 0]
    return row["argv"]


def replace_arg(argv, original, replacement):
    require(argv.count(str(original)) == 1, "The original native build argument is required")
    return [str(replacement) if arg == str(original) else arg for arg in argv]


def dynamic_identity(binary):
    text = subprocess.check_output(["readelf", "-d", str(binary)], text=True, timeout=15)
    return sorted(line.split(")", 1)[1].strip() for line in text.splitlines()
                  if any(tag in line for tag in ("(NEEDED)", "(RPATH)", "(RUNPATH)")))


def build(ui, fixture, fixture_sha, output):
    ui_spec = read(pin(ui / "EXECUTION_SPEC.json", UI_SPEC))
    lease = read(ui / "control/LEASE.json")
    require(lease["status"] == "CLOSED" and lease["spec_sha256"] == UI_SPEC and not lease["worker_exit_code"],
            "Exact closed historical build required")
    viewer = ui / "candidate-01"
    pin(viewer / "RESULT.json", UI_RECEIPT)
    baseline = Path(ui_spec["baseline_candidate"])
    pin(viewer / "rl_sim_mujoco", BASELINE_BINARY)
    main = pin(baseline / "rl_sim_mujoco.cpp", BASELINE_MAIN)
    utils = pin(baseline / "mujoco_utils.hpp", BASELINE_UTILS)
    assets, unavailable = {}, {}
    for name, digest in ui_spec["frozen_files"].items():
        path = Path(name)
        archives = ARCHIVED_TRACES.get(path.parent.parent.name, set())
        if not path.exists() and path.parent.name == "native-01" and path.name in archives:
            # These historic logs were inputs to UI validation, not the native compiler.
            unavailable[digest] = path.name
        else:
            assets[name] = pin(path, digest)
    fixtures = read(pin(fixture, fixture_sha))
    require(fixtures["physics_integrations"] == 0 and len(fixtures["optional_phase_fixture"]) == 6
            and all(not row["remaining_epoch_mismatches"] for row in fixtures["optional_phase_fixture"]),
            "A closed phase verification on all six archived comparisons is required")
    m, u = overlay(main.read_text(), utils.read_text())
    output.mkdir(mode=0o700)
    (output / "warmup_pd_phase_v64.hpp").write_text(HEADER)
    (output / main.name).write_text(m)
    (output / utils.name).write_text(u)
    for name in ("rl_sim_mujoco.hpp", "native_home.hpp", "native_startup.hpp", "init_trace.hpp",
                 "control_clock.hpp", "applied_control_clock.hpp", "command_pair.hpp", "state_snapshot.hpp"):
        shutil.copyfile(baseline / name, output / name)
    profiles, windows = check_header(output, fixtures)
    commands = exact_command(baseline / "COMMANDS.private.json", "main_compile")
    commands = replace_arg(commands, main, output / main.name)
    old_object = baseline / "rl_sim_mujoco.cpp.o"
    new_object = output / old_object.name
    commands = replace_arg(commands, old_object, new_object)
    commands.insert(1,"-I"+str(output))
    link = exact_command(viewer / "COMMANDS.private.json", "native_link")
    rebound_fastcdr = not FASTCDR_OLD.exists()
    if rebound_fastcdr:
        rebound = pin(FASTCDR_SITE, FASTCDR_SITE_SHA)
        link = replace_arg(link, FASTCDR_OLD, rebound)
    link = replace_arg(link, old_object, new_object)
    binary = output / "rl_sim_mujoco"
    link = replace_arg(link, viewer / binary.name, binary)
    for label, argv in (("main_compile",commands), ("native_link",link), ("native_ldd",["ldd",str(binary)])):
        process = subprocess.run(argv, cwd=ui_spec["native_build_directory"], text=True, capture_output=True, timeout=180)
        (output / f"{label}.log").write_text(process.stdout+process.stderr)
        require(not process.returncode, f"{label} failed. See its retained output.")
    require("not found" not in (output / "native_ldd.log").read_text(), "Candidate native dependencies unresolved")
    require(dynamic_identity(binary) == dynamic_identity(viewer / binary.name),
            "Candidate ELF adds or changes a dynamic dependency or search path")
    for name, path in assets.items():
        pin(path, ui_spec["frozen_files"][name])
    receipt = dict(status="COMPILED_AND_PHASE_LOGIC_CHECKED", original_binary_sha256=BASELINE_BINARY,
                   new_binary_sha256=sha(binary), fixture_sha256=fixture_sha,
                   archive_profiles=profiles, clock_windows_checked=windows,
                   exact_original_without_phase_hooks=True, native_starts=0, native_integrations=0,
                   same_dynamic_dependency_identity=True,
                   site_build_link_input_sha256=FASTCDR_SITE_SHA if rebound_fastcdr else None,
                   unreadable_historical_ui_evidence=unavailable, native_state_qualified=False)
    write(output / "RESULT.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("ui","fixture","fixture_sha","output"):
        cli.add_argument(name, type=str if name == "fixture_sha" else Path)
    build(**vars(cli.parse_args()))
