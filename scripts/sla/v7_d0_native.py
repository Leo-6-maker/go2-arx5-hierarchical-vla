"""Connect the two V7 early Recovery cards to the exact prior native runner."""

import argparse
import shutil
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write


PLAN = "67d15fb8fad688a2bbd82e225111bb49ec34559b"
RECOVERY_PLAN = "531f826b36c1195db8c253ca71497f1aeb569c80"
PROSPECTIVE = "4e10cfdd865ade7fe6e360664d2601530ba5737e9160e14d1c9abdde4edfe761"
NOISE = "1073abde8dae989e6534fe298b30063a6c770767808add685ce565ea4e7f48e6"
FIFTH = "a140b5a5577807582d2009ce24999821ac9397acf189bc6180a3d81f495834cf"
REPLAY = "e524b5d6d8e239e019dfe575ea848e8ceb88fbc56abce481be7bd4d7dbda81de"
PARENT = "1e19b5cdce0d957d9da4b976d2f3b80cca0b3c972c372b683f665465d6687050"
ENGINE = {
    "v64_m38_recovery_wave5.py": "ab1447fd333e9a90799429264aaeb654f5262db6e2234722f703b81480ff4279",
    "v64_recovery_launcher.py": "bf0f724e6436e4909a83213ec652be70203313dd57d29a89754decb45f3b9cad",
    "v64_recovery_cohorts.py": "e3e14c36546bbfed55d2aedbba088a60bac54da963b6ae0cfc28686bc48a0a19",
    "v64_recovery_terminal.py": "b7a332aae00a99691be859429d6a8d0040dbaab5beb5c5ff34829727aa549eb0",
}
EARLY = {
    "v64_recovery_native.py": "026e8dd1369b0085a674d94e457c321b332fbf99361e2c496303fe28945119c9",
    "v64_recovery_carry.py": "81c9fc9ab586d6e5d2f2e494b3c0cc0818bab41387d15777097af424abba3732",
    "v64_recovery_early_teacher.py": "6396fad5ca8732f101a464a1c34c51bc4f7cf24317b0a11e41a3b51294365695",
    "v64_recovery_release_prefix.py": "5a6534dae66e5ebaf0fb0c4e6013fd85b951c59e0b83855ce7528eb9486c9373",
}
IDS = ("V7_D0_EARLY_TRAIN_11", "V7_D0_EARLY_TRAIN_12")
STAGES = ("LIFT", "PREGRASP")
ROSTER_NAME = "V7-D0-NATIVE-ROSTER.private.json"


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def bound_roster():
    return Path(__file__).resolve().parents[2] / ROSTER_NAME


def read_runtime(path, expected=None):
    if Path(path).resolve() != bound_roster():
        return None
    roster = read(path)
    source = roster["prospective"]
    evidence = read(pin(source["path"],PROSPECTIVE))
    require(source["sha256"] == PROSPECTIVE and roster["plan"] == RECOVERY_PLAN and roster["science_plan"] == PLAN
            and roster["split"] == "TRAIN" and roster["cases"] == evidence["cases"]
            and roster["student_provider"] == evidence["student_provider"]
            and roster["checkpoint_sha256"] == evidence["checkpoint_sha256"]
            and roster["protected_scene_sha256"] == evidence["protected_scene_sha256"]
            and roster["max_native_requests"] == 2 and roster["max_native_ticks"] == evidence["max_ticks"]
            and roster["handoff_protocol"] == "EARLY_UNINSTALLED_G7_ONLY"
            and roster["early_handoff_replay"]["sha256"] == REPLAY
            and tuple(row["id"] for row in roster["cases"]) == IDS
            and tuple(row["required_stage"] for row in roster["cases"]) == STAGES,
            "Early Recovery binding differs from V7 scientific allocation")
    checksum = sha(path)
    require(expected is None or checksum == expected, "Bound Recovery runtime roster changed")
    return roster, checksum


def register(roster_path, teacher_root, parent_path, noise_path):
    selected = read_runtime(roster_path)
    if selected is None:
        return None
    roster, _ = selected
    parent = read(pin(parent_path,PARENT))
    noise = read(pin(noise_path,NOISE))
    require(parent["comparison_model"] == roster["student_provider"] and parent["v4_model_role"] == "R4_DEV_0200"
            and parent["gpu_noise_sha256"] == noise["prior_noise_manifest_sha256"]
            and noise["roster_sha256"] == PROSPECTIVE and noise["plan"] == PLAN and noise["split"] == "TRAIN"
            and noise["matched_archived_cuda_sentinels"] == 3 and not noise["model_forwards"]
            and not noise["native_integrations"] and not noise["optimizer_updates"],
            "Expected the exact frozen R4 and realized CUDA noise")
    expected = [(card["id"],q,(card["policy_seed"]+q) % 2**31)
                for card in roster["cases"] for q in range(card["maximum_queries"])]
    observed = [(row["case"],row["query"],row["noise_id"]) for row in noise["tensors"]]
    require(observed == expected and len({row["gpu_sha256"] for row in noise["tensors"]}) == len(expected),
            "CUDA noise differs from the approved early roster")
    teachers = {}
    for card in roster["cases"]:
        wave = Path(roster["teacher_sources"][card["id"]]["teacher_wave"]).resolve()
        require(wave.parent == teacher_root.resolve(), "The teacher wave is outside the admitted project")
        case = wave / "cases" / card["original_teacher"]
        proof = card["teacher_provenance"]
        result = read(pin(case/"RESULT.private.json",proof["result_sha256"]))
        spec = pin(case/"EXECUTION_SPEC.json",proof["spec_sha256"])
        lease = read(case/"control/LEASE.json")
        require(result["shutdown_clean"] and result["detail"]["score"]["stable_release_condition"]
                and lease["status"] == "CLOSED" and lease["spec_sha256"] == proof["spec_sha256"],
                "Teacher source must be the closed, complete TRAIN success")
        teachers[card["id"]] = ref(spec)
    return roster, parent, teachers


def rewrite(path, edits):
    text = path.read_text()
    for before, after in edits:
        require(text.count(before) == 1, f"Unrecognized original implementation in {path.name}")
        text = text.replace(before,after)
    compile(text,str(path),"exec")
    require(not path.is_symlink(),"Only a private overlay file may be updated")
    path.write_text(text)


def prepare(site, prospective_path, output):
    prospective = read(pin(prospective_path,PROSPECTIVE))
    older = read(pin(site/"RECOVERY-TRAIN-ROSTER-05.json",FIFTH))
    replay = pin(site/"v64-recovery-early-teacher-code-01/ACTUAL-EARLY-TEACHER-GUARDED-CLOCK-01.json",REPLAY)
    source = site/"V64-RECOVERY-OVERLAY-06"
    early = site/"v64-early-recovery-entry-code-02"
    for name, checksum in ENGINE.items():
        pin(source/"scripts/sla"/name,checksum)
    for name, checksum in EARLY.items():
        pin(early/name,checksum)
    shutil.copytree(source,output,symlinks=True)
    scripts = output/"scripts/sla"
    for name in EARLY:
        require(not (scripts/name).is_symlink(),"Require a new writable source overlay")
        shutil.copyfile(early/name,scripts/name)
    shutil.copyfile(Path(__file__),scripts/Path(__file__).name)
    manifest = output/ROSTER_NAME
    write(manifest,dict(plan=RECOVERY_PLAN,science_plan=PLAN,split="TRAIN",cases=prospective["cases"],
          prospective=ref(prospective_path),checkpoint_sha256=prospective["checkpoint_sha256"],
          student_provider=prospective["student_provider"],protected_scene_sha256=prospective["protected_scene_sha256"],
          teacher_sources={r["id"]:older["teacher_sources"][r["historical_failure"]] for r in prospective["cases"]},
          max_native_requests=2,max_native_ticks=prospective["max_ticks"],
          handoff_protocol="EARLY_UNINSTALLED_G7_ONLY",early_handoff_replay=ref(replay)))
    rewrite(scripts/"v64_recovery_launcher.py", (
        ("    roster, cohort = read_roster(roster_path)\n",
         "    from v7_d0_native import register\n"
         "    if (early := register(roster_path, teacher_wave, parent_path, noise_path)) is not None:\n"
         "        return early\n"
         "    roster, cohort = read_roster(roster_path)\n"),
    ))
    rewrite(scripts/"v64_recovery_cohorts.py", (
        ("def read_roster(path, expected=None):\n",
         "from v7_d0_native import bound_roster, read_runtime, NOISE as D0_NOISE\n"
         "NOISE[sha(bound_roster())] = D0_NOISE\n\n"
         "def read_roster(path, expected=None):\n"
         "    if (early := read_runtime(path, expected)) is not None:\n"
         "        return early\n"),
    ))
    rewrite(scripts/"v64_recovery_native.py", (
        ('        self.early_enabled = early_replay is not None\n',
         '        self.early_enabled = early_replay is not None\n'
         '        self.early_only = manifest.get("handoff_protocol") == "EARLY_UNINSTALLED_G7_ONLY"\n'
         '        self.required_early_stage = card["required_stage"] if self.early_only else None\n'
         '        if self.early_only and (not self.early_enabled or self.required_early_stage not in ("LIFT", "PREGRASP")):\n'
         '            raise PermissionError("Require the registered early-only stage and replay")\n'),
        ('            self.last_consumed_query = path\n        sample = adapter.measured_task_state()\n',
         '            self.last_consumed_query = path\n'
         '        if self.early_only:\n'
         '            return None\n'
         '        sample = adapter.measured_task_state()\n'),
        ('        consumed = read(self.last_consumed_query)\n',
         '        stage = "LIFT" if sample["native_grasp"] else "PREGRASP"\n'
         '        if self.early_only and stage != self.required_early_stage:\n'
         '            return None\n'
         '        consumed = read(self.last_consumed_query)\n'),
    ))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for f in ("site","prospective_path","output"):
        parser.add_argument(f,type=Path)
    print(prepare(**vars(parser.parse_args())))
