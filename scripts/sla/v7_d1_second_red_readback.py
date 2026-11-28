"""Independently reconcile the second red D1 score, actual command continuity and closed owners."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys
from tempfile import mkdtemp

from m38_gpu_preflight_v1 import pin, read, sha
from v64_adapt_pregrasp_attribution import journal


def run(work):
    live = work/"V7-D1-EARLY-NATIVE-03"
    case = live/"V7_D1_EARLY_TRAIN_19"
    closed = read(pin(work/"V7-D1-EARLY-SEALED-04/RESULT.json","8f4e1cbf851949b3e370e5d42472257e169240d8cfcceb9f0c3652bdd26a3624"))
    result = read(pin(case/"RESULT.private.json","f7884d82843070ef80204e2466de595a72932ab6f652e1935963f54015f5c435"))
    assert closed["complete_reconciliation"] and closed["train_recovery_count"] == closed["terminal_cases"] == 1
    assert closed["actual_forwards"] == closed["worker_actual_forwards"] == result["model_generations"] == 48
    assert result["completed"] and result["shutdown_clean"] and not result["unsafe"] and not result["technical_failure"]
    assert not result["physics_stops"] and not result["detail"]["task_success"] and result["detail"]["recovery_task_success"]
    spec = read(pin(case/"EXECUTION_SPEC.json",read(case/"control/LEASE.json")["spec_sha256"]))
    intent = read(pin(case/"native-01/RECOVERY-HANDOFF-INTENT.private.json","865a1e5070537a25eecd99f6279252f1a756bc7a134ddc7724a137260dd92532"))
    assert intent["native_step"] == 17703 and intent["first_teacher_gripper_sequence"] == 608
    assert spec["mobile_condition"]["reset"] == 932461869 and spec["comparison_policy_seed"] == 1934487174
    assert all(read(root/"control/LEASE.json")["status"] == "CLOSED" for root in (live,live/"worker",case))
    assert not Path("/proc/2553450").exists() and not (live/"V7_D1_EARLY_TRAIN_20/EXECUTION_SPEC.json").exists()
    samples = list(journal(case/"native-01/MOBILE_SAMPLES.private.jsonl"))
    before, = [r for r in samples if r["native_end_step"] == 17703]
    teacher = [r for r in samples if r["native_start_step"] >= 17703]
    assert before["gripper_command"]["sequence"] == 607 and len(teacher) == 1140
    assert [r["native_start_step"] for r in teacher] == list(range(17703,46203,25))
    assert [r["gripper_command"]["sequence"] for r in teacher] == list(range(608,1748))
    assert all(r["native_end_step"] == r["native_start_step"]+25 and r["ticks"] == 25 and r["safety_valid"] for r in teacher)
    cap = read(pin(result["detail"]["capture"]["path"],result["detail"]["capture"]["sha256"]))
    queries = read(pin(cap["labels"]["path"],cap["labels"]["sha256"]))
    assert len(queries) == result["detail"]["completed_teacher_P10"] == 114
    assert all(q["supervision_role"] == "RECOVERY_TEACHER" and q["partial_interval_ticks"] == 0
               and len(q["intervals"]) == 10 for q in queries)
    score = result["detail"]["score"]
    assert score["task_success"] and score["ever_target_grasp"] and score["ever_target_lift20mm"]
    assert score["released"] and score["inside_selected_region"] and score["stable_release_condition"]
    assert score["maximum_lift_m"] >= .020 and score["longest_grasp_lift_hold_s"] >= 1.
    assert score["TCP_retreat_m"] >= .050 and score["longest_stable_released_retreat_s"] >= 1.
    worker = read(work/"ADAPT0250-NATIVE-PLACE-01/worker/EXECUTION_SPEC.json")
    with Path(worker["neural_owner_lock"]).open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(f,fcntl.LOCK_UN)
    assert subprocess.run(["pgrep","-x","go2_x5_ocs2_node"],capture_output=True).returncode != 0
    value = dict(status="SECOND_RED_D1_NATIVE_SCORE_AND_CONSUMPTION_INDEPENDENTLY_VERIFIED",
        native_result_sha256=sha(case/"RESULT.private.json"),reconciliation_sha256=sha(work/"V7-D1-EARLY-SEALED-04/RESULT.json"),
        actual_ticks=46203,actual_frozen_forwards=48,teacher_queries=114,actual_teacher_atoms=1140,
        handoff_tick=17703,handoff_wrist_gap_mm=intent["wrist_target_mm"],previous_student_sequence=607,
        first_teacher_sequence=608,last_teacher_sequence=1747,sequence_and_tick_continuity=True,
        score=score,student_autonomous_success=False,teacher_assisted_success=True,
        recovery_attempts_total=17,recovery_admissions_total=6,recovery_windows_total=369,
        admitted_early_takeover_starts=2,new_independent_layouts=0,blue_generalization_established=False,
        unsafe=False,technical_failure=False,all_leases_closed=True,process_gone=True,neural_owner_lock_released=True,
        trace_audit_status=result["trace_audit"]["status"],
        source_spec_sha256=sha(case/"EXECUTION_SPEC.json"),capture_sha256=sha(result["detail"]["capture"]["path"]),
        actual_samples_sha256=sha(case/"native-01/MOBILE_SAMPLES.private.jsonl"),
        query_labels_sha256=sha(cap["labels"]["path"]),new_native_integrations=0,new_model_forwards=0,new_optimizer_updates=0)
    output = Path(mkdtemp(prefix="v7-d1-second-red-site-readback.",dir=work))/"ACTUAL.json"
    output.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    print(output,sha(output),flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
