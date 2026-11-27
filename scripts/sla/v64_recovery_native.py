"""Run a registered TRAIN-only Recovery suffix on the original native owner."""

from pathlib import Path
import argparse

from scripts.pi0292.durable import write
from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
from go2arx5_vla.world_model.mfree_capture_v1 import LiveWriter
from go2arx5_vla.world_model.mobile_boundary_clock_v1 import boundary
from go2arx5_vla.world_model.multitask_camera_v6 import TwoObjectCameraSession
from go2arx5_vla.world_model.multitask_state_v6 import score_through
from go2arx5_vla.world_model.multitask_teacher_episode_v6 import confirmed_context
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest

from .v64_recovery_carry import ATOM_TICKS, PREFIX_TICKS, CarryStall, prepare, prepare_supported_release


PLAN = "531f826b36c1195db8c253ca71497f1aeb569c80"
OPEN_REPLAY = "bfda1426f0e91c6cbc8efe81df2ab3c590973152c034582d404a11dff805de49"
EARLY_REPLAY = "e524b5d6d8e239e019dfe575ea848e8ceb88fbc56abce481be7bd4d7dbda81de"


class Pilot:
    def __init__(self, adapter):
        spec = adapter.spec
        manifest = read_pin(spec["v64_recovery_train"])
        scenes = {row["id"]: row for row in manifest["cases"]}
        card = scenes[spec["panel_entry_id"]]
        bound_scene = spec["frozen_files"][spec["model_path"]]
        if (manifest["plan"] != PLAN or manifest["split"] != "TRAIN"
                or not len(scenes) == len(manifest["cases"]) == 2
                or len({row["scene_sha256"] for row in scenes.values()}) != 2
                or bound_scene != card["scene_sha256"]
                or bound_scene in manifest["protected_scene_sha256"]
                or spec["collection_split"] != "TRAIN" or spec["sim_only_role"] != "STUDENT"
                or spec["native_backend"] != "MUJOCO_ONLY" or spec["hardware_connections_allowed"]
                or spec["physics_dt_s"] != .002 or spec["WBC_changes"] or spec["safety_changes"]
                or spec["mobile_condition"]["mode"] != "M_FREE"
                or spec["new_physical_requests_budget"] != 1 or spec["max_ticks"] != 50703
                or spec["policy_queries"] != 180 or spec["new_optimizer_updates_budget"]
                or spec["comparison_model"] != manifest["student_provider"]
                or card["policy_seed"] != spec["comparison_policy_seed"]
                or card["reset"] != spec["mobile_condition"]["reset"]
                or card["task"] != adapter.task.instruction or digest(Path(spec["model_path"])) != bound_scene):
            raise PermissionError("Require the independent two-card Recovery TRAIN allocation")
        self.stall = CarryStall(adapter.task, spec["multitask_geometry"])
        self.open_enabled = manifest.get("handoff_protocol") == "CARRY_OR_SUPPORTED_OPEN"
        if self.open_enabled:
            replay = read_pin(manifest["supported_open_replay"])
            if (manifest["supported_open_replay"]["sha256"] != OPEN_REPLAY
                    or replay["new_native_integrations"] or replay["new_model_forwards"]
                    or len(replay["cases"]) != 4 or not any(row["reached"] for row in replay["cases"])):
                raise PermissionError("The supported OPEN route requires the sealed four-case boundary replay")
        early_replay = manifest.get("early_handoff_replay")
        self.early_enabled = early_replay is not None
        self.last_consumed_query = None
        if self.early_enabled:
            replay = read_pin(early_replay)
            if (early_replay["sha256"] != EARLY_REPLAY or len(replay["cases"]) != 2
                    or any(replay[k] for k in ("native_integrations", "model_forwards", "optimizer_updates",
                                               "new_training_labels"))):
                raise PermissionError("The early route requires the sealed two-case guard and clock replay")
        self.rejected = False

    def maybe_prepare(self, adapter, score, initial, query, folder):
        from scripts.pi0292.durable import read

        if self.rejected:
            return None
        adapter.guard()
        path = Path(folder) / "QUERY_ACTUAL.private.json"
        row = read(path)
        if query != row["query"] or not row["full_P10"]:
            raise ValueError("The recovery check must follow the current genuine P10 return")
        if self.early_enabled:
            self.last_consumed_query = path
        sample = adapter.measured_task_state()
        step = sample["native_step"]
        result = self.stall.observe(score, sample, row)
        if step + PREFIX_TICKS > adapter.spec["max_ticks"]:
            return None
        if result is None:
            if not self.open_enabled:
                return None
            return self.maybe_open(adapter, score, initial, row, sample, query)
        previous = confirmed_context(adapter, step, sample["time_s"])
        try:
            teacher, program, evidence, margin = prepare(initial, sample, adapter.spec["mobile_condition"],
                adapter.spec["multitask_geometry"], adapter.task.instruction, row, previous,
                adapter.base_pose_hold, adapter.actual_state.model)
        except ValueError as error:
            self.rejected = True
            write(adapter.output / "RECOVERY-HANDOFF-REJECTED.private.json",
                  dict(result, reason=str(error), first_install_sent=False, qualification_stage="FIRST_CARRY_H40"))
            return None
        return dict(event=result, teacher=teacher, program=program, evidence=evidence, margin=margin,
                    witness=row["confirmed_intervals"][-1]["accepted_boundary"], query=query)

    def maybe_open(self, adapter, score, initial, consumed, sample, query):
        scored = score.result()
        target = sample["objects"][adapter.task.color]
        if (not scored["inside_selected_region"] or not scored["ever_target_lift20mm"]
                or not target["table_contact"] or not target["native_grasp"]):
            return None
        step = sample["native_step"]
        previous = confirmed_context(adapter, step, sample["time_s"])
        try:
            teacher, program, evidence, margin = prepare_supported_release(
                initial, sample, scored, adapter.spec["mobile_condition"], adapter.spec["multitask_geometry"],
                adapter.task.instruction, consumed, previous, adapter.base_pose_hold, adapter.actual_state.model)
        except ValueError:
            return None
        event = dict(trigger="SUPPORTED_OPEN", native_step=step, native_time_s=sample["time_s"],
                     actual_previous_width_m=previous["width_m"], physically_supported=True)
        return dict(event=event, teacher=teacher, program=program, evidence=evidence, margin=margin,
                    witness=consumed["confirmed_intervals"][-1]["accepted_boundary"], query=query)

    def maybe_early(self, adapter, score, initial, query, path):
        """Consider only a pre-install G7 refusal at the still-paused Student boundary."""
        from scripts.pi0292.durable import read

        if not self.early_enabled or self.last_consumed_query is None:
            return None
        from .v64_recovery_early_teacher import G7_PREFLIGHT_REFUSAL, prepare_early

        rejected = read(path)
        if rejected["query"] != query or rejected["rejection"]["reason"] != G7_PREFLIGHT_REFUSAL:
            return None
        adapter.guard()
        sample = adapter.measured_task_state()
        step = sample["native_step"]
        if step + PREFIX_TICKS > adapter.spec["max_ticks"] or query + 1 >= adapter.spec["policy_queries"]:
            return None
        consumed = read(self.last_consumed_query)
        previous = confirmed_context(adapter, step, sample["time_s"])
        try:
            teacher, program, evidence, margin = prepare_early(
                initial, sample, score.result(), adapter.spec, rejected, consumed, previous, adapter.base_pose_hold)
        except ValueError as error:
            write(adapter.output / "RECOVERY-EARLY-HANDOFF-REJECTED.private.json",
                  dict(native_step=step, query=query, reason=str(error), first_install_sent=False))
            return None
        event = dict(trigger="EARLY_UNINSTALLED_G7", native_step=step, native_time_s=sample["time_s"],
                     actual_previous_width_m=previous["width_m"], holding_at_handoff=sample["native_grasp"],
                     rejected_query_sha256=digest(path), consumed_query_sha256=digest(self.last_consumed_query))
        return dict(event=event, teacher=teacher, program=program, evidence=evidence, margin=margin,
                    witness=consumed["confirmed_intervals"][-1]["accepted_boundary"], query=query)


def finish(adapter, prepared, frame, stream, reader, score):
    from scripts.wm02.mobile_boundary_runner_v1 import consume, install

    adapter.guard()
    teacher, witness = (prepared[name] for name in ("teacher", "witness"))
    reference = adapter.reference_history[-1]["pose"].copy()
    write(adapter.output/"RECOVERY-HANDOFF-INTENT.private.json", dict(prepared["event"],
          actual_student_queries=prepared["query"]+1, clock_owner_unchanged=True,
          student_only_result=False, prior_role="STUDENT", next_role="TEACHER"))
    writer = LiveWriter(adapter, frame, camera_factory=TwoObjectCameraSession)
    scored, completed_prefixes, consumed_atoms, rejection = score.result(), 0, 0, None
    try:
        for query in range(prepared["query"]+1, adapter.spec["policy_queries"]):
            adapter.guard()
            sample = adapter.measured_task_state()
            prior = confirmed_context(adapter, sample["native_step"], sample["time_s"])
            try:
                if query == prepared["query"]+1:
                    program, evidence, margin = (prepared[name] for name in ("program", "evidence", "margin"))
                else:
                    program, evidence = teacher.decide(sample, reference, adapter.base_pose_hold)
                    margin = label_margin_report(program.decoded_H40, prior)
                if margin["training_margin_exceeded"] or margin["original_execution_speed_exceeded"]:
                    raise ValueError("Recovery Teacher failed the original H40 margin")
            except ValueError as error:
                rejection = str(error)
                break
            row = writer.capture(sample, program, evidence, witness, teacher)
            row.update(instruction=adapter.task.instruction, previous_confirmed_command=prior,
                       original_label_G7_margin=margin, supervision_role="RECOVERY_TEACHER")
            adapter.mobile_contact_stage = teacher.stage.lower()
            adapter.mobile_program = program
            adapter.mobile_program_start_step = sample["native_step"]
            for i in range(PREFIX_TICKS//ATOM_TICKS):
                preview = program.preview(i)
                if i in (0, 7):
                    install(adapter, preview, None)
                atom = dict(atom=i, start_native_step=adapter.observe()["step"],
                            installed_reference_sha256=adapter.active_reference["wire_sha256"], confirmed_ticks=0)
                row["intervals"].append(atom)
                sample, _ = consume(adapter, preview, sequence=128+query*10+i)
                reference, witness = boundary(adapter)
                adapter.reference_history[-1]["pose"] = reference.copy()
                atom.update(end_native_step=sample["native_step"], confirmed_ticks=ATOM_TICKS, boundary=witness)
                row["consumed_mask"][i] = True
                consumed_atoms += 1
            teacher.commit(program, sample["native_step"])
            row["next_query_boundary"] = witness
            completed_prefixes += 1
            scored = score_through(stream, reader, score, sample["native_step"])
            write(adapter.output/"RECOVERY-SCORE-PROGRESS.private.json", scored)
            if scored["task_success"]:
                break
    finally:
        capture = writer.close()
    success = scored["task_success"]
    return dict(schema="V64_ACTUAL_TRAIN_RECOVERY", arm=adapter.spec["arm"], teacher_takeover=consumed_atoms > 0,
                task_instruction=adapter.task.instruction, score=scored, completed_teacher_P10=completed_prefixes,
                actual_SLA_queries=prepared["query"]+1, expert_queries=len(writer.rows),
                task_success=False, recovery_task_success=success, recovery_class=(
                    "RECOVERY_COMPLETE_CANDIDATE" if success else "RECOVERY_PARTIAL_CANDIDATE"
                    if consumed_atoms else "NO_CONFIRMED_TEACHER_CONSUMPTION"),
                consumed_teacher_atoms=consumed_atoms, terminal_native_step=adapter.observe()["step"],
                teacher_rejection=rejection, capture=capture, actual_training_admission=False,
                cancel=adapter.cancel_reference())


def student_source(text):
    """Add the optional Recovery hook after the original complete P10 return."""
    edits = (
        ("    queries = atoms = 0\n    rejection = None\n",
         "    queries = atoms = 0\n    rejection = None\n    recovery = None\n"
         "    if spec.get('v64_recovery_train') is not None:\n"
         "        from scripts.sla.v64_recovery_native import Pilot, finish\n"
         "        recovery = Pilot(adapter)\n"),
        ("                if result['task_success']:\n                    break\n",
         "                if result['task_success']:\n                    break\n"
         "                if recovery is not None:\n"
         "                    folder = Path(prepared['metadata']['input']['path']).parent\n"
         "                    handoff = recovery.maybe_prepare(adapter, score, initial, query, folder)\n"
         "                    if handoff is not None:\n"
         "                        camera.close()\n"
         "                        return finish(adapter, handoff, frame, stream, reader, score)\n"),
        ("                               full_H40_compiled=rejection['full_H40_compiled'],rejection=rejection))\n"
         "                    break\n",
         "                               full_H40_compiled=rejection['full_H40_compiled'],rejection=rejection))\n"
         "                    if recovery is not None:\n"
         "                        path = Path(prepared['metadata']['input']['path']).parent/'QUERY_ACTUAL.private.json'\n"
         "                        handoff = recovery.maybe_early(adapter, score, initial, query, path)\n"
         "                        if handoff is not None:\n"
         "                            camera.close()\n"
         "                            return finish(adapter, handoff, frame, stream, reader, score)\n"
         "                    break\n"),
    )
    for before, after in edits:
        if text.count(before) != 1:
            raise ValueError("The original student source seam changed")
        text = text.replace(before, after)
    compile(text, "<v64-recovery-student>", "exec")
    return text


def stage_student(source, destination):
    with destination.open("x") as stream:
        stream.write(student_source(source.read_text()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    stage_student(**vars(parser.parse_args()))
