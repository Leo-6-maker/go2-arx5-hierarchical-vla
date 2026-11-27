"""Seal actual Reach/Orient TRAIN captures under the original R4 reader contract."""

import argparse
import json
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from scripts.sla.v64_reach_qual import train_language
from v64_a78_delta import boundary_reads, statistics
from v64_recovery_seal import accepted_rows, actual_queries, ref, typed_labels
from v63_adapt_data import physical_aux


def export_nominal(case, spec, capture, queries, label, destination, stats):
    from go2arx5_vla.sla_adapter.codec_v1 import Quantiles, eef, rigid
    from go2arx5_vla.sla_adapter.dense_g7_v6 import CLOCK_FIELDS, label_margin_report
    from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
    from scripts.sla.export_multitask_g7_v6 import fp32_audit
    from scripts.sla.multitask_g7_capture_v6 import copy_shards

    frame = rigid(queries[0]["episode_fixed_T_policy_from_mount"])
    transform = rigid(spec["policy_from_world"])
    codec = GripLogitCodec(frame)
    command_log = case / "native-01/MOBILE_SAMPLES.private.jsonl"
    with command_log.open() as stream:
        samples = [json.loads(line) for line in stream]
    joined = {row["native_start_step"]: row for row in samples}
    require(len(joined) == len(samples), "Require unique native 50ms intervals")
    state7, latent7, nominal, contexts = [], [], [], []
    for index, query in enumerate(queries):
        tick = 5703 + 250 * index
        previous = joined[tick - 25]
        context = query["previous_confirmed_command"]
        world = eef(query["nominal_world_H40"], (40, 10))
        margin = label_margin_report(world, context)
        state = eef(query["state_eef10"], (10,))
        measured = query["actual_state"]
        witnessed = pose10(transform @ np.asarray(measured["tcp"]), measured["gripper_fraction"])
        require(query["native_step"] == tick and abs(query["native_time_s"] - tick * .002) <= 1e-8
                and query["instruction"] == spec["task_instruction_id"] and query["phase"] == "PREGRASP"
                and query["full_EEF_H40_planner_check"] is True and query["supervision_mask"] == [True] * 40
                and query["object_phase_memory_in_student_inputs"] is False and previous["native_end_step"] == tick
                and previous["ticks"] == 25 and context["width_m"] == .044 * previous["installed_EEF10"][9]
                and context["source_native_s"] == previous["gripper_command"]["source_native_s"]
                and context["query_native_s"] == query["native_time_s"]
                and margin == query["original_label_G7_margin"] and not margin["original_execution_speed_exceeded"]
                and state[9] == witnessed[9] and np.allclose(state[:9], witnessed[:9], rtol=0, atol=1e-12)
                and all(joined[tick+25*atom]["gripper_command"]["sequence"] == 128+index*10+atom
                        for atom in range(10)), "The original H40/G7 margin, phase, camera or real command identity changed")
        target = pose10(transform @ matrix_from10(world), world[:, 9])
        s, a = codec.encode(state, target)
        state7.append(s)
        latent7.append(a)
        nominal.append(target)
        contexts.append([context[key] for key in CLOCK_FIELDS])
    arrays = dict(state7=np.asarray(state7), latent7=np.asarray(latent7), state_eef10=label["state_eef10"],
                  nominal_policy_H40=np.asarray(nominal), native_step=label["native_step"],
                  previous_actual_command=np.asarray(contexts), frame_index=np.arange(len(queries)),
                  phase=label["phase"], supervision_mask=np.ones((len(queries),40), dtype=bool),
                  consumed_mask=np.tile([True]*10+[False]*30, (len(queries),1)))
    quantiles = {key: Quantiles(item["q01"], item["q99"]) for key, item in stats.items()}
    audit = fp32_audit([(case.name, codec, arrays, arrays["previous_actual_command"])], quantiles)
    require(not any(g["over_original_gate"] or (g["codec_path"] == "fp32_codec" and g["over_margin"])
                    for g in audit["label_margin_by_domain_codec_phase"]), "Original G7 FP32 margin")
    destination.mkdir(mode=0o700)
    shards = copy_shards(case / "native-01/LIVE_DATA", capture, arrays, destination)
    path = destination / "LABELS.npz"
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    return frame, shards, dict(file=f"{case.name}/LABELS.npz", sha256=sha(path))


def build(wave, state, actions, template, out):
    roster = ("EF5_T_REACH_01", "EF5_T_REACH_02")
    parent = read(template)
    stats_refs = {"state_statistics": parent["state_sha256"], "action_statistics": parent["action_sha256"]}
    stats = statistics(dict(statistics_sha256=stats_refs), state, actions)
    terminal_path = wave / "RESULT.private.json"
    group = read(terminal_path)
    rows = group["records"]
    require(read(wave / "control/LEASE.json")["status"] == "CLOSED" and group["unreconciled"] is None
            and rows == read(wave / "RECORDS.private.json") and 0 < group["paid_requests"] <= len(roster)
            and tuple(row["condition"] for row in rows) == roster[:len(rows)]
            and group["paid_requests"] == len(rows), "Require the original reconciled Reach TRAIN roster")
    out.mkdir(mode=0o700)
    entries, rejected = [], []
    for row in rows:
        case = wave / "cases" / row["condition"]
        lease = read(case / "control/LEASE.json")
        source = case / "EXECUTION_SPEC.json"
        spec = read(pin(source, lease["spec_sha256"]))
        result = read(pin(case / "RESULT.private.json", row["result"]["sha256"]))
        require(lease["status"] == "CLOSED" and row["split"] == spec["collection_split"] == "TRAIN"
                and spec["expert_training_eligible"] and row["requests"] == result["physical_requests"] == 1,
                "Only a real closed TRAIN case may be sealed")
        detail = result["detail"]
        if (not result["completed"] or not result["shutdown_clean"] or result["unsafe"]
                or result["technical_failure"] or row["hard_STOP"]
                or not detail.get("reach_orientation_success")):
            rejected.append(dict(case=case.name, reason=row["first_failure"], result=row["result"]["sha256"]))
            continue
        color = spec["task_instruction_id"].removeprefix("HOLD-")
        task = detail["training_instruction"]
        language = train_language(color, case.name)
        score = detail["score"]
        end = result["actual_native_integrations"]
        require(task == f"REACH-ORIENT-{color}" and not detail["scored_as_original_hold"]
                and score["task_success"] and score["native_step"] == detail["terminal_native_step"] == end
                and not (score["object_hand_contact_ticks"] or score["unsafe_ticks"] or score["missing_native_intervals"])
                and score["longest_dwell_s"] >= .5 and result["trace_audit"]["status"] == "RECORDED_TRACE_CHECKS_PASS"
                and result["trace_audit"]["native_step_records"] == end and not result["optimizer_updates"]
                and not result["model_generations"] and not result["training_visits"], "Reach score or native trace invalid")
        try:
            capture, queries = actual_queries(case, detail, dict(spec, language=language))
            require(capture["final_native_step"] == end and capture["data_contract"] == spec["collection_contract"]
                    and len(queries) == detail["expert_queries"] <= spec["policy_queries"]
                    and detail["consumed_task_atoms"] == 10 * len(queries), "Require the actual capture denominator")
            world, sample_ref = accepted_rows(case, queries, end)
            labels, residual, p10 = typed_labels(spec, queries, world)
            dest = out / case.name
            frame, shards, raw = export_nominal(case, spec, capture, queries, labels, dest, stats)
            corrected = dest / "R4_ACTION.npz"
            with corrected.open("xb") as stream:
                np.savez_compressed(stream, **labels)
            aux = physical_aux(case, corrected, dest / "R4_AUX.npz")
        except (KeyError, ValueError) as error:
            rejected.append(dict(case=case.name, reason=str(error), result=row["result"]["sha256"]))
            continue
        episode = dict(id=case.name, split="TRAIN", task=task, family=spec["mobile_condition"]["family"],
                       language=language, windows=len(queries), mount=dict(T_policy_from_mount=frame.tolist()),
                       scene_sha256=capture["frozen_scene_sha256"], reset=spec["mobile_condition"]["reset"],
                       label_kind="expert_query_plan_H40", expert_stage_audit_only=labels["phase"].tolist(),
                       labels=raw, frame_shards=shards, native_result_sha256=row["result"]["sha256"], individual_pass=True)
        entries.append(dict(episode=episode, corrected=dict(file=str(corrected.relative_to(out)), sha256=sha(corrected)),
                            aux=aux, native_result_sha256=row["result"]["sha256"],
                            original_execution_spec_sha256=sha(source), population="newest",
                            actual_native_samples=sample_ref, maximum_roundtrip_mm_deg=residual,
                            maximum_P10_roundtrip_mm_deg=p10))
    require(len({entry["episode"]["scene_sha256"] for entry in entries}) == len(entries), "Duplicate Reach scenes")
    manifest = dict(schema="V64_REACH_ORIENT_R4_ACTUAL_CONSUMED", wave=ref(terminal_path), paid=group["paid_requests"],
                    records=ref(wave / "RECORDS.private.json"), episodes=entries, rejected=rejected,
                    statistics_sha256=stats_refs, train=len(entries), windows=sum(e["episode"]["windows"] for e in entries),
                    full_H40=sum(e["aux"]["complete_H40"] for e in entries), optimizer_visits=0, new_native_integrations=0)
    actual = boundary_reads(out, manifest, stats) if entries else []
    write(out / "MANIFEST.json", manifest)
    write(out / "CORE_READBACK.json", dict(manifest_sha256=sha(out / "MANIFEST.json"),
          actual_RGB_boundaries=actual, verified_episodes=len(entries), actual_optimizer_visits=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wave", "state", "actions", "template", "out"):
        parser.add_argument(name, type=Path)
    build(**vars(parser.parse_args()))
