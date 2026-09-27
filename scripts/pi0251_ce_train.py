#!/usr/bin/env python3
"""Run the PI0251 C/E pair through the pinned PI0247 training lifecycle."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any


AUTH = "PI0251_XYZ_EXCESS_RISK_TO_TRAIN_SIM_GO"
CANONICAL = 5858091531
POINTER = 5858094328
SEED = 22301
STEPS = 2000
CHECKPOINTS = (1000, 2000)
REVISION = "c289992952c08161b8c37e215ceef98b6cb38b70"
TREE_SHA256 = "e8c8ac97f6b455204decc93ec753ab90e0bbe8f380260488fdfdc20c50b86467"
FAMILIES = {
    "C": "PI0251_C_X_CONTINUATION_M64_2K_V1",
    "E": "PI0251_E_WINDOW_XYZ_EXCESS_M64_2K_V1",
}
LOSSES = {
    "C": "L_native + 0.4*ell_pos(a_free_S4) + 0.1*ell_delta(a_free_S4) + 0.1*ell_rot(a_free_S4)",
    "E": "L_native + 0.4*ell_pos(a_free_S4) + 0.1*ell_delta(a_free_S4) + 0.1*ell_rot(a_free_S4) + 0.4*relu(ell_pos-0.49)",
}
COEFFICIENTS = {
    "C": {"ell_pos": 0.4, "ell_delta": 0.1, "ell_rot": 0.1, "ell_excess": 0.0},
    "E": {"ell_pos": 0.4, "ell_delta": 0.1, "ell_rot": 0.1, "ell_excess": 0.4, "ell_excess_threshold_scaled": 0.49},
}
life: Any = None
helper: Any = None
card: dict[str, Any] | None = None
resume_after = 0
runtime_state: dict[str, Any] = {}
base_validate_initializer: Any = None
base_verify_code: Any = None


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise RuntimeError(f"refusing to replace conflicting PI0251 evidence: {path}")
        return
    path.write_text(data, encoding="utf-8")


def _replace(path: Path, value: Any) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = path.with_name(path.name + ".pi0251.tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def _load_life(paths: dict[str, Any]) -> Any:
    repo = Path(paths["repo_code_root"])
    sys.path[:0] = [str(repo / "scripts"), str(Path(paths["execution_driver_root"]))]
    import pi0247_lu_train
    pi0247_lu_train.STEPS = STEPS
    pi0247_lu_train.CHECKPOINTS = CHECKPOINTS
    return pi0247_lu_train


def _load_helper(paths: dict[str, Any]) -> Any:
    driver = Path(paths.get("legacy_execution_driver_root") or paths["execution_driver_root"]) / "pi0248_cr_train.py"
    spec = importlib.util.spec_from_file_location("pi0249_pi0248_rotation_adapter", driver)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen PI0248 rotation adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.life = life
    sys.path.insert(0, str(driver.parent))
    return module


def _processor_identity(rotation_check: dict[str, Any]) -> dict[str, Any]:
    identity = rotation_check.get("processor_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("PI0248 frozen C_T processor identity is missing")
    return identity


def _verify_pair_binding(run_card: Path, doc: dict[str, Any]) -> None:
    source = doc["source"]
    binding = _json(Path(source["pair_run_binding_path"]))
    if (binding.get("schema") != "PI0251_CE_PAIR_RUN_BINDING_V1"
            or binding.get("status") != "SEALED"
            or binding.get("selected_checkpoint") != "NONE"
            or binding.get("pair_recipe_binding_sha256") != source["pair_recipe_binding_sha256"]):
        raise RuntimeError("PI0251 C/E pair run binding is not the sealed shared recipe")
    cards = binding.get("cards", {})
    for arm in ("C", "E"):
        item = cards.get(arm, {})
        path = Path(item.get("path", ""))
        if not path.is_file() or _sha(path) != item.get("sha256"):
            raise RuntimeError(f"PI0251 {arm} run-card identity is not closed")
    own = cards[doc["arm"]]
    if own["path"] != str(run_card) or own["sha256"] != _sha(run_card):
        raise RuntimeError("PI0251 pair binding differs from this sealed run card")
    if _sha(Path(source["pair_recipe_binding_path"])) != source["pair_recipe_binding_sha256"]:
        raise RuntimeError("PI0251 shared pair recipe binding changed")
    parent_path = Path(source["x2000_parent_run_card_path"])
    parent_sha = _sha(parent_path) if parent_path.is_file() else None
    parent_seal = _json(parent_path.with_name("run_card.seal.json"))
    if (parent_sha != source["x2000_parent_run_card_sha256"]
            or parent_sha != source["x2000_parent_run_card_seal_sha256"]
            or parent_seal.get("sha256") != parent_sha
            or _json(parent_path).get("arm") != "X"
            or _json(parent_path).get("family") != "PI0249_X_XYZ4_PRIORITY_M64_2K_V1"):
        raise RuntimeError("PI0251 fixed X2000 ancestor card/seal changed")
    integrity_path = Path(source["x2000_integrity_path"])
    integrity = _json(integrity_path)
    if (integrity.get("status") != "PASS_PI0249_KX_TRAINING_INTEGRITY"
            or integrity.get("arm") != "X" or integrity.get("run_card_sha256") != parent_sha
            or integrity.get("optimizer_updates") != STEPS or integrity.get("selected_checkpoint") != "NONE"):
        raise RuntimeError("PI0251 X2000 ancestor integrity does not match its sealed parent card")


def _verify_card(path: Path, arm: str) -> dict[str, Any]:
    doc = _json(path)
    seal = _json(path.with_name("run_card.seal.json"))
    expected_prefix = f"arms/PI0251/xyz-excess-risk-20260928/{arm}"
    if (arm not in FAMILIES or doc.get("schema") != "PI0251_CE_RUN_CARD_V1"
            or doc.get("status") != "SEALED_READY_NOT_STARTED"
            or doc.get("authorization") != AUTH or doc.get("execution_authorization") != AUTH
            or doc.get("canonical_issue_comment") != CANONICAL
            or doc.get("pointer_pr_comment") != POINTER or doc.get("arm") != arm
            or doc.get("family") != FAMILIES[arm] or doc.get("loss") != LOSSES[arm]
            or doc.get("loss_coefficients") != COEFFICIENTS[arm]
            or doc.get("seed") != SEED or doc.get("selected_checkpoint") != "NONE"
            or doc.get("d1_runtime_id") != "S4"
            or doc.get("stage") != {"name": "M64_CE_CONTINUATION", "steps": STEPS,
                                     "checkpoint_steps": list(CHECKPOINTS), "source_revision": REVISION}
            or doc.get("optimizer", {}).get("steps") != STEPS
            or doc.get("optimizer", {}).get("checkpoint_steps") != list(CHECKPOINTS)
            or doc.get("optimizer", {}).get("learning_rate") != 1e-5
            or doc.get("optimizer", {}).get("warmup_steps") != 10
            or doc.get("cohort", {}).get("count") != 64
            or doc.get("schedule", {}).get("count") != STEPS
            or doc.get("archive", {}).get("path_prefix") != expected_prefix
            or seal.get("sha256") != _sha(path)):
        raise RuntimeError("PI0251 run card or seal differs from the fixed C/E authorization")
    _verify_pair_binding(path, doc)
    src = doc["source"]
    if (_sha(Path(src["x2000_integrity_path"])) != src["x2000_integrity_sha256"]
            or _sha(Path(src["x2000_durable_receipt_path"])) != src["x2000_durable_receipt_sha256"]
            or _sha(Path(src["x2000_hf_readback_path"])) != src["x2000_hf_readback_sha256"]):
        raise RuntimeError("PI0251 fixed X2000 integrity/readback identities changed")
    exec_root = Path(doc["paths"]["execution_driver_root"])
    legacy_root = Path(doc["paths"]["legacy_execution_driver_root"])
    custom_paths = {
        "pi0251_ce_train.py": exec_root / "pi0251_ce_train.py",
        "pi0251_step_rng.py": exec_root / "pi0251_step_rng.py",
        "pi0251_ce_train_h200.pbs": Path(doc["paths"]["pbs_script"]),
        "pi0248_cr_train.py": legacy_root / "pi0248_cr_train.py",
        "pi0248_rotation_cpu.py": legacy_root / "pi0248_rotation_cpu.py",
    }
    for name, expected in src["custom_execution_code_sha256"].items():
        if name not in custom_paths or not custom_paths[name].is_file() or _sha(custom_paths[name]) != expected:
            raise RuntimeError(f"PI0251 execution source changed: {name}")
    if _sha(Path(src["g0_source_index_path"])) != src["g0_source_index_sha256"]:
        raise RuntimeError("PI0251 G0 source index changed")
    if _sha(Path(src["g1_cpu_selfcheck_path"])) != src["g1_cpu_selfcheck_sha256"]:
        raise RuntimeError("PI0251 G1 CPU self-check changed")
    if _json(Path(src["g1_cpu_selfcheck_path"])).get("status") != "PASS_PI0251_G1_CPU_FIXTURES":
        raise RuntimeError("PI0251 G1 CPU self-check did not pass")
    if _sha(Path(doc["schedule"]["path"])) != doc["schedule"]["sha256"]:
        raise RuntimeError("PI0251 fixed 64/2000 exposure schedule changed")
    if _sha(Path(src["step_key_table_path"])) != src["step_key_table_sha256"]:
        raise RuntimeError("PI0251 step-addressed key table changed")
    if doc.get("initializer", {}).get("screen_revision") != REVISION or doc["initializer"].get("full_checkpoint_tree_sha256") != TREE_SHA256:
        raise RuntimeError("PI0251 initializer is not the pinned X2000 checkpoint")
    if arm == "E":
        peer = Path(src["peer_run_card_path"])
        if not peer.is_file() or _sha(peer) != src["peer_run_card_sha256"]:
            raise RuntimeError("PI0251 E is not bound to the sealed C card")
    life.STEPS, life.CHECKPOINTS = STEPS, CHECKPOINTS
    return doc


def _validate_initializer(doc: dict[str, Any]) -> None:
    init, source = doc["initializer"], doc["source"]
    receipt_path = Path(source["x2000_durable_receipt_path"])
    readback_path = Path(source["x2000_hf_readback_path"])
    view_path = Path(source["initializer_view_receipt_path"])
    if (_sha(receipt_path) != init["durable_receipt_sha256"]
            or _sha(readback_path) != source["x2000_hf_readback_sha256"]
            or _sha(view_path) != source["initializer_view_receipt_sha256"]):
        raise RuntimeError("PI0251 X2000 durable/view receipts changed")
    receipt, readback, view = _json(receipt_path), _json(readback_path), _json(view_path)
    record = receipt.get("source_record", {})
    inventory = record.get("inventory", {})
    if (receipt.get("status") != "DURABLE_VERIFIED" or receipt.get("checkpoint_step") != 2000
            or receipt.get("family") != init.get("source_family")
            or receipt.get("hf_manifest_commit_id") != REVISION
            or inventory.get("tree_sha256") != TREE_SHA256
            or inventory.get("total_bytes") != int(init["full_checkpoint_bytes"])
            or readback.get("status") != "PASS_PT_FULL_HF_READBACK"
            or readback.get("revision") != REVISION or readback.get("source_tree_sha256") != TREE_SHA256
            or readback.get("repo_id") != doc["archive"]["repo_id"]
            or view.get("status") != "PASS_PI0251_X2000_CT_VIEW"
            or view.get("revision") != REVISION or view.get("checkpoint_tree_sha256") != TREE_SHA256
            or view.get("source_durable_receipt_sha256") != _sha(receipt_path)):
        raise RuntimeError("PI0251 X2000 durable receipt or model view identity mismatch")
    root = Path(init["path"])
    entries = {row["path"]: row for row in inventory.get("entries", [])}
    model_names = {"config.json", "model.safetensors.index.json",
                   "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
    observed = {str(row.get("path", "")).split(f"/checkpoint-2000/", 1)[-1]: row
                for row in readback.get("files", []) if "/checkpoint-2000/" in str(row.get("path", ""))}
    if set(observed) != set(entries):
        raise RuntimeError("PI0251 X2000 HF readback does not cover the full durable Trainer checkpoint")
    for name, entry in entries.items():
        row = observed[name]
        if int(row.get("bytes", -1)) != int(entry["bytes"]) or row.get("sha256") != entry["sha256"]:
            raise RuntimeError(f"PI0251 X2000 HF readback differs from durable inventory: {name}")
    for name in sorted(model_names):
        entry, local = entries.get(name), root / name
        if (entry is None or not local.is_file() or local.stat().st_size != int(entry["bytes"])
                or _sha(local) != entry["sha256"]
                or int(observed.get(name, {}).get("bytes", -1)) != int(entry["bytes"])
                or observed.get(name, {}).get("sha256") != entry["sha256"]):
            raise RuntimeError(f"PI0251 X2000 model file differs from its fixed HF readback: {name}")
    bindings = source["processor_bindings"]
    identity = {key: value for key, value in bindings.items() if key != "view_files"}
    rotation = _json(Path(source["rotation_check_path"]))
    if (_sha(Path(source["rotation_check_path"])) != source["rotation_check_sha256"]
            or rotation.get("status") != "PASS_CPU_DECODER_AND_GRADIENT"
            or _processor_identity(rotation) != identity):
        raise RuntimeError("PI0251 frozen C_T processor binding differs from the PI0248 decoder receipt")
    for name, expected in bindings["view_files"].items():
        local = root / name
        if not local.is_file() or _sha(local) != expected:
            raise RuntimeError(f"PI0251 frozen C_T view changed: {name}")
    dataset = Path(doc["paths"]["train_dataset"]) / "meta"
    for name, expected in (("stats.json", identity["NEW_EMBODIMENT_stats_sha256"]),
                           ("relative_stats.json", identity["relative_stats_sha256"])):
        if _sha(dataset / name) != expected:
            raise RuntimeError(f"PI0251 frozen TRAIN statistics changed: {name}")
    legacy = copy.deepcopy(doc)
    legacy["schema"] = "PI0247_M64_RUN_CARD_V1"
    base_validate_initializer(legacy)


def _verify_code(doc: dict[str, Any], repo: Path, shared: Path, training: Path) -> None:
    schema = doc["schema"]
    doc["schema"] = "PI0247_M64_RUN_CARD_V1"
    try:
        base_verify_code(doc, repo, shared, training)
    finally:
        doc["schema"] = schema


def _append(path: Path, row: dict[str, Any], *, sync: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        if sync:
            os.fsync(stream.fileno())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rows_for_root(output: Path, root: Path, name: str, resume: int) -> list[dict[str, Any]]:
    current = _jsonl(root / name)
    if root == output / "training":
        return current
    if len(current) == STEPS:
        return current
    prefix = _jsonl(output / "training" / name)
    if len(prefix) != resume or len(current) != STEPS - resume:
        raise RuntimeError(f"PI0251 exact-resume audit is incomplete: {name}")
    return prefix + current


def _last_jsonl(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if end == 0:
            raise RuntimeError(f"PI0251 live sample receipt is empty: {path}")
        start = end - 1
        while start >= 0:
            stream.seek(start)
            if stream.read(1) == b"\n" and start < end - 1:
                break
            start -= 1
        stream.seek(start + 1)
        line = stream.read(end - start - 1).decode("utf-8").strip()
    return json.loads(line)


def _peer_rows(doc: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    peer_card = _json(Path(doc["source"]["peer_run_card_path"]))
    peer_out = Path(peer_card["paths"]["output_root"])
    integrity_path = peer_out / "pi0251_training_integrity.json"
    integrity = _json(integrity_path)
    if (integrity.get("status") != "PASS_PI0251_C_TRAINING_INTEGRITY"
            or integrity.get("arm") != "C"
            or integrity.get("run_card_sha256") != doc["source"]["peer_run_card_sha256"]
            or integrity.get("optimizer_updates") != STEPS):
        raise RuntimeError("PI0251 E cannot start before C has a complete integrity PASS")
    root = helper._winning_training_root(peer_card)
    resume = 0 if root == peer_out / "training" else int(_json(root.parent / "resume_record.json")["resume_after_step"])
    _, overlays = helper._overlay_rows(root)
    rows = (_rows_for_root(peer_out, root, "pi0251_step_rng_audit.jsonl", resume),
            _rows_for_root(peer_out, root, "pi0241_sample_consumption.jsonl", resume),
            _rows_for_root(peer_out, root, "pi0251_pair_identity.jsonl", resume))
    if len(overlays) != STEPS or any(len(items) != STEPS for items in rows):
        raise RuntimeError("PI0251 C actual S4/source/input reference is incomplete")
    if any(row.get("optimizer_step") != i or row.get("status") != "REFERENCE_CAPTURED"
           or row.get("before_backward") is not True for i, row in enumerate(rows[2], 1)):
        raise RuntimeError("PI0251 C reference identity log is not a complete actual capture")
    return overlays, *rows


def _install_overlay_runtime(module: Any, doc: dict[str, Any], step_rng: Any) -> None:
    import torch

    arm = doc["arm"]
    schedule = _json(Path(doc["schedule"]["path"]))["rows"]
    if arm == "E":
        peer_overlay, peer_rng, peer_samples, peer_pairs = _peer_rows(doc)
    else:
        peer_overlay = peer_rng = peer_samples = peer_pairs = []
    output = Path(doc["paths"]["output_root"])

    def before_forward(active: dict[str, Any]) -> dict[str, Any]:
        step = int(active["optimizer_step"])
        if step < 1 or step > STEPS:
            raise RuntimeError(f"PI0251 pre-forward guard received invalid optimizer step {step}")
        training_root = runtime_state.get("training_root")
        sample = _last_jsonl(Path(training_root) / "pi0241_sample_consumption.jsonl") if training_root else {}
        plan = schedule[step - 1]
        actual_source = {key: sample.get(key) for key in
                         ("optimizer_step", "fixed_index", "episode_index", "source_episode_id", "observation_index")}
        expected_source = {key: plan.get(key) for key in
                           ("optimizer_step", "fixed_index", "episode_index", "source_episode_id")}
        expected_source["observation_index"] = 0
        key, seed = step_rng.step_key_seed(step)
        actual = {"source": actual_source, "key": active.get("key"), "seed": active.get("seed"),
                  "model_input_sha256": active.get("model_input_sha256")}
        expected = {"source": expected_source, "key": key, "seed": seed,
                    "model_input_sha256": active.get("model_input_sha256")}
        fields = ["source", "key", "seed", "model_input_sha256"]
        if not actual.get("model_input_sha256"):
            expected["model_input_sha256"] = "REQUIRED"
        if arm == "E":
            peer_sample = peer_samples[step - 1]
            peer_source = {key: peer_sample.get(key) for key in actual_source}
            expected.update({"source": peer_source, "key": peer_rng[step - 1].get("key"),
                             "seed": peer_rng[step - 1].get("seed"),
                             "model_input_sha256": peer_rng[step - 1].get("model_input_sha256")})
        event = {"optimizer_step": step, "phase": "BEFORE_NATIVE_RANDOM_SAMPLING",
                 "actual": actual, "expected": expected, "status": "PASS"}
        try:
            step_rng.verify_identity(actual, expected, tuple(fields))
        except RuntimeError as error:
            event.update({"status": "FAIL", "error": str(error)})
            _append(Path(training_root) / "pi0251_pre_forward_guard.jsonl", event, sync=True)
            raise
        _append(Path(training_root) / "pi0251_pre_forward_guard.jsonl", event, sync=step % 100 == 0)
        return {"status": "PASS", "event": "BEFORE_NATIVE_RANDOM_SAMPLING"}

    runtime_state["before_forward_guard"] = before_forward
    original_install = module.install

    def install(head_cls: type, **kwargs: Any) -> dict[str, Any]:
        audit = original_install(head_cls, **kwargs)
        original_physical = module._physical_losses
        original_append = module._append_json
        original_forward = head_cls.forward
        state: dict[str, Any] = {"position": None, "tau_identity": None}

        def physical(*args: Any, **kw: Any) -> Any:
            position, delta, details = original_physical(*args, **kw)
            state["position"] = position
            return position, delta, details

        def append(path: Path, row: dict[str, Any], count: int) -> None:
            context = step_rng.active_step_context()
            step = int(context["optimizer_step"]) if context else -1
            position = state.get("position")
            if position is None or not state.get("tau_identity"):
                raise RuntimeError("PI0251 actual ell_pos or native tau identity is missing before backward")
            scaled = float(position.detach().float().item())
            excess = max(scaled - 0.49, 0.0)
            penalty = 0.4 * excess if arm == "E" else 0.0
            row = {**row, "native_tau_identity": state["tau_identity"],
                   "pi0251_arm": arm, "pi0251_ell_pos_scaled_xyz_mse": scaled,
                   "pi0251_ell_excess": excess, "pi0251_excess_penalty": penalty,
                   "pi0251_actual_training_loss": float(row["total_loss"])
                   + 0.1 * float(row["physical_rotation_loss"]) + penalty}
            current_root = runtime_state.get("training_root")
            sample = _last_jsonl(Path(current_root) / "pi0241_sample_consumption.jsonl") if current_root else {}
            actual = {
                "source": {key: sample.get(key) for key in
                           ("optimizer_step", "fixed_index", "episode_index", "source_episode_id", "observation_index")},
                "key": context.get("key") if context else None,
                "seed": context.get("seed") if context else None,
                "model_input_sha256": context.get("model_input_sha256") if context else None,
                "native_tau": row.get("native_tau"), "native_tau_identity": state["tau_identity"],
                "native_epsilon": row.get("native_epsilon"),
                "native_epsilon_sha256": row.get("native_epsilon_sha256"),
                "stock_initial_noise": row.get("stock_initial_noise"),
            }
            fields = []
            expected = None
            if step < 1 or step > STEPS or count != step or int(row.get("forward_index", -1)) != step:
                fields.append("optimizer_step")
            if arm == "E" and 1 <= step <= STEPS:
                ref_overlay, ref_rng, ref_sample, ref_pair = (
                    peer_overlay[step - 1], peer_rng[step - 1], peer_samples[step - 1], peer_pairs[step - 1])
                expected = {
                    "source": {key: ref_sample.get(key) for key in actual["source"]},
                    "key": ref_rng.get("key"), "seed": ref_rng.get("seed"),
                    "model_input_sha256": ref_rng.get("model_input_sha256"),
                    "native_tau": ref_overlay.get("native_tau"),
                    "native_tau_identity": ref_overlay.get("native_tau_identity"),
                    "native_epsilon": ref_overlay.get("native_epsilon"),
                    "native_epsilon_sha256": ref_overlay.get("native_epsilon_sha256"),
                    "stock_initial_noise": ref_overlay.get("stock_initial_noise"),
                }
                fields.extend(field for field in expected if actual.get(field) != expected.get(field))
                if fields:
                    try:
                        step_rng.verify_identity(actual, expected, tuple(expected))
                    except RuntimeError:
                        pass
            pair_row = {"optimizer_step": step, "phase": "AFTER_NATIVE_SAMPLING_BEFORE_BACKWARD",
                        "status": ("REFERENCE_CAPTURED" if arm == "C" else "PASS" if not fields else "FAIL"),
                        "before_backward": True, "mismatch_fields": fields,
                        "expected_identity": expected, **actual}
            pair_path = Path(current_root) / "pi0251_pair_identity.jsonl"
            _append(pair_path, pair_row, sync=bool(fields) or step % 100 == 0)
            if fields:
                raise RuntimeError(f"PI0251 C/E actual pre-backward identity failed at step {step}: {fields}")
            original_append(path, row, count)

        def forward(self: Any, *args: Any, **kw: Any) -> Any:
            state.update(position=None, tau_identity=None)
            had_instance = "sample_time" in self.__dict__
            original_sample_time = self.sample_time

            def capture_tau(*sample_args: Any, **sample_kw: Any) -> Any:
                value = original_sample_time(*sample_args, **sample_kw)
                state["tau_identity"] = module._tensor_identity(value)
                return value

            self.sample_time = capture_tau
            try:
                result = original_forward(self, *args, **kw)
            finally:
                if had_instance:
                    self.sample_time = original_sample_time
                else:
                    del self.sample_time
            if arm == "E":
                if state["position"] is None or not isinstance(result, dict) or "loss" not in result:
                    raise RuntimeError("PI0251 E excess term lacks the shared physical position tensor")
                result["loss"] = result["loss"] + 0.4 * torch.relu(state["position"] - 0.49)
            return result

        module._physical_losses = physical
        module._append_json = append
        head_cls.forward = forward
        return audit

    module.install = install


def _install_adapters(run_card: Path, arm: str, resume: int) -> tuple[dict[str, Any], Any]:
    global life, helper, card, resume_after, runtime_state, base_validate_initializer, base_verify_code
    preliminary = _json(run_card)
    life = _load_life(preliminary["paths"])
    helper = _load_helper(preliminary["paths"])
    card = _verify_card(run_card, arm)
    resume_after = resume
    runtime_state = {}
    base_validate_initializer = life._validate_initializer_runtime
    base_verify_code = life._verify_code
    life._verify_card = _verify_card
    life._validate_initializer_runtime = _validate_initializer
    life._verify_code = _verify_code

    original_load = life.load_module

    def load_module(path: Path, name: str) -> Any:
        loaded = original_load(path, name)
        if resume and name == "pi0247_lu_train_core":
            original_build = loaded.build_runtime_config

            def build_runtime_config(*args: Any, **kwargs: Any) -> Any:
                pi0228, config, summary = original_build(*args, **kwargs)

                original_run = pi0228.run_official

                def run_official(run_config: Any, groot_source: Path) -> Any:
                    from gr00t.experiment import experiment as experiment_module

                    original_save = experiment_module.save_run_config_artifacts
                    checkpoint = Path(card["paths"]["output_root"]) / "training" / f"checkpoint-{resume}"

                    def save_run_config_artifacts(*save_args: Any, **save_kwargs: Any) -> Any:
                        cfg = save_args[2] if len(save_args) > 2 else save_kwargs["config"]
                        resume_path = cfg.training.resume_from_checkpoint
                        if not isinstance(resume_path, str) or Path(resume_path).resolve() != checkpoint.resolve():
                            raise RuntimeError("PI0251 exact-resume config differs from its pinned checkpoint")
                        cfg.training.resume_from_checkpoint = True
                        try:
                            return original_save(*save_args, **save_kwargs)
                        finally:
                            cfg.training.resume_from_checkpoint = resume_path

                    experiment_module.save_run_config_artifacts = save_run_config_artifacts
                    try:
                        return original_run(run_config, groot_source)
                    finally:
                        experiment_module.save_run_config_artifacts = original_save

                pi0228.run_official = run_official
                return pi0228, config, summary

            loaded.build_runtime_config = build_runtime_config
        if path.name == "pi0247_u_rollout_overlay.py":
            loaded.LAMBDA_POS = float(card["loss_coefficients"]["ell_pos"])
            helper._patch_overlay(loaded, {
                "arm": "R", "paths": {"output_root": card["paths"]["output_root"]},
                "source": {"rotation_check_path": card["source"]["rotation_check_path"]},
                "embodiment_tag": card["embodiment_tag"],
            }, resume)
            rng_path = Path(card["paths"]["execution_driver_root"]) / "pi0251_step_rng.py"
            rng_spec = importlib.util.spec_from_file_location("pi0251_step_rng_runtime", rng_path)
            if rng_spec is None or rng_spec.loader is None:
                raise RuntimeError("PI0251 step RNG module is unavailable")
            rng_module = importlib.util.module_from_spec(rng_spec)
            rng_spec.loader.exec_module(rng_module)
            _install_overlay_runtime(loaded, card, rng_module)
            runtime_state["step_rng"] = rng_module
        return loaded

    life.load_module = load_module
    original_chain = life._install_u_chain_check

    def install_chain(audit: dict[str, Any], output: Path, *, mode_path: Path | None = None) -> None:
        mode_path = mode_path or output / "u_mode_restoration_checks.jsonl"
        if resume:
            mode_source = output / "u_mode_restoration_checks.jsonl"
            rows = _jsonl(mode_source)
            if len(rows) != resume:
                raise RuntimeError("PI0251 exact resume lacks its complete mode-restoration prefix")
            if not mode_path.exists():
                shutil.copy2(mode_source, mode_path)
            elif _sha(mode_path) != _sha(mode_source):
                raise RuntimeError("PI0251 exact-resume mode prefix conflicts")
        original_chain(audit, output, mode_path=mode_path)
        training_root = mode_path.parent / "training"
        runtime_state["training_root"] = training_root
        rng_path = training_root / "pi0251_step_rng_audit.jsonl"
        if resume:
            source = Path(card["paths"]["output_root"]) / "training" / rng_path.name
            prefix = _jsonl(source)
            if len(prefix) != resume:
                raise RuntimeError("PI0251 exact resume lacks the complete keyed-RNG prefix")
            if not rng_path.exists():
                shutil.copy2(source, rng_path)
            elif _sha(rng_path) != _sha(source):
                raise RuntimeError("PI0251 exact-resume keyed-RNG prefix conflicts")
            pre_source = Path(card["paths"]["output_root"]) / "training" / "pi0251_pre_forward_guard.jsonl"
            pre_target = training_root / pre_source.name
            if len(_jsonl(pre_source)) != resume:
                raise RuntimeError("PI0251 exact resume lacks the complete pre-forward identity prefix")
            if not pre_target.exists():
                shutil.copy2(pre_source, pre_target)
            pair_source = Path(card["paths"]["output_root"]) / "training" / "pi0251_pair_identity.jsonl"
            if card["arm"] in ("C", "E"):
                pair_rows = _jsonl(pair_source)
                expected_status = "REFERENCE_CAPTURED" if card["arm"] == "C" else "PASS"
                if len(pair_rows) != resume or any(row.get("status") != expected_status for row in pair_rows):
                    raise RuntimeError("PI0251 exact resume lacks its complete paired-identity prefix")
                shutil.copy2(pair_source, training_root / pair_source.name)
        from gr00t.experiment import experiment as experiment_module
        experiment_module.Gr00tTrainer = runtime_state["step_rng"].install_training_step_rng(
            experiment_module.Gr00tTrainer, rng_path, resume_after_step=resume,
            before_forward_guard=runtime_state["before_forward_guard"])

    life._install_u_chain_check = install_chain
    original_finalize = life.finalize

    def finalize(args: argparse.Namespace) -> None:
        verified = _preverify_training(card, run_card)
        helper._normalize_resume_overlay(verified)
        key = "PI0247_FINALIZER_CODE_COMMIT"
        previous = os.environ.get(key)
        os.environ[key] = card["source"]["legacy_execution_code_commit"]
        try:
            original_finalize(args)
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        _write_integrity(card, run_card, verified)

    life.finalize = finalize
    return card, life


def _preverify_training(doc: dict[str, Any], run_card: Path) -> dict[str, Any]:
    output = Path(doc["paths"]["output_root"])
    root = helper._winning_training_root(doc)
    resume = 0 if root == output / "training" else int(_json(root.parent / "resume_record.json")["resume_after_step"])
    audit_path, rows = helper._overlay_rows(root)
    if len(rows) != STEPS or [int(row.get("forward_index", -1)) for row in rows] != list(range(1, STEPS + 1)):
        raise RuntimeError("PI0251 full S4 objective audit is incomplete")
    coeff = doc["loss_coefficients"]
    for index, row in enumerate(rows, 1):
        if (row.get("runtime_id") != "S4" or row.get("rollout_count") != 1
                or row.get("rollout_steps") != 4 or row.get("checkpointed_euler_steps") != 4
                or row.get("backward_recomputation_head_evaluations_expected") != 4
                or row.get("expert_action_in_condition") is not False
                or row.get("active_action_modified") is not False
                or len(row.get("euler_states", [])) != 4 or len(row.get("euler_velocities", [])) != 4
                or row.get("lambda_position") != coeff["ell_pos"]
                or row.get("lambda_delta") != coeff["ell_delta"]
                or row.get("lambda_rotation") != coeff["ell_rot"]
                or row.get("theta_ref_rad") != 0.05):
            raise RuntimeError(f"PI0251 S4/loss-identity evidence is incomplete at step {index}")
        values = {key: float(row[key]) for key in
                  ("native_loss", "physical_position_loss", "physical_delta_loss", "physical_rotation_loss", "total_loss")}
        base = values["native_loss"] + coeff["ell_pos"] * values["physical_position_loss"] + coeff["ell_delta"] * values["physical_delta_loss"]
        total = base + coeff["ell_rot"] * values["physical_rotation_loss"]
        if (not all(math.isfinite(value) for value in (*values.values(), float(row.get("total_loss_with_rotation", math.nan))))
                or not math.isclose(base, values["total_loss"], rel_tol=1e-5, abs_tol=1e-7)
                or not math.isclose(total, float(row["total_loss_with_rotation"]), rel_tol=1e-5, abs_tol=1e-7)):
            raise RuntimeError(f"PI0251 base objective terms do not sum at step {index}")
        scaled = float(row.get("pi0251_ell_pos_scaled_xyz_mse", math.nan))
        excess = max(scaled - 0.49, 0.0)
        penalty = 0.4 * excess if doc["arm"] == "E" else 0.0
        actual_loss = float(row.get("pi0251_actual_training_loss", math.nan))
        expected_loss = float(row.get("total_loss_with_rotation", math.nan)) + penalty
        if (row.get("pi0251_arm") != doc["arm"] or not math.isfinite(scaled)
                or not math.isclose(float(row.get("pi0251_ell_excess", math.nan)), excess, rel_tol=0, abs_tol=1e-6)
                or not math.isclose(float(row.get("pi0251_excess_penalty", math.nan)), penalty, rel_tol=1e-5, abs_tol=1e-7)
                or not math.isclose(actual_loss, expected_loss, rel_tol=1e-5, abs_tol=1e-7)):
            raise RuntimeError(f"PI0251 excess objective is inconsistent at step {index}")
        if (not row.get("native_epsilon_sha256") or not row.get("stock_initial_noise", {}).get("sha256")
                or not row.get("native_tau_identity", {}).get("sha256")):
            raise RuntimeError(f"PI0251 actual tau/epsilon/stock-noise identity is absent at step {index}")
    rng_path = root / "pi0251_step_rng_audit.jsonl"
    rng_rows = _rows_for_root(output, root, rng_path.name, resume)
    key_table = _json(Path(doc["source"]["step_key_table_path"]))["rows"]
    if len(rng_rows) != STEPS or len(key_table) != STEPS:
        raise RuntimeError("PI0251 keyed RNG audit or key table is incomplete")
    for index, (actual, expected) in enumerate(zip(rng_rows, key_table), 1):
        if (actual.get("optimizer_step") != index or actual.get("key") != expected.get("key")
                or actual.get("seed") != expected.get("seed") or not actual.get("model_input_sha256")
                or expected.get("optimizer_step") != index
                or actual.get("pre_forward_guard", {}).get("status") != "PASS"):
            raise RuntimeError(f"PI0251 step-addressed RNG/input audit differs at update {index}")
    schedule = _json(Path(doc["schedule"]["path"]))["rows"]
    consumption = _rows_for_root(output, root, "pi0241_sample_consumption.jsonl", resume)
    if len(consumption) != STEPS:
        raise RuntimeError("PI0251 actual sample consumption is incomplete")
    for index, (sample, planned) in enumerate(zip(consumption, schedule), 1):
        if (int(sample.get("optimizer_step", -1)) != index
                or int(sample.get("fixed_index", -1)) != int(planned["fixed_index"])
                or sample.get("source_episode_id") != planned["source_episode_id"]
                or int(sample.get("episode_index", -1)) != int(planned["episode_index"])
                or int(sample.get("observation_index", -1)) != 0):
            raise RuntimeError(f"PI0251 actual source differs from the sealed exposure row at update {index}")
    pre_rows = _rows_for_root(output, root, "pi0251_pre_forward_guard.jsonl", resume)
    if (len(pre_rows) != STEPS or any(row.get("optimizer_step") != i
                                      or row.get("phase") != "BEFORE_NATIVE_RANDOM_SAMPLING"
                                      or row.get("status") != "PASS" for i, row in enumerate(pre_rows, 1))):
        raise RuntimeError("PI0251 pre-forward source/key guard evidence is incomplete")
    pair_rows = _rows_for_root(output, root, "pi0251_pair_identity.jsonl", resume)
    expected_pair_status = "REFERENCE_CAPTURED" if doc["arm"] == "C" else "PASS"
    if (len(pair_rows) != STEPS or any(row.get("optimizer_step") != i
            or row.get("status") != expected_pair_status
            or row.get("before_backward") is not True for i, row in enumerate(pair_rows, 1))):
        raise RuntimeError("PI0251 actual pre-backward identity receipt is incomplete or failed")
    checkpoint_receipts = _require_checkpoint_receipts(doc, root, resume)
    parity_path = output / "pi0247_U_sampler_parity.json"
    parity = _json(parity_path)
    if (parity.get("status") != "PASS_PI0247_U_S4_PARITY" or not parity.get("gate_decision_equal")
            or not parity.get("gate_issues_equal") or parity.get("expert_action_in_condition") is not False):
        raise RuntimeError("PI0251 native S4/gate parity did not pass")
    chain_path = output / "u_first_update_chain_gradient.json"
    chain = _json(chain_path)
    gradients = chain.get("state_gradient_norms", {})
    if (chain.get("status") != "PASS_PI0247_U_FIRST_UPDATE_FULL_EULER_CHAIN"
            or set(gradients) != {"1", "2", "3", "4"}
            or chain.get("checkpointed_recompute_steps") != 4
            or chain.get("mode_restored_after_forward") is not True
            or chain.get("mode_restored_after_backward") is not True
            or any(not math.isfinite(float(value)) or float(value) <= 0 for value in gradients.values())):
        raise RuntimeError("PI0251 first-update four-step full-gradient evidence is incomplete")
    mode_path = root.parent / "u_mode_restoration_checks.jsonl"
    modes = _jsonl(mode_path)
    if (len(modes) != STEPS or any(row.get("optimizer_step") != i or row.get("restored") is not True
                                   or int(row.get("rollout_forward_head_evaluations", -1)) != 4 * i
                                   or int(row.get("backward_recompute_head_evaluations", -1)) != 4 * i
                                   for i, row in enumerate(modes, 1))):
        raise RuntimeError("PI0251 S4 mode restoration did not pass for all updates")
    return {"root": root, "rows": rows, "audit_path": audit_path, "resume_after_step": resume,
            "rng_path": rng_path, "rng_rows": rng_rows, "sample_rows": consumption,
            "pre_path": root / "pi0251_pre_forward_guard.jsonl", "pre_rows": pre_rows,
            "pair_path": root / "pi0251_pair_identity.jsonl", "pair_rows": pair_rows,
            "checkpoint_receipts": checkpoint_receipts,
            "mode_path": mode_path, "mode_rows": modes,
            "parity_path": parity_path, "chain_path": chain_path}


def _write_integrity(doc: dict[str, Any], run_card: Path, verified: dict[str, Any]) -> None:
    output, root = Path(doc["paths"]["output_root"]), verified["root"]
    manifest_path = output / "training_manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("status") != "PASS_TRAINING_INTEGRITY":
        raise RuntimeError("shared PI0247 read-only finalizer did not pass")
    manifest.update({"schema": "PI0251_CE_TRAINING_MANIFEST_V1", "stage": "M64_CE_CONTINUATION",
                     "authorization": AUTH, "execution_authorization": AUTH,
                     "execution_code_commit": doc["source"]["execution_code_commit"],
                     "arm": doc["arm"], "family": doc["family"], "objective": doc["loss"],
                     "loss_coefficients": doc["loss_coefficients"], "selected_checkpoint": "NONE"})
    pi_manifest_path = output / "pi0251_training_manifest.json"
    _write(pi_manifest_path, manifest)
    result = {
        "schema": "PI0251_CE_TRAINING_INTEGRITY_V1",
        "status": f"PASS_PI0251_{doc['arm']}_TRAINING_INTEGRITY",
        "arm": doc["arm"], "family": doc["family"], "run_card_sha256": _sha(run_card),
        "training_manifest_path": str(pi_manifest_path), "training_manifest_sha256": _sha(pi_manifest_path),
        "shared_lifecycle_manifest_sha256": _sha(manifest_path),
        "execution_code_commit": doc["source"]["execution_code_commit"],
        "initializer_revision": REVISION, "initializer_tree_sha256": TREE_SHA256,
        "schedule_sha256": doc["schedule"]["sha256"], "step_key_table_sha256": doc["source"]["step_key_table_sha256"],
        "optimizer_updates": STEPS, "checkpoint_steps": list(CHECKPOINTS), "selected_checkpoint": "NONE",
        "objective_audit_path": str(verified["audit_path"]), "objective_audit_sha256": _sha(verified["audit_path"]),
        "objective_first": verified["rows"][0], "objective_last": verified["rows"][-1],
        "step_rng_audit_path": str(verified["rng_path"]), "step_rng_audit_sha256": _sha(verified["rng_path"]),
        "step_rng_rows": len(verified["rng_rows"]), "actual_sample_consumption_rows": len(verified["sample_rows"]),
        "pre_forward_guard_path": str(verified["pre_path"]), "pre_forward_guard_sha256": _sha(verified["pre_path"]),
        "pre_forward_guard_rows": len(verified["pre_rows"]),
        "online_identity_path": str(verified["pair_path"]), "online_identity_sha256": _sha(verified["pair_path"]),
        "online_identity_rows": len(verified["pair_rows"]),
        "checkpoint_receipts": verified["checkpoint_receipts"],
        "mode_audit_path": str(verified["mode_path"]), "mode_audit_sha256": _sha(verified["mode_path"]),
        "mode_restoration_rows": len(verified["mode_rows"]),
        "parity_sha256": _sha(verified["parity_path"]), "gradient_chain_sha256": _sha(verified["chain_path"]),
        "paired_identity_status": "REFERENCE_CAPTURED_ALL_2000" if doc["arm"] == "C" else "PASS_PRE_BACKWARD_ALL_2000",
        "paired_identity_sha256": _sha(verified["pair_path"]),
        "checkpoints": manifest.get("checkpoints"), "resume_after_step": verified["resume_after_step"],
        "selected_checkpoint": "NONE", "sim_admitted": False,
    }
    integrity_path = output / "pi0251_training_integrity.json"
    _write(integrity_path, result)
    state_path = output / "execution_state.json"
    state = _json(state_path)
    state.update({"status": "TRAINING_COMPLETE", "arm": doc["arm"], "family": doc["family"],
                  "training_integrity_sha256": _sha(integrity_path),
                  "selected_checkpoint": "NONE"})
    _replace(state_path, state)


def _require_checkpoint_receipts(doc: dict[str, Any], root: Path, resume: int) -> list[dict[str, Any]]:
    output = Path(doc["paths"]["output_root"])
    closed = []
    for step in CHECKPOINTS:
        archive_root = output if resume and step <= resume else root.parent
        receipt_path = archive_root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
        readback_path = archive_root / "hf_readbacks" / f"checkpoint-{step}.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"PI0251 {doc['arm']} checkpoint-{step} durable receipt missing")
        if not readback_path.is_file():
            raise RuntimeError(f"PI0251 {doc['arm']} checkpoint-{step} HF readback missing")
        receipt, readback = _json(receipt_path), _json(readback_path)
        record = receipt.get("source_record", {})
        inventory = record.get("inventory", {})
        if (receipt.get("status") != "DURABLE_VERIFIED" or receipt.get("family") != doc["family"]
                or receipt.get("checkpoint_step") != step
                or receipt.get("hf_manifest_commit_id") != readback.get("revision")
                or not inventory.get("tree_sha256")
                or readback.get("status") != "PASS_PT_FULL_HF_READBACK"
                or readback.get("source_tree_sha256") != inventory.get("tree_sha256")):
            raise RuntimeError(f"PI0251 {doc['arm']} checkpoint-{step} durable/readback identity mismatch")
        entries = {item["path"]: item for item in inventory.get("entries", [])}
        observed = {str(item.get("path", "")).split(f"/checkpoint-{step}/", 1)[-1]: item
                    for item in readback.get("files", []) if f"/checkpoint-{step}/" in str(item.get("path", ""))}
        if set(observed) != set(entries) or any(
                int(observed[name].get("bytes", -1)) != int(item["bytes"])
                or observed[name].get("sha256") != item["sha256"] for name, item in entries.items()):
            raise RuntimeError(f"PI0251 {doc['arm']} checkpoint-{step} HF readback differs from durable inventory")
        closed.append({"step": step, "receipt_sha256": _sha(receipt_path),
                       "readback_sha256": _sha(readback_path), "tree_sha256": inventory["tree_sha256"]})
    return closed


def _dispatch_lifecycle(args: argparse.Namespace, lifecycle: Any) -> Any:
    if args.hf_preflight:
        return lifecycle.hf_preflight(args)
    if args.train:
        return lifecycle.train(args)
    if args.archive_sidecar:
        if args.step not in CHECKPOINTS:
            raise RuntimeError(f"PI0251 sidecar step must be one of {CHECKPOINTS}")
        return lifecycle.archive_sidecar(args)
    if args.finalize:
        return lifecycle.finalize(args)
    raise RuntimeError("PI0251 lifecycle mode was not selected")


def _g1_cpu_selfcheck(output_path: Path, runtime_scripts: Path) -> dict[str, Any]:
    import tempfile
    from types import SimpleNamespace

    runtime_scripts = runtime_scripts.resolve()
    source_files = {name: runtime_scripts / name for name in
                    ("pi0247_u_rollout_overlay.py", "pi0245_physical_delta.py")}
    if any(not path.is_file() for path in source_files.values()):
        raise RuntimeError("PI0251 G1 CPU fixture lacks the frozen S4 overlay or physical decoder")
    sys.path.insert(0, str(runtime_scripts))
    rng_spec = importlib.util.spec_from_file_location("pi0251_g1_rng", Path(__file__).with_name("pi0251_step_rng.py"))
    if rng_spec is None or rng_spec.loader is None:
        raise RuntimeError("PI0251 G1 CPU fixture cannot load its step RNG source")
    rng = importlib.util.module_from_spec(rng_spec)
    rng_spec.loader.exec_module(rng)
    result = rng.cpu_self_check()
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
        temp_root = Path(temporary)
        class Routed:
            def __init__(self) -> None:
                self.calls = []
            def hf_preflight(self, args: Any) -> None: self.calls.append(("hf_preflight", args.arm, None))
            def train(self, args: Any) -> None: self.calls.append(("train", args.arm, None))
            def archive_sidecar(self, args: Any) -> None: self.calls.append(("archive_sidecar", args.arm, args.step))
            def finalize(self, args: Any) -> None: self.calls.append(("finalize", args.arm, None))
        routed = Routed()
        for arm in FAMILIES:
            for step in CHECKPOINTS:
                args = SimpleNamespace(hf_preflight=False, train=False, archive_sidecar=True,
                                       finalize=False, arm=arm, step=step)
                _dispatch_lifecycle(args, routed)
            args = SimpleNamespace(hf_preflight=False, train=False, archive_sidecar=False,
                                   finalize=True, arm=arm, step=None)
            _dispatch_lifecycle(args, routed)
        expected_routes = []
        for arm in FAMILIES:
            expected_routes.extend(("archive_sidecar", arm, step) for step in CHECKPOINTS)
            expected_routes.append(("finalize", arm, None))
        if routed.calls != expected_routes:
            raise AssertionError("PI0251 sidecar/finalizer route changed arm or checkpoint identity")

        missing_doc = {"arm": "C", "family": FAMILIES["C"], "paths": {"output_root": str(temp_root / "C")}}
        missing_root = Path(missing_doc["paths"]["output_root"]) / "training"
        missing_root.mkdir(parents=True)
        try:
            _require_checkpoint_receipts(missing_doc, missing_root, 0)
        except RuntimeError as error:
            if str(error) != "PI0251 C checkpoint-1000 durable receipt missing":
                raise AssertionError(f"PI0251 missing-receipt status changed: {error}") from error
        else:
            raise AssertionError("PI0251 finalizer accepted a missing checkpoint receipt")

        for step in CHECKPOINTS:
            receipt_root = Path(missing_doc["paths"]["output_root"])
            (receipt_root / "durable_receipts").mkdir(parents=True, exist_ok=True)
            (receipt_root / "hf_readbacks").mkdir(parents=True, exist_ok=True)
            entry = {"path": "trainer_state.json", "bytes": 1, "sha256": "sha"}
            tree = f"tree-{step}"
            _write(receipt_root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json", {
                "status": "DURABLE_VERIFIED", "family": FAMILIES["C"], "checkpoint_step": step,
                "hf_manifest_commit_id": f"revision-{step}", "source_record": {"inventory": {
                    "tree_sha256": tree, "entries": [entry]}},
            })
            _write(receipt_root / "hf_readbacks" / f"checkpoint-{step}.json", {
                "status": "PASS_PT_FULL_HF_READBACK", "revision": f"revision-{step}",
                "source_tree_sha256": tree, "files": [{"path": f"arms/C/checkpoint-{step}/trainer_state.json",
                                                        "bytes": 1, "sha256": "sha"}],
            })
        if len(_require_checkpoint_receipts(missing_doc, missing_root, 0)) != len(CHECKPOINTS):
            raise AssertionError("PI0251 synthetic checkpoint receipts did not close")

    result.update({"status": "PASS_PI0251_G1_CPU_FIXTURES",
                   "trainer_invoked_pre_forward_guard": True,
                   "online_guard_rejects_source_key_tau_epsilon": True,
                   "cpu_runtime_sources": {name: {"path": str(path), "sha256": _sha(path)}
                                           for name, path in source_files.items()},
                   "finalizer_sidecar_routes": "PASS",
                   "sidecar_steps": list(CHECKPOINTS), "finalizer_arms": list(FAMILIES)})
    _write(output_path, result)
    return result


def _materialize_x2000_view(
    root: Path, repo_id: str, entries: dict[str, dict[str, Any]],
    processor_root: Path, processor_files: dict[str, str],
) -> list[dict[str, Any]]:
    import huggingface_hub

    names = ("config.json", "model.safetensors.index.json",
             "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        entry = entries.get(name)
        if entry is None:
            raise RuntimeError(f"PI0251 X2000 archive is missing model-view file {name}")
        local = root / name
        if local.exists() and (local.stat().st_size != int(entry["bytes"]) or _sha(local) != entry["sha256"]):
            raise RuntimeError(f"PI0251 refusing to replace conflicting X2000 view file: {local}")
    missing = [name for name in names if not (root / name).is_file()]
    for name in missing:
        huggingface_hub.hf_hub_download(
            repo_id=repo_id, filename=name, revision=REVISION, token=False, local_dir=str(root),
        )
    files = []
    for name in names:
        local, entry = root / name, entries[name]
        if not local.is_file() or local.stat().st_size != int(entry["bytes"]) or _sha(local) != entry["sha256"]:
            raise RuntimeError(f"PI0251 anonymous pinned X2000 download failed readback: {name}")
        files.append({"path": name, "bytes": local.stat().st_size, "sha256": _sha(local)})
    for name, expected in processor_files.items():
        source, local = processor_root / name, root / name
        if not source.is_file() or _sha(source) != expected:
            raise RuntimeError(f"PI0251 frozen C_T processor source changed: {name}")
        if local.exists() and _sha(local) != expected:
            raise RuntimeError(f"PI0251 refusing to replace conflicting C_T view file: {local}")
        if not local.exists():
            shutil.copy2(source, local)
        if not local.is_file() or _sha(local) != expected:
            raise RuntimeError(f"PI0251 copied C_T view file differs from its frozen hash: {name}")
        files.append({"path": name, "bytes": local.stat().st_size, "sha256": _sha(local)})
    return files


def prepare(args: argparse.Namespace) -> None:
    task_root, exec_root = args.task_root.resolve(), args.execution_root.resolve()
    parent_path = args.parent_card.resolve()
    parent = _json(parent_path)
    if parent.get("arm") != "X" or parent.get("family") != "PI0249_X_XYZ4_PRIORITY_M64_2K_V1" or parent.get("selected_checkpoint") != "NONE":
        raise RuntimeError("PI0251 must bind the sealed PI0249 X card with selected_checkpoint=NONE")
    parent_seal = _json(parent_path.with_name("run_card.seal.json"))
    parent_sha = _sha(parent_path)
    if parent_seal.get("sha256") != parent_sha:
        raise RuntimeError("PI0249 X run-card seal mismatch")
    output_x = Path(parent["paths"]["output_root"])
    x_integrity_path = output_x / "pi0249_training_integrity.json"
    x_integrity = _json(x_integrity_path)
    if (x_integrity.get("status") != "PASS_PI0249_KX_TRAINING_INTEGRITY"
            or x_integrity.get("arm") != "X" or x_integrity.get("run_card_sha256") != parent_sha
            or x_integrity.get("optimizer_updates") != STEPS
            or x_integrity.get("selected_checkpoint") != "NONE"):
        raise RuntimeError("PI0251 fixed X2000 initializer lacks the sealed parent integrity PASS")

    g0_path = args.g0_index.resolve()
    g0 = _json(g0_path)
    classification = g0.get("pair_identity_classification", {})
    x_screen = next((row for row in g0.get("screen_artifacts", []) if row.get("arm") == "X2000"), None)
    if (g0.get("status") != "PASS_WITH_SOURCE_COPY_VARIANCE_NOTED"
            or classification.get("classification") != "OBSERVED_POSTHOC_CAPTURED_IDENTITIES"
            or classification.get("online_pre_backward_proof") is not False
            or x_screen is None or x_screen.get("checkpoint_revision") != REVISION
            or x_screen.get("checkpoint_tree_sha256") != TREE_SHA256
            or x_screen.get("selected_checkpoint") != "NONE"):
        raise RuntimeError("PI0251 G0 source index does not bind the preserved X2000/posthoc boundary")

    matches = []
    for receipt_path in sorted(output_x.glob("recovery_attempts/*/durable_receipts/checkpoint-2000.durable_receipt.json")):
        receipt = _json(receipt_path)
        attempt = receipt_path.parent.parent
        readback_path = attempt / "hf_readbacks/checkpoint-2000.json"
        if not readback_path.is_file():
            continue
        readback = _json(readback_path)
        inventory = receipt.get("source_record", {}).get("inventory", {})
        if (receipt.get("status") == "DURABLE_VERIFIED" and receipt.get("checkpoint_step") == 2000
                and receipt.get("family") == "PI0249_X_XYZ4_PRIORITY_M64_2K_V1"
                and receipt.get("hf_manifest_commit_id") == REVISION
                and inventory.get("tree_sha256") == TREE_SHA256
                and readback.get("status") == "PASS_PT_FULL_HF_READBACK"
                and readback.get("revision") == REVISION
                and readback.get("source_tree_sha256") == TREE_SHA256):
            matches.append((receipt_path, readback_path, receipt, readback))
    if len(matches) != 1:
        raise RuntimeError(f"PI0251 expected one full X2000 durable/readback pair, found {len(matches)}")
    receipt_path, readback_path, receipt, readback = matches[0]
    receipt_sha, readback_sha, integrity_sha = _sha(receipt_path), _sha(readback_path), _sha(x_integrity_path)
    record = receipt["source_record"]
    entries = {row["path"]: row for row in record["inventory"]["entries"]}
    observed = {str(row.get("path", "")).split("/checkpoint-2000/", 1)[-1]: row
                for row in readback.get("files", []) if "/checkpoint-2000/" in str(row.get("path", ""))}
    if set(observed) != set(entries) or any(
            int(observed[name].get("bytes", -1)) != int(entry["bytes"])
            or observed[name].get("sha256") != entry["sha256"] for name, entry in entries.items()):
        raise RuntimeError("PI0251 X2000 full HF readback differs from its durable Trainer inventory")
    if x_screen.get("hf_readback_receipt_sha256") != readback_sha:
        raise RuntimeError("PI0251 G0 source index and X2000 HF readback receipt differ")

    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "logs").mkdir(exist_ok=True)
    exec_root.mkdir(parents=True, exist_ok=True)
    for arm in FAMILIES:
        out = task_root / arm
        if (out / "run_card.json").exists() or (out / "run_card.seal.json").exists():
            raise RuntimeError(f"PI0251 sealed {arm} card already exists; refusing to rewrite it")
        out.mkdir(parents=True, exist_ok=True)
        (out / "logs").mkdir(exist_ok=True)
    evidence_root = task_root / "evidence"
    evidence_root.mkdir(exist_ok=True)
    g0_copy = evidence_root / "PI0251_G0_SOURCE_INDEX_20260928.json"
    if g0_path != g0_copy:
        if g0_copy.exists() and _sha(g0_copy) != _sha(g0_path):
            raise RuntimeError("PI0251 task evidence already contains a different G0 source index")
        if not g0_copy.exists():
            shutil.copy2(g0_path, g0_copy)
    g1_path = args.g1_receipt.resolve()
    g1 = _json(g1_path)
    if (g1.get("status") != "PASS_PI0251_G1_CPU_FIXTURES"
            or g1.get("trainer_invoked_pre_forward_guard") is not True
            or g1.get("online_guard_rejects_source_key_tau_epsilon") is not True
            or g1.get("finalizer_sidecar_routes") != "PASS"):
        raise RuntimeError("PI0251 G1 CPU numerical/guard/lifecycle fixtures did not pass")

    view_root = task_root / "initializer_X2000"
    processor_bindings = parent["source"]["processor_bindings"]
    model_files = _materialize_x2000_view(
        view_root, readback["repo_id"], entries, Path(parent["paths"]["base_model_path"]),
        processor_bindings["view_files"],
    )
    identity = {key: value for key, value in processor_bindings.items() if key != "view_files"}
    rotation_path = Path(parent["source"]["rotation_check_path"])
    rotation = _json(rotation_path)
    if (_sha(rotation_path) != parent["source"]["rotation_check_sha256"]
            or rotation.get("status") != "PASS_CPU_DECODER_AND_GRADIENT"
            or _processor_identity(rotation) != identity):
        raise RuntimeError("PI0251 frozen C_T processor/rotation receipt changed")
    for name, expected in processor_bindings["view_files"].items():
        local = view_root / name
        if not local.is_file() or _sha(local) != expected:
            raise RuntimeError(f"PI0251 X2000 processor view differs from frozen C_T: {name}")
    view_receipt = task_root / "initializer_X2000_view_receipt.json"
    _write(view_receipt, {"status": "PASS_PI0251_X2000_CT_VIEW", "revision": REVISION,
                          "checkpoint_tree_sha256": TREE_SHA256, "model_files": model_files,
                          "source_durable_receipt_sha256": receipt_sha,
                          "source_hf_readback_sha256": readback_sha,
                          "processor_identity": identity})

    schedule_ref = copy.deepcopy(parent["schedule"])
    schedule_rows = _json(Path(schedule_ref["path"]))["rows"]
    if len(schedule_rows) != STEPS or _sha(Path(schedule_ref["path"])) != schedule_ref["sha256"]:
        raise RuntimeError("PI0251 fixed 64/2000 exposure schedule changed")
    rng_path = exec_root / "pi0251_step_rng.py"
    rng_spec = importlib.util.spec_from_file_location("pi0251_rng_prepare", rng_path)
    if rng_spec is None or rng_spec.loader is None:
        raise RuntimeError("PI0251 step-addressed RNG source is missing")
    rng_module = importlib.util.module_from_spec(rng_spec)
    rng_spec.loader.exec_module(rng_module)
    key_table_path = task_root / "pi0251_step_key_table.json"
    _write(key_table_path, {"schema": "PI0251_STEP_KEY_TABLE_V1", "seed": SEED,
                            "schedule_sha256": schedule_ref["sha256"],
                            "rows": rng_module.build_key_table(schedule_rows)})
    exec_files = {
        "pi0251_ce_train.py": exec_root / "pi0251_ce_train.py",
        "pi0251_step_rng.py": rng_path,
        "pi0251_ce_train_h200.pbs": Path(args.pbs_script).resolve(),
        "pi0248_cr_train.py": Path(parent["paths"]["legacy_execution_driver_root"]) / "pi0248_cr_train.py",
        "pi0248_rotation_cpu.py": Path(parent["paths"]["legacy_execution_driver_root"]) / "pi0248_rotation_cpu.py",
    }
    if any(not path.is_file() for path in exec_files.values()):
        raise RuntimeError("PI0251 execution sources are incomplete before card preparation")
    code_hashes = {name: _sha(path) for name, path in exec_files.items()}

    global life, helper
    life = _load_life(parent["paths"])
    helper = _load_helper(parent["paths"])
    life.STEPS, life.CHECKPOINTS = STEPS, CHECKPOINTS
    base_paths = dict(parent["paths"])
    legacy_root = Path(parent["paths"]["legacy_execution_driver_root"])
    base_paths.update({
        "output_base": str(task_root), "base_model_path": str(view_root),
        "execution_driver_root": str(exec_root), "execution_driver": str(exec_root / "pi0251_ce_train.py"),
        "pbs_script": str(exec_files["pi0251_ce_train_h200.pbs"]),
        "initializer_view_receipt_path": str(view_receipt),
    })
    initializer = {
        "path": str(view_root), "screen_revision": REVISION, "screen_manifest_sha256": readback_sha,
        "full_checkpoint_tree_sha256": TREE_SHA256, "full_checkpoint_bytes": int(record["inventory"]["total_bytes"]),
        "durable_receipt_path": str(receipt_path), "durable_receipt_sha256": receipt_sha,
        "model_files": model_files, "source_family": receipt["family"],
    }
    source_base = copy.deepcopy(parent["source"])
    source_base.update({
        "execution_code_commit": args.execution_commit,
        "legacy_execution_code_commit": parent["source"]["legacy_execution_code_commit"],
        "custom_execution_code_sha256": code_hashes,
        "legacy_execution_driver_root": str(legacy_root),
        "initializer_view_receipt_path": str(view_receipt),
        "initializer_view_receipt_sha256": _sha(view_receipt),
        "x2000_parent_run_card_path": str(parent_path), "x2000_parent_run_card_sha256": parent_sha,
        "x2000_parent_run_card_seal_sha256": parent_seal["sha256"],
        "x2000_integrity_path": str(x_integrity_path), "x2000_integrity_sha256": integrity_sha,
        "x2000_durable_receipt_path": str(receipt_path), "x2000_durable_receipt_sha256": receipt_sha,
        "x2000_hf_readback_path": str(readback_path), "x2000_hf_readback_sha256": readback_sha,
        "processor_bindings": processor_bindings,
        "rotation_check_path": str(rotation_path), "rotation_check_sha256": _sha(rotation_path),
        "g0_source_index_path": str(g0_copy), "g0_source_index_sha256": _sha(g0_copy),
        "g1_cpu_selfcheck_path": str(g1_path), "g1_cpu_selfcheck_sha256": _sha(g1_path),
        "step_key_table_path": str(key_table_path), "step_key_table_sha256": _sha(key_table_path),
        "pair_recipe_binding_path": str(task_root / "pi0251_pair_recipe_binding.json"),
        "pair_run_binding_path": str(task_root / "pi0251_pair_run_binding.json"),
    })
    recipe = {
        "schema": "PI0251_CE_PAIR_RECIPE_BINDING_V1", "authorization": AUTH,
        "canonical_issue_comment": CANONICAL, "pointer_pr_comment": POINTER,
        "start_revision": REVISION, "start_tree_sha256": TREE_SHA256,
        "schedule_sha256": schedule_ref["sha256"], "step_key_table_sha256": _sha(key_table_path),
        "runtime_id": "S4", "optimizer_steps_each": STEPS, "new_adamw_each_arm": True,
        "learning_rate": 1e-5, "checkpoint_steps": list(CHECKPOINTS),
        "losses": {arm: {"family": FAMILIES[arm], "loss": LOSSES[arm],
                         "coefficients": COEFFICIENTS[arm]} for arm in FAMILIES},
        "shared_noise_rule": "PI0251_STEP_ADDRESSED_TORCH_RNG_V1",
        "pairing": ["source", "key", "model_input_sha256", "native_tau_identity",
                    "native_epsilon", "stock_initial_noise"],
        "pairing_excludes": ["stock_vl_features", "model_output", "gradient"],
    }
    recipe_path = Path(source_base["pair_recipe_binding_path"])
    _write(recipe_path, recipe)
    source_base["pair_recipe_binding_sha256"] = _sha(recipe_path)

    initializer["model_files"] = model_files
    card_paths = {}
    configs = {}
    parent_config = copy.deepcopy(parent["resolved_config"])
    parent_config["start_from_checkpoint"] = str(view_root)
    for arm in FAMILIES:
        out = task_root / arm
        paths = {**base_paths, "output_root": str(out)}
        fixed = life.load_module(Path(paths["code_root"]) / "scripts/pi0241_fixed16_learning.py", f"pi0251_fixed_{arm}")
        fixed.MAX_STEPS, fixed.CHECKPOINT_STEPS = STEPS, CHECKPOINTS
        runtime = Path(paths["stage10_root"]) / "runtime"
        sys.path[:0] = [str(runtime), paths["training_runtime_root"], str(Path(paths["code_root"]) / "scripts"),
                        str(Path(paths["code_root"]) / "src"), paths["groot_source"]]
        fixed.bind_runtime(life._card_with_p_runtime_paths({**parent, "paths": paths, "initializer": initializer}))
        train_core = life.load_module(Path(paths["training_runtime_root"]) / "pi0246_gate_pass_train.py", f"pi0251_core_{arm}")
        train_core.STEPS, train_core.CHECKPOINTS = STEPS, CHECKPOINTS
        _, _, summary = train_core.build_runtime_config(paths, out / "training", fixed)
        life._check_summary(summary, parent_config, str(view_root))
        configs[arm] = summary
        card_paths[arm] = out / "run_card.json"

    for arm in FAMILIES:
        out = task_root / arm
        paths = {**base_paths, "output_root": str(out)}
        archive = {**copy.deepcopy(parent["archive"]),
                   "path_prefix": f"arms/PI0251/xyz-excess-risk-20260928/{arm}",
                   "checkpoint_steps": list(CHECKPOINTS), "full_trainer_checkpoints": True}
        doc = copy.deepcopy(parent)
        doc.update({
            "schema": "PI0251_CE_RUN_CARD_V1", "status": "SEALED_READY_NOT_STARTED",
            "authorization": AUTH, "execution_authorization": AUTH,
            "canonical_issue_comment": CANONICAL, "pointer_pr_comment": POINTER,
            "arm": arm, "family": FAMILIES[arm], "loss": LOSSES[arm],
            "loss_coefficients": COEFFICIENTS[arm], "seed": SEED, "selected_checkpoint": "NONE",
            "d1_runtime_id": "S4",
            "stage": {"name": "M64_CE_CONTINUATION", "steps": STEPS,
                      "checkpoint_steps": list(CHECKPOINTS), "source_revision": REVISION},
            "initializer": copy.deepcopy(initializer), "paths": paths, "archive": archive,
            "schedule": schedule_ref, "resolved_config": configs[arm], "source": copy.deepcopy(source_base),
            "loss_contract": {
                "objective": LOSSES[arm],
                "generation": "complete pure-noise label-free S4 rollout before physical losses",
                "ell_pos": "whole-window physical XYZ mean squared Euclidean error / (0.05 m)^2",
                "ell_excess": "relu(ell_pos - 0.49)" if arm == "E" else "not applied",
                "ell_excess_lambda": 0.4 if arm == "E" else 0.0,
                "horizon": 40, "physical_dt_s": 0.05, "euler_steps": 4,
                "backpropagate_through_all_euler_steps": True,
                "expert_prefix_input": False, "active_action_smoothing": False,
                "position_units": "m", "delta_dt_s": 0.05, "rotation_theta_ref_rad": 0.05,
                "S4_full_gradient": True,
            },
            "optimizer": {
                "steps": STEPS, "checkpoint_steps": list(CHECKPOINTS), "learning_rate": 1e-5,
                "weight_decay": 1e-5, "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8,
                "max_grad_norm": 1.0, "batch_size": 1, "gradient_accumulation_steps": 1,
                "workers": 0, "scheduler": "cosine", "warmup_steps": 10,
                "optimizer_state": "fresh AdamW; no X optimizer resume",
            },
            "paired_arm_root": str(task_root / ("E" if arm == "C" else "C")),
        })
        if arm == "E":
            doc["source"]["peer_run_card_path"] = str(card_paths["C"])
            doc["source"]["peer_run_card_sha256"] = _sha(card_paths["C"])
        card_path = card_paths[arm]
        _write(card_path, doc)
        _write(out / "run_card.seal.json", {"sha256": _sha(card_path)})
        _write(out / "preparation.json", {
            "status": "READY_NOT_STARTED", "selected_checkpoint": "NONE",
            "run_card_sha256": _sha(card_path), "config": configs[arm],
            "schedule_sha256": schedule_ref["sha256"], "initializer_tree_sha256": TREE_SHA256,
        })
    cards = {arm: {"path": str(card_paths[arm]), "sha256": _sha(card_paths[arm])} for arm in FAMILIES}
    _write(task_root / "pi0251_pair_run_binding.json", {
        "schema": "PI0251_CE_PAIR_RUN_BINDING_V1", "status": "SEALED",
        "pair_recipe_binding_sha256": _sha(recipe_path), "cards": cards, "selected_checkpoint": "NONE",
    })
    print(json.dumps({"status": "READY_NOT_STARTED", "cards": cards,
                      "key_table_sha256": _sha(key_table_path), "initializer_tree_sha256": TREE_SHA256}, indent=2))

def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--hf-preflight", action="store_true")
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--archive-sidecar", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    modes.add_argument("--g1-selfcheck", action="store_true")
    parser.add_argument("--parent-card", type=Path)
    parser.add_argument("--task-root", type=Path)
    parser.add_argument("--execution-root", type=Path)
    parser.add_argument("--execution-commit")
    parser.add_argument("--pbs-script", type=Path)
    parser.add_argument("--g0-index", type=Path)
    parser.add_argument("--g1-receipt", type=Path)
    parser.add_argument("--g1-output", type=Path)
    parser.add_argument("--g1-runtime-scripts", type=Path)
    parser.add_argument("--run-card", type=Path)
    parser.add_argument("--arm", choices=("C", "E"))
    parser.add_argument("--resume-after-step", type=int, choices=(0, 1000), default=0)
    parser.add_argument("--step", type=int, choices=(1000, 2000))
    parser.add_argument("--archive-root", type=Path)
    args = parser.parse_args()
    if args.g1_selfcheck:
        if args.g1_output is None or args.g1_runtime_scripts is None:
            parser.error("--g1-selfcheck requires --g1-runtime-scripts and --g1-output")
        print(json.dumps(_g1_cpu_selfcheck(args.g1_output, args.g1_runtime_scripts), indent=2, sort_keys=True))
        return 0
    if args.prepare:
        required = (args.parent_card, args.task_root, args.execution_root, args.execution_commit,
                    args.pbs_script, args.g0_index, args.g1_receipt)
        if any(value is None for value in required):
            parser.error("--prepare requires parent, task/execution roots, commit, PBS, G0 index, and G1 receipt")
        prepare(args)
        return 0
    if args.run_card is None or args.arm is None:
        parser.error("execution mode requires --run-card and --arm")
    if args.train and args.resume_after_step not in (0, 1000):
        parser.error("PI0251 exact resume cursor must be 0 or 1000")
    if args.archive_sidecar and args.step is None:
        parser.error("--archive-sidecar requires --step")
    card_doc = _json(args.run_card)
    global life
    life = _load_life(card_doc["paths"])
    _install_adapters(args.run_card, args.arm, args.resume_after_step)
    _validate_initializer(card)
    _verify_code(card, Path(card["paths"]["repo_code_root"]), Path(card["paths"]["code_root"]),
                 Path(card["paths"]["training_runtime_root"]))
    life._verify_runtime_bindings(card["paths"], card["source"]["runtime_bindings"])
    _dispatch_lifecycle(args, life)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
