#!/usr/bin/env python3
"""One authorized FULL stage, using the existing G objective and durable lifecycle."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pi0251_ce_train as ce
import pi0253_step_rng as rng

AUTH = "PI0253_LIVE_BRIDGE_TO_FULL_SUPPORT_GO"
RELEASE = "82554098fb55b39ceca4b955252f8952c91c0dc0"
FAMILY = "PI0253_FULL_G_SUPPORT_V1"
REVISION = "efecea485ae77000599d19171b02491f8d514f42"
TREE = "cfbfeb54bea2b5ac266047dfd764cfa6b041a8a77b9688737ea70b56935d5226"
STEPS, SAVES = 10000, (2500, 5000, 7500, 10000)
COEFF = {"ell_pos": .4, "ell_delta": .1, "ell_rot": .1, "ell_speed_margin": .1}
ROOT = Path("/scratch/e1192548/Projects/go2-arx5/experiments/PI0253")
PARENT = ROOT.parent / "PI0252/r1-speed-margin-20260928/G/run_card.json"
ce.STEPS, ce.CHECKPOINTS, ce.REVISION, ce.TREE_SHA256 = STEPS, SAVES, REVISION, TREE


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_card(path, arm):
    doc = ce._json(path)
    require(ce._json(path.with_name("run_card.seal.json"))["sha256"] == ce._sha(path), "FULL card seal")
    require(doc["schema"] == "PI0253_FULL_RUN_CARD_V1" and doc["protocol"] == "PI0253"
            and doc["authorization"] == doc["execution_authorization"] == AUTH
            and doc["authority_commit"] == RELEASE and arm == doc["arm"] == "G"
            and doc["family"] == FAMILY and doc["selected_checkpoint"] == "NONE", "FULL authority/family")
    require(doc["stage"]["steps"] == STEPS and tuple(doc["stage"]["checkpoint_steps"]) == SAVES
            and doc["cohort"]["count"] == 9792 and doc["schedule"]["count"] == STEPS
            and doc["loss_coefficients"] == COEFF and doc["d1_runtime_id"] == "S4"
            and doc["source"]["paired_arms"] == ["G"] and doc["source"]["intervention_arm"] is None,
            "FULL fixed recipe")
    for key in ("cohort", "schedule"):
        require(ce._sha(Path(doc[key]["path"])) == doc[key]["sha256"], f"FULL {key} identity")
    for name, item in doc["source"]["bound_files"].items():
        require(ce._sha(Path(item["path"])) == item["sha256"], f"FULL bound file changed: {name}")
    for name, digest in doc["source"]["custom_execution_code_sha256"].items():
        require(ce._sha(Path(doc["paths"]["execution_driver_root"]) / name) == digest,
                f"FULL execution source changed: {name}")
    return doc


def validate_initializer(doc):
    src, init = doc["source"], doc["initializer"]
    parent = ce._json(Path(src["bound_files"]["G_card"]["path"]))
    integrity = ce._json(Path(src["bound_files"]["G_integrity"]["path"]))
    require(parent["arm"] == "G" and integrity["status"] == "PASS_PI0252_G_TRAINING_INTEGRITY"
            and integrity["optimizer_updates"] == 2000 and integrity["run_card_sha256"] == src["bound_files"]["G_card"]["sha256"],
            "G training integrity")
    require(ce._json(Path(src["bound_files"]["G_card"]["path"]).with_name("run_card.seal.json"))["sha256"]
            == integrity["run_card_sha256"], "G parent seal")
    receipt = ce._json(Path(init["durable_receipt_path"]))
    readback = ce._json(Path(src["bound_files"]["G_readback"]["path"]))
    inventory = receipt["source_record"]["inventory"]
    require(receipt["status"] == "DURABLE_VERIFIED" and receipt["checkpoint_step"] == 2000
            and receipt["hf_manifest_commit_id"] == readback["revision"] == REVISION
            and inventory["tree_sha256"] == readback["source_tree_sha256"] == TREE
            and readback["status"] == "PASS_PT_FULL_HF_READBACK", "G pinned full checkpoint")
    expected = {r["path"]: r for r in inventory["entries"]}
    observed = {r["path"].split("/checkpoint-2000/", 1)[-1]: r for r in readback["files"]}
    require(set(expected) == set(observed), "G full readback coverage")
    for name, row in expected.items():
        require(row["bytes"] == observed[name]["bytes"] and row["sha256"] == observed[name]["sha256"], "G full readback bytes")
    for row in init["model_files"]:
        local = Path(init["path"]) / row["path"]
        require(local.stat().st_size == row["bytes"] and ce._sha(local) == row["sha256"], f"G/CT initializer: {local.name}")
    bindings = src["processor_bindings"]
    require(bindings == parent["source"]["processor_bindings"], "G frozen C_T binding")
    for name, key in (("stats.json", "NEW_EMBODIMENT_stats_sha256"), ("relative_stats.json", "relative_stats_sha256")):
        require(ce._sha(Path(doc["paths"]["train_dataset"]) / "meta" / name) == bindings[key], "TRAIN statistics")
    legacy = copy.deepcopy(doc)
    legacy["schema"] = "PI0247_M64_RUN_CARD_V1"
    ce.base_validate_initializer(legacy)


def full_config(core):
    original = core.build_runtime_config

    def build(*args, **kwargs):
        pi0228, config, summary = original(*args, **kwargs)
        config.training.save_steps = 2500
        summary.update(save_steps=2500, save_step_set=list(SAVES))
        require(config.training.max_steps == STEPS and config.training.warmup_steps == 10
                and summary["learning_rate"] == 1e-5 and summary["scheduler"] == "cosine", "FULL config endpoint")
        return pi0228, config, summary

    core.build_runtime_config = build


def capacity(doc, *, resume=False):
    output = Path(doc["paths"]["output_root"]).resolve()
    require(resume or not (output / "training").exists(), "fresh FULL output already exists")
    device = output.stat().st_dev
    probes = [(str(p), os.statvfs(p).f_bavail * os.statvfs(p).f_frsize)
              for p in (output, *output.parents) if p.exists() and p.stat().st_dev == device]
    path, free = min(probes, key=lambda row: row[1])
    needed = int(doc["initializer"]["full_checkpoint_bytes"]) + (10 << 30)
    require(free >= needed, f"CHECKPOINT_STORAGE_BLOCKED free={free} required={needed} actual_fileset={path}")
    return dict(persistent_filesystem=str(output), persistent_capacity_probe_path=path,
                persistent_free_bytes_before_training=free, required_persistent_bytes=needed,
                quota_headroom="UNKNOWN", reserve_bytes=10 << 30, checkpoint_steps=list(SAVES),
                checkpoint_storage_mode="HF_PUBLIC_ROLLING_SINGLE_CHECKPOINT", tmpfs_is_not_a_checkpoint_source=True)


def route_sidecar(command, doc, path):
    routed = original_route(command, doc, path)
    if routed != command:
        routed = list(routed)
        routed[1] = str(Path(doc["paths"]["execution_driver_root"]) / "pi0253_full_train.py")
    return routed


original_route = ce._route_archive_sidecar
ce._route_archive_sidecar = route_sidecar
ce._verify_card = verify_card
ce._validate_initializer = validate_initializer


def install(path, resume):
    doc, life = ce._install_adapters(path, "G", resume)
    ce.helper.STEPS, ce.helper.CHECKPOINTS = STEPS, SAVES
    life._validate_storage = capacity
    original_load = life.load_module

    def load(source, name):
        module = original_load(source, name)
        if source.name == "pi0246_gate_pass_train.py":
            full_config(module)
        if source.name == "pi0247_u_rollout_overlay.py" and not resume:
            original_install = module.install

            def inherited_install(*args, **kwargs):
                audit = original_install(*args, **kwargs)
                parity_source = Path(doc["source"]["bound_files"]["G_parity"]["path"])
                parity = ce._json(parity_source)
                require(parity["status"] == "PASS_PI0247_U_S4_PARITY" and parity["gate_decision_equal"]
                        and parity["gate_issues_equal"] and parity["expert_action_in_condition"] is False,
                        "inherited G sampler parity")
                audit["parity"] = parity
                audit["parity_inherited_sha256"] = ce._sha(parity_source)
                audit["parity_nonselection_rollouts"] = 0
                ce._write(kwargs["parity_path"], parity)
                return audit

            module.install = inherited_install
        return module

    life.load_module = load
    original_archive = life._install_hf_rolling_archive

    def archives(card, persistence, card_path, core):
        original_archive(card, persistence, card_path, core)
        from transformers import TrainerCallback
        from gr00t.experiment import experiment as experiment_module

        class ObservationAndBarrier(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if int(state.global_step) == 1000:
                    capacity(card, resume=True)
                    target = Path(args.output_dir).parent / "model-only-1000"
                    require(not target.exists(), "first-valid 1000 snapshot exists; repair its archive separately")
                    kwargs["model"].save_pretrained(target, safe_serialization=True)
                    for name in card["source"]["processor_bindings"]["view_files"]:
                        shutil.copy2(Path(card["initializer"]["path"]) / name, target / name)
                    result = subprocess.run([sys.executable, str(Path(__file__)), "--archive-observation", "--run-card", str(card_path)],
                                            env={**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})
                    require(result.returncode == 0, "model-only 1000 retained on persistent scratch; archive failed")
                return control

            def on_save(self, args, state, control, **kwargs):
                if int(state.global_step) == 2500 and not resume:
                    require((Path(args.output_dir).parent / "durable_receipts/checkpoint-2500.durable_receipt.json").is_file(),
                            "2500 barrier requires a durable full checkpoint")
                    control.should_training_stop = True
                    ce._write(Path(card["paths"]["output_root"]) / "barrier_wait.json",
                              dict(schema="PI0253_FIXED_2500_BARRIER_WAIT_V1", status="WAIT_FIXED_2500_BARRIER",
                                   run_card_sha256=ce._sha(card_path), canonical_updates=2500, scheduler_horizon=STEPS))
                return control

        base = experiment_module.Gr00tTrainer

        class FullTrainer(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.add_callback(ObservationAndBarrier())

        experiment_module.Gr00tTrainer = FullTrainer

    life._install_hf_rolling_archive = archives
    original_finalize = life.finalize

    def finalize(args):
        output = Path(doc["paths"]["output_root"])
        if args.train and not resume and (output / "barrier_wait.json").is_file():
            state = ce._json(output / "execution_state.json")
            state.update(status="WAIT_FIXED_2500_BARRIER", canonical_updates=2500)
            ce._replace(output / "execution_state.json", state)
            return
        original_finalize(args)

    life.finalize = finalize
    original_integrity = ce._write_integrity

    def integrity(card, card_path, verified):
        original_integrity(card, card_path, verified)
        target = Path(card["paths"]["output_root"]) / "pi0253_training_integrity.json"
        receipt = ce._json(target)
        receipt.pop("paired_identity_status", None)
        receipt.pop("paired_identity_sha256", None)
        receipt.update(actual_random_capture_status=f"REFERENCE_CAPTURED_ALL_{STEPS}",
                       paired_arm_comparison="NOT_APPLICABLE_SINGLE_ARM",
                       new_nonselection_parity_rollouts=0,
                       inherited_parity_sha256=card["source"]["bound_files"]["G_parity"]["sha256"])
        ce._replace(target, receipt)
        state_path = target.parent / "execution_state.json"
        state = ce._json(state_path)
        state["training_integrity_sha256"] = ce._sha(target)
        ce._replace(state_path, state)

    ce._write_integrity = integrity
    return doc, life


def archive_observation(path):
    """Fixed first-valid snapshot upload/readback; never load a model or optimizer."""
    import requests
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_url
    doc = verify_card(path, "G")
    out = Path(doc["paths"]["output_root"])
    receipt_path, intent_path = out / "model-only-1000.hf_readback.json", out / "model-only-1000.upload.json"
    if receipt_path.exists():
        receipt = ce._json(receipt_path)
        require(receipt["status"] == "PASS_FULL_FILE_READBACK" and receipt["run_card_sha256"] == ce._sha(path), "1000 receipt")
        return
    source = out / "model-only-1000"
    files = [dict(path=p.relative_to(source).as_posix(), bytes=p.stat().st_size, sha256=ce._sha(p))
             for p in sorted(source.rglob("*")) if p.is_file()]
    require(files and not any(r["path"] in {"optimizer.pt", "scheduler.pt", "trainer_state.json"} for r in files),
            "1000 must be model-only")
    token = Path(doc["archive"]["token_file"]).read_text().strip()
    api = HfApi(token=token)
    repo, prefix = doc["archive"]["repo_id"], doc["archive"]["path_prefix"] + "/model-only-1000"
    require(api.model_info(repo).private is False, "HF must be public")
    if intent_path.exists():
        intent = ce._json(intent_path)
        require(intent["files"] == files and intent["run_card_sha256"] == ce._sha(path), "1000 upload identity")
        revision = intent["revision"]
    else:
        revision = api.create_commit(repo_id=repo, repo_type="model", operations=[
            CommitOperationAdd(path_in_repo=prefix + "/" + r["path"], path_or_fileobj=str(source / r["path"])) for r in files],
            commit_message="PI0253 FULL step1000 model-only observation; not a candidate or resume source").oid
        ce._write(intent_path, dict(files=files, revision=revision, run_card_sha256=ce._sha(path)))
    for row in files:
        digest, size = hashlib.sha256(), 0
        with requests.get(hf_hub_url(repo, filename=prefix + "/" + row["path"], revision=revision),
                          headers={"Authorization": "Bearer " + token}, stream=True, timeout=(30,180)) as response:
            response.raise_for_status()
            for block in response.iter_content(8 << 20):
                digest.update(block)
                size += len(block)
        require(size == row["bytes"] and digest.hexdigest() == row["sha256"], "1000 full readback")
    ce._write(receipt_path, dict(schema="PI0253_MODEL_ONLY_OBSERVATION_HF_READBACK_V1", status="PASS_FULL_FILE_READBACK",
                                step=1000, candidate=False, resumable=False, files=files, repo_id=repo, prefix=prefix,
                                revision=revision, run_card_sha256=ce._sha(path)))


def prepare(args):
    import numpy as np
    parent = ce._json(PARENT)
    out, exe = ROOT / "FULL_G", Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    require(not (out / "run_card.json").exists(), "sealed FULL card already exists")
    data = Path(parent["paths"]["train_dataset"]).resolve()
    episodes = ce._jsonl(data / "meta/episodes.jsonl")
    prov = ce._jsonl(data / "meta/stage19_episode_provenance.jsonl")
    by_id = {r["source_episode_id"]: r for r in prov}
    episodes.sort(key=lambda r: (int(r["episode_index"]), r["source_episode_id"].encode()))
    require(len(episodes) == len(by_id) == 9792 and all(r["split"] == "train" for r in episodes), "original full TRAIN only")
    cohort, dts = [], []
    for i, ep in enumerate(episodes):
        p = by_id[ep["source_episode_id"]]
        task = re.search(r'persistent_train_([a-d])_', ep["source_episode_id"])[1]
        delta = np.diff(p["sample_timestamps"])
        require(p["desired_label"] and p["source_frame"] == "NATIVE_WBC_RELATIVE_REFERENCE"
                and len(delta) == 39 and np.allclose(delta, .05, rtol=0, atol=1e-9), "FULL desired/time semantics")
        cohort.append(dict(episode_index=int(ep["episode_index"]), source_episode_id=ep["source_episode_id"],
                           source_rollout_id=ep["source_episode_id"].split(":obs-")[0], target_family=task,
                           split="TRAIN", phase=(int(ep["episode_index"]) % 16) // 4))
        dts.append(dict(fixed_index=i, episode_index=int(ep["episode_index"]), source_episode_id=ep["source_episode_id"], dt_s=delta.tolist()))
    cohort_path, schedule_path = ROOT / "full_cohort.json", ROOT / "full_schedule.json"
    ce._write(cohort_path, dict(schema="PI0253_FULL_TRAIN_COHORT_V1", rows=cohort))
    pcg = np.random.Generator(np.random.PCG64(22301))
    order = np.concatenate([pcg.permutation(9792), pcg.permutation(9792)[:208]])
    schedule = [dict(optimizer_step=i, fixed_index=int(j), **{k: cohort[int(j)][k] for k in
                  ("episode_index", "source_episode_id", "target_family", "phase")}) for i,j in enumerate(order,1)]
    ce._write(schedule_path, dict(schema="PI0253_FULL_EXPOSURE_SCHEDULE_V1", cohort_sha256=ce._sha(cohort_path), rows=schedule))
    dt_path, key_path = ROOT / "full_actual_dt.json", ROOT / "full_step_keys.json"
    ce._write(dt_path, dict(schema="PI0253_FULL_ACTUAL_DT_V1", rows=dts))
    ce._write(key_path, dict(schema=rng.SCHEMA, rows=rng.build_key_table(schedule)))
    doc = copy.deepcopy(parent)
    doc.pop("paired_arm_root", None)
    doc.update(schema="PI0253_FULL_RUN_CARD_V1", protocol="PI0253", authorization=AUTH, execution_authorization=AUTH,
               authority_commit=RELEASE, canonical_issue_comment=5870278574, pointer_pr_comment=5870286088,
               family=FAMILY, arm="G", integrity_schema="PI0253_FULL_TRAINING_INTEGRITY_V1", training_manifest_family="FULL",
               loss="L_native + .4*ell_pos + .1*ell_delta + .1*ell_rot + .1*ell_speed_margin", loss_coefficients=COEFF,
               stage=dict(name="FULL", steps=STEPS, checkpoint_steps=list(SAVES), model_only_observation_step=1000, source_revision=REVISION))
    for key in list(doc["source"]):
        if key.startswith(("peer_", "pair_", "c2000_", "g0_", "g1_")):
            del doc["source"][key]
    paths = doc["paths"]
    paths.update(output_root=str(out), output_base=str(ROOT), train_dataset=str(data), base_model_path=str(ROOT / "initializer_G2000"),
                 execution_driver_root=str(exe), execution_driver=str(exe / "pi0253_full_train.py"), pbs_script=str(exe / "pi0253_full_train_h200.pbs"))
    doc["archive"].update(path_prefix="arms/PI0253/full-g-support-20260928/G", token_file="/nfs/home/svu/e1192548/.config/huggingface/token",
                          checkpoint_steps=list(SAVES))
    view_path = ROOT / "initializer_G2000_view_receipt.json"
    view = ce._json(view_path)
    receipt_path, readback_path = PARENT.parent / "durable_receipts/checkpoint-2000.durable_receipt.json", PARENT.parent / "hf_readbacks/checkpoint-2000.json"
    receipt = ce._json(receipt_path)
    doc["initializer"] = dict(path=paths["base_model_path"], screen_revision=REVISION, screen_manifest_sha256=ce._sha(readback_path),
                             full_checkpoint_tree_sha256=TREE, full_checkpoint_bytes=receipt["source_record"]["inventory"]["total_bytes"],
                             durable_receipt_path=str(receipt_path), durable_receipt_sha256=ce._sha(receipt_path), model_files=view["model_files"], source_family=parent["family"])
    doc["cohort"] = dict(path=str(cohort_path), sha256=ce._sha(cohort_path), count=9792)
    doc["schedule"] = dict(path=str(schedule_path), sha256=ce._sha(schedule_path), count=STEPS)
    src = doc["source"]
    src.update(audit_prefix="pi0253", paired_arms=["G"], reference_arm="G", intervention_arm=None,
               authorization=AUTH, canonical_issue_comment=5870278574, pointer_pr_comment=5870286088,
               step_rng_module="pi0253_step_rng.py", step_rng_module_name="pi0253_full_rng_runtime",
               dt_table_path=str(dt_path), dt_table_sha256=ce._sha(dt_path), step_key_table_path=str(key_path), step_key_table_sha256=ce._sha(key_path),
               execution_code_commit=args.execution_commit, initializer_view_receipt_path=str(view_path), initializer_view_receipt_sha256=ce._sha(view_path))
    bound = dict(G_card=PARENT, G_integrity=PARENT.parent / "pi0252_training_integrity.json", G_receipt=receipt_path,
                 G_readback=readback_path, G_parity=PARENT.parent / "pi0247_U_sampler_parity.json", initializer_view=view_path,
                 full_support=ROOT / "evidence/full_support.json", dt_table=dt_path, step_keys=key_path,
                 rotation=Path(src["rotation_check_path"]))
    src["bound_files"] = {name: dict(path=str(p), sha256=ce._sha(p)) for name,p in bound.items()}
    names = ("pi0253_full_train.py", "pi0253_step_rng.py", "pi0251_step_rng.py", "pi0251_ce_train.py", "pi0252_speed_margin.py", "pi0253_full_train_h200.pbs")
    src["custom_execution_code_sha256"] = {name: ce._sha(exe / name) for name in names}
    doc["optimizer"].update(steps=STEPS, checkpoint_steps=list(SAVES), optimizer_state="fresh AdamW/scheduler warm start from G2000; exact full state on subsequent resume")
    doc["resolved_config"].update(max_steps=STEPS, save_steps=2500, save_step_set=list(SAVES), start_from_checkpoint=paths["base_model_path"])
    doc["storage_contract"] = dict(reserve_bytes=10 << 30, quota_headroom="UNKNOWN", persistent_only=True)
    doc["execution_contract"] = dict(pause_after_durable_2500_for_fixed_barrier=True, model_only_1000_candidate=False,
                                      new_parity_queries=0, inherited_parity=src["bound_files"]["G_parity"], single_arm=True)
    card_path = out / "run_card.json"
    ce._write(card_path, doc)
    ce._write(out / "run_card.seal.json", dict(sha256=ce._sha(card_path)))
    print(json.dumps(dict(status="SEALED_READY_NOT_STARTED", run_card=str(card_path), sha256=ce._sha(card_path))))


def cpu_check(path, doc, life):
    """Actual FULL adapter checks with CPU tensors and explicitly synthetic logs."""
    import tempfile
    import torch
    from unittest.mock import patch
    validate_initializer(doc)
    life._verify_code(doc, Path(doc["paths"]["repo_code_root"]), Path(doc["paths"]["code_root"]), Path(doc["paths"]["training_runtime_root"]))
    life._verify_runtime_bindings(doc["paths"], doc["source"]["runtime_bindings"])
    paths = doc["paths"]
    sys.path[:0] = [str(Path(paths["stage10_root"]) / "runtime"), paths["training_runtime_root"],
                    str(Path(paths["code_root"]) / "scripts"), str(Path(paths["code_root"]) / "src"), paths["groot_source"]]
    fixed = life.load_module(Path(paths["code_root"]) / "scripts/pi0241_fixed16_learning.py", "pi0253_cpu_fixed")
    fixed.MAX_STEPS, fixed.CHECKPOINT_STEPS = STEPS, SAVES
    fixed.bind_runtime(life._card_with_p_runtime_paths(doc))
    core = life.load_module(Path(paths["training_runtime_root"]) / "pi0246_gate_pass_train.py", "pi0253_cpu_core")
    _, config, summary = core.build_runtime_config(paths, Path(paths["output_root"]) / "training", fixed)
    require(summary == doc["resolved_config"], "actual FULL runtime config differs from card")
    schedule = ce._json(Path(doc["schedule"]["path"]))["rows"]
    keys = ce._json(Path(doc["source"]["step_key_table_path"]))["rows"]
    require(len(schedule) == STEPS and len({r["fixed_index"] for r in schedule[:9792]}) == 9792, "FULL PCG exposure")
    for cursor in (0,2500,5000,7500):
        suffix = fixed.fixed_schedule_suffix(schedule, cursor)
        require(len(suffix) == STEPS-cursor and suffix[0]["optimizer_step"] == cursor+1
                and suffix[-1]["optimizer_step"] == STEPS, "FULL resume schedule cursor")
    require(keys == rng.build_key_table(schedule) and keys[-1]["key"] == "PI0253|22301|stage=FULL|optimizer_step=10000", "FULL RNG terminal")
    from transformers import get_scheduler
    probe = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([probe], lr=1e-5, betas=(.9,.999), eps=1e-8, weight_decay=1e-5)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=10, num_training_steps=STEPS)
    class FakeTrainer:
        def __init__(self, cursor):
            self.args = SimpleNamespace(gradient_accumulation_steps=1)
            self.state = SimpleNamespace(global_step=cursor)
        def add_callback(self, callback):
            self.callback = callback
        def training_step(self, model, inputs, *args, **kwargs):
            return torch.randn(1), torch.distributions.Beta(1.5,1.0).sample((1,))
    with tempfile.TemporaryDirectory(dir=ROOT / "evidence", prefix="cpu-fixture-") as temporary:
        for cursor in (2500,5000,7500):
            optimizer.param_groups[0]["lr"] = scheduler.base_lrs[0] * scheduler.lr_lambdas[0](cursor)
            scheduler.last_epoch = cursor
            saved = Path(temporary) / f"state-{cursor}.pt"
            torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=torch.get_rng_state(), cursor=cursor), saved)
            restored = torch.load(saved, weights_only=False)
            optimizer.load_state_dict(restored["optimizer"])
            scheduler.load_state_dict(restored["scheduler"])
            torch.set_rng_state(restored["rng"])
            require(scheduler.last_epoch == restored["cursor"] == cursor and math.isclose(optimizer.param_groups[0]["lr"],
                    1e-5 * scheduler.lr_lambdas[0](cursor), abs_tol=1e-15), "FULL optimizer/scheduler restore")
            audit = Path(temporary) / f"rng-{cursor}.jsonl"
            audit.write_text("".join(json.dumps({k:r[k] for k in ("optimizer_step","key","seed")})+'\n' for r in keys[:cursor]))
            wrapped = rng.install_training_step_rng(FakeTrainer, audit, resume_after_step=cursor)
            trainer = wrapped(cursor)
            model = torch.nn.Linear(1,1)
            first = trainer.training_step(model, {})
            with rng.rng_context(cursor+1):
                expected = torch.randn(1), torch.distributions.Beta(1.5,1.0).sample((1,))
            require(all(torch.equal(a,b) for a,b in zip(first,expected)), "FULL actual Trainer resume step RNG")
    require(scheduler.lr_lambdas[0](STEPS) == 0, "FULL cosine horizon")
    command = [sys.executable, str(Path(paths["repo_code_root"]) / "scripts/pi0247_lu_train.py"),
               "--archive-sidecar", "--run-card", str(path), "--arm", "G", "--step", "10000", "--archive-root", paths["output_root"]]
    routed = ce._route_archive_sidecar(command, doc, path)
    require(Path(routed[1]).name == "pi0253_full_train.py", "FULL actual archive callback routing")
    # ponytail: reuse the real finalizer validator; only its I/O is a synthetic CPU fixture.
    out = Path(paths["output_root"])
    root = out / "training"
    base = dict(runtime_id="S4", rollout_count=1, rollout_steps=4, checkpointed_euler_steps=4,
                backward_recomputation_head_evaluations_expected=4, expert_action_in_condition=False,
                active_action_modified=False, euler_states=[{}]*4, euler_velocities=[{}]*4,
                lambda_position=.4, lambda_delta=.1, lambda_rotation=.1, theta_ref_rad=.05,
                native_loss=1., physical_position_loss=2., physical_delta_loss=3., physical_rotation_loss=4.,
                total_loss=2.1, total_loss_with_rotation=2.5, pi0253_actual_training_loss=2.519,
                pi0253_arm="G", pi0253_peak_z=1., pi0253_peak_speed_mps=2., pi0253_peak_segment=0,
                pi0253_speed_margin=.19, pi0253_speed_penalty=.019, pi0253_lambda_speed_margin=.1,
                native_epsilon_sha256="synthetic", stock_initial_noise={"sha256":"synthetic"}, native_tau_identity={"sha256":"synthetic"})
    rows = [dict(base, forward_index=i) for i in range(1,STEPS+1)]
    fixtures = {
        "pi0253_step_rng_audit.jsonl": [dict(r, model_input_sha256="synthetic", pre_forward_guard={"status":"PASS"}) for r in keys],
        "pi0241_sample_consumption.jsonl": [dict(r, observation_index=0) for r in schedule],
        "pi0253_pre_forward_guard.jsonl": [dict(optimizer_step=i, phase="BEFORE_NATIVE_RANDOM_SAMPLING", status="PASS") for i in range(1,STEPS+1)],
        "pi0253_pair_identity.jsonl": [dict(optimizer_step=i, status="REFERENCE_CAPTURED", before_backward=True) for i in range(1,STEPS+1)],
    }
    parity = ce._json(Path(doc["source"]["bound_files"]["G_parity"]["path"]))
    chain = dict(status="PASS_PI0247_U_FIRST_UPDATE_FULL_EULER_CHAIN", state_gradient_norms={str(i):1. for i in range(1,5)},
                 checkpointed_recompute_steps=4, mode_restored_after_forward=True, mode_restored_after_backward=True)
    original_json = ce._json
    def fixture_json(p):
        if p.name == "pi0247_U_sampler_parity.json": return parity
        if p.name == "u_first_update_chain_gradient.json": return chain
        return original_json(p)
    with patch.object(ce.helper, "_winning_training_root", return_value=root), \
         patch.object(ce.helper, "_overlay_rows", return_value=(root / "synthetic.jsonl", rows)), \
         patch.object(ce, "_rows_for_root", side_effect=lambda o,r,n,s: fixtures[n]), \
         patch.object(ce, "_json", side_effect=fixture_json), \
         patch.object(ce, "_require_checkpoint_receipts", return_value=[dict(step=s, synthetic=True) for s in SAVES]), \
         patch.object(ce, "_jsonl", return_value=[dict(optimizer_step=i, restored=True, rollout_forward_head_evaluations=4*i,
                                                    backward_recompute_head_evaluations=4*i) for i in range(1,STEPS+1)]):
        verified = ce._preverify_training(doc,path)
        require(len(verified["rows"]) == STEPS, "actual FULL finalizer terminal")
        rows[-1]["pi0253_actual_training_loss"] = 2.5
        try: ce._preverify_training(doc,path)
        except RuntimeError: pass
        else: raise AssertionError("FULL finalizer accepted missing terminal speed loss")
        rows[-1]["pi0253_actual_training_loss"] = 2.519
    result = dict(schema="PI0253_FULL_ENTRY_CPU_CHECK_V1", status="PASS", run_card_sha256=ce._sha(path),
                  initializer_revision=REVISION, initializer_tree=TREE, actual_resolved_config=summary,
                  endpoint=STEPS, resume_cursors=[2500,5000,7500], scheduler_horizon=STEPS, full_objective_finalizer="PASS_TERMINAL_AND_POISON",
                  archive_callback_routing="PASS", observation1000_model_only=True, project_model_queries=0,
                  project_optimizer_updates=0, fixture_boundary="synthetic CPU tensors/logs, no project training or inference",
                  storage=capacity(doc))
    ce._write(ROOT / "evidence/FULL_ENTRY_CPU_CHECK.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "actual_resolved_config"}))


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag in ("prepare", "cpu-check", "hf-preflight", "train", "finalize", "archive-sidecar", "archive-observation"):
        modes.add_argument("--" + flag, action="store_true")
    parser.add_argument("--run-card", type=Path)
    parser.add_argument("--arm", choices=["G"], default="G")
    parser.add_argument("--resume-after-step", type=int, choices=[0,2500,5000,7500], default=0)
    parser.add_argument("--step", type=int, choices=SAVES)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--execution-commit")
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
    elif args.archive_observation:
        archive_observation(args.run_card)
    else:
        require(args.run_card is not None, "run card required")
        doc, life = install(args.run_card, args.resume_after_step)
        if args.cpu_check:
            cpu_check(args.run_card, doc, life)
        else:
            if args.train and args.resume_after_step:
                barrier = ce._json(Path(doc["paths"]["output_root"]) / "barrier_verdict.json")
                require(barrier["status"] == "CONTINUE_FIXED_FULL" and barrier["run_card_sha256"] == ce._sha(args.run_card), "2500 barrier admission")
            ce._dispatch_lifecycle(args, life)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
