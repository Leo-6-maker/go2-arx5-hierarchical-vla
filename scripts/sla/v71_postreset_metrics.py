"""Read existing fixed37 outputs: physical speed, phase errors and language response."""

import json
from pathlib import Path
import sys
from tempfile import mkdtemp

import numpy as np
from scipy.spatial.transform import Rotation

from m38_gpu_preflight_v1 import pin, read, sha
from scripts.pi0291_arx5_data.native_dataset import matrix_from10


MODELS = ("R4", "LAMBDA_ZERO", "SHARED_L2SP", "R4_SHARED_ZERO_IO")


def percentiles(values):
    return dict(zip(("p50", "p95", "maximum"), map(float, np.percentile(values, (50, 95, 100)))))


def run(work):
    root = work / "V7-POSTRESET-EXPERT-37-01"
    registry = read(pin(root / "REGISTRY.json", "5ad324eb7d8c9377ea3139d5bee8eaa8ed04a60fa98878e0630150aec6500d0a"))
    actual = read(pin(root / "OUTPUT/ACTUAL.json", "9708cb9e2b7bca384a79e4364d3fc98ccd090cc94901d371e39c562303bd95b0"))
    summary = read(pin(root / "OUTPUT/SUMMARY.json", "2a4aba8a7fe1dae70dc7ed303e7cad00692fba36fef4ae7959caad018d7f92af"))
    frames = {f["id"]: f for f in registry["frames"]}
    baseline = {(r["model"], r["frame"], r["repeat"]): r for r in registry["original_outputs"]["rows"]}
    hybrid = {(r["frame"], r["repeat"]): r for r in actual["rows"]}
    metrics = {(r["model"], r["frame"]): dict(r) for r in summary["rows"]}
    actions = {}
    for frame in frames.values():
        for model in MODELS:
            h = model == MODELS[-1]
            row = hybrid[frame["id"], 0] if h else baseline[model, frame["id"], 0]
            folder = root / "OUTPUT" if h else Path(registry["original_outputs"]["root"])
            digest = row["artifact_sha256"] if h else row["action_sha256"]
            with np.load(pin(folder / row["artifact"], digest), allow_pickle=False) as f:
                action = f["physical_H40"].copy()
                actions[model, frame["id"]] = action
                if not h and frame["has_real_teacher_label"]:
                    teacher, valid = f["teacher_H40"], f["valid"][:10]
                    angle = np.degrees(Rotation.from_matrix(
                        matrix_from10(action[:10])[:, :3, :3]
                        @ np.swapaxes(matrix_from10(teacher[:10])[:, :3, :3], -1, -2)).magnitude())
                    metrics[model, frame["id"]]["p10_teacher_rotation_mean_deg"] = float(angle[valid].mean())
        if frame["has_real_teacher_label"]:
            label_row = baseline["R4", frame["id"], 0]
            with np.load(Path(registry["original_outputs"]["root"]) / label_row["artifact"], allow_pickle=False) as f:
                teacher, valid = f["teacher_H40"], f["valid"][:10]
                action = actions[MODELS[-1], frame["id"]]
                angle = np.degrees(Rotation.from_matrix(
                    matrix_from10(action[:10])[:, :3, :3]
                    @ np.swapaxes(matrix_from10(teacher[:10])[:, :3, :3], -1, -2)).magnitude())
                metrics[MODELS[-1], frame["id"]]["p10_teacher_rotation_mean_deg"] = float(angle[valid].mean())
    models = {}
    for model in MODELS:
        rows = [metrics[model, f] for f in frames]
        labelled = [r for r in rows if "p10_teacher_xyz_rms_mm" in r]
        assert len(rows) == 37 and len(labelled) == 30
        legal = [r for r in rows if not metrics["R4", r["frame"]]["raw_illegal"]]
        assert len(legal) == 32
        keys = ("p10_xyz_rms_from_R4_mm", "p10_jaw_mae_from_R4_mm", "p10_rotation_mean_from_R4_deg")
        label_keys = ("p10_teacher_xyz_rms_mm", "p10_teacher_jaw_mae_mm", "p10_teacher_rotation_mean_deg")
        models[model] = dict(
            all37_first_seam_mm_s=percentiles([r["first_seam_mm_s"] for r in rows]),
            all37_minimum_seam_margin_to_40_mm_s=40-max(r["first_seam_mm_s"] for r in rows),
            r4_legal32_first_seam_mm_s=percentiles([r["first_seam_mm_s"] for r in legal]),
            r4_legal32_minimum_seam_margin_to_40_mm_s=40-max(r["first_seam_mm_s"] for r in legal),
            all37_mean_drift_from_R4={k: float(np.mean([r[k] for r in rows])) for k in keys},
            labelled30_mean_teacher_error={k: float(np.mean([r[k] for r in labelled])) for k in label_keys},
            teacher_phase_means={phase: dict(conditions=len(subset), **{
                k: float(np.mean([r[k] for r in subset])) for k in label_keys})
                for phase in sorted({r["phase"] for r in labelled})
                for subset in [[r for r in labelled if r["phase"] == phase]]})
    language = []
    for task in ("PLACE", "HOLD"):
        suffix = "-right" if task == "PLACE" else ""
        red, blue = (f"RED_PREGRASP-{task}-{colour}{suffix}" for colour in ("red", "blue"))
        assert frames[red]["input"] == frames[blue]["input"]
        assert frames[red]["noise_id"] == frames[blue]["noise_id"]
        language.append(dict(task=task, red_frame=red, blue_frame=blue,
            identical_observation_and_noise=True, direction_correctness_scored=False,
            reason="No target-object geometry is present in this frozen model-output diagnostic",
            blue_minus_red_per_model={model: dict(
                mean_P10_xyz_mm=(1000*(actions[model, blue][:10, :3]-actions[model, red][:10, :3]).mean(axis=0)).tolist(),
                last_P10_xyz_mm=(1000*(actions[model, blue][9, :3]-actions[model, red][9, :3])).tolist(),
                mean_P10_jaw_mm=float(40*(actions[model, blue][:10, 9]-actions[model, red][:10, 9]).mean()))
                for model in MODELS}))
    special = [metrics[model, frame] for frame in frames
               if frame in ("R4_SAFE_GRASP", "R4_SAFE_LIFT") or frames[frame]["phase"] in ("LIFT", "OPEN")
               for model in MODELS]
    value = dict(status="FIXED37_REUSED_OUTPUT_PHYSICAL_METRICS_COMPLETE", code_sha256=sha(Path(__file__)),
        registry_sha256=sha(root/"REGISTRY.json"), forward_sha256=sha(root/"OUTPUT/ACTUAL.json"),
        summary_sha256=sha(root/"OUTPUT/SUMMARY.json"), condition_count=37, labelled_conditions=30,
        aggregation="Equal weight per condition; P10 XYZ component RMS, physical jaw MAE, SO3 angle mean on valid P10 points",
        first_seam_definition="Absolute width difference from previous actually consumed command divided by 0.05 seconds",
        models=models, language_pairs=language, lift_open_and_safe_rows=special,
        repeats_are_independent_samples=False, train_selection_not_heldout=True,
        no_native_skill_or_stage_completion_score=True, new_model_forwards=0, optimizer_updates=0, physics_integrations=0)
    p = Path(mkdtemp(prefix="v71-postreset-metrics.", dir=work))/"ACTUAL.json"
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(p, sha(p), flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
