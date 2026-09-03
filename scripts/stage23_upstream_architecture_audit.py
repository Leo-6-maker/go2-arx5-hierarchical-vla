#!/usr/bin/env python3
"""Generate the pinned Stage23 upstream architecture review.

The audit reads source files from immutable git objects with ``git show``.  It
does not import either upstream project, construct a model, read a dataset,
run inference, submit a job, or execute a controller.  The generated JSON is
an index of source evidence and bounded Stage23 facts already recorded in the
repository; it intentionally contains no model weights, credentials, or
private runtime files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable


ISAAC_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
WBC_SHA = "a0732b642c0333077e127a2f56ab0014c196bca4"


def _run_git(root: Path, args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {root}: {result.stderr.strip()}"
        )
    return result.stdout


class PinnedTree:
    def __init__(self, root: Path, revision: str):
        self.root = root
        self.revision = revision

    def bytes(self, path: str) -> bytes | None:
        result = subprocess.run(
            ["git", "-C", str(self.root), "show", f"{self.revision}:{path}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            return None
        return result.stdout

    def blob(self, path: str) -> str | None:
        output = _run_git(self.root, ["ls-tree", self.revision, "--", path])
        for line in output.splitlines():
            if "\t" in line and line.split("\t", 1)[1] == path:
                return line.split()[2]
        output = _run_git(self.root, ["ls-tree", "-r", self.revision, "--", path])
        for line in output.splitlines():
            if "\t" in line and line.split("\t", 1)[1] == path:
                return line.split()[2]
        return None

    def paths(self, prefix: str) -> list[str]:
        output = _run_git(self.root, ["ls-tree", "-r", "--name-only", self.revision, "--", prefix])
        return [line for line in output.splitlines() if line]

    def commit_metadata(self) -> dict[str, str]:
        return {
            "sha": self.revision,
            "subject": _run_git(self.root, ["show", "-s", "--format=%s", self.revision]).strip(),
            "date": _run_git(self.root, ["show", "-s", "--format=%cI", self.revision]).strip(),
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _marker_lines(text: str, patterns: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    lines = text.splitlines()
    result = []
    for name, pattern in patterns:
        try:
            matches = [index for index, line in enumerate(lines, 1) if re.search(pattern, line)]
        except re.error as exc:
            raise ValueError(f"invalid marker regex {pattern!r}: {exc}") from exc
        result.append({"name": name, "pattern": pattern, "lines": matches})
    return result


def _inventory_entry(tree: PinnedTree, path: str, patterns: Iterable[tuple[str, str]] = ()) -> dict[str, Any]:
    payload = tree.bytes(path)
    if payload is None:
        return {"path": path, "present": False, "blob_sha1": None, "sha256": None, "line_count": None, "markers": []}
    text = payload.decode("utf-8", errors="replace")
    return {
        "path": path,
        "present": True,
        "blob_sha1": tree.blob(path),
        "sha256": _sha256(payload),
        "byte_count": len(payload),
        "line_count": len(text.splitlines()),
        "markers": _marker_lines(text, patterns),
    }


def _refs(inventory: dict[str, dict[str, Any]], paths: Iterable[str]) -> list[dict[str, Any]]:
    refs = []
    for path in paths:
        entry = inventory.get(path, {})
        marker_lines = {
            marker["name"]: marker["lines"]
            for marker in entry.get("markers", [])
            if marker.get("lines")
        }
        refs.append(
            {
                "path": path,
                "blob_sha1": entry.get("blob_sha1"),
                "sha256": entry.get("sha256"),
                "line_count": entry.get("line_count"),
                "marker_lines": marker_lines,
            }
        )
    return refs


def _project_status(repo: Path) -> dict[str, Any]:
    status = _run_git(repo, ["status", "--short"])
    expected = {
        "reports/STAGE23_UPSTREAM_ARCHITECTURE_REVIEW.json",
        "reports/STAGE23_UPSTREAM_ARCHITECTURE_REVIEW.md",
        "scripts/stage23_upstream_architecture_audit.py",
    }
    unexpected = []
    for line in status.splitlines():
        path = line[3:] if len(line) >= 4 else line
        if path not in expected:
            unexpected.append(line)
    return {
        "branch": _run_git(repo, ["branch", "--show-current"]).strip(),
        "head": _run_git(repo, ["rev-parse", "HEAD"]).strip(),
        "status_short": status.splitlines(),
        "unexpected_status_short": unexpected,
        "preexisting_tree_clean": not unexpected,
        "remote": _run_git(repo, ["remote", "-v"]).splitlines(),
    }


ISAAC_SPECS: dict[str, list[tuple[str, str]]] = {
    "gr00t/model/gr00t_n1d7/gr00t_n1d7.py": [
        ("action_head_class", r"class Gr00tN1d7ActionHead"),
        ("action_head_components", r"self\.state_encoder|self\.action_encoder|self\.action_decoder"),
        ("state_dropout", r"self\.state_dropout_prob|state_dropout_prob"),
        ("flow_noise", r"noise\s*=\s*torch\.randn"),
        ("flow_time", r"sample_time|t\s*=\s*self\.sample_time"),
        ("noisy_trajectory", r"noisy_trajectory\s*="),
        ("diT_call", r"model_output\s*=\s*self\.model|self\.model\("),
        ("masked_flow_loss", r"action_loss\s*=\s*F\.mse_loss|action_mask"),
        ("inference_noise", r"actions\s*=\s*torch\.randn"),
        ("euler_step", r"actions\s*\+=\s*dt"),
        ("rtc_options", r"rtc_overlap_steps|rtc_frozen_steps|rtc_ramp_rate"),
        ("action_return", r"action_pred\s*=|return action_pred"),
    ],
    "gr00t/configs/model/gr00t_n1d7.py": [
        ("cosmos_backbone", r"model_name:\s*str\s*=\s*\"nvidia/Cosmos-Reason2-2B\""),
        ("max_dimensions", r"max_state_dim|max_action_dim|action_horizon"),
        ("four_step_sampler", r"num_inference_timesteps:\s*int\s*=\s*4"),
        ("model_tuning_defaults", r"tune_projector|tune_diffusion_model|tune_vlln"),
        ("raw_state_dropout_default", r"state_dropout_prob:\s*float\s*=\s*0\.8"),
        ("relative_default", r"use_relative_action:\s*bool\s*=\s*False"),
    ],
    "gr00t/model/modules/embodiment_conditioned_mlp.py": [
        ("category_specific_init", r"0\.02\s*\*\s*torch\.randn|torch\.zeros\(num_categories"),
        ("category_selection", r"selected_W\s*=\s*self\.W\[cat_ids\]|selected_b\s*=\s*self\.b\[cat_ids\]"),
        ("category_mlp", r"class CategorySpecificMLP"),
        ("multi_embodiment_action_encoder", r"class MultiEmbodimentActionEncoder"),
    ],
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py": [
        ("projector_id_table", r"_PROJECTOR_INDEX_GROUPS|new_embodiment|unitree_g1_sonic"),
        ("inference_action_mask", r"action_mask\s*=\s*torch\.zeros"),
        ("inference_valid_horizon", r"action_horizon\s*=\s*len\(action_config\.delta_indices\)"),
        ("decode_slice", r"action_horizon\]|action_horizon\)"),
        ("training_action_pad", r"Pad action to max_action_dim|Pad action to max_action_horizon"),
        ("training_mask", r"action_mask\[action_horizon:\]\s*=\s*0|action_mask\[:, action_dim:\]\s*=\s*0"),
        ("processor_state_dropout", r"self\.state_dropout_prob|random\.random\(\)"),
        ("save_stats_config", r"statistics_file|state_dropout_prob"),
    ],
    "gr00t/data/state_action/state_action_processor.py": [
        ("relative_reference", r"reference_state\s*=\s*state\[state_key\]\[-1\]"),
        ("apply_relative", r"_convert_to_relative_action"),
        ("unapply_absolute", r"_convert_to_absolute_action"),
    ],
    "gr00t/data/state_action/pose.py": [
        ("relative_pose_composition", r"Mathematically:.*T_relative|T_relative\s*=\s*relative_transformation"),
    ],
    "gr00t/data/state_action/action_chunking.py": [
        ("absolute_pose_composition", r"T_absolute\s*=\s*T_ref\s*@\s*T_relative"),
        ("slerp_interpolation", r"linear interpolation for translation and SLERP"),
    ],
    "gr00t/configs/finetune_config.py": [
        ("official_flags", r"tune_projector|tune_diffusion_model|state_dropout_prob"),
        ("official_dropout_default", r"state_dropout_prob:\s*float\s*=\s*0\.2"),
        ("official_batch_default", r"global_batch_size:\s*int\s*=\s*64"),
        ("official_lr", r"learning_rate:\s*float\s*=\s*1e-4"),
        ("official_warmup", r"warmup_ratio"),
        ("official_max_steps", r"max_steps:\s*int\s*=\s*10000"),
    ],
    "gr00t/data/embodiment_tags.py": [
        ("new_embodiment_tag", r"NEW_EMBODIMENT"),
        ("finetune_only_tags", r"FINETUNE_ONLY_TAGS"),
        ("sonic_posttrain_tag", r"UNITREE_G1_SONIC"),
    ],
    "gr00t/experiment/launch_finetune.py": [
        ("launcher_relative_action", r"use_relative_action\s*=\s*True"),
        ("launcher_dropout", r"state_dropout_prob"),
        ("launcher_tuning_flags", r"tune_projector|tune_diffusion_model|tune_llm|tune_visual"),
    ],
    "gr00t/model/gr00t_n1d7/setup.py": [
        ("model_from_checkpoint", r"AutoModel\.from_pretrained"),
        ("processor_from_checkpoint", r"AutoProcessor\.from_pretrained"),
    ],
    "gr00t/model/modules/qwen3_backbone.py": [
        ("backbone_tuning_flags", r"tune_llm|tune_visual|set_trainable_parameters"),
    ],
    "gr00t/__init__.py": [
        ("local_pretrained_construction", r"from_pretrained|no_init_weights|_zero_no_weight_model_parameters"),
    ],
    "gr00t/configs/data/embodiment_configs.py": [
        ("sonic_modality", r"unitree_g1_sonic|motion_token"),
        ("sonic_horizon", r"list\(range\(40\)\)"),
        ("official_so100_horizon", r"list\(range\(16\)\)"),
    ],
    "getting_started/real_world_deployment.md": [
        ("dataset_recommendation", r"at least \*\*100 valid episodes\*\*|200\+ episodes"),
        ("relative_action_recommendation", r"State-relative Action Prediction"),
        ("chunk_recommendation", r"Action Chunk Size|at least 32"),
        ("jitter_taxonomy", r"Inside each chunk|Between consecutive chunks|Chunks look smooth"),
        ("async_receding_rtc", r"Asynchronous Inference|Receding Horizon|Real-Time Chunking"),
        ("rtc_not_policy_wired", r"not wired into `Gr00tPolicy`"),
        ("intra_chunk_metric", r"Mean intra-chunk acceleration"),
    ],
    "getting_started/finetune_new_embodiment.md": [
        ("so100_horizon", r"predict 16 future steps|range\(0, 16\)"),
        ("so100_relative_joint", r"RELATIVE = delta from current state"),
        ("official_finetune_scale", r"max-steps 2000|global-batch-size 32"),
        ("official_augmentation", r"color-jitter-params"),
        ("open_loop_execution", r"execution-horizon 16"),
        ("training_curve_guidance", r"Plot overlap on a training trajectory|Error decreases with training"),
    ],
    "getting_started/policy.md": [
        ("execution_horizon_definition", r"executed before the script re-plans"),
        ("base_horizon", r"action_horizon: 40"),
    ],
    "README.md": [
        ("relative_eef_claim", r"Relative EEF Action Space"),
        ("expanded_dimensions", r"132|action horizon"),
        ("inference_chunk_distinction", r"rollout CLI flag|action chunk"),
    ],
    "gr00t/eval/open_loop_eval.py": [
        ("execution_horizon_arg", r"execution-horizon|execution_horizon"),
        ("denoising_steps_arg", r"denoising-steps|denoising_steps"),
    ],
}


WBC_SPECS: dict[str, list[tuple[str, str]]] = {
    "gear_sonic/trl/modules/universal_token_modules.py": [
        ("fsq_config", r"num_fsq_levels|fsq_level_list|down_t|num_future_frames"),
        ("token_shape", r"token_total_dim|max_num_tokens|token_dim"),
        ("encoder_decoder_contract", r"encode.*motion|decode action and motion|decoder output"),
        ("encode_motion", r"encode motion|self\.encode\("),
        ("assemble_tokens", r"assemble_all_tokens"),
        ("decoder_input", r"decode_input_dict|decoded_outputs"),
    ],
    "gear_sonic/data/features_sonic_vla.py": [
        ("sonic_fps", r"FPS\s*=\s*50"),
        ("motion_token_feature", r"action\.motion_token"),
        ("motion_token_width", r"64"),
        ("eef_state_feature", r"observation\.eef_state"),
        ("wrist_video_features", r"left_wrist|right_wrist"),
    ],
    "gear_sonic/scripts/run_data_exporter.py": [
        ("exporter_frequency", r"data_collection_frequency: int = 50"),
        ("state_and_eef_labels", r"observation\.state|observation\.eef_state|action\.wbc"),
        ("motion_token_label", r"action\.motion_token|token_state"),
        ("sonic_stale_threshold", r"sonic_latency_ms <= 100\.0"),
        ("planner_stale_threshold", r"planner_latency_ms <= 200\.0"),
    ],
    "gear_sonic/scripts/run_vla_inference.py": [
        ("action_publish_rate", r"action_publish_rate: int = 50"),
        ("action_horizon", r"action_horizon: int = 40"),
        ("inference_rate", r"rate: float = 1 / 0\.4"),
        ("latent_message_shape", r"Shape ``\[64\]``|token_state"),
        ("action_bound_gate", r"max\(\) > 1\.25"),
        ("latency_index", r"calculate_latency_compensated_index"),
        ("async_worker", r"inference_worker_thread|inference_queue"),
        ("chunk_indexing", r"current_idx|min\(action_chunk_index"),
    ],
    "gear_sonic/scripts/launch_inference.py": [
        ("launch_publish_rate", r"action_publish_rate: int = 50"),
        ("launch_horizon", r"action_horizon: int = 40"),
        ("simulation_switch", r"sim: bool = False"),
    ],
    "gear_sonic/utils/inference/vla_utils.py": [
        ("latency_to_index", r"raw_index = np\.round|inference_delay \* control_freq"),
        ("stale_prefix_description", r"stale|skip the first few"),
        ("trigger_interval", r"inference_interval|time_since_last_inference"),
    ],
    "docs/source/tutorials/vla_workflow.md": [
        ("compact_token_description", r"64-dimensional|50 Hz"),
        ("vla_rate_and_horizon", r"2\.5 Hz|64-dim x 40"),
        ("full_action_dim", r"78-dimensional"),
        ("data_labeling", r"teleop demonstrations|LeRobot dataset"),
        ("data_scale", r"50–100"),
    ],
    "docs/source/tutorials/vla_inference.md": [
        ("controller_lookahead", r"4 future frames|10 future frames|80 ms|200 ms"),
        ("sonic_action_space", r"78-dimensional"),
    ],
    "docs/source/tutorials/data_collection.md": [
        ("collection_path", r"LeRobot|motion_token|50"),
    ],
    "docs/source/model_card.md": [
        ("default_lookahead", r"10 future frames|200 ms"),
        ("low_latency_lookahead", r"4 future frames|80 ms"),
        ("controller_rate", r"50 Hz"),
        ("latency_not_lookahead", r"not measurements of total end-to-end"),
    ],
    "decoupled_wbc/control/policy/interpolation_policy.py": [
        ("max_change_rate", r"max_change_rate"),
        ("target_time", r"target_time"),
        ("garbage_collection", r"interpolation_garbage_collection_time"),
        ("waypoint_schedule", r"schedule_waypoint"),
        ("continuous_output", r"def get_action|self\.interp\(time\)"),
    ],
    "decoupled_wbc/control/policy/g1_decoupled_whole_body_policy.py": [
        ("high_level_goal_keys", r"target_upper_body_pose|target_time|navigate_cmd|base_height_command"),
        ("upper_open_loop", r"open loop \(just interpolation\)"),
        ("lower_body_policy", r"lower_body_policy|get_action"),
        ("teleop_timeout", r"time_since_goal|safe_goal"),
    ],
    "decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py": [
        ("timestamp", r"data\[\"timestamp\"\]"),
        ("target_time", r"data\[\"target_time\"\]"),
        ("initial_pose_duration", r"time_to_get_to_initial_pose"),
    ],
    "decoupled_wbc/control/utils/sync_sim_utils.py": [
        ("target_pose_feature", r"observation\.sim\.target_upper_body_pose"),
        ("nav_feature", r"teleop\.navigate_command|navigate_cmd"),
        ("base_height_feature", r"teleop\.base_height_command"),
        ("target_time_not_recorded", r"target_time.*commented|# .*target_time"),
    ],
    "decoupled_wbc/control/policy/g1_gear_wbc_policy.py": [
        ("teleop_commands", r"use_teleop_policy_cmd|interpolated_navigate_cmd"),
        ("lower_body_selection", r"standing policy|walking policy"),
        ("body_action", r"body_action"),
    ],
    "decoupled_wbc/control/policy/wbc_policy_factory.py": [
        ("interpolation_factory", r"InterpolationPolicy|upper_body_policy_type"),
        ("gear_lower_body", r"G1GearWbcPolicy|lower_body_policy_type"),
    ],
}


def _render_markers(entry: dict[str, Any]) -> str:
    if not entry.get("present"):
        return "missing"
    marker_text = []
    for marker in entry.get("markers", []):
        if marker["lines"]:
            marker_text.append(f"{marker['name']}={','.join(map(str, marker['lines']))}")
    return "; ".join(marker_text) if marker_text else "no requested marker match"


def _markdown(data: dict[str, Any]) -> str:
    isaac = data["source_inventory"]["isaac_gr00t"]
    wbc = data["source_inventory"]["groot_whole_body_control"]
    project = data["project_context"]

    def source_table(entries: dict[str, Any]) -> str:
        rows = ["| pinned source | blob SHA-1 | content SHA-256 | lines / markers |", "|---|---|---|---|"]
        for path, entry in entries.items():
            rows.append(
                f"| `{path}` | `{entry.get('blob_sha1') or 'MISSING'}` | "
                f"`{entry.get('sha256') or 'MISSING'}` | {entry.get('line_count') or '—'} / {_render_markers(entry)} |"
            )
        return "\n".join(rows)

    options = data["architecture_options"]
    option_rows = ["| option | scientific upside | engineering/data cost | falsifier / strengthening evidence |", "|---|---|---|---|"]
    for option in options:
        option_rows.append(
            f"| **{option['id']}** | {option['scientific_upside']} | {option['cost_and_data']} | {option['falsifier_or_strengthener']} |"
        )

    attribution_rows = ["| hypothesis | evidence status | interpretation boundary |", "|---|---|---|"]
    for item in data["stage23_claim_reassessment"]["attribution_matrix"]:
        attribution_rows.append(f"| `{item['hypothesis']}` | **{item['status']}** | {item['boundary']} |")

    comparison_rows = [
        "| system | high-level representation | active dims / 132 | H / sampling | duration / update | execution and latency | smoothness / label owner |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in data["comparison_table"]:
        comparison_rows.append(
            f"| {row['system']} | {row['representation']} | {row['active_dims_over_132']} | "
            f"{row['horizon_and_sample']} | {row['duration_and_update']} | {row['execution_and_latency']} | "
            f"{row['smoothness_and_labels']} |"
        )

    return f"""# Stage23 Upstream Architecture Review

Date: {data['generated_date']} (Asia/Singapore)

## Scope and execution boundary

This is a source/documentation-only review of the two immutable upstream
revisions. Source text was read from pinned Git objects, not imported or
vendored. No model was constructed and no dataset, validation `e`, holdout
`f`, checkpoint, PBS/Hopper job, WBC runtime, ROS/ZMQ path, or hardware path
was executed.

Pinned revisions:

- Isaac-GR00T: `{data['pins']['isaac_gr00t_sha']}`
- GR00T-WholeBodyControl: `{data['pins']['groot_whole_body_control_sha']}`

Current project evidence anchor: `{project['head']}` on
`{project['branch']}`. Pre-existing working tree at generation: `{'clean' if project['preexisting_tree_clean'] else 'not clean'}`; the three expected audit outputs may be pending.

## Executive findings

1. The N1.7 path is a dense flow-matching action head: processor
   normalization/padding → category-conditioned state/action projectors →
   Gaussian noisy H×132 trajectory → DiT velocity → masked MSE flow loss →
   four-step Euler sampling → processor unapply/decode. The action head does
   not receive an explicit timestamp tensor; temporal semantics enter through
   modality `delta_indices` and the dataset/configuration contract.
2. `NEW_EMBODIMENT` maps to category/projector ID 10, a finetune-only custom
   slot. The audited source initializes category-specific matrices with an
   independent small random initialization and contains no special
   new-embodiment copy/transfer initializer. Shared backbone/DiT weights can
   be loaded from a base checkpoint, but the custom category-specific path
   still requires embodiment data and learning. This is a source-level
   conclusion; exact checkpoint tensor values were not inspected.
3. Stage23 O1@200 remains a real bounded negative for its exact recipe and
   the free-running temporal-profile failure remains strongly observed. The
   earlier phrase “single-sample memorization failure” is too strong as a
   general claim: O1 exposed one example for 200 optimizer steps, while the
   official custom example uses batch 32 and 2000 steps, a 320× larger nominal
   example-presentation count, a shorter H16 target, and different defaults.
4. The observed oscillatory profile is therefore not yet attributable to the
   four-step sampler alone. State dropout, augmentation, VLLN/projector
   settings, horizon/padding, batch/LR schedule, custom-slot learning, data
   scale, and free-running flow behavior must remain in the attribution set.
5. Modern SONIC is a materially different ABI: GR00T predicts a 64D learned
   motion token plus two 7D hand streams; a SONIC decoder/controller consumes
   tokens at 50 Hz. Its VLA runtime is asynchronous and latency-indexed, and
   its controller—not the VLA—owns low-level smooth whole-body realization.
6. The old decoupled path likewise separates sparse/time-stamped upper-body
   and navigation intent from continuous output: `target_time` waypoints are
   scheduled through an explicit interpolation policy with `max_change_rate`,
   while a lower-body Gear WBC/RL policy generates body actions.

## A. Exact N1.7 action-head trace

The pinned source supports this contract:

```text
physical action streams
  -> StateActionProcessor: relative transform (when enabled) and normalization
  -> pad to max_action_horizon=40 and max_action_dim=132
  -> action_mask: zero invalid temporal slots and inactive width
  -> category-specific state/action encoders selected by embodiment_id
  -> x_t=(1-t) noise + t action, v_target=action-noise
  -> action-conditioned DiT / AlternateVLDiT
  -> category-specific velocity decoder
  -> masked velocity MSE
  -> inference Gaussian H40x132 + 4 Euler velocity steps
  -> unnormalize / relative-to-absolute / slice configured modalities
  -> physical action dictionaries
```

The padding is not absent from the forward graph: padded H×D values are
constructed and flow through the action head, while the training mask removes
their direct loss contribution. Therefore 122 inactive width dimensions and
any unused temporal slots are masked targets, not proof that those channels
are computationally absent. This is a plausible nuisance-channel question,
not a demonstrated bug.

For relative EEF, the processor uses the last observed state as one fixed
reference for the chunk and composes `T_reference^-1 @ T_target`; it is not a
frame-to-frame delta chain. Gripper/action streams can remain absolute per
modality config. The upstream model/action head has no explicit `0.05 s`
input; Stage22's 0.05 s was a project-owned TEST_ONLY timing contract.

### Pinned source evidence

{source_table(isaac)}

### `NEW_EMBODIMENT` finding

The source tags `NEW_EMBODIMENT` as finetune-only and maps it to category ID
10; `UNITREE_G1_SONIC` maps to a different posttrain category. The category
specific linear layers allocate `W[num_categories, input_dim, hidden_dim]`
with `0.02 * randn` and zero bias, then select `W[cat_ids]`/`b[cat_ids]` at
runtime. No audited loading or initialization path copies an existing
embodiment slot into ID 10. The safe conclusion is therefore:

```text
shared pretrained backbone/DiT: loadable from base checkpoint
custom category-specific slot: no source-level copy initializer; must be learned
exact slot tensor state in a particular checkpoint: not inspected in this review
```

### Effective state dropout and O1 scale

The Stage20S/22/23 launcher explicitly passes `state_dropout_prob=0.0` to the
model and processor. The official finetune CLI default is `0.2`; the raw N1.7
model-config default is `0.8`. Both the processor and action head contain
state-dropout logic, so this is a real recipe difference, not merely a log
label. Zero dropout is sensible for a deliberate one-example diagnostic but
means O1 is not a direct reproduction of the official regularized recipe.

O1 uses batch 1 and 200 optimizer steps: 200 optimizer-example presentations
and 200 stochastic flow noise/time draws when gradient accumulation is one.
The official SO100 command uses global batch 32 and 2000 steps: 64,000
nominal example presentations/noise-time draws, or 320× O1. That count is a
presentation budget, not 64,000 unique windows; the official dataset has
multiple episodes/windows. O1@200 is consequently a bounded exact-recipe
negative, not evidence of architectural unlearnability.

### Horizon/mask and RTC

The official SO100 modality asks for 16 action indices while N1.7's maximum
head horizon is 40. Training pads to H40 and width 132, then zeros the mask
after the valid horizon and after the active action width. In inference, the
head produces its configured H40 universal tensor, while the processor slices
the configured `delta_indices` horizon and un-applies only the configured
physical streams. The open-loop evaluator executes only a separate
`execution_horizon` (16 in the official example).

RTC is an inpainting-style inter-chunk primitive. It feeds the previous action
tail into the new action overlap, freezes an initial overlap prefix, and ramps
the remaining overlap velocity. It is intended for latency and inter-chunk
continuity. It can reduce a boundary jump, but it does not explain or repair
high-frequency oscillation already present inside a standalone predicted
chunk. The upstream docs also say this primitive is not wired into
`Gr00tPolicy`/server-client and has no ready-made example at this revision.

## B. Official recommendations and symptom mapping

The pinned deployment guide recommends state-relative actions for easier
convergence/inter-chunk consistency, treats both joint and task space as
valid (with low-DoF arms often using joint space), recommends H16 by default
and H≥32 with RTC, and separates inference/replanning frequency from action
execution/camera frequency. It recommends asynchronous inference plus
receding-horizon execution and RTC when latency causes stop-and-go.

Its jitter taxonomy is directly useful for Stage20–23:

| upstream case | source meaning | Stage20–23 mapping |
|---|---|---|
| intra-chunk jitter | model undertrained or data quality issue | Stage20T/23's long, oscillatory H40 model profile is consistent with this phenotype; causality remains open |
| inter-chunk jitter | adjacent chunk predictions inconsistent | Stage21's old-prediction/new-chunk inconsistency is consistent with this case; achieved-to-new-first remained small, so it is prediction inconsistency rather than an adapter jump |
| smooth chunks, robot shakes | hardware/low-level control | not tested and not a current claim |
| stop-and-go | end-to-end latency exceeds execution period | Stage21 synchronous paused loop deliberately did not qualify real-time behavior |

The custom-embodiment guide uses a five-episode SO100 fixture, H16 joint
relative arm plus absolute gripper, batch 32 and 2000 steps, color jitter,
and train-trajectory open-loop curves as a primary sanity check. The guide
also warns that poor held-out performance with only five episodes is data
scarcity, not necessarily a code bug. These are upstream examples and
recommendations, not robot-specific guarantees.

## C. Modern SONIC-backed whole-body VLA path

The pinned path is:

```text
VR/teleop/controller motion and proprioception
  -> universal-token encoder / FSQ representation
  -> controller token_state
  -> LeRobot action.motion_token (64D) plus hand streams
  -> GR00T UNITREE_G1_SONIC action chunk, H40
  -> asynchronous Python inference worker
  -> latency-compensated action_chunk_index
  -> one 64D token + 7D left/right hand action per 50-Hz publish step
  -> SONIC C++ decoder/controller
  -> whole-body joint command
```

The SONIC action ABI is 64D motion token + 7D + 7D hands = 78D per step.
Relative to N1.7's universal width, 64/132 = 48.48% for the motion token and
78/132 = 59.09% if the hand streams are counted together. The project
10D physical EEF+gripper row is 7.58% of 132, but the ratios are only a width
comparison—not a claim that the spaces are semantically interchangeable.

The source declares VLA inference at 2.5 Hz (0.4 s interval), action publish
at 50 Hz (20 ms), and H40. If all H40 outputs are consumed at 50 Hz, one VLA
chunk spans 0.8 s between its first and last sample window; the 0.4 s update
interval creates nominal temporal overlap. This is distinct from SONIC's
controller reference lookahead: the released controller uses 4 or 10 frames
at 20 ms, approximately 80 or 200 ms. Those lookahead values are not total
end-to-end latency.

The runtime marks prefix steps stale by rounding inference delay × 50 Hz and
starting at that index, while a background worker obtains fresh observations
and chunks. The exporter records `action.motion_token` from controller
`token_state` (or zeros when unavailable); the upstream docs describe VR
whole-body teleoperation demonstrations as the label source. Sensor-side
stale checks use 100 ms for SONIC pose and 200 ms for planner messages.
SONIC's pretrained decoder/controller owns the conversion from token to
smooth, dynamically feasible full-body joint motion. This is materially
different from the current dense EEF H40 path, where a task-space adapter and
native WBC own realization.

## D. Older decoupled-WBC path

The old design passes high-level fields including `target_upper_body_pose`,
`target_time`, `navigate_cmd`, and `base_height_command`. The teleop loop
attaches a monotonic timestamp and target completion time; the upper-body
policy is explicitly described as open-loop interpolation. `InterpolationPolicy`
stores time-stamped waypoints, drops waypoints before
`interpolation_garbage_collection_time`, and schedules each waypoint subject
to `max_change_rate`. `get_action(time)` evaluates the interpolator and emits
the current continuous pose. The composed policy then supplies these
interpolated upper-body values to a lower-body Gear WBC policy, which selects
standing/walking RL policies based on navigation command magnitude and emits
body joint action.

Therefore this interface is sparse/time-stamped intent followed by an
explicit declared interpolation/controller layer; it is not a high-level
model emitting every low-level smooth step. This is an architectural analogy
for option C below, not a recommendation to hide post-hoc correction in the
current benchmark.

## E. Architecture comparison

{chr(10).join(comparison_rows)}

For the current project row, the exact frozen description is:

```text
Vision+Language+StateV1
 -> NEW_EMBODIMENT GR00T
 -> dense H40 x 10D EEF+gripper
 -> 0.05 s TEST_ONLY sample period (~2.0 s nominal chunk; 1.95 s endpoint span)
 -> task-space trajectory adapter
 -> native WBC
```

## F. Stage23 claim reassessment

The following prior conclusions are retained but bounded:

- `FREE_RUNNING_TEMPORAL_PROFILE_FAILURE_STRONGLY_SUPPORTED` remains an
  observed phenotype: O1@200's sealed profiles show approximately 25.57×
  expert median path length, median adjacent direction cosine -0.5493,
  back-and-forth rate 0.7632, and high-frequency first-difference energy
  fraction 0.8436; the three O1@200 outputs were finite but failed the
  declared test-only SafetyGate speed bound.
- `O1@200` remains a valid bounded negative for the exact `TINY1_H2STATS`,
  batch-1, 200-step, Stage23 recipe. It does not establish that a custom
  embodiment is unlearnable or that the representation is wrong.
- The phrase “single-sample memorization failure” should be demoted to
  “single-sample, bounded-recipe overfit diagnostic did not produce a healthy
  free-running profile.” The official example has 320× the nominal optimizer
  example/noise-time exposure and materially different H, augmentation,
  dropout, and default trainability settings.

### Attribution matrix

{chr(10).join(attribution_rows)}

In particular, the four-step Euler sampler is only one unresolved factor.
The source review finds no action-mask defect, no relative-transform semantic
ambiguity in the audited path, and no evidence that the Stage23 evaluator's
identity test is wrong. It does find legitimate recipe differences: O1 uses
state dropout 0 rather than official CLI 0.2, no explicit color jitter while
the official example enables it, freezes VLLN explicitly while the raw model
default is trainable, uses B=1, uses H40 rather than the official H16 example,
and relies on the custom category slot. These differences are confounds to
control, not reasons to erase the oscillation observation.

## G. Candidate next-stage options (analysis only)

{chr(10).join(option_rows)}

No option was implemented or selected by this audit. The evidence most
directly supports a PI choice between a narrowly isolating fixed-noise /
teacher-forced flow diagnostic and a properly controlled official-scale
custom-embodiment replication. A future runtime/representation change should
wait for that decision.

## Required return

```text
ROLE: CODEX_CORELAB
STAGE: Stage23 upstream architecture review before Stage24
GIT_SHA: {project['head']}
ISAAC_GROOT_SHA: {data['pins']['isaac_gr00t_sha']}
GROOT_WBC_SHA: {data['pins']['groot_whole_body_control_sha']}
C1_EXECUTED: false
NEW_MODEL_FORWARD: false
NEW_TRAINING: false
HOPPER_PBS: false
VALIDATION_E_READ: false
HOLDOUT_F_READ: false
UPSTREAM_ACTION_HEAD_TRACE: COMPLETE; dense H40x132 flow head with masked padding/loss and four-step Euler inference
NEW_EMBODIMENT_PROJECTOR_FINDING: category ID 10 finetune-only custom slot; no source-level copy initializer; embodiment-specific path requires learning
EFFECTIVE_STATE_DROPOUT_FINDING: Stage23 launcher/model/processor 0.0; official CLI 0.2; raw model default 0.8
O1_VS_OFFICIAL_FINETUNE_SCALE_FINDING: O1 200 batch-1 presentations versus official SO100 nominal 64,000 batch-32 presentations; 320x exposure difference; O1 is bounded negative only
SONIC_VLA_ABI_FINDING: 64D motion token + 7D/7D hands, H40, 50Hz publish, 2.5Hz inference, SONIC decoder owns whole-body realization
ASYNC_LATENCY_RTC_FINDING: SONIC async worker skips stale prefix by latency-compensated index; N1.7 RTC is low-level/manual and mainly inter-chunk
DECOUPLED_INTERPOLATION_FINDING: sparse/time-stamped upper-body/nav intent becomes continuous output through max-change-rate interpolation; lower body is separate Gear WBC/RL
STAGE23_CLAIM_REASSESSMENT: phenotype remains strongly observed; O1@200 is a bounded negative; prior general single-sample memorization wording was too strong; sampler-only attribution is unsupported
ARCHITECTURE_OPTIONS: A/B/C/D compared analytically; none implemented or selected
RECOMMENDED_NEXT_EXPERIMENT_FOR_PI_REVIEW: PI review should choose between the already-prepared fixed-noise teacher-forced C1 diagnostic and a controlled official-scale custom-embodiment replication; no experiment executed here
RESULT: UPSTREAM_ARCHITECTURE_REVIEW_COMPLETE
BLOCKER: NONE_FOR_SOURCE_REVIEW; C1 remains on hold pending PI architecture decision
NEXT_GATE_REQUEST: GPT_PI_ARCHITECTURE_REVIEW_AND_STAGE24_DECISION
```

## Reproducibility

```text
python scripts/stage23_upstream_architecture_audit.py
```

The companion JSON records every inspected path's pinned Git blob ID,
content SHA-256, line count, and requested marker line numbers.
"""


def _build_data(repo: Path, isaac_root: Path, wbc_root: Path) -> dict[str, Any]:
    isaac = PinnedTree(isaac_root, ISAAC_SHA)
    wbc = PinnedTree(wbc_root, WBC_SHA)

    isaac_specs = dict(ISAAC_SPECS)
    for path in isaac.paths("gr00t/data/state_action"):
        isaac_specs.setdefault(path, [])

    isaac_inventory = {
        path: _inventory_entry(isaac, path, patterns)
        for path, patterns in isaac_specs.items()
    }
    wbc_inventory = {
        path: _inventory_entry(wbc, path, patterns)
        for path, patterns in WBC_SPECS.items()
    }
    project_launcher = _inventory_entry(
        PinnedTree(repo, _project_status(repo)["head"]),
        "scripts/stage20s_train_gr00t.py",
        [
            ("launcher_dropout_zero", r"state_dropout_prob\s*=\s*0\.0"),
            ("launcher_horizon_40", r"action_horizon\s*=\s*40"),
            ("launcher_relative_action", r"use_relative_action\s*=\s*True"),
            ("launcher_vlln_frozen", r"tune_vlln\s*=\s*False"),
            ("launcher_diffusion_flag", r"tune_diffusion_model"),
            ("launcher_batch_one", r"global_batch_size\s*=\s*1"),
            ("launcher_warmup", r"warmup_ratio\s*=\s*0\.05"),
        ],
    )

    comparison_table = [
        {
            "system": "Current Go2+ARX5 ABI",
            "representation": "Vision+Language+StateV1 -> NEW_EMBODIMENT GR00T -> dense EEF XYZ+rot6d+gripper",
            "active_dims_over_132": "10/132 = 7.58%",
            "horizon_and_sample": "H40; 0.05 s TEST_ONLY",
            "duration_and_update": "~2.0 s nominal window / 1.95 s first-to-last; production TBD",
            "execution_and_latency": "task-space adapter -> native WBC; Stage21 paused/receding diagnostic, no production async",
            "smoothness_and_labels": "native WBC; expert desired EEF + gripper labels, achieved retained separately",
        },
        {
            "system": "Isaac-GR00T N1.7 custom example",
            "representation": "NEW_EMBODIMENT SO100 joint-relative arm + absolute gripper",
            "active_dims_over_132": "configured physical prefix; remaining width padded/masked",
            "horizon_and_sample": "H16 configured in modality; model max H40",
            "duration_and_update": "source does not impose a universal physical period; execution_horizon=16",
            "execution_and_latency": "open-loop evaluator executes prefix then replans; RTC primitive manual at pinned revision",
            "smoothness_and_labels": "model action head plus robot controller; LeRobot demo trajectories",
        },
        {
            "system": "GEAR-SONIC VLA",
            "representation": "64D motion token + 7D left hand + 7D right hand",
            "active_dims_over_132": "64/132 = 48.48%; full 78/132 = 59.09% width comparison",
            "horizon_and_sample": "H40; 20 ms / 50 Hz publish",
            "duration_and_update": "~0.8 s if H40 is consumed at 50 Hz; VLA update 2.5 Hz / 0.4 s",
            "execution_and_latency": "async worker; latency-compensated stale-prefix index; SONIC C++ decoder/controller",
            "smoothness_and_labels": "SONIC owns whole-body smoothness/dynamic realization; token_state from teleop/controller exporter",
        },
        {
            "system": "Older decoupled-WBC",
            "representation": "target_upper_body_pose + target_time + navigate_cmd + base_height_command",
            "active_dims_over_132": "not a 132D GR00T tensor; sparse structured intent",
            "horizon_and_sample": "time-stamped waypoints; interpolation evaluated at controller time",
            "duration_and_update": "target-time dependent; no fixed VLA H",
            "execution_and_latency": "max_change_rate interpolation -> upper-body command; lower-body Gear WBC/RL",
            "smoothness_and_labels": "interpolator/controller owns continuous motion; teleop/planner supplies waypoints",
        },
    ]

    attribution_matrix = [
        {"hypothesis": "input drift", "status": "NOT DEMONSTRATED", "boundary": "No new model run was performed; source review cannot close the prior run's input provenance."},
        {"hypothesis": "GPU/resource", "status": "LOW PLAUSIBILITY FOR PHASE A", "boundary": "Existing sealed Phase B evidence reports H200 resource/OOM ruled out; this review did not rerun it."},
        {"hypothesis": "optimizer/LR schedule", "status": "PLAUSIBLE UNRESOLVED", "boundary": "O1 has only 200 steps and each max-step run has its own warmup/decay schedule."},
        {"hypothesis": "flow objective fit", "status": "NOT DEMONSTRATED", "boundary": "Free-running profile failure does not prove teacher-forced objective failure."},
        {"hypothesis": "normalization", "status": "NO DEFECT FOUND IN AUDIT", "boundary": "Tiny-stat degeneracy is retained as a diagnostic warning, not a demonstrated cause of H2."},
        {"hypothesis": "temporal oscillation", "status": "STRONGLY OBSERVED PHENOTYPE", "boundary": "High path/tortuosity/frequency metrics support the phenotype, not one unique mechanism."},
        {"hypothesis": "132D inactive-noise burden", "status": "MECHANISTICALLY PLAUSIBLE", "boundary": "Analytic inactive/active noise energy ratio 12.2x is not causal proof."},
        {"hypothesis": "4-step Euler sampler", "status": "UNRESOLVED", "boundary": "Sampler is a candidate factor; no C1 or sampler comparison was authorized."},
        {"hypothesis": "data scarcity/generalization", "status": "PLAUSIBLE HIGH", "boundary": "Official custom guidance uses substantially more presentations and multiple demonstration windows."},
        {"hypothesis": "state dropout / augmentation / trainability recipe", "status": "REAL RECIPE CONFOUNDS", "boundary": "Stage23 differs from official defaults in dropout, color jitter, VLLN, batch, H, and custom-slot exposure; none is proven causal here."},
    ]

    options = [
        {
            "id": "A",
            "scientific_upside": "Directly tests whether the dense H40 EEF hypothesis becomes healthy under official-scale custom-embodiment training and matching runtime semantics.",
            "cost_and_data": "Highest immediate GPU/data cost; requires a larger robot-specific or controlled simulation dataset and careful timing lock.",
            "falsifier_or_strengthener": "Healthy held-out profiles and overlap under the same dense ABI strengthen the current architecture; persistent oscillation after controlled scale weakens dense H40."},
        {
            "id": "B",
            "scientific_upside": "Separates prediction horizon/profile burden from execution latency by using shorter state-relative chunks, async scheduling, and RTC where valid.",
            "cost_and_data": "Moderate implementation/runtime cost; requires explicit timing, chunk-age, and inter-chunk accounting, and can mask full-chunk quality if not audited.",
            "falsifier_or_strengthener": "Healthy raw chunks plus stable receding execution strengthen a runtime explanation; raw intra-chunk oscillation remains unchanged and falsifies a runtime-only explanation."},
        {
            "id": "C",
            "scientific_upside": "Tests whether the VLA is better suited to sparse task-space intent while an explicit, declared interpolation/rate-limit layer owns smoothness before WBC.",
            "cost_and_data": "Requires a new waypoint/target-time ABI, interpolation contract, and matched evaluation; increases interface design burden and may reduce endpoint expressivity.",
            "falsifier_or_strengthener": "Stable waypoints plus transparent interpolation with improved task-space outcomes strengthen it; failure before interpolation or dependence on hidden correction weakens it."},
        {
            "id": "D",
            "scientific_upside": "Moves temporal smoothness into a learned compact task-space manifold, inspired by SONIC without requiring the Go2+ARX5 VLA to emit low-level whole-body channels.",
            "cost_and_data": "Largest research cost: trajectory corpus, tokenizer/decoder training, representation validation, and a new checkpoint/ABI lifecycle.",
            "falsifier_or_strengthener": "A frozen token decoder reconstructing WBC-compatible smooth references and improving held-out profile/overlap metrics strengthens it; no reconstruction gain or embodiment leakage weakens it."},
    ]

    project = _project_status(repo)
    return {
        "schema": "STAGE23_UPSTREAM_ARCHITECTURE_REVIEW_V1",
        "generated_date": str(date.today()),
        "pins": {
            "isaac_gr00t_sha": ISAAC_SHA,
            "groot_whole_body_control_sha": WBC_SHA,
            "isaac_gr00t_commit": isaac.commit_metadata(),
            "groot_whole_body_control_commit": wbc.commit_metadata(),
        },
        "scope": {
            "source_only": True,
            "model_constructed": False,
            "new_model_forward": False,
            "new_inference": False,
            "new_training": False,
            "checkpoint_manipulation": False,
            "hopper_pbs": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc_execution": False,
            "ros": False,
            "zmq_runtime": False,
            "hardware": False,
        },
        "project_context": project,
        "source_inventory": {
            "isaac_gr00t": isaac_inventory,
            "groot_whole_body_control": wbc_inventory,
            "project_stage23_launcher": project_launcher,
        },
        "findings": {
            "action_head_trace": {
                "universal_shape": [40, 132],
                "physical_project_shape": [40, 10],
                "inactive_width": 122,
                "sampler_steps": 4,
                "relative_eef_reference": "last observed state, fixed reference for chunk",
                "timestamp_input": "not an explicit action-head tensor; supplied by modality/dataset semantics",
                "loss": "masked velocity MSE over action_mask",
            },
            "new_embodiment": {
                "tag": "NEW_EMBODIMENT",
                "category_id": 10,
                "classification": "finetune-only custom category",
                "initializer": "CategorySpecificLinear uses independent 0.02*randn weights and zero bias; no audited copy initializer",
                "shared_pretrained_components": ["Cosmos/Qwen visual-language backbone when loaded", "DiT/action head shared weights when present in checkpoint"],
                "custom_learning_requirement": "category-specific state/action encoder/decoder/projector path requires embodiment learning",
                "exact_checkpoint_slot_values_inspected": False,
            },
            "state_dropout": {
                "stage23_launcher_and_processor": 0.0,
                "official_finetune_cli_default": 0.2,
                "raw_model_config_default": 0.8,
                "note": "processor and action-head contain state-dropout logic; recipe difference matters for one-sample diagnostics",
            },
            "exposure_scale": {
                "o1_batch": 1,
                "o1_steps": 200,
                "o1_presentations": 200,
                "o1_noise_time_draws": 200,
                "official_batch": 32,
                "official_steps": 2000,
                "official_nominal_presentations": 64000,
                "official_to_o1_nominal_ratio": 320.0,
                "caveat": "presentations are not unique windows; official fixture has multiple episodes/windows",
            },
            "horizon_mask": {
                "n1d7_max_action_horizon": 40,
                "official_so100_configured_horizon": 16,
                "max_action_dim": 132,
                "training_behavior": "zero-pad temporal and width; action_mask zeros invalid temporal suffix and inactive width",
                "inference_behavior": "head returns configured universal horizon; processor slices configured delta-index horizon and physical streams",
            },
            "rtc": {
                "overlap": "previous action tail inpainted into new overlap",
                "frozen": "initial overlap velocity frozen for latency-aligned prefix",
                "ramp": "remaining overlap velocity receives exponential ramp",
                "primary_target": "inter-chunk continuity and latency",
                "not_primary_target": "intra-chunk high-frequency oscillation",
                "official_integration_status": "low-level/manual primitive, not wired into Gr00tPolicy/server-client at pin",
            },
            "official_guidance": {
                "relative": "state-relative recommended for convergence/inter-chunk consistency",
                "space": "joint and task space both valid; low-DoF arms often joint space in guide",
                "episodes": "at least 100 valid for single-task; ~30 may suffice for narrow domain; 200+ more stable",
                "training": "SO100 custom example B32/N2000; generic config B64/N10000; source is guidance not guarantee",
                "chunk": "default H16; H>=32 with RTC",
                "async": "asynchronous inference + receding horizon + RTC recommended for latency/stop-and-go",
                "jitter": "intra-chunk vs inter-chunk vs smooth-chunk low-level issue",
            },
        },
        "sonic_findings": {
            "motion_token_dim": 64,
            "hand_dims": {"left": 7, "right": 7},
            "full_action_dim": 78,
            "motion_token_over_132": 64 / 132,
            "full_action_over_132": 78 / 132,
            "vla_horizon": 40,
            "action_publish_hz": 50,
            "action_period_s": 0.02,
            "vla_inference_hz": 2.5,
            "vla_inference_period_s": 0.4,
            "implied_h40_span_if_consumed_at_50hz_s": 0.8,
            "controller_reference_lookahead_s": {"low_latency_4_frames": 0.08, "default_10_frames": 0.2},
            "latency_handling": "round(inference_delay*50Hz), clip to horizon, start from stale-compensated prefix",
            "label_source": "controller/teleop token_state copied to LeRobot action.motion_token",
            "smoothness_owner": "SONIC decoder/controller at 50Hz",
            "stale_sensor_thresholds_ms": {"sonic_pose": 100, "planner": 200},
        },
        "decoupled_findings": {
            "high_level_fields": ["target_upper_body_pose", "target_time", "navigate_cmd", "base_height_command"],
            "interface_class": "sparse/time-stamped intent followed by explicit interpolation/controller",
            "interpolation": "PoseTrajectoryInterpolator with waypoint garbage collection and max_change_rate",
            "lower_body": "Gear WBC ONNX standing/walking policy selected from navigation command magnitude",
            "continuous_model_steps": False,
        },
        "comparison_table": comparison_table,
        "stage23_claim_reassessment": {
            "phenotype": "FREE_RUNNING_TEMPORAL_PROFILE_FAILURE_STRONGLY_SUPPORTED",
            "o1_scope": "bounded exact-recipe negative; not architectural unlearnability",
            "single_sample_wording": "too strong as a general claim relative to official custom baseline",
            "observed_profile": {
                "o1_at_200_median_path_ratio": 25.57,
                "o1_at_200_median_direction_cosine": -0.5493,
                "o1_at_200_back_and_forth_rate": 0.7632,
                "o1_at_200_high_frequency_first_difference_fraction": 0.8436,
                "safety_gate": "all three O1@200 outputs rejected for test-only linear-speed bound",
            },
            "recipe_confounders": ["state_dropout", "image_augmentation", "VLLN/projector trainability", "batch/LR schedule", "H40 versus official H16", "custom category slot", "data/exposure scale", "four-step Euler sampler"],
            "attribution_matrix": attribution_matrix,
        },
        "architecture_options": options,
        "required_return": {
            "role": "CODEX_CORELAB",
            "stage": "Stage23 upstream architecture review before Stage24",
            "c1_executed": False,
            "result": "UPSTREAM_ARCHITECTURE_REVIEW_COMPLETE",
            "blocker": "NONE_FOR_SOURCE_REVIEW; C1 remains on hold pending PI architecture decision",
            "next_gate_request": "GPT_PI_ARCHITECTURE_REVIEW_AND_STAGE24_DECISION",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--isaac-source", type=Path, default=Path("/home/liuyu-nfs/Projects/isaac-gr00t"))
    parser.add_argument("--wbc-source", type=Path, default=Path("/home/liuyu-nfs/Projects/GR00T-WholeBodyControl"))
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo.resolve()
    markdown_path = args.markdown or repo / "reports" / "STAGE23_UPSTREAM_ARCHITECTURE_REVIEW.md"
    json_path = args.json_path or repo / "reports" / "STAGE23_UPSTREAM_ARCHITECTURE_REVIEW.json"

    data = _build_data(repo, args.isaac_source.resolve(), args.wbc_source.resolve())
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(data), encoding="utf-8")
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {markdown_path}")
    print(f"wrote {json_path}")
    print(f"project_git_sha={data['project_context']['head']}")
    print(f"isaac_groot_sha={ISAAC_SHA}")
    print(f"groot_wbc_sha={WBC_SHA}")
    print("execution_boundary=model_constructed=false,new_model_forward=false,new_training=false,hopper_pbs=false,wbc=false,hardware=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
