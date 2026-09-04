"""Deterministic tests for the Stage26 H1 C1 preparation boundary."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "stage26_h1_c1_fixed_noise_local_flow.py"
SPEC = importlib.util.spec_from_file_location("stage26_h1_c1_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_index(checkpoint: Path, shard_name: str = "model.safetensors") -> Path:
    index_path = checkpoint / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {"fixture.weight": shard_name}}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return index_path


def _file_record(checkpoint: Path, relative: str) -> dict[str, object]:
    path = checkpoint / relative
    payload = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "regular_file": True,
        "symlink": False,
    }


def _o1_identity_fixture(root: Path) -> tuple[Path, Path, Path, dict[str, dict[str, object]]]:
    checkpoint = root / "o1"
    checkpoint.mkdir()
    index_path = _write_index(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"o1-shard-v1")
    experiment_cfg = checkpoint / "experiment_cfg"
    experiment_cfg.mkdir()
    history: dict[str, dict[str, object]] = {}
    for relative in MODULE.EXPECTED_O1_SUPPLEMENTAL_METADATA_ALLOWLIST:
        payload = f"historical:{relative}".encode("utf-8")
        path = checkpoint / relative
        path.write_bytes(payload)
        record = _file_record(checkpoint, relative)
        history[relative] = {
            "status": "EXACT_HISTORICAL_HASH_FOUND",
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "evidence": ["synthetic Stage23 historical fixture"],
        }

    core_paths = ("model.safetensors.index.json", "model.safetensors")
    seal = root / "checkpoint_file_seal.tsv"
    rows = []
    for relative in core_paths:
        record = _file_record(checkpoint, relative)
        rows.append(
            "candidate_200\t"
            f"{relative}\tregular file\t{record['bytes']}\t{record['sha256']}"
        )
    seal.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return checkpoint, seal, index_path, history


class Stage26H1C1Tests(unittest.TestCase):
    def test_static_plan_reuses_exact_stage23_grid(self) -> None:
        plan = MODULE.protocol_plan()
        MODULE.assert_protocol_static(plan)
        self.assertEqual(plan["schema"], MODULE.STAGE26_SCHEMA)
        self.assertTrue(plan["stage23_c1_protocol_reused"])
        self.assertEqual(plan["probe_contract"]["probe_count"], 9)
        self.assertEqual(plan["execution_budget"]["maximum_c1_flow_evaluations"], 18)
        self.assertTrue(plan["readiness"]["h1_c1_execution_ready"])
        self.assertTrue(plan["readiness"]["runtime_identity_revalidation_required"])
        self.assertFalse(plan["readiness"]["execution_started"])
        self.assertEqual(
            plan["fixed_identity"]["base_tree_sha256"],
            MODULE.EXPECTED_BASE_TREE_SHA256,
        )
        self.assertEqual(
            plan["fixed_identity"]["o1_seal_sha256"],
            MODULE.EXPECTED_O1_SEAL_SHA256,
        )
        self.assertEqual(
            plan["fixed_identity"]["o1_supplemental_metadata_allowlist"],
            list(MODULE.EXPECTED_O1_SUPPLEMENTAL_METADATA_ALLOWLIST),
        )
        self.assertEqual(
            [(item["noise_seed"], item["t_discretized"]) for item in plan["probe_contract"]["probes"]],
            [
                (2223, 67),
                (2223, 369),
                (2223, 783),
                (2224, 67),
                (2224, 369),
                (2224, 783),
                (2225, 67),
                (2225, 369),
                (2225, 783),
            ],
        )

    def test_static_plan_has_no_enabled_execution_boundary(self) -> None:
        plan = MODULE.protocol_plan()
        self.assertTrue(all(value is False for value in plan["execution_boundary"].values()))
        self.assertTrue(plan["output_schema"]["descriptive_only"])
        self.assertTrue(plan["output_schema"]["no_new_scientific_pass_threshold"])
        self.assertEqual(
            plan["runtime_precision"]["device_cast"],
            "model.to(device=args.device, dtype=torch.bfloat16)",
        )

    def test_source_map_and_direct_runner_are_pre_sampler(self) -> None:
        source_map = MODULE._source_map_static()
        self.assertEqual(source_map["status"], "PASS")
        self.assertTrue(source_map["pre_sampler"])
        self.assertFalse(source_map["complete_model_forward_called"])
        self.assertFalse(source_map["free_running_sampler_called"])

        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_teacher_forced_local_flow"
        )
        calls = [
            node.func.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertNotIn("get_action", calls)
        self.assertNotIn("get_action_with_features", calls)
        self.assertNotIn("forward", calls)

    def test_canonical_probes_are_created_once_and_reused(self) -> None:
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        run_function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_one_checkpoint"
        )
        run_canonical_calls = [
            node
            for node in ast.walk(run_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_canonical_probe_inputs"
        ]
        self.assertEqual(run_canonical_calls, [])

        execute_function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "execute"
        )
        execute_canonical_calls = [
            node
            for node in ast.walk(execute_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_canonical_probe_inputs"
        ]
        self.assertEqual(len(execute_canonical_calls), 1)

        run_fixed_probe_keywords = [
            keyword.arg
            for node in ast.walk(execute_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_one_checkpoint"
            for keyword in node.keywords
        ]
        self.assertEqual(run_fixed_probe_keywords.count("fixed_probes"), 2)

    def test_tree_digest_is_order_independent_but_bytes_sensitive(self) -> None:
        entries = [
            {"path": "b", "bytes": 2, "sha256": "bb"},
            {"path": "a", "bytes": 1, "sha256": "aa"},
        ]
        reversed_entries = list(reversed(entries))
        self.assertEqual(
            MODULE._tiny_tree_digest(entries),
            MODULE._tiny_tree_digest(reversed_entries),
        )
        changed = [dict(item) for item in entries]
        changed[0]["bytes"] = 3
        self.assertNotEqual(
            MODULE._tiny_tree_digest(entries),
            MODULE._tiny_tree_digest(changed),
        )

    def test_regular_file_manifest_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "regular").write_bytes(b"ok")
            link = root / "link"
            try:
                link.symlink_to(root / "regular")
            except (NotImplementedError, OSError):
                self.skipTest("symlinks are unavailable in this test environment")
            with self.assertRaises(ValueError):
                MODULE._regular_file_entries(root)

    def test_checkpoint_preflight_rejects_wrong_index_without_tensor_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "model.safetensors.index.json").write_text(
                '{"weight_map": {}}\\n', encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                MODULE._checkpoint_preflight("BASE", checkpoint)

    def test_base_preflight_rejects_shard_mutation_with_unchanged_index(self) -> None:
        stage25 = MODULE._load_stage25_identity_module()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "base"
            checkpoint.mkdir()
            index_path = _write_index(checkpoint)
            shard = checkpoint / "model.safetensors"
            shard.write_bytes(b"base-shard-v1")
            index_sha = MODULE._sha256_file(index_path)
            tree_sha = stage25.stage24_tree_sha256(
                stage25.walk_regular_files(checkpoint),
                MODULE.EXPECTED_BASE_TREE_PREFIX,
            )
            with patch.object(MODULE, "EXPECTED_BASE_INDEX_SHA256", index_sha), patch.object(
                MODULE, "EXPECTED_BASE_TREE_SHA256", tree_sha
            ), patch.object(stage25, "EXPECTED_BASE_INDEX_SHA256", index_sha), patch.object(
                stage25, "EXPECTED_BASE_TREE_SHA256", tree_sha
            ):
                identity = MODULE._checkpoint_preflight("BASE", checkpoint)
                self.assertTrue(identity["full_byte_identity_verified_before_model_load"])
                shard.write_bytes(b"base-shard-mutated")
                with self.assertRaisesRegex(ValueError, "BASE canonical tree SHA mismatch"):
                    MODULE._checkpoint_preflight("BASE", checkpoint)

    def test_o1_preflight_rejects_shard_and_supplemental_mutations(self) -> None:
        stage25 = MODULE._load_stage25_identity_module()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, seal, _, history = _o1_identity_fixture(Path(temporary))
            index_sha = MODULE._sha256_file(checkpoint / "model.safetensors.index.json")
            seal_sha = MODULE._sha256_file(seal)
            patchers = (
                patch.object(MODULE, "EXPECTED_O1_INDEX_SHA256", index_sha),
                patch.object(MODULE, "EXPECTED_O1_SEAL_SHA256", seal_sha),
                patch.object(stage25, "EXPECTED_O1_INDEX_SHA256", index_sha),
                patch.object(stage25, "EXPECTED_O1_SEAL_SHA256", seal_sha),
                patch.object(stage25, "SUPPLEMENTAL_METADATA_PRIOR_BINDING", history),
            )
            with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4]:
                identity = MODULE._checkpoint_preflight(
                    "O1@200", checkpoint, o1_seal=seal
                )
                self.assertTrue(identity["full_byte_identity_verified_before_model_load"])

                shard = checkpoint / "model.safetensors"
                shard.write_bytes(b"o1-shard-mutated")
                with self.assertRaisesRegex(ValueError, "O1 two-layer provenance mismatch"):
                    MODULE._checkpoint_preflight(
                        "O1@200", checkpoint, o1_seal=seal
                    )

                shard.write_bytes(b"o1-shard-v1")
                supplemental = checkpoint / "experiment_cfg" / "config.yaml"
                supplemental.write_bytes(b"supplemental-mutated")
                with self.assertRaisesRegex(ValueError, "O1 two-layer provenance mismatch"):
                    MODULE._checkpoint_preflight(
                        "O1@200", checkpoint, o1_seal=seal
                    )

    def test_preflight_identity_gate_precedes_model_construction(self) -> None:
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        execute_function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "execute"
        )
        call_lines: dict[str, list[int]] = {}
        for node in ast.walk(execute_function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                call_lines.setdefault(node.func.id, []).append(node.lineno)
        self.assertLess(
            max(call_lines["_checkpoint_preflight"]),
            min(call_lines["_build_runtime_processor"]),
        )
        self.assertLess(
            max(call_lines["_checkpoint_preflight"]),
            min(call_lines["_run_one_checkpoint"]),
        )


if __name__ == "__main__":
    unittest.main()
