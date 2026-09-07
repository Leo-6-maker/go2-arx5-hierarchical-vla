from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
B1 = (ROOT / "hpc" / "pbs" / "stage40_b1.pbs").read_text(encoding="utf-8")
LIVE = (ROOT / "hpc" / "pbs" / "stage40_live.pbs").read_text(encoding="utf-8")
PY = (ROOT / "scripts" / "stage40_live_one_handoff.py").read_text(encoding="utf-8")
NATIVE = (ROOT / "scripts" / "stage21_native_receding_horizon.py").read_text(encoding="utf-8")


class Stage40RuntimeContractTest(unittest.TestCase):
    def test_b1_uses_native_python38_runtime(self):
        self.assertIn("stage36_runtime_reconstruction_20260905/torch_cu121_v1/site-packages", B1)
        self.assertIn("stage36_runtime_reconstruction_20260905/torch_cu121_v1/ninja_target_v4/bin/ninja", B1)
        self.assertIn("/opt/miniconda3/envs/isaacgym/bin/python", B1)
        self.assertIn("CONTAINER_NINJA=/runtime/ninja/bin", B1)
        self.assertNotIn("stage22_h200_runtime_deb920_20260902/runtime/venv/lib/python3.12", B1)

    def test_live_has_separate_runtime_commands(self):
        self.assertIn("NATIVE_CMD=(", LIVE)
        self.assertIn("GROOT_CMD=(", LIVE)
        self.assertNotIn("BASE_CMD=(", LIVE)
        native = LIVE.split("NATIVE_CMD=(", 1)[1].split("GROOT_CMD=(", 1)[0]
        groot = LIVE.split("GROOT_CMD=(", 1)[1].split('WORKER_LOG=', 1)[0]
        self.assertIn("NATIVE_SITE", native)
        self.assertIn("GROOT_SITE", groot)
        self.assertNotIn("GROOT_SITE", native)
        self.assertNotIn("NATIVE_SITE", groot)
        self.assertIn("pytorch_25.12-py3.sif", LIVE)
        self.assertIn("stage36_runtime_reconstruction_20260905", LIVE)
        self.assertIn("--mailbox", LIVE)

    def test_scientific_contract_is_unchanged(self):
        self.assertIn("SEEDS = (2223, 2228, 2233, 2238)", PY)
        self.assertIn("HANDOFF_S = 0.9750000000000001", PY)
        self.assertIn("CHUNK_DURATION_S = 1.9500000000000002", PY)
        self.assertIn('PI_GATE = "PI_GATE_STATE_0060"', PY)

    def test_c0_and_c_fail_closed_contracts(self):
        self.assertIn('choices=("b1", "c0", "live", "live-seed")', PY)
        self.assertIn("STAGE40_NONUNIFORM_REFERENCE_TIMESTAMPS", PY)
        self.assertIn("SCIENTIFIC_Q0_TERMINATION", PY)
        self.assertIn("q1_requested", PY)
        self.assertIn("reference_completion", PY)

    def test_c0_safety_gate_is_diagnostic_only(self):
        self.assertIn("def _c0_runtime_checks_pass", PY)
        self.assertIn('"safety_gate_diagnostic_only": True', PY)
        self.assertIn('"c0_admission_semantics": "RUNTIME_ONLY_SAFETYGATE_DIAGNOSTIC"', PY)
        self.assertNotIn(
            'all((body_ok, wrist_ok, state_ok, ids_ok, model_ok, native_ok, gate_ok))',
            PY,
        )

    def test_reference_lifecycle_has_explicit_two_phase_api(self):
        self.assertIn("def initial_bind_reference", NATIVE)
        self.assertIn("def load_reference_chunk", NATIVE)
        self.assertIn("REFERENCE_UPDATE_BEFORE_FIRST_WBC_FORWARD", NATIVE)
        self.assertIn("INITIAL_BIND_AFTER_FIRST_WBC_FORWARD", NATIVE)
        self.assertIn("native.initial_bind_reference(q0)", PY)
        self.assertIn("installed_chunk_a = native.stage21_current_chunk", PY)
        self.assertNotIn("native.load_reference_chunk(chunk_a)", PY)

    def test_initial_anchor_is_derived_from_newly_loaded_chunk(self):
        self.assertIn("pre_anchor_start = self.traj_batch.gamma_p[env_ids, 0].clone()", NATIVE)
        self.assertIn("post_anchor_start = self.traj_batch.gamma_p[env_ids, 0].clone()", NATIVE)
        self.assertIn("anchor = post_anchor_start - pre_anchor_start", NATIVE)
        self.assertNotIn("before_anchor = self.traj_batch.gamma_p[env_ids, 0].clone()", NATIVE)


if __name__ == "__main__":
    unittest.main()
