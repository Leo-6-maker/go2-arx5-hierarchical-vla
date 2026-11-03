import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.wm02.env_preflight import inspect, inspect_members, main


def spec(member):
    return dict(schema="WM02_RESOLVED_SPEC_V1", target_blockers=["TARGET_S1_T2_QUARANTINE_NOT_RELEASED"],
                target=dict(backend="GO2_ARX5_MUJOCO_OCS2", frozen_members=[member],
                            permission={"status": "CLOSED", "native_safety_quarantine_preserved": True},
                            full_control_restore_available=False,
                            scorer={"actual_WM02_standing_task_binding": "UNQUALIFIED"}))


class Preflight(unittest.TestCase):
    def test_real_file_hash_and_no_dynamic_grant(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            file = root / "src" / "ctrl" / "unit.bin"
            file.parent.mkdir(parents=True)
            file.write_bytes(b"frozen identity")
            member = dict(role="src/ctrl/unit.bin", expected=hashlib.sha256(file.read_bytes()).hexdigest(), match=False)
            result = inspect(spec(member), root=root)
            self.assertEqual(result["frozen_member_check"][0]["status"], "MATCH")
            self.assertFalse(result["target_dynamic_authorized"])
            self.assertEqual(result["native_target_admission"], "NOT_PERFORMED")
            self.assertIn("TARGET_S1_T2_QUARANTINE_NOT_RELEASED", result["target_blockers"])
            file.write_bytes(b"unexpected changed state")
            result = inspect(spec(member), root=root)
            self.assertEqual(result["frozen_member_check"][0]["status"], "MISMATCH")
            self.assertIn("TARGET_FROZEN_MEMBER_BYTES_NOT_MATCHED", result["target_blockers"])

    def test_absent_root_is_not_byte_verification(self):
        m = dict(role="src/ctrl/unit.bin", expected="a"*64, match=True)
        result = inspect(spec(m))
        self.assertEqual(result["frozen_member_check"][0]["status"], "NOT_CHECKED_NO_ROOT")
        self.assertIn("NO_LOCAL_TARGET_ROOT_BYTES_CHECKED", result["target_blockers"])

    def test_reject_path_traversal(self):
        for name in ("../out", "/tmp/out", "src/../out"):
            with self.assertRaises(ValueError):
                inspect_members([dict(role=name, expected="a"*64)], Path("/tmp"))

    def test_wrong_backend_fails_before_inspection(self):
        s = spec(dict(role="src/ctrl/unit.bin", expected="a"*64))
        s["target"]["backend"] = "ISAAC_GYM"
        with self.assertRaises(ValueError):
            inspect(s)

    def test_cli_output_not_overwritten_or_mistaken_for_permission(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sp = root / "spec.json";out = root / "result.json"
            sp.write_text(json.dumps(spec(dict(role="unit.bin", expected="a"*64))))
            self.assertEqual(main(["--spec",str(sp),"--output",str(out)]),0)
            self.assertFalse(json.loads(out.read_text())["target_dynamic_authorized"])
            with self.assertRaises(FileExistsError):
                main(["--spec",str(sp),"--output",str(out)])

    def test_help_is_import_only_no_model_or_ros(self):
        cmd = [sys.executable, "-m", "scripts.wm02.env_preflight", "--help"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--target-root", r.stdout)


if __name__ == "__main__":
    unittest.main()
