from pathlib import Path
import tempfile
import unittest
from .task_capture_runtime import require_task_asset


class TerminalTests(unittest.TestCase):
    def test_final_fixed_failure_blocks_physics_without_relabeling_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'TASK_PILOT_CLOSED.json').write_text('{}')
            with self.assertRaisesRegex(PermissionError,'FIXED_TASK_PILOTS_TERMINAL_SUCCESS_DATA_NOT_ADMITTED'):
                require_task_asset(root,{},'FULLBODY_TASK_OVERLAY_V2_WORKING_STROKE')
            self.assertFalse((root/'NATIVE_SAFETY_FAIL_V2.json').exists())
