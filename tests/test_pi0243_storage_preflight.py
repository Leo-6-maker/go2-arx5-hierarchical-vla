import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pi0243_storage_preflight", ROOT / "scripts" / "pi0243_storage_preflight.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StoragePreflightTests(unittest.TestCase):
    def test_same_fileset_binding_is_explicitly_scoped_to_scratch(self):
        self.assertTrue(MODULE.same_fileset_binding(Path("/scratch/u/work"), Path("/scratch/u/durable")))
        self.assertFalse(MODULE.same_fileset_binding(Path("/scratch/u/work"), Path("/Project_Storage/u/durable")))


if __name__ == "__main__":
    unittest.main()
