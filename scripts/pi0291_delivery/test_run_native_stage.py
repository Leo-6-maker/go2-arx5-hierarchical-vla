"""Wrapper lifecycle unit fixtures, not an official data qualification."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_native_stage as wrapper


class WrapperTests(unittest.TestCase):
    def fixture(self, root, body):
        script = root/'code/scripts/pi0291_arx5_data/export_dataset.py'
        script.parent.mkdir(parents=True)
        script.write_text(body)
        (root/'evidence').mkdir()
        (root/'RESOURCE_ACCOUNTING.json').write_text(json.dumps(dict(CPU_s=0,unmetered_dependency_setup_CPU_s_upper=0)))
        (root/'TEACHER_AND_LAYOUT_SEAL.json').write_text(json.dumps(dict(source_sha='UNIT_FIXTURE_NOT_RUNTIME',
            code_files={'scripts/pi0291_arx5_data/export_dataset.py':hashlib.sha256(script.read_bytes()).hexdigest()})))

    def invoke(self, root):
        affinity = os.sched_getaffinity(0)
        argv = sys.argv[:]
        search = sys.path[:]
        cpu_limit = wrapper.resource.getrlimit(wrapper.resource.RLIMIT_CPU)
        try:
            with patch.object(wrapper,'install_deadline'), patch.object(wrapper.signal,'signal'):
                wrapper.run(root,'export_dataset',[])
        finally:
            os.sched_setaffinity(0,affinity)
            sys.argv = argv
            sys.path[:] = search
            wrapper.resource.setrlimit(wrapper.resource.RLIMIT_CPU,cpu_limit)

    def test_success_records_only_uncovered_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            body = "from pathlib import Path\nimport json\nroot=Path("+repr(temporary)+")\n(root/'native_dataset').mkdir()\n(root/'native_dataset/EXPORT_RECEIPT.json').write_text(json.dumps({'CPU_s':0}))\n"
            self.fixture(root,body)
            self.invoke(root)
            receipts = list((root/'evidence').glob('CPU_AUX_NATIVE_WRAPPER*.json'))
            self.assertEqual(len(receipts),1)
            value = json.loads(receipts[0].read_text())
            self.assertEqual(value['CPU_s'],value['invocation_CPU_s'])
            self.assertEqual(value['already_covered_by_stage_receipt_CPU_s'],0)

    def test_failure_preserves_chargeable_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root,"raise ValueError('unit fixture failure, not data')\n")
            with self.assertRaises(ValueError):
                self.invoke(root)
            receipt = next((root/'evidence').glob('CPU_AUX_NATIVE_WRAPPER*.json'))
            self.assertEqual(json.loads(receipt.read_text())['status'],'FAILED_PRESERVED')

    def test_changed_source_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root,"raise RuntimeError('must never execute')\n")
            (root/'code/scripts/pi0291_arx5_data/export_dataset.py').write_text('changed')
            with self.assertRaises(AssertionError):
                self.invoke(root)


if __name__ == '__main__':
    unittest.main()
