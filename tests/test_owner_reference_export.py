import ast
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from scripts.maintenance.export_owner_reference import export,redact,guarded


class OwnerReferenceTests(unittest.TestCase):
    def test_roles_and_future_import_guard_no_execution(self):
        text='from __future__ import annotations\nraise RuntimeError("must never run")\np="/home/example/private/source"\n'
        clean,counts=redact(text,{'/home/example':'@LOCAL_HOME@'})
        body,syntax=guarded(clean);ast.parse(body)
        self.assertEqual(syntax,'PARSE_PASS');self.assertEqual(counts,{'@LOCAL_HOME@':1})
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'reference.py.txt';p.write_text(body)
            with self.assertRaisesRegex(SystemExit,'OWNER_REFERENCE_ONLY'):runpy.run_path(str(p))

    def test_secrets_unknown_path_and_private_ip_fail_closed(self):
        for text in ('token="hf_'+'a'*32+'"','p="/scratch/unknown/private"','ip="10.1.2.3"',
                     'password="not-a-placeholder"','-----BEGIN OPENSSH PRIVATE KEY-----'):
            with self.subTest(text_type=text[:5]),self.assertRaises(ValueError):
                redact(text,{'/home/example':'@LOCAL_HOME@'})

    def test_only_python_sources_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'private';root.mkdir();out=Path(folder)/'public'
            (root/'owner.py').write_text('p="/home/example/work"\n')
            (root/'secrets.json').write_text('{"excluded":true}')
            value=export(root,out,{'/home/example':'@LOCAL_HOME@'})
            self.assertEqual(value['source_file_count'],1)
            self.assertFalse((out/'secrets.json').exists())
            self.assertEqual(json.loads((out/'MANIFEST.json').read_text())['files'][0]['executable'],False)
            with self.assertRaises(FileExistsError):export(root,out,{'/home/example':'@LOCAL_HOME@'})

    def test_secret_preflight_leaves_no_partial_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'private';root.mkdir();out=Path(folder)/'public'
            (root/'a.py').write_text('a=1\n');(root/'z.py').write_text('token="hf_'+'a'*32+'"\n')
            with self.assertRaises(ValueError):export(root,out,{'/home/example':'@LOCAL_HOME@'})
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
