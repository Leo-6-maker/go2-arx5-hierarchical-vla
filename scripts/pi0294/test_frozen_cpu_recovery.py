"""Frozen CPU recovery rejects mutation; no model, physics or remote execution."""
import tempfile
from argparse import Namespace
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.pi0292.durable import read,write
from scripts.pi0294.cpu_pipeline import run,verify_frozen_bank
from scripts.pi0294.native_dataset import sha


class FrozenRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        write(self.root/'DATASET_MANIFEST.json',dict(schema='PI0294_FROZEN_MULTITASK_NATIVE_V1',
            version='v3.1',frozen=True,fixture=False,statistics_fit_count=1,
            all_planned_first_attempts_terminal=True,partial_extension_archive_only=False))
        write(self.root/'statistics.json',dict(TRAIN_only=True))
        self.manifest=sha(self.root/'DATASET_MANIFEST.json')
        self.statistics=sha(self.root/'statistics.json')
        self.seal()

    def seal(self):
        files={p.name:sha(p) for p in self.root.iterdir() if p.name!='BANK_SEAL.json'}
        write(self.root/'BANK_SEAL.json',dict(schema='PI0294_BANK_SEAL_V1',version='v3.1',fixture=False,
            statistics_fit_count=1,manifest_sha256=self.manifest,
            statistics_sha256=self.statistics,file_sha256=files))
        self.seal_hash=sha(self.root/'BANK_SEAL.json')

    def verify(self):
        return verify_frozen_bank(self.root,self.seal_hash,self.manifest,self.statistics)

    def test_exact_bytes_reused_without_mutation(self):
        before={p.name:p.read_bytes() for p in self.root.iterdir()}
        self.assertTrue(self.verify()['frozen'])
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})

    def test_changed_member_rejected(self):
        write(self.root/'statistics.json',dict(DEV_refit=True))
        with self.assertRaisesRegex(ValueError,'MEMBER_CHANGED'):self.verify()

    def test_changed_seal_rejected(self):
        write(self.root/'BANK_SEAL.json',dict(fake=True))
        with self.assertRaisesRegex(ValueError,'SEAL_CHANGED'):self.verify()

    def test_extra_or_missing_member_rejected(self):
        write(self.root/'extra.json',{})
        with self.assertRaisesRegex(ValueError,'MEMBER_SET'):self.verify()
        (self.root/'extra.json').unlink()
        (self.root/'statistics.json').unlink()
        with self.assertRaisesRegex(ValueError,'MEMBER_SET'):self.verify()

    def test_partial_or_unfrozen_bank_rejected_even_with_new_seal(self):
        for key,value in (('all_planned_first_attempts_terminal',False),
                          ('partial_extension_archive_only',True),('frozen',False),
                          ('statistics_fit_count',2)):
            original=read(self.root/'DATASET_MANIFEST.json')
            changed=dict(original);changed[key]=value
            write(self.root/'DATASET_MANIFEST.json',changed)
            self.manifest=sha(self.root/'DATASET_MANIFEST.json');self.seal()
            with self.assertRaisesRegex(ValueError,'FULL_TERMINAL'):self.verify()
            write(self.root/'DATASET_MANIFEST.json',original)
            self.manifest=sha(self.root/'DATASET_MANIFEST.json');self.seal()

    def test_escaping_or_symlink_member_rejected(self):
        outside=self.root.parent/(self.root.name+'-outside.json')
        write(outside,{})
        self.addCleanup(outside.unlink)
        (self.root/'linked.json').symlink_to(outside)
        self.seal()
        with self.assertRaisesRegex(ValueError,'MEMBER_CHANGED'):self.verify()

    def test_actual_failed_attempt_recovers_without_export_or_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            previous=root/'evidence/bank-qualification-v3_1'
            previous.mkdir(parents=True)
            write(previous/'CPU_ATTEMPT.json',dict(status='FAILED_PRESERVED'))
            write(root/'scene.json',dict(version='v3.1'))
            write(root/'run.json',dict(data=dict(manifest_sha256=self.manifest),
                interfaces=dict(statistics_sha256=self.statistics)))
            args=Namespace(root=str(root),reservation='fixture',manifest=str(root/'run.json'),
                scene_plan=str(root/'scene.json'),qualification_attempt=2,
                frozen_bank_seal_sha256=self.seal_hash,output=str(self.root))
            before={p.name:p.read_bytes() for p in self.root.iterdir()}
            with patch('scripts.pi0294.cpu_pipeline.validate_cpu_ticket',return_value=dict(deadline_epoch=1791939600)), \
                patch('scripts.pi0294.cpu_pipeline.build') as build, \
                patch('scripts.pi0294.cpu_pipeline.qualify',return_value=dict(status='PASS_FIXTURE_NOT_ACTUAL')) as qualify, \
                patch('scripts.pi0294.cpu_pipeline.signal.alarm'):
                self.assertEqual(run(args)['status'],'PASS_FIXTURE_NOT_ACTUAL')
            build.assert_not_called()
            qualify.assert_called_once()
            self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})
            self.assertTrue((previous/'CPU_ATTEMPT.json').exists())
            self.assertTrue((root/'evidence/bank-qualification-v3_1-02/CPU_ATTEMPT.json').exists())

    def test_successful_first_attempt_cannot_be_requalified(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);previous=root/'evidence/bank-qualification-v3_1'
            previous.mkdir(parents=True)
            write(previous/'CPU_ATTEMPT.json',dict(status='PASS_ALREADY_QUALIFIED'))
            write(root/'scene.json',dict(version='v3.1'))
            args=Namespace(root=str(root),reservation='fixture',manifest='unused',
                scene_plan=str(root/'scene.json'),qualification_attempt=2,
                frozen_bank_seal_sha256=self.seal_hash,output=str(self.root))
            with patch('scripts.pi0294.cpu_pipeline.validate_cpu_ticket',return_value=dict(deadline_epoch=1791939600)), \
                patch('scripts.pi0294.cpu_pipeline.build') as build, \
                patch('scripts.pi0294.cpu_pipeline.qualify') as qualify, \
                patch('scripts.pi0294.cpu_pipeline.signal.alarm'):
                with self.assertRaisesRegex(ValueError,'FAILED_FIRST_ATTEMPT'):run(args)
            build.assert_not_called();qualify.assert_not_called()


if __name__=='__main__':unittest.main()
