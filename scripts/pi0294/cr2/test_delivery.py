"""Local archive and fake-network transaction tests, never an HF receipt."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0292.durable import read,write
from scripts.pi0294.fd1 import put,sha
from scripts.pi0294.pv1.data_delivery import pack_archive
from . import delivery
from .bank import LOCAL
from .plan import R1


def fixture(folder):
    parent=Path(folder);source=parent/'children/pack';bank=source/'datasets/CR2_FIXED_R1_P12'
    output=source/'delivery';output.mkdir(parents=True);bank.mkdir(parents=True);(source/'evidence').mkdir()
    put(bank/'DATASET_MANIFEST.json',dict(statistics_sha256=R1,fixture_not_real_data=True))
    put(bank/'statistics.json',dict(fixture_not_actual_R1_bytes=True))
    payload=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in bank.iterdir()]
    put(bank/'PAYLOAD_MANIFEST.json',payload)
    put(bank/'LOCAL_FROZEN.json',dict(status=LOCAL,statistics_sha256=R1,payload_manifest_sha256=sha(bank/'PAYLOAD_MANIFEST.json')))
    members={p.name:p for p in bank.iterdir()}
    proof=pack_archive(output/'metadata.tar.gz',members,lambda:None)
    index=dict(schema='PI0294_CR2_ARCHIVE_INDEX_V1',archives={'metadata.tar.gz':proof},statistics_sha256=R1,
        dataset_manifest_sha256=sha(bank/'DATASET_MANIFEST.json'),source_local_frozen_sha256=sha(bank/'LOCAL_FROZEN.json'))
    put(output/delivery.INDEX,index)
    put(source/'evidence/PACK_RESULT.private.json',dict(status='PASS_CR2_LOCAL_BANK_AND_PACK_NOT_HF_READY',
        gate=dict(status='PASS_CR2_DATA_BANK_GATE'),HF_READY=False,dataset_manifest_sha256=index['dataset_manifest_sha256'],
        archive_index_sha256=sha(output/delivery.INDEX)))
    return parent,source,sha(source/'evidence/PACK_RESULT.private.json')


class PackageTests(unittest.TestCase):
    def test_archive_and_each_member_hashes_really_read(self):
        with tempfile.TemporaryDirectory() as folder:
            _,source,result=fixture(folder)
            index,members=delivery.checked_package(source,result,lambda:None)
            self.assertEqual(set(members),set(index['archives']['metadata.tar.gz']['member_sha256']))
            p=source/'datasets/CR2_FIXED_R1_P12/statistics.json';p.chmod(0o600);p.write_text('changed');p.chmod(0o400)
            with self.assertRaises(ValueError):delivery.checked_package(source,result,lambda:None)

    def test_extra_or_existing_READY_not_uploaded_as_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            _,source,result=fixture(folder);put(source/'delivery/READY.json',{})
            with self.assertRaises(PermissionError):delivery.checked_package(source,result,lambda:None)


class TransactionTests(unittest.TestCase):
    def run_fixture(self,folder,*,commit_unknown=False,corrupt=False):
        parent,source,result=fixture(folder);root=parent/'children/delivery';(root/'plans').mkdir(parents=True)
        (root/'evidence').mkdir();events=[];snapshots={};prefix='PI0294/original-cr2/CR2_FIXED_R1_P12'
        put(parent/'CR2_ADOPTION_AND_SCOPE.json',dict(run_uuid='original-cr2'))
        write(parent/'RESOURCE_LEDGER.json',dict(network_bytes_upper=64*1024**2))
        put(root/'plans/WORK_PLAN.private.json',dict(operation='CR2_FIXED_REVISION_HF_DELIVERY',repo_id=delivery.REPO,
            maximum_transfer_bytes=16*1024**3,closed_pack_root=str(source),source_pack_result_sha256=result,HF_prefix=prefix))
        write(root/'RESOURCE_LEDGER.json',dict(status='RESERVED',unknown_effects=[]))
        class API:
            token=None
            def repo_info(self,*a,**k):return SimpleNamespace(private=False,sha='0'*40)
            def list_repo_files(self,*a,**k):return []
            def create_commit(self,*a,**kwargs):
                operations=kwargs['operations'];kind='READY' if len(operations)==1 and operations[0].path_in_repo.endswith('/READY.json') else 'DATA'
                events.append('COMMIT_'+kind)
                if commit_unknown:raise ConnectionError('synthetic unknown commit outcome')
                revision=('1' if kind=='DATA' else '2')*40
                before=dict(snapshots.get(kwargs['parent_commit'],{}))
                for op in operations:before[op.path_in_repo]=Path(op.path_or_fileobj).read_bytes()
                snapshots[revision]=before
                return SimpleNamespace(oid=revision)
        def download(repo,path,**kwargs):
            self.assertTrue(kwargs['force_download']);events.append('READ_'+Path(path).name)
            p=Path(kwargs['local_dir'])/path;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(b'corrupt' if corrupt else snapshots[kwargs['revision']][path]);return str(p)
        module=SimpleNamespace(HfApi=lambda **k:API(),CommitOperationAdd=lambda **k:SimpleNamespace(**k),hf_hub_download=download)
        stack=ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch.dict(sys.modules,{'huggingface_hub':module}))
        for name,replacement in (
            ('validate',lambda *a: (dict(limits=dict(operation_deadline_epoch=time.time()+60,CPU_s=120)),{})),
            ('closed_child',lambda *a:source),('bind_cpu',lambda *a:None),('wait_priority',lambda *a:None),
            ('data_network',lambda:lambda:None),('existing_token',lambda *a:None),
            ('verify_remote',lambda api,repo,prefix,revision,files,*a: self.assertEqual(set(snapshots[revision]),{f['path'] for f in files})),
        ):stack.enter_context(patch.object(delivery,name,replacement))
        stack.enter_context(patch.object(delivery.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)))
        return root,events

    def test_full_payload_readback_precedes_READY_commit_and_final_READY_readback(self):
        with tempfile.TemporaryDirectory() as folder:
            root,events=self.run_fixture(folder);v=delivery.run(root)
            self.assertEqual(v['revision'],'2'*40)
            self.assertLess(events.index('READ_metadata.tar.gz'),events.index('COMMIT_READY'))
            self.assertEqual(events[-1],'READ_READY.json')
            receipt=read(root/'evidence/DELIVERY_RESULT.private.json')
            self.assertTrue(receipt['READY_claim']);self.assertEqual(read(root/'RESOURCE_LEDGER.json')['unknown_effects'],[])

    def test_unknown_commit_is_preserved_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            root,events=self.run_fixture(folder,commit_unknown=True)
            with self.assertRaises(ConnectionError):delivery.run(root)
            self.assertEqual(events,['COMMIT_DATA'])
            self.assertEqual(read(root/'RESOURCE_LEDGER.json')['unknown_effects'],['HF_MUTATION_OUTCOME_UNKNOWN'])
            self.assertFalse(read(root/'evidence/HF_TRANSFER.private.json')['READY_claim'])

    def test_corrupt_actual_payload_does_not_get_READY(self):
        with tempfile.TemporaryDirectory() as folder:
            root,events=self.run_fixture(folder,corrupt=True)
            with self.assertRaises(ValueError):delivery.run(root)
            self.assertNotIn('COMMIT_READY',events)
            self.assertFalse(read(root/'evidence/HF_TRANSFER.private.json')['READY_claim'])


if __name__=='__main__':unittest.main()
