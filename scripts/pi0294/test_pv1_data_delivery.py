import os
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1 import data_delivery as delivery
from scripts.pi0294.native_dataset import sha


class DeliveryContracts(unittest.TestCase):
    def test_hardlinked_raw_pixels_are_regular_archive_members_and_all_hashes_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a.png';b=root/'b.png';a.write_bytes(b'CPU fixture, not native image');os.link(a,b)
            archive=root/'parents-000.tar.gz';calls=[]
            proof=delivery.pack_archive(archive,{'raw/fixture/body_rgb/0000.png':a,'raw/fixture/wrist_rgb/0000.png':b},lambda:calls.append(1))
            self.assertEqual(proof['files'],2);self.assertEqual(len(calls),2)
            self.assertEqual(set(proof['member_sha256'].values()),{sha(a)})
            with tarfile.open(archive,'r:gz') as handle:self.assertTrue(all(m.isfile() and not m.islnk() for m in handle.getmembers()))
            with self.assertRaises(FileExistsError):delivery.pack_archive(archive,{'a':a},lambda:None)

    def test_public_metadata_does_not_leak_paths_or_authentication(self):
        delivery.safe_json(dict(id='PV1-TRAIN-F0-000-T1',sha256='a'*64,origin='actual committed source data'))
        for value in ('/home/private/data','/scratch/private/model','/opt/private','hf_'+'a'*32):
            with self.subTest(value=value),self.assertRaises(PermissionError):delivery.safe_json(dict(payload=[value]))

    def test_unknown_bank_or_login_node_cannot_start_download_or_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);outside=root/'outside';outside.mkdir();bank=root/'bank';bank.symlink_to(outside,target_is_directory=True)
            with self.assertRaisesRegex(PermissionError,'REAL_OWN_BANK'):delivery.bank_members(bank,lambda:None)
            with patch.dict(os.environ,{'PBS_JOBID':''}),self.assertRaisesRegex(PermissionError,'ONLY_INSIDE_OWN_NATIVE'):
                delivery.consume(root,dict(job_id='123.hopper-m-02'),lambda:None)

    def test_only_the_fixed_dataset_download_temporarily_opens_hub_network_and_restores_offline(self):
        constants=SimpleNamespace(HF_HUB_OFFLINE=True);calls=[]
        def get(*args,**kwargs):
            self.assertFalse(constants.HF_HUB_OFFLINE);self.assertEqual(os.environ['HF_HUB_OFFLINE'],'0')
            calls.append(kwargs);return '/CPU-fixture-not-an-actual-download'
        module=SimpleNamespace(constants=constants,hf_hub_download=get)
        with patch.dict('sys.modules',{'huggingface_hub':module}),patch.dict(os.environ,{'HF_HUB_OFFLINE':'1'}):
            delivery.download_dataset('fixture','prefix/READY.json',revision='a'*40,local_dir='/CPU-fixture')
            self.assertTrue(constants.HF_HUB_OFFLINE);self.assertEqual(os.environ['HF_HUB_OFFLINE'],'1')
            self.assertEqual(calls[0]['repo_type'],'dataset')
            module.hf_hub_download=lambda *a,**kw:(_ for _ in ()).throw(TimeoutError('READ_FAILED'))
            with self.assertRaises(TimeoutError):delivery.download_dataset('fixture','prefix/READY.json',revision='a'*40)
            self.assertTrue(constants.HF_HUB_OFFLINE);self.assertEqual(os.environ['HF_HUB_OFFLINE'],'1')

    def test_fixed_hf_readback_requires_exact_members_size_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'small.json';path.write_text('{}\n')
            r=dict(path='prefix/small.json',bytes=path.stat().st_size,sha256=sha(path))
            api=SimpleNamespace(token=None,list_repo_tree=lambda *a,**kw:[SimpleNamespace(path=r['path'],size=r['bytes'],lfs=None)])
            download=lambda *a,**kw:str(path)
            with patch.dict('sys.modules',{'huggingface_hub':SimpleNamespace(hf_hub_download=download)}):
                delivery.verify_remote(api,'fixture','prefix','a'*40,[r],lambda:None,root/'readback')
                for change in ('extra_file','wrong_size','wrong_hash'):
                    actual=SimpleNamespace(path=r['path'],size=r['bytes'],lfs=None)
                    rows=[actual];expected=[dict(r)]
                    if change=='extra_file':rows.append(SimpleNamespace(path='prefix/extra',size=1,lfs=None))
                    if change=='wrong_size':actual.size+=1
                    if change=='wrong_hash':expected[0]['sha256']='f'*64
                    api.list_repo_tree=lambda *a,**kw:rows
                    with self.subTest(change=change),self.assertRaises(ValueError):
                        delivery.verify_remote(api,'fixture','prefix','a'*40,expected,lambda:None,root/'readback')
