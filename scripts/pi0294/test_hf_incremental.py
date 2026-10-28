"""CPU fixture identities only; no credentials, network, model or physics."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0292.durable import read,write
from scripts.pi0294.hf import V3_0_REVISION,previous_bank,reused_raw,package,download
from scripts.iteration.queue import digest


class PreviousBankTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.folder=self.root/'delivery/v3_0';self.folder.mkdir(parents=True)
        (self.root/'evidence').mkdir()
        self.prefix='PI0294/'+self.root.name+'/bank-v3_0-2fd28e00'
        self.rows=[dict(id=f'fake-{i:03d}',origin='new',first_attempt=True,admitted=i!=3,
            split='TRAIN' if i<64 else 'DEV',samples_sha256='a'*64) for i in range(80)]
        self.admission=dict(episodes=copy.deepcopy(self.rows)+[dict(id='extension',origin='new',first_attempt=True)])
        self.ready=dict(status='READY_FIXED_REVISION_CPU_QUALIFIED',version='v3.0',
            repo_id='fixture/private',prefix=self.prefix,data_revision='b'*40)
        write(self.folder/'READY.json',self.ready)
        write(self.folder/'BANK_ADMISSION.json',dict(episodes=self.rows))
        self.proofs={r['id']:dict(member_sha256={r['id']+'/samples.json':'a'*64}) for r in self.rows}
        write(self.folder/'ARCHIVE_MEMBER_MANIFEST.json',self.proofs)
        files=[dict(path=self.prefix+'/'+name,bytes=(self.folder/name).stat().st_size,
            sha256=digest(self.folder/name)) for name in ('READY.json','BANK_ADMISSION.json','ARCHIVE_MEMBER_MANIFEST.json')]
        files += [dict(path=self.prefix+'/episodes/'+r['id']+'.tar.gz',bytes=1,sha256='c'*64) for r in self.rows]
        self.transfer=dict(status='PI0294_PRIVATE_FIXED_REVISION_READY_LAST_FULL_SHA_PASS',READY_claim=True,
            revision=V3_0_REVISION,prefix=self.prefix,repo_id='fixture/private',data_revision='b'*40,
            manifest_sha256='d'*64,statistics_sha256='e'*64,files=files)
        write(self.root/'evidence/HF_v3_0.json',self.transfer)

    def test_preserves_all_eighty_including_failed_attempts(self):
        value,proofs=previous_bank(self.root,self.admission)
        self.assertEqual(len(value['episodes']),80)
        self.assertFalse(value['episodes']['fake-003']['admitted'])
        self.assertFalse(value['raw_uploaded_again'])
        self.assertEqual(proofs,self.proofs)
        self.assertEqual(value['revision'],V3_0_REVISION)

    def test_mutable_revision_wrong_run_or_changed_receipt_rejected(self):
        for key,value in [('revision','main'),('prefix','PI0294/other/bank-v3_0-2fd28e00'),
                          ('READY_claim',False),('data_revision','f'*40)]:
            with self.subTest(key=key):
                write(self.root/'evidence/HF_v3_0.json',dict(self.transfer,**{key:value}))
                with self.assertRaises(ValueError):previous_bank(self.root,self.admission)
        write(self.root/'evidence/HF_v3_0.json',self.transfer)
        write(self.folder/'BANK_ADMISSION.json',dict(episodes=self.rows,note='changed'))
        with self.assertRaisesRegex(ValueError,'RECEIPT_CHANGED'):previous_bank(self.root,self.admission)

    def test_changed_old_outcome_or_split_is_not_reclassified_as_new(self):
        for change in [dict(admitted=True),dict(split='DEV'),dict(first_attempt=False),dict(samples_sha256='f'*64)]:
            with self.subTest(change=change):
                admission=copy.deepcopy(self.admission);admission['episodes'][3].update(change)
                with self.assertRaisesRegex(ValueError,'OUTCOME_CHANGED'):previous_bank(self.root,admission)

    def test_missing_archive_reference_rejected(self):
        changed=copy.deepcopy(self.transfer);changed['files']=changed['files'][:-1]
        write(self.root/'evidence/HF_v3_0.json',changed)
        with self.assertRaisesRegex(ValueError,'MEMBER_PROOF_REQUIRED'):previous_bank(self.root,self.admission)


class LocalReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.native=self.root/'native';self.native.mkdir()
        self.raw=self.root/'raw';self.raw.mkdir()
        (self.raw/'samples.json').write_bytes(b'fixture samples')
        (self.raw/'image.png').write_bytes(b'fixture pixels')
        self.row=dict(id='fake',seed=1,group='paired',split='TRAIN',task='T2',family='F0',origin='new',
            samples_sha256=digest(self.raw/'samples.json'))
        write(self.native/'DATASET_MANIFEST.json',dict(episodes=[self.row]))
        write(self.native/'statistics.json',dict(fixture=True))
        self.previous=dict(manifest_sha256=digest(self.native/'DATASET_MANIFEST.json'),
                           statistics_sha256=digest(self.native/'statistics.json'))
        self.bindings=self.root/'bindings.json';write(self.bindings,{'fake':str(self.raw)})
        self.proof=dict(member_sha256={'fake/'+name:digest(self.raw/name) for name in ('samples.json','image.png')})

    def check(self,row=None,proof=None):
        return reused_raw(row or self.row,self.previous,self.native,self.bindings,proof or self.proof)

    def test_actual_member_bytes_checked_without_any_download(self):
        self.assertEqual(self.check(),self.raw.resolve())
        (self.raw/'image.png').write_bytes(b'changed pixels')
        with self.assertRaisesRegex(ValueError,'MEMBER_CHANGED'):self.check()

    def test_changed_task_or_split_or_stats_rejected(self):
        for change in (dict(task='T1'),dict(split='DEV'),dict(seed=2),dict(group='other')):
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError,'EPISODE_IDENTITY_CHANGED'):self.check(dict(self.row,**change))
        write(self.native/'statistics.json',dict(fixture='changed'))
        with self.assertRaisesRegex(ValueError,'NATIVE_IDENTITY_CHANGED'):self.check()

    def test_member_prefix_and_path_escape_rejected(self):
        with self.assertRaisesRegex(ValueError,'PREFIX_REQUIRED'):
            self.check(proof=dict(member_sha256={'other/image.png':'a'*64}))
        with self.assertRaises(ValueError):
            self.check(proof=dict(member_sha256={'fake/../outside':'a'*64}))


class PackageReuseTests(PreviousBankTests):
    def test_missing_local_previous_inputs_fails_before_any_download_or_output(self):
        path=self.root/'transfer.json';write(path,dict(status='PI0294_PRIVATE_FIXED_REVISION_READY_LAST_FULL_SHA_PASS',
            READY_claim=True,version='v3.1'))
        output=self.root/'never-created'
        for kwargs in ({},{'previous_native':self.root/'absent','previous_bindings':self.root/'absent.json'}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError,'EXISTING_LOCAL_V3_0'):
                    download(path,output,self.root/'old-native',self.root/'old-bindings',**kwargs)
                self.assertFalse(output.exists())

    def test_packaging_skips_old_eighty_raw_archives(self):
        native=self.root/'native';native.mkdir()
        write(native/'DATASET_MANIFEST.json',dict(schema='PI0294_FROZEN_MULTITASK_NATIVE_V1',frozen=True,
            version='v3.1',episodes=[],original_statistics_sha256='f'*64))
        write(native/'statistics.json',dict(fixture=True))
        write(native/'BANK_SEAL.json',dict(manifest_sha256=digest(native/'DATASET_MANIFEST.json'),
            statistics_sha256=digest(native/'statistics.json'),file_sha256={}))
        write(self.root/'evidence/BANK_v3_1_ADMISSION.json',dict(episodes=self.rows))
        old=self.root/'old.json'
        from scripts.pi0294.hf import OLD_REVISION,OLD_PREFIX
        write(old,dict(revision=OLD_REVISION,prefix=OLD_PREFIX,repo_id='fixture/private',files=[]))
        calls=[]
        def fake_tar(path,members):
            calls.append(path.name);path.write_bytes(b'fixture archive')
            return dict(member_sha256={name:digest(p) for name,p in members.items()})
        with patch('scripts.pi0294.hf.verified_tar',side_effect=fake_tar):
            payload,proofs,_=package(self.root,native,old,self.root/'out')
        self.assertEqual(calls,['native_dataset.tar.gz'])
        self.assertFalse(any(name.startswith('episodes/') for name in payload))
        self.assertEqual(len(proofs),81)
        self.assertIn('PREVIOUS_BANK_DEPENDENCY.json',payload)
        self.assertEqual(len(read(payload['PREVIOUS_BANK_DEPENDENCY.json'])['episodes']),80)


if __name__=='__main__':unittest.main()
