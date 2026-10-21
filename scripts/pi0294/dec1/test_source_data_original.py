import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.pi0294.dec1 import source_data_original as c
from scripts.pi0294.dec1 import source_data_consumer as entry
from scripts.pi0294.dec1.test_source_data_consumer import SourceCPUContracts,put


class OriginalDataContracts(unittest.TestCase):
    def test_original_126_selection_retains_exact_legacy_train64_and_never_dev_raw(self):
        manifest=dict(schema='PI0294_FROZEN_MULTITASK_NATIVE_V1',episodes=
            [dict(id=f'T291{i:03d}',split='TRAIN',origin='legacy') for i in range(64)]+
            [dict(id='new-'+str(i),split='TRAIN',origin='new') for i in range(62)]+
            [dict(id='DEV-'+str(i),split='DEV',origin='new') for i in range(32)])
        result=c.train_rows(manifest);self.assertEqual(len(result),126)
        self.assertTrue(all(x['split']=='TRAIN' and not x['id'].startswith('DEV-') for x in result))
        for index in (0,63,125):
            bad=copy.deepcopy(manifest);bad['episodes'][index]['split']='DEV'
            with self.subTest(index=index),self.assertRaises(PermissionError):c.train_rows(bad)

    def test_fixed_clients_verify_bytes_and_preserve_first_download_without_token_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'member.json';raw.write_text('{}\n')
            spec=dict(repo_id=c.REPO,revision=c.V3,prefix='PI0294/fixture',files=[
                dict(path='PI0294/fixture/member.json',bytes=raw.stat().st_size,sha256=c.sha(raw))])
            calls=[]
            def get(repo,name,**kw):calls.append((repo,name,kw));return raw
            client=c.FixedFiles(root/'download',spec,lambda *a:None,get=get)
            self.assertEqual(client.fetch('member.json'),raw);self.assertEqual(client.bytes,3)
            self.assertIsNone(calls[0][2]['token']);self.assertEqual(calls[0][2]['revision'],c.V3)
            with self.assertRaises(FileExistsError):client.fetch('member.json')
            with self.assertRaises(PermissionError):client.fetch('../member.json')
            bad=copy.deepcopy(spec);bad['revision']='a'*40
            with self.assertRaises(PermissionError):c.FixedFiles(root,bad,lambda *a:None,get=get)
            bad=copy.deepcopy(spec);bad['files'][0]['sha256']='a'*64
            with self.assertRaises(ValueError):c.FixedFiles(root,bad,lambda *a:None,get=get).fetch('member.json')

    def test_correction_ready_is_a_different_bound_schema_not_a_fake_list_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ready=root/'ready';ready.write_text('{}\n')
            spec=dict(repo_id=c.REPO,revision=c.C1,prefix='PI0294/fixture',READY_sha256=c.sha(ready),
                files={'MEMBERS.json':dict(bytes=3,sha256=c.sha(ready))})
            client=c.FixedFiles(root,spec,lambda *a:None,get=lambda *a,**kw:ready)
            self.assertEqual(client.fetch('READY.json'),ready)
            self.assertEqual(set(client.files),{'READY.json','MEMBERS.json'})

    def test_new_profile_requires_distinct_prepayment_and_all_three_pinned_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)/'child';r.mkdir();m,b,seals=SourceCPUContracts().fixture(r)
            m['source_data_profile']='ORIGINAL126_C1'
            m['sealed_inputs']['BINDINGS.private.json']=put(r/'BINDINGS.private.json',dict(official_source='/CPU-fixture'))
            inputs={n:put(r/n,dict(CPU_fixture=True)) for n in c.INPUTS}
            m['sealed_inputs'].update(inputs)
            name='capsule/code/scripts/pi0294/dec1/source_data_original.py'
            m['sealed_inputs'][name]=put(r/name,'CPU_fixture')
            put(r/'RUN_MANIFEST.json',m);b['run_manifest_sha256']=entry.sha(r/'RUN_MANIFEST.json');put(r/'OUTPUT_BINDING.private.json',b)
            with patch.object(entry,'__file__',str(r/'capsule/code/scripts/pi0294/dec1/source_data_consumer.py')),\
                    patch.object(entry,'ADOPTION',seals['plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json']),\
                    patch.object(c,'INPUTS',inputs),patch.dict('os.environ',dict(CUDA_VISIBLE_DEVICES='',PBS_JOBID='')),\
                    patch.object(entry.subprocess,'check_output',return_value='51d4c89f72fda44cbf77285c6a8114b52676b8a1\n'),\
                    patch.object(entry.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as clean:
                with self.assertRaisesRegex(PermissionError,'PREPAYMENT'):entry.validate(r,now=lambda:1792560000)
                paid=entry.read(r/'CPU_BUDGET_PREPAID.json');paid['source_data_profile']='ORIGINAL126_C1'
                m['sealed_inputs']['CPU_BUDGET_PREPAID.json']=put(r/'CPU_BUDGET_PREPAID.json',paid)
                put(r/'RUN_MANIFEST.json',m);b['run_manifest_sha256']=entry.sha(r/'RUN_MANIFEST.json');put(r/'OUTPUT_BINDING.private.json',b)
                self.assertIsNone(entry.validate(r,now=lambda:1792560000)[2])
                clean.return_value=SimpleNamespace(returncode=1)
                with self.assertRaisesRegex(PermissionError,'OFFICIAL_PROCESSOR'):entry.validate(r,now=lambda:1792560000)
                clean.return_value=SimpleNamespace(returncode=0)
                put(r/'plans/C1_TRANSFER.json',dict(changed=True))
                with self.assertRaises(PermissionError):entry.validate(r,now=lambda:1792560000)


if __name__=='__main__':unittest.main()
