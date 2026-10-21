"""Synthetic fixed bytes only; no Hub, model, PBS or running process changes."""
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.fd1 import sha
from . import source_original_cache as c
from .source_data_original import FixedFiles,REPO,V3


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.fixture')
    temporary.write_bytes(value if isinstance(value,bytes) else (json.dumps(value)+'\n').encode());temporary.chmod(0o400)
    os.replace(temporary,path)
    return sha(path)


class OriginalCacheTests(unittest.TestCase):
    def test_only_exact_zero_model_history_guard_failure_can_be_reused(self):
        result=dict(failure=dict(error_type='CalledProcessError',
            error="Command '['/opt/pbs/bin/qstat', '-x', '-f', '-F', 'json', '1471903.stdct-mgmt-02']' returned non-zero exit status 1."))
        self.assertTrue(c.reusable_failure(result,c.HISTORY_REASON))
        self.assertFalse(c.reusable_failure(result,c.REASON))
        result['failure']['error']=result['failure']['error'].replace('1471903','OTHER_JOB')
        self.assertFalse(c.reusable_failure(result,c.HISTORY_REASON))
        self.assertFalse(c.reusable_failure(dict(failure=dict(error='scientific failure')),c.HISTORY_REASON))

    def test_reuse_and_new_members_have_identical_bytes_without_full_tree_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);old=base/'old';new=base/'new';old.mkdir();new.mkdir()
            files={'E/images/0.png':b'original-png','E/samples.json':b'original-samples'}
            archive=base/'raw.tar.gz'
            with tarfile.open(archive,'w:gz') as handle:
                for name,value in files.items():
                    m=tarfile.TarInfo(name);m.size=len(value);handle.addfile(m,io.BytesIO(value))
            put(old/'raw/E/images/0.png',files['E/images/0.png'])
            expected={name:__import__('hashlib').sha256(value).hexdigest() for name,value in files.items()}
            cache=c.Cache(new,old,lambda *a:None,upper=1024**3)
            with patch.object(Path,'rglob',side_effect=AssertionError('NO_FULL_TREE_RESCAN')):
                result=cache.extract(archive,new/'raw',expected)
            self.assertEqual(result,dict(members=2,reused_hash_verified_members=1,new_members=1))
            self.assertTrue(os.path.samefile(old/'raw/E/images/0.png',new/'raw/E/images/0.png'))
            self.assertEqual(sha(new/'raw/E/samples.json'),expected['E/samples.json'])
            self.assertEqual(cache.budget.reserved,64*1024**2+sum(map(len,files.values())))
            stat=(new/'raw/E/samples.json').stat()
            self.assertEqual(cache.incremental,dict(new_regular_files=1,
                new_regular_file_allocated_bytes=max(stat.st_size,stat.st_blocks*512)))
            with self.assertRaises(PermissionError):cache.extract(archive,new/'raw',expected)

    def test_fixed_download_reuses_only_exact_bytes_and_counts_only_new_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);old=base/'old';new=base/'new';old.mkdir();new.mkdir()
            name='PI0294/fixture/member.json';original=old/'download/v3'/name
            checksum=put(original,b'{}\n')
            spec=dict(repo_id=REPO,revision=V3,prefix='PI0294/fixture',files=[dict(path=name,bytes=3,sha256=checksum)])
            cache=c.Cache(new,old,lambda *a:None,upper=1024**3)
            client=FixedFiles(new/'download/v3',spec,lambda *a:None,cache=cache,
                get=lambda *a,**kw:self.fail('EXACT_CLOSED_MEMBER_NOT_DOWNLOADED'))
            actual=client.fetch('member.json')
            self.assertEqual(client.bytes,3);self.assertEqual(client.new_bytes,0)
            self.assertTrue(os.path.samefile(original,actual));self.assertEqual(len(cache.reused_downloads),1)
            with self.assertRaises(FileExistsError):client.fetch('member.json')
            spec['files'].append(dict(path='PI0294/fixture/new.json',bytes=3,sha256=checksum))
            calls=[]
            def get(repo,path,**kw):
                calls.append((repo,path,kw));out=Path(kw['local_dir'])/path;put(out,b'{}\n');return out
            client=FixedFiles(new/'download/v3b',spec,lambda *a:None,cache=cache,get=get)
            client.fetch('new.json');self.assertEqual(client.new_bytes,3);self.assertIsNone(calls[0][2]['token'])
            self.assertEqual(cache.incremental['new_regular_files'],1)
            self.assertGreaterEqual(cache.incremental['new_regular_file_allocated_bytes'],3)

    def fixture(self,base):
        old=base/'old';root=base/'new';old.mkdir();root.mkdir()
        zero=dict(GPU=0,PBS=0,models=0,queries=0,updates=0,physics=0)
        m=dict(source_data_profile='ORIGINAL126_C1',authority_release='fixed-authority',original_T0_epoch=1,zero_effects=zero)
        put(old/'RUN_MANIFEST.json',m)
        put(old/'RESULT.private.json',dict(status='FAILED_PRESERVED_NO_MODEL_OR_HF_MUTATION',
            failure=dict(error='SOURCE_DATA_CPU_STORAGE_CLOCK_OR_STOP_BOUND')))
        put(old/'STOP.json',dict(reason=c.REASON,administrative_CPU_only=True,new_GPU_or_PBS_or_physics=0))
        put(old/'PROCESS.private.json',dict(pid=999999999,proc_start='fixture-terminal'))
        lease=old/'lease.json';put(lease,dict(status='CLOSED_FAILED_CPU_STAGING_PRESERVED',run_uuid=old.name,
            owner='EXISTING_LOCAL_PC',authority_release=m['authority_release']))
        put(old/'OUTPUT_BINDING.private.json',dict(lease=str(lease)))
        for name in ('plans/V3_TRANSFER.json','plans/C0_TRANSFER.json','plans/C1_TRANSFER.json'):
            put(old/name,{});put(root/name,{})
        plan=dict(schema='CLOSED_ORIGINAL126_C1_STAGING_CACHE_V1',source_root=str(old),
            source_manifest_sha256=sha(old/'RUN_MANIFEST.json'),source_result_sha256=sha(old/'RESULT.private.json'),
            source_lease_sha256=sha(lease),source_stop_sha256=sha(old/'STOP.json'))
        digest=put(root/c.PLAN,plan);put(root/'RUN_MANIFEST.json',dict(m,sealed_inputs={c.PLAN:digest}))
        return old,root

    def test_only_exact_terminal_administrative_cache_not_live_or_safety_stop(self):
        for change in (None,'lease','reason','profile','delivery','plan_mode'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                old,root=self.fixture(Path(tmp))
                if change=='lease':put(old/'lease.json',dict(status='ACTIVE'))
                if change=='reason':put(old/'STOP.json',dict(reason='REAL_SAFETY_STOP',administrative_CPU_only=True,new_GPU_or_PBS_or_physics=0))
                if change=='profile':
                    value=json.loads((root/'RUN_MANIFEST.json').read_text());value['source_data_profile']='PV1_139';put(root/'RUN_MANIFEST.json',value)
                if change=='delivery':put(root/'plans/C1_TRANSFER.json',dict(changed=True))
                if change=='plan_mode':(root/c.PLAN).chmod(0o600)
                if change is None:self.assertEqual(c.previous(root),old)
                else:
                    with self.assertRaises(PermissionError):c.previous(root)


if __name__=='__main__':unittest.main()
