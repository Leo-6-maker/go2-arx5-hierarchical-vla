"""Local files only: verify cache reuse, preserved partials and archive safety."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.pi0294.dec1 import source_data_cache as c


class CacheTests(unittest.TestCase):
    def archive(self,path,rows):
        with tarfile.open(path,'w:gz') as archive:
            for name,data in rows:
                member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))

    def test_exact_existing_members_link_and_partial_is_preserved_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);old=r/'old';old.mkdir();out=r/'new';archive=r/'x.tar.gz'
            (old/'good').write_bytes(b'good');(old/'partial').write_bytes(b'wrong')
            rows=[('good',b'good'),('partial',b'correct'),('missing',b'new')]
            self.archive(archive,rows);expected={n:hashlib.sha256(b).hexdigest() for n,b in rows}
            result=c.extract(archive,out,expected,lambda *a:None,old_target=old,budget=c.OutputBudget(100*1024**2))
            self.assertEqual(result['reused_hash_verified_members'],1);self.assertEqual(result['new_members'],2)
            self.assertEqual((old/'partial').read_bytes(),b'wrong')
            self.assertEqual((out/'good').stat().st_ino,(old/'good').stat().st_ino)
            self.assertTrue(all(c.sha(out/n)==h for n,h in expected.items()))
            with self.assertRaises(PermissionError):
                c.extract(archive,out,expected,lambda *a:None,old_target=old,budget=c.OutputBudget(100*1024**2))

    def test_unsafe_duplicate_missing_hash_and_symlink_fail_closed(self):
        cases=[([('../escape',b'x')],{'../escape':hashlib.sha256(b'x').hexdigest()}),
            ([('a',b'x'),('a',b'x')],{'a':hashlib.sha256(b'x').hexdigest()}),
            ([('a',b'x')],{'a':'f'*64}),([('a',b'x')],{'a':hashlib.sha256(b'x').hexdigest(),'b':'f'*64})]
        for rows,expected in cases:
            with self.subTest(rows=rows),tempfile.TemporaryDirectory() as tmp:
                r=Path(tmp);(r/'old').mkdir();self.archive(r/'x.tar.gz',rows)
                with self.assertRaises((PermissionError,ValueError)):
                    c.extract(r/'x.tar.gz',r/'new',expected,lambda *a:None,old_target=r/'old',budget=c.OutputBudget(100*1024**2))
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);(r/'old').mkdir();(r/'foreign').write_bytes(b'x');(r/'old/a').symlink_to(r/'foreign')
            self.archive(r/'x.tar.gz',[('a',b'x')])
            with self.assertRaises(PermissionError):c.extract(r/'x.tar.gz',r/'new',{'a':c.sha(r/'foreign')},lambda *a:None,old_target=r/'old',budget=c.OutputBudget(100*1024**2))

    def test_cached_archive_is_full_hash_verified_and_only_missing_downloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);old=r/'old';folder=r/'new/download';p=old/'download/prefix/a';p.parent.mkdir(parents=True);p.write_bytes(b'archive')
            rows=[dict(path='prefix/a',bytes=7,sha256=c.sha(p)),dict(path='prefix/b',bytes=3,sha256=hashlib.sha256(b'new').hexdigest())]
            calls=[]
            def get(repo,path,**kwargs):calls.append(path);return 'FIXTURE_NETWORK_NOT_CALLED'
            receipt=dict(repo_id='fixture',revision='a'*40,files=rows)
            fetch,reused,new=c.cached_download(r/'new',old,receipt,get,c.OutputBudget(100*1024**2),lambda *a:None)
            out=fetch('fixture','prefix/a',revision='a'*40,local_dir=folder,token=False)
            self.assertEqual(out.stat().st_ino,p.stat().st_ino);self.assertEqual(reused,['prefix/a']);self.assertFalse(calls)
            fetch('fixture','prefix/b',revision='a'*40,local_dir=folder,token=False)
            self.assertEqual(calls,['prefix/b']);self.assertEqual(new,['prefix/b'])
            with self.assertRaises(PermissionError):fetch('fixture','prefix/a',revision='b'*40,local_dir=folder,token=False)

    def test_budget_is_cumulative_and_negative_or_overflow_is_refused(self):
        b=c.OutputBudget(64*1024**2+10);b.reserve(7);b.reserve(3)
        for size in (1,-1,True):
            with self.assertRaises(PermissionError):b.reserve(size)


if __name__=='__main__':unittest.main()
