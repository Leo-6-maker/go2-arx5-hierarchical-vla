import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from .data_consumer import extract_checked


class TestExtraction(unittest.TestCase):
    def archive(self,path,names):
        with tarfile.open(path,'w:gz') as archive:
            for name in names:
                member=tarfile.TarInfo(name);member.size=3
                archive.addfile(member,io.BytesIO(b'abc'))

    def test_all_members_and_retained_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'data.tar.gz';self.archive(path,['T/a','T/b'])
            expected={name:hashlib.sha256(b'abc').hexdigest() for name in ('T/a','T/b')}
            reserved=[]
            self.assertEqual(extract_checked(path,root/'raw',expected,reserved.append),2)
            self.assertEqual(reserved[0],6)
            self.assertTrue(path.exists())
            self.assertEqual(extract_checked(path,root/'raw',expected,reserved.append),2)

    def test_traversal_duplicate_missing_and_bad_hash_rejected(self):
        for names,expected in [(['../escape'],{'../escape':'bad'}),(['a','a'],{'a':hashlib.sha256(b'abc').hexdigest()}),
                (['a'],{'a':hashlib.sha256(b'abc').hexdigest(),'missing':'bad'}),(['a'],{'a':'bad'})]:
            with self.subTest(names=names),tempfile.TemporaryDirectory() as d:
                root=Path(d);path=root/'data.tar.gz';self.archive(path,names)
                with self.assertRaises(ValueError):extract_checked(path,root/'raw',expected,lambda n:None)


if __name__=='__main__':unittest.main()
