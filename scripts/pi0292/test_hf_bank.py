import hashlib
from pathlib import Path
import tempfile
import unittest

from .hf_bank import contained,verified_tar
from .data_consumer import extract_checked


class TestBankArchive(unittest.TestCase):
    def test_verified_archive_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input';source.write_bytes(b'original bytes')
            archive=root/'archive.tar.gz';proof=verified_tar(archive,{'T/source':source})
            self.assertEqual(proof['member_sha256']['T/source'],hashlib.sha256(source.read_bytes()).hexdigest())
            extract_checked(archive,root/'readback',proof['member_sha256'],lambda n:None)
            self.assertEqual((root/'readback/T/source').read_bytes(),source.read_bytes())
            with self.assertRaises(FileExistsError):verified_tar(archive,{'T/source':source})

    def test_unbound_and_symlink_paths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';base.mkdir();(root/'outside').write_bytes(b'x')
            (base/'link').symlink_to(root/'outside')
            for name in ('../outside','/etc/passwd','missing','link'):
                with self.subTest(name=name),self.assertRaises(ValueError):contained(base,name)


if __name__=='__main__':unittest.main()
