"""Real temporary-file audit I/O; no model, GPU, external service or native run."""
import hashlib
import os
from pathlib import Path
from unittest.mock import patch
import pytest
from .readonly_checkpoint_hash import StreamHashes


def test_full_stream_and_hardlinked_aliases_only_read_one_physical_copy(tmp_path):
    path=tmp_path/'model';data=b'whole member bytes'*100;path.write_bytes(data)
    alias=tmp_path/'alias';os.link(path,alias)
    h=StreamHashes(chunk_bytes=13);digest=hashlib.sha256(data).hexdigest()
    assert h(path)==h(alias)==h(path)==digest
    proof=h.witnesses();assert proof['unique_streamed_inodes']==1
    assert proof['bytes_streamed']==2*len(data) and len(proof['paths'])==2
    assert proof['all_unique_inodes_independently_rehashed'] is True
    with pytest.raises(PermissionError):h(path)
    with pytest.raises(PermissionError):h.witnesses()


def test_same_bytes_in_distinct_files_are_both_streamed(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.write_bytes(b'same');b.write_bytes(b'same')
    h=StreamHashes();assert h(a)==h(b)
    assert h.witnesses()['unique_streamed_inodes']==2 and h.bytes_read==16


@pytest.mark.parametrize('fault',['contents','replace','mode_ctime','alias_change','symlink'])
def test_cached_content_never_masks_path_or_inode_changes(tmp_path,fault):
    path=tmp_path/'member';path.write_bytes(b'original');h=StreamHashes();h(path)
    if fault=='contents':path.write_bytes(b'changed!')
    elif fault=='replace':
        other=tmp_path/'new';other.write_bytes(b'original');other.replace(path)
    elif fault=='mode_ctime':path.chmod(0o400)
    elif fault=='alias_change':
        alias=tmp_path/'alias';os.link(path,alias);alias.write_bytes(b'changed!')
    else:
        target=tmp_path/'target';path.rename(target);path.symlink_to(target)
    # A filesystem can give both writes the same timestamp. Cache hits are
    # tentative; the mandatory independent final stream must still reject.
    try:h(path)
    except PermissionError:pass
    with pytest.raises(PermissionError):h.witnesses()


def test_mutation_after_read_before_final_witness_is_rejected(tmp_path):
    path=tmp_path/'member';path.write_bytes(b'original');h=StreamHashes();h(path)
    path.write_bytes(b'modified')
    with pytest.raises(PermissionError):h.witnesses()


def test_changed_file_during_stream_is_not_a_cached_pass(tmp_path):
    path=tmp_path/'member';path.write_bytes(b'original'*10);h=StreamHashes(chunk_bytes=3)
    original=Path.open
    class Changed:
        def __enter__(self):self.file=original(path,'rb');return self
        def __exit__(self,*args):self.file.close()
        def fileno(self):return self.file.fileno()
        def read(self,size):
            value=self.file.read(size)
            if value:
                with original(path,'ab') as writer:writer.write(b'x')
                # Change once; the original stream's EOF must remain finite.
                self.read=self.file.read
            return value
    with patch.object(Path,'open',lambda *args,**kwargs:Changed()),pytest.raises(PermissionError):h(path)
    assert not h.inodes


def test_symlink_and_other_uid_rejected_before_data_read(tmp_path):
    path=tmp_path/'member';path.write_bytes(b'bytes');link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises(PermissionError):StreamHashes()(link)
    with pytest.raises(PermissionError):StreamHashes(uid=os.getuid()+1)(path)


def test_no_cache_is_carried_into_another_audit(tmp_path):
    path=tmp_path/'member';path.write_bytes(b'old');first=StreamHashes();first(path)
    path.write_bytes(b'new');second=StreamHashes()
    assert second(path)==hashlib.sha256(b'new').hexdigest() and second.bytes_read==3


@pytest.mark.parametrize('size',[0,-1,True,8*1024**2+1])
def test_stream_memory_bound(size):
    with pytest.raises(ValueError):StreamHashes(chunk_bytes=size)
