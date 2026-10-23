"""Full streaming checkpoint reads, reused only for unchanged physical files.

This reduces repeated audit I/O for hardlinked frozen members. Cached results
are tentative: witnesses() MUST independently stream each unique inode again.
It is not a validator, admission proof, or cache across audit processes.
"""
import hashlib
import os
from pathlib import Path
import stat


def signature(value):
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,
            value.st_mode,value.st_nlink,value.st_uid)


class StreamHashes:
    def __init__(self, *, uid=None, chunk_bytes=8*1024**2):
        if type(chunk_bytes) is not int or not 0<chunk_bytes<=8*1024**2:
            raise ValueError('BOUNDED_STREAM_CHUNK')
        self.uid=os.getuid() if uid is None else uid
        self.chunk_bytes=chunk_bytes
        self.paths={};self.inodes={};self.bytes_read=0;self.finished=False

    def __call__(self, name):
        if self.finished:raise PermissionError('AUDIT_ALREADY_FINALIZED')
        path=Path(name);before=path.lstat();identity=signature(before)
        if not stat.S_ISREG(before.st_mode) or before.st_uid!=self.uid:
            raise PermissionError('OWN_REGULAR_AUDIT_FILE_NO_SYMLINK')
        key=str(path.absolute());inode=identity[:2]
        if key in self.paths and self.paths[key]!=identity:
            raise PermissionError('AUDIT_PATH_CHANGED_SINCE_FIRST_READ')
        if inode in self.inodes:
            saved,digest=self.inodes[inode]
            if saved!=identity:
                raise PermissionError('AUDIT_INODE_CHANGED_SINCE_FULL_READ')
            # Check again even on reuse. A cache hit never makes a later stat
            # optional or silently re-hashes a changed checkpoint as acceptable.
            if signature(path.lstat())!=identity:
                raise PermissionError('AUDIT_FILE_CHANGED_DURING_REUSE')
            self.paths[key]=identity
            return digest
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            if signature(os.fstat(stream.fileno()))!=identity:
                raise PermissionError('AUDIT_OPENED_FILE_IDENTITY_MISMATCH')
            for chunk in iter(lambda:stream.read(self.chunk_bytes),b''):
                digest.update(chunk);self.bytes_read+=len(chunk)
            if signature(os.fstat(stream.fileno()))!=identity:
                raise PermissionError('AUDIT_OPEN_FILE_CHANGED_DURING_STREAM')
        if signature(path.lstat())!=identity:
            raise PermissionError('AUDIT_PATH_CHANGED_DURING_FULL_READ')
        result=digest.hexdigest();self.inodes[inode]=(identity,result);self.paths[key]=identity
        return result

    def witnesses(self):
        """Independently rehash; timestamps alone cannot detect same-tick writes."""
        if self.finished:raise PermissionError('AUDIT_ALREADY_FINALIZED')
        rows=[];verified=set()
        for path,identity in sorted(self.paths.items()):
            if signature(Path(path).lstat())!=identity:
                raise PermissionError('AUDIT_FILE_CHANGED_BEFORE_FINAL_WITNESS')
            saved,digest=self.inodes[identity[:2]]
            if identity!=saved:
                raise PermissionError('AUDIT_ALIAS_NO_LONGER_MATCHES_STREAMED_FILE')
            if identity[:2] not in verified:
                actual=hashlib.sha256()
                with Path(path).open('rb') as stream:
                    if signature(os.fstat(stream.fileno()))!=identity:
                        raise PermissionError('FINAL_OPEN_FILE_IDENTITY_CHANGED')
                    for chunk in iter(lambda:stream.read(self.chunk_bytes),b''):
                        actual.update(chunk);self.bytes_read+=len(chunk)
                    if signature(os.fstat(stream.fileno()))!=identity:
                        raise PermissionError('FINAL_FILE_CHANGED_DURING_STREAM')
                if actual.hexdigest()!=digest or signature(Path(path).lstat())!=identity:
                    raise PermissionError('FINAL_PHYSICAL_BYTES_DIFFER_FROM_FIRST_STREAM')
                verified.add(identity[:2])
            rows.append(dict(path=path,device=identity[0],inode=identity[1],bytes=identity[2],
                             mtime_ns=identity[3],ctime_ns=identity[4],sha256=digest))
        self.finished=True
        return dict(paths=rows,unique_streamed_inodes=len(self.inodes),bytes_streamed=self.bytes_read,
                    all_paths_rechecked=True,all_unique_inodes_independently_rehashed=True,
                    reused_unmodified_physical_bytes_only=True)
