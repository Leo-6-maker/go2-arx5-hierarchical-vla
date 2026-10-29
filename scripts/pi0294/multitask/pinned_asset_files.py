"""Finite private download of explicit upstream Git blobs; no asset execution.

The existing owner reserves CPU/storage/transfer first. No retry, overwrite,
archive expansion, package installation, Git config or credential operation.
"""
import hashlib,json,os,re,time
from pathlib import Path,PurePosixPath
from urllib.request import urlopen
from urllib.parse import urlparse


def download(root,*,repository,commit,members,maximum_file_bytes=8*1024**2,
             maximum_total_bytes=32*1024**2):
    if (repository!='Lifelong-Robot-Learning/LIBERO'
            or commit!='8f1084e3132a39270c3a13ebe37270a43ece2a01'
            or not 0<len(members)<=4 or not 0<maximum_file_bytes<=16*1024**2
            or not maximum_file_bytes<=maximum_total_bytes<=32*1024**2):
        raise PermissionError('EXPLICIT_PINNED_MINIMUM_LIBERO_ASSETS_ONLY')
    root=Path(root).resolve(strict=True);result={};total=0;started=time.process_time()
    for name,blob in members.items():
        rel=PurePosixPath(name)
        if (str(rel)!=name or rel.is_absolute()or '..'in rel.parts
                or not name.startswith('libero/libero/assets/turbosquid_objects/')
                or not re.fullmatch('[0-9a-f]{40}',blob)):
            raise PermissionError('PINNED_ASSET_MEMBER_ESCAPE_OR_MISSING_GIT_IDENTITY')
        destination=root/name
        for parent in reversed(destination.parents):
            if parent==root or root in parent.parents:
                if parent.is_symlink():raise PermissionError('PRIVATE_ASSET_PARENT_SYMLINK')
                parent.mkdir(exist_ok=True,mode=0o700)
        partial=destination.with_name(destination.name+'.partial')
        intent=destination.with_name(destination.name+'.download-intent.json')
        if any(os.path.lexists(p)for p in(destination,partial,intent)):
            raise FileExistsError('KEEP_EXISTING_ASSET_OR_UNKNOWN_DOWNLOAD_NO_REPLAY')
        url='https://raw.githubusercontent.com/'+repository+'/'+commit+'/'+name
        fd=os.open(intent,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
        with os.fdopen(fd,'w')as f:
            json.dump(dict(repository=repository,commit=commit,path=name,git_blob_sha=blob,
                maximum_file_bytes=maximum_file_bytes,maximum_total_bytes=maximum_total_bytes,
                automatic_retry=False,observed_epoch=time.time()),f);f.flush();os.fsync(f.fileno())
        fd=os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
        count=0
        with os.fdopen(fd,'wb')as f,urlopen(url,timeout=25)as response:
            if urlparse(response.url).hostname!='raw.githubusercontent.com' or response.status!=200:
                raise PermissionError('NO_UNPINNED_REDIRECT_OR_PARTIAL_ASSET_RESPONSE')
            expected=response.headers.get('Content-Length')
            if expected is not None and not 0<int(expected)<=maximum_file_bytes:
                raise PermissionError('ASSET_DECLARED_LENGTH_OUTSIDE_RESERVED_UPPER')
            while True:
                block=response.read(min(128*1024,maximum_file_bytes-count+1))
                if not block:break
                count+=len(block);total+=len(block)
                if count>maximum_file_bytes or total>maximum_total_bytes:
                    raise PermissionError('ASSET_TRANSFER_HARD_UPPER_NO_ARCHIVE_OR_RETRY')
                f.write(block)
            f.flush();os.fsync(f.fileno())
        if expected is not None and count!=int(expected):raise ValueError('TRUNCATED_ASSET_RESPONSE')
        b=partial.read_bytes();actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if actual!=blob or b.startswith(b'version https://git-lfs.github.com/'):
            raise ValueError('EXACT_ORIGINAL_GIT_BLOB_NOT_LFS_POINTER_OR_ALTERNATE_CONTENT')
        if os.path.lexists(destination):raise FileExistsError('NO_OVERWRITE_AFTER_DOWNLOAD')
        os.rename(partial,destination)
        result[name]=dict(bytes=len(b),git_blob_sha=actual,sha256=hashlib.sha256(b).hexdigest())
    return dict(status='EXACT_PINNED_MINIMUM_ASSETS_SAVED_PRIVATELY',repository=repository,
        commit=commit,files=result,total_bytes=total,PC_CPU_s=time.process_time()-started,
        new_physics=0,new_model_calls=0)
