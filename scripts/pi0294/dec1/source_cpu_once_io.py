"""Once-only metadata CAS for an already authorized owner's resource proposal.

This primitive does not admit resources, register a child or run physics. The
owner first obtains the complete typed proposal under fresh native admission.
An existing intent is never replayed, even when readback sees the old bytes.
All uncertain files remain available for independent read-only reconciliation.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def _encode(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def _file(path):
    path=Path(path)
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise PermissionError('SOURCE_ONCE_NO_SYMLINK_PATH')
    info=path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid():
        raise PermissionError('SOURCE_ONCE_OWNED_REGULAR_METADATA')
    return path.read_bytes()


def _exclusive(path,raw,mode=0o400):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
    with os.fdopen(fd,'wb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    if _file(path)!=raw:raise OSError('SOURCE_ONCE_NEW_FILE_READBACK_UNKNOWN')


def _directory_sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


@contextmanager
def _lock(path):
    fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'r+b') as f:
        info=os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid():
            raise PermissionError('SOURCE_ONCE_OWNED_LOCK')
        fcntl.flock(f,fcntl.LOCK_EX)
        yield


def commit(path,*,expected_before_sha256,proposed_bytes,request_id,intent_path,receipt_path,lock_path):
    """CAS exactly one ledger file; exceptions after intent mean UNKNOWN.

    The surrounding owner controls fresh permissions and the complete proposal.
    It must reserve both original parent and native DEC1 before any dispatch;
    success here is only one confirmed metadata write, never cross-site GO.
    """
    path,intent,receipt,lock=map(Path,(path,intent_path,receipt_path,lock_path))
    if (type(proposed_bytes) is not bytes or not proposed_bytes or type(request_id) is not str
            or not request_id or len(request_id)>160 or any(c not in
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in request_id)):
        raise PermissionError('SOURCE_ONCE_EXACT_PROPOSAL_AND_SAFE_REQUEST_ID')
    # Neither input can masquerade as the other, and all auxiliary writes
    # stay beside the exact ledger. No caller-supplied temporary directory.
    if (len({p.absolute() for p in (path,intent,receipt,lock)})!=4
            or any(p.parent.resolve()!=path.parent.resolve() for p in (intent,receipt,lock))):
        raise PermissionError('SOURCE_ONCE_DISTINCT_LEDGER_AND_LOCAL_AUXILIARIES')
    for p in (path,intent,receipt,lock):
        if any(q.is_symlink() for q in (p,*p.parents)):
            raise PermissionError('SOURCE_ONCE_NO_SYMLINK_PATH')
    # Validate serialization before creating any intent. Cost/authority checks
    # belong to the typed pure proposal and are not replaced by JSON validity.
    json.loads(proposed_bytes,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))
    temporary=path.with_name(path.name+'.source-once-'+request_id)
    with _lock(lock):
        if intent.exists() or receipt.exists() or temporary.exists():
            raise FileExistsError('SOURCE_ONCE_INTENT_EXISTS_NO_AUTOMATIC_RETRY')
        before=_file(path)
        if _sha(before)!=expected_before_sha256:
            raise PermissionError('SOURCE_ONCE_ACTUAL_LEDGER_CHANGED_BEFORE_INTENT')
        proposed_sha=_sha(proposed_bytes)
        if proposed_sha==expected_before_sha256:
            raise PermissionError('SOURCE_ONCE_REAL_NEW_RESERVATION_REQUIRED')
        intent_value=dict(schema='DEC1_SOURCE_CPU_ONCE_METADATA_INTENT_V1',request_id=request_id,
            target=str(path.absolute()),before_sha256=expected_before_sha256,
            proposed_sha256=proposed_sha,temporary=str(temporary.absolute()),
            native_dispatch_admitted=False)
        _exclusive(intent,_encode(intent_value));_directory_sync(path.parent)
        _exclusive(temporary,proposed_bytes)
        # Compare again immediately before replace, in case a noncooperating
        # historical writer changed the file despite the owner's lock.
        if _sha(_file(path))!=expected_before_sha256:
            raise PermissionError('SOURCE_ONCE_CONCURRENT_CHANGE_INTENT_RETAINED')
        os.replace(temporary,path);_directory_sync(path.parent)
        if _file(path)!=proposed_bytes:raise OSError('SOURCE_ONCE_LEDGER_WRITE_READBACK_UNKNOWN')
        value=dict(schema='DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1',request_id=request_id,
            target=str(path.absolute()),before_sha256=expected_before_sha256,
            after_sha256=proposed_sha,intent_sha256=_sha(_file(intent)),
            status='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION',native_dispatch_admitted=False)
        _exclusive(receipt,_encode(value));_directory_sync(path.parent)
        return value


def reconcile(path,*,intent_path,receipt_path):
    """Read-only actual bytes; never reissue a write or release a reservation."""
    path=Path(path);intent_raw=_file(intent_path);intent=json.loads(intent_raw)
    if (intent.get('schema')!='DEC1_SOURCE_CPU_ONCE_METADATA_INTENT_V1'
            or intent.get('target')!=str(path.absolute()) or intent.get('native_dispatch_admitted') is not False):
        raise PermissionError('SOURCE_ONCE_EXACT_INTENT_TARGET')
    actual=_sha(_file(path));receipt_path=Path(receipt_path)
    receipt=json.loads(_file(receipt_path)) if receipt_path.exists() else None
    verified=(type(receipt) is dict and receipt.get('schema')=='DEC1_SOURCE_CPU_ONCE_METADATA_RECEIPT_V1'
        and receipt.get('target')==intent['target'] and receipt.get('request_id')==intent['request_id']
        and receipt.get('before_sha256')==intent['before_sha256']
        and receipt.get('after_sha256')==intent['proposed_sha256']
        and receipt.get('intent_sha256')==_sha(intent_raw)
        and receipt.get('status')=='CONFIRMED_SINGLE_METADATA_WRITE_NOT_CROSS_SITE_ADMISSION'
        and receipt.get('native_dispatch_admitted') is False)
    if actual==intent['proposed_sha256']:
        status='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT' if verified else 'CURRENT_PROPOSED_BYTES_RECEIPT_UNCONFIRMED'
    elif actual==intent['before_sha256']:status='CURRENT_OLD_BYTES_NO_RETRY_AUTHORIZATION'
    else:status='UNKNOWN_OR_CONCURRENT_CHANGE_KEEP_RESERVATION'
    return dict(status=status,request_id=intent['request_id'],actual_sha256=actual,
        receipt_binding_valid=verified,read_only=True,native_dispatch_admitted=False,
        automatic_retry_allowed=False)
