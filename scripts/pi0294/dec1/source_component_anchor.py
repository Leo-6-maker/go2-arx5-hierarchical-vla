"""One post-close/reconcile owner anchor. Called only by existing Core owner.

No STOP, snapshot, model, integration or ledger debit. Uses actual live reader
state; never derives a trusted last sequence/hash from the supplied file itself.
"""
import hashlib
from pathlib import Path
from scripts.pi0292.durable import read,write
from scripts.pi0294.dec1.source_component_ack import _bytes
from scripts.pi0294.dec1.source_component_evidence import verify_event_bytes
from scripts.pi0294.dec1.mapped_ack import _hash

SCHEMA='DEC1_SOURCE_COMPONENT_TERMINAL_BYTE_ANCHOR_V1'


def capture_terminal_anchor(factory,*,manifest_sha256):
    f=factory
    if getattr(f,'_terminal_anchor_attempted',False):
        raise PermissionError('TERMINAL_ANCHOR_ALREADY_ATTEMPTED_NO_RETRY')
    f._terminal_anchor_attempted=True
    try:
        f.ledger.guard(closeout=True)
        if (not getattr(f,'_outer_reconciled',False) or not f.persist_confirmed or not f.closed
                or f.unknown or f.evidence is None or not f.evidence.closed or f.evidence.unknown
                or f.actual_stop is None or f.barrier.state()['unknown']
                or not f.barrier.state()['stop_acknowledged']):
            raise PermissionError('ACTUAL_KNOWN_CLOSE_RECONCILE_AND_STOP_REQUIRED')
        f._propagate_unknown()
        if f.unknown:raise PermissionError('UNKNOWN_COMPONENT_CANNOT_SEAL_TERMINAL_ANCHOR')
        folder=f.folder;manifest=f.ledger.root/'RUN_MANIFEST.json'
        def digest(path):
            if path.is_symlink() or not path.is_file():raise PermissionError('EXACT_REGULAR_AUDIT_FILE_REQUIRED')
            h=hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda:handle.read(1024**2),b''):h.update(block)
            return h.hexdigest()
        if digest(manifest)!=_hash(manifest_sha256):raise PermissionError('TERMINAL_MANIFEST_BYTES_CHANGED')
        if read(manifest)['code']['vla_commit']!=f.ack_options['runtime_release']:
            raise PermissionError('TERMINAL_RUNTIME_RELEASE_CHANGED')
        events=f.evidence.read_events()
        # These are copied from actual owner-confirmed memory BEFORE readback.
        offset,sequence,previous=f.evidence.offset,f.evidence.sequence,f.evidence.previous
        with f.evidence._open_read() as handle:raw=handle.read(f.evidence.maximum+1)
        if verify_event_bytes(raw,episode_id=f.episode_id,executor_id=f.executor_id,
                maximum_bytes=f.evidence.maximum,expected_offset=offset,expected_sequence=sequence,
                expected_previous=previous)!=events:
            raise PermissionError('OWNER_CONFIRMED_EVENT_BYTES_CHANGED')
        paths=[folder/p for p in ('RESULT.json','S1_RESULT.json','SOURCE_COMPONENT_STOP.json',
            'SOURCE_COMPONENT_EVENTS.jsonl','highrate.npz','source_component_ack/OWNER_COMPONENT_BINDING.json')]
        for name,suffix in (('plans','.json'),('acks','.json'),('physics','.npz')):
            base=folder/'source_component_ack'/name
            if base.is_symlink() or not base.is_dir():raise PermissionError('ORIGINAL_ACK_DIRECTORY_REQUIRED')
            members=list(base.iterdir())
            if len(members)>80:raise PermissionError('ORIGINAL_MAXIMUM_80_PREFIX_FILES')
            for p in members:
                if p.suffix!=suffix or len(p.stem)!=3 or not p.stem.isdigit() or not 0<=int(p.stem)<80:
                    raise PermissionError('ORIGINAL_PREFIX_AUDIT_FILENAME_REQUIRED')
            paths.extend(members)
        files={p.relative_to(folder).as_posix():digest(p) for p in paths}
        event_sha=hashlib.sha256(raw).hexdigest()
        if files['SOURCE_COMPONENT_EVENTS.jsonl']!=event_sha:raise PermissionError('TERMINAL_EVENT_FILE_CHANGED')
        document=dict(schema=SCHEMA,condition_id=f.ack_options['condition_id'],episode_id=f.episode_id,
            executor_id=f.executor_id,interface=f.adapter.mode,manifest_sha256=manifest_sha256,
            component_plan_sha256=f.ack_options['expected_plan_sha256'],
            plan_owner_release=f.ack_options['owner_release'],runtime_release=f.ack_options['runtime_release'],
            engine_sha256=f.stop_options['source_step_sha256'],owner_closeout_known=True,
            events=dict(offset=offset,sequence=sequence,previous=previous,bytes_sha256=event_sha),files=files)
        path=folder/'SOURCE_COMPONENT_TERMINAL_BYTE_ANCHOR.json'
        if path.exists() or path.is_symlink():raise FileExistsError('PRESERVE_FIRST_TERMINAL_ANCHOR')
        write(path,document)
        if (read(path)!=document or path.read_bytes()!=_bytes(document)
                or f.evidence.read_events()!=events or any(digest(folder/p)!=value for p,value in files.items())):
            raise PermissionError('ACTUAL_TERMINAL_ANCHOR_AND_FILES_READBACK_REQUIRED')
        f._terminal_anchor_confirmed=True
        return dict(relative=path.name,sha256=digest(path))
    except BaseException:
        f._mark_unknown()
        raise
