"""Cold-owner heartbeat publication/readback with an exact RPC intent token.

Call only through a registered single owner with the existing dual-platform
campaign view. No scheduler calls, retries, model effects or watchdog changes.
"""
import hashlib
import math
from pathlib import Path
import re
import time

from scripts.pi0292.durable import OwnerLock, read, write


def _guard(root, directory, job, view, ledger, identity, now):
    root, directory = Path(root), Path(directory)
    if root.is_symlink() or directory.is_symlink() or not directory.resolve().is_relative_to(root.resolve()):
        raise PermissionError('REGISTERED_RUN_AND_REQUEST_DIRECTORY')
    request_path = directory / 'REQUEST.json'
    if request_path.is_symlink():
        raise PermissionError('ORIGINAL_REQUEST_NOT_SYMLINK')
    request = read(request_path)
    binding = read(root / 'OUTPUT_BINDING.private.json')
    lease = read(binding['lease'])
    expected = dict(run_uuid=root.name, job_id=job, source_sha=request['source_sha'],
                    request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest())
    if expected != identity:
        raise PermissionError('EXACT_REGISTERED_HEARTBEAT_IDENTITY')
    if (lease.get('status') != 'ACTIVE' or lease.get('run_uuid') != root.name
            or (root/'STOP.json').exists() or (root.parent.parent/'STOP.json').exists()
            or ledger.get('unknown_effects') != [] or ledger.get('pending_updates') != []):
        raise PermissionError('ORIGINAL_ACTIVE_LEASE_STOP_AND_EFFECT_GUARDS')
    epoch = view.get('observed_epoch')
    from . import vanda_campaign as single
    vanda_only=view.get('schema')==single.SCHEMA
    if vanda_only:single.validate(view,ledger=ledger,now_epoch=now)
    if (type(now) not in (int, float) or not math.isfinite(now)
            or type(epoch) not in (int, float) or not math.isfinite(epoch)
            or not 0 <= now-epoch < 120
            or view.get('status') != (single.PASS if vanda_only else 'PASS_TWO_PLATFORM_CAMPAIGN_VIEW')):
        raise PermissionError('FRESH_ORIGINAL_TWO_PLATFORM_CAMPAIGN_VIEW')
    own = [v for v in view['campaign_jobs'] if v['job_id'] == job]
    if len(own) != 1 or own[0]['state'] not in ('Q', 'R'):
        raise PermissionError('ONE_EXACT_LIVE_NATIVE_JOB')
    return expected, own[0]


def _receipt(root, view, heartbeat):
    value=dict(heartbeat=heartbeat, lease_status='ACTIVE', run_STOP=False,
                parent_STOP=False, campaign_view_status=view['status'],
                campaign_view_observed_epoch=view['observed_epoch'],
                unknown_effects=[], pending_updates=[])
    from . import vanda_campaign as single
    if view.get('schema')==single.SCHEMA:value['vanda_campaign_view']=view
    return value


def publish(root, directory, job, view, ledger, intent, *, clock=time.time):
    """One publication, never called again after an unknown RPC result."""
    operation = intent['operation_id']
    if re.fullmatch('[0-9a-f]{32}', operation) is None or intent.get('effect') != 'HEARTBEAT':
        raise PermissionError('EXACT_HEARTBEAT_PUBLICATION_INTENT')
    now = clock()
    started = intent.get('started_epoch')
    if (type(started) not in (int, float) or not math.isfinite(started)
            or not 0 <= now-started <= 30):
        raise PermissionError('HEARTBEAT_PUBLICATION_RPC_DEADLINE')
    expected, own = _guard(root, directory, job, view, ledger, intent['identity'], now)
    root = Path(root)
    control = root/'control'
    if control.is_symlink() or not control.is_dir():
        raise PermissionError('EXISTING_REGISTERED_CONTROL_DIRECTORY')
    path = control/'OWNER_HEARTBEAT.json'
    if path.is_symlink():
        raise PermissionError('ORIGINAL_HEARTBEAT_NOT_SYMLINK')
    lock = OwnerLock(control/'HEARTBEAT_PUBLICATION.lock')
    try:
        intents = control/'HEARTBEAT_PUBLICATION_INTENTS'
        if intents.is_symlink():
            raise PermissionError('NO_PUBLICATION_INTENT_SYMLINK')
        intents.mkdir(exist_ok=True)
        marker = intents/(operation+'.json')
        if marker.exists() or path.exists() and read(path).get('rpc_intent_id') == operation:
            raise PermissionError('DUPLICATE_PUBLICATION_INTENT_NO_REPLAY')
        # A crash after this marker may leave an unresolved attempt, never a retry.
        write(marker, intent)
        _guard(root, directory, job, view, ledger, intent['identity'], clock())
        beat = dict(expected, owner='EXISTING_LOCAL_PC', rpc_intent_id=operation,
                    global_unknown_effects=[], original_parent_STOP=False,
                    native_job_state=own['state'], source_pair_platform='VANDA_A40',
                    grant_sha256=view['grant_sha256'], observed_epoch=clock())
        from . import vanda_campaign as single
        if view.get('schema')==single.SCHEMA:
            from .dual_platform import GRANT_SHA
            # Preserve the logical scientific owner/base concurrency grant.
            # Explicitly record where its new controller actually executes.
            beat.update(grant_sha256=GRANT_SHA,campaign_view_schema=single.SCHEMA,
                control_execution_platform='VANDA_A40',agent_liveness_required=False,
                migration_grant_sha256=view['grant_sha256'],
                current_authority_global_canonical_sha256=view['global_ledger_canonical_sha256'],
                migration_marker_sha256=view['migration_marker_sha256'],
                control_authority_proof=view['authority'],queue_observed_epoch=view['queue_observed_epoch'])
        if not 0 <= beat['observed_epoch']-started <= 30:
            raise PermissionError('HEARTBEAT_PUBLICATION_RPC_DEADLINE')
        write(path, beat)
        path.chmod(0o400)
        # Guard after the write too: racing STOP/lease changes cannot yield success.
        _guard(root, directory, job, view, ledger, intent['identity'], clock())
        return _receipt(root, view, read(path))
    finally:
        lock.close()


def readback(root, directory, job, view, ledger, intent, *, clock=time.time):
    """No heartbeat renewal; a missing/old/different intent remains unresolved."""
    _guard(root, directory, job, view, ledger, intent['identity'], clock())
    path = Path(root)/'control/OWNER_HEARTBEAT.json'
    if path.is_symlink() or path.parent.is_symlink():
        raise PermissionError('ORIGINAL_HEARTBEAT_NOT_SYMLINK')
    return _receipt(root, view, read(path))
