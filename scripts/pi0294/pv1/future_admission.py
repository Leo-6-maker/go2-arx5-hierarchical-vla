"""Read-only bridge from sealed PV1 evidence to fresh budget recomputation.

This helper neither submits a job nor supplies native authority. The existing
owner, neural scope, stage, site, lease and unknown-effect guards remain required.
Fresh owner metadata is kept outside sealed scientific inputs, like the existing
job-bound project attestation; a queued forecast is not a pre-training release.
"""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import time

from . import future_budget as budget
from .allocation import native_request_binding
from .decision import AUTHORITY


INDEX = 'plans/PV1_FUTURE_BUDGET_INPUT_INDEX.json'
EVIDENCE_KEYS = ('result', 'adjudication', 'audit', 'causal_chain', 'cost_readback', 'raw_file_manifest')
HOPPER_KEYS = ('H200_s', 'PBS_submissions', 'optimizer_updates', 'neural_queries')
TERMINAL = ('NATIVE_TERMINAL', 'NATIVE_QSUB_REJECTED_NO_JOB')
SNAPSHOT='evidence/FUTURE_BUDGET_PREQUEUE_SNAPSHOT.private.json'
PROGRESS='evidence/FUTURE_BUDGET_PREQUEUE_PROGRESS.private.json'


def _fail(message):
    raise PermissionError('PV1_FUTURE_ADMISSION_'+message)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def _file(root, name):
    root = Path(root)
    relative = Path(name)
    if (type(name) is not str or relative.is_absolute() or '..' in relative.parts
            or relative.as_posix() != name or not relative.parts or root.is_symlink()):
        _fail('EXACT_OWN_RELATIVE_INPUT_PATH')
    path = root
    for part in relative.parts:
        path = path/part
        if path.is_symlink():
            _fail('NO_SYMLINK_INPUT_COMPONENT')
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        _fail('EXISTING_OWN_REGULAR_INPUT')
    before = path.stat()
    if before.st_uid != os.getuid() or before.st_mode & 0o222:
        _fail('OWN_READ_ONLY_EVIDENCE_INPUT')
    raw = path.read_bytes();after = path.stat()
    if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
        _fail('INPUT_CHANGED_DURING_READ')
    return raw


def _index(raw):
    value, _ = budget._read(raw)
    if (value.get('schema') != 'PI0294_PV1_FUTURE_BUDGET_INPUT_INDEX_V1'
            or value.get('authority_release') != AUTHORITY
            or value.get('runtime_refresh_policy') != 'SAME_OWNER_SAME_JOB_FRESH_RECOMPUTATION_V1'
            or not value.get('case_observations')):
        _fail('EXPLICIT_SEALED_INPUT_INDEX')
    names = [value[k] for k in ('baseline_admission', 'assumptions', 'training_jobs', 'request_work_binding')]
    for row in value['case_observations']:
        if set(row) != set(EVIDENCE_KEYS):
            _fail('ALL_ORIGINAL_CASE_AND_COST_EVIDENCE_PATHS_REQUIRED')
        names.extend(row.values())
    if len(set(names)) != len(names) or INDEX in names:
        _fail('NO_ALIASED_OR_SELF_REFERENTIAL_INPUT_PATHS')
    return value, names


def preseal(root, index_bytes):
    """Return actual input hashes for the owner to add to RUN_MANIFEST; no writes."""
    _, names = _index(index_bytes)
    return dict(schema='PI0294_PV1_FUTURE_INPUT_HASHES_PREPARATION_V1',
        sealed_inputs={INDEX:_digest(index_bytes), **{name:_digest(_file(root, name)) for name in names}},
        manifest_written=False, scientific_or_native_execution=False)


def _inputs(root, request_bytes):
    manifest_bytes = _file(root, 'RUN_MANIFEST.json')
    manifest, _ = budget._read(manifest_bytes)
    request, request_sha = budget._read(request_bytes)
    if (manifest.get('schema') != 'PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1'
            or manifest.get('owner') != 'EXISTING_LOCAL_PC' or manifest.get('run_uuid') != Path(root).name
            or manifest.get('authority', {}).get('protocol_ref') != AUTHORITY
            or request.get('schema') != 'PI0294_PV1_NATIVE_ALLOCATION_REQUEST_V1'
            or request.get('authority_release') != AUTHORITY or request.get('run_uuid') != Path(root).name
            or request.get('run_manifest_sha256') != _digest(manifest_bytes)
            or request.get('source_sha') != manifest.get('code', {}).get('vla_commit')):
        _fail('EXACT_ORIGINAL_MANIFEST_REQUEST_AND_SOURCE')
    sealed = manifest['execution']['sealed_inputs']
    index_bytes = _file(root, INDEX);index, names = _index(index_bytes)
    if (sealed.get(INDEX) != _digest(index_bytes)
            or index['campaign_run_uuid'] != request['campaign_run_uuid']
            or index['T0_epoch'] != manifest['limits']['PV1_T0_epoch']):
        _fail('SEALED_INDEX_SINGLE_ORIGINAL_CAMPAIGN')
    files = {name:_file(root, name) for name in set(names)|set(sealed)}
    if any(_digest(raw) != sealed.get(name) for name, raw in files.items()):
        _fail('EVERY_ORIGINAL_EVIDENCE_AND_INPUT_MUST_BE_MANIFEST_SEALED')
    if set(sealed)-set(request.get('sealed_inputs', {})):
        _fail('REQUEST_CANNOT_DROP_MANIFEST_INPUTS')
    for name, digest in request['sealed_inputs'].items():
        if _digest(files[name] if name in files else _file(root, name)) != digest:
            _fail('REQUEST_SEALED_INPUT_CHANGED')
    return request, request_sha, manifest, index, files


def _case_cost(index, files, assumptions, now):
    observations = [];provenance = []
    for paths in index['case_observations']:
        originals = {key:files[paths[key]] for key in EVIDENCE_KEYS}
        result, _ = budget._read(originals['result'])
        cost, cost_sha = budget._read(originals['cost_readback'])
        bound = dict(result_sha256=_digest(originals['result']),
            adjudication_sha256=_digest(originals['adjudication']),
            independent_audit_sha256=_digest(originals['audit']),
            causal_chain_sha256=_digest(originals['causal_chain']),
            raw_file_manifest_sha256=_digest(originals['raw_file_manifest']))
        if (cost.get('schema') != 'PI0294_PV1_TERMINAL_COST_READBACK_V1'
                or cost.get('status') != 'PASS_READ_ONLY_TERMINAL_COST_READBACK'
                or cost.get('authority_release') != AUTHORITY
                or cost.get('campaign_run_uuid') != index['campaign_run_uuid']
                or cost.get('attempt_id') != result['id']
                or any(cost.get(key) != digest for key, digest in bound.items())
                or cost.get('terminal_process_verified') is not True
                or cost.get('inventory_stable') is not True or cost.get('unknown_effects') != []
                or any(cost.get(k) != 0 for k in ('new_integrations', 'new_model_queries', 'new_native_constructors'))):
            _fail('ACTUAL_TERMINAL_COST_READBACK_AND_ALL_ORIGINAL_HASHES')
        budget._number(cost['observed_epoch'])
        if not index['T0_epoch'] <= cost['observed_epoch'] <= now:
            _fail('COST_READBACK_OBSERVATION_IN_ORIGINAL_WINDOW')
        budget._number(cost['regular_file_count'], integer=True, positive=True)
        size = budget._number(cost['regular_file_bytes'], integer=True, positive=True)
        allocated = budget._number(cost.get('allocated_file_bytes', size), integer=True, positive=True)
        budget._sha(cost['inventory_metadata_sha256'])
        observer_cpu = budget._number(cost['observer_CPU_s'])
        observer_wall = budget._number(cost['observer_wall_s'])
        extra = {};unknown = []
        for dimension in ('CPU', 'wall'):
            measured = cost['prior_audit_'+dimension+'_s']
            if measured is None:
                extra[dimension] = budget._number(assumptions['unmeasured_audit_'+dimension+'_s_upper'], positive=True)
                unknown.append(dimension)
            else:
                extra[dimension] = budget._number(measured)
        # This is a new, explicitly labelled derived cost view, NOT an original
        # worker file. Raw original bytes and the actual inventory receipt stay
        # separately hashed and are never modified to invent worker output keys.
        view = dict(result, schema='PI0294_PV1_DERIVED_TERMINAL_COST_VIEW_V1',
            derivation='RESULT_PLUS_MEASURED_OBSERVER_AND_MEASURED_OR_DECLARED_AUDIT_UPPER',
            NOT_ORIGINAL_WORKER_RECEIPT=True, cost_readback_sha256=cost_sha,
            wall_s=budget._number(result['wall_s'], positive=True)+observer_wall+extra['wall'],
            cpu_s=budget._number(result['cpu_s'], positive=True)+observer_cpu+extra['CPU'],
            owned_bytes=max(size, allocated), output_bytes=max(size, allocated))
        observations.append(dict(result=originals['result'], adjudication=originals['adjudication'],
            audit=originals['audit'], causal_chain=originals['causal_chain'],
            worker_receipt=_bytes(view), observed_epoch=now))
        provenance.append(dict(attempt_id=result['id'], cost_readback_sha256=cost_sha,
            original_hashes=bound, derived_view_sha256=_digest(_bytes(view)),
            original_worker_receipt_claimed=False, unmeasured_prior_audit_dimensions=unknown,
            logical_file_bytes=size, allocated_file_bytes=allocated))
    value = budget.case_upper(observations, floor=assumptions['case_floor'],
        multiplier=assumptions['case_headroom_multiplier'], now_epoch=now)
    value['actual_read_side_cost_provenance'] = provenance
    value['original_worker_receipt_claimed'] = False
    return value


def _ledger(root, snapshot, progress):
    path = Path(root).parent.parent/'HOPPER_RESOURCE_LEDGER.json'
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
        _fail('ACTUAL_OWN_HOPPER_LEDGER_REQUIRED')
    raw = path.read_bytes();value, digest = budget._read(raw)
    if (value.get('schema') != 'PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1'
            or value.get('authority_release') != AUTHORITY
            or value.get('campaign_run_uuid') != snapshot['campaign_run_uuid']
            or value.get('unknown_effects') != [] or value.get('pending_updates') != []
            or snapshot.get('hopper_ledger_sha256') != digest
            or snapshot.get('hopper_ledger_revision') != value.get('revision')
            or value.get('generation_queries') != value['used'].get('neural_queries', 0)
            or value.get('optimizer_updates') != value['used'].get('optimizer_updates', 0)
            or value['used'].get('PBS_submissions', 0) != len(value['PBS_submissions'])):
        _fail('FRESH_EXACT_HOPPER_LEDGER_PENDING_UNKNOWN_AND_PROJECTION')
    for field in ('used', 'reserved'):
        native = budget._vector(value[field]);projected = budget._vector(snapshot[field])
        if any(native[k] != projected[k] for k in HOPPER_KEYS):
            _fail('NO_UNDERCOUNTED_OR_STALE_HOPPER_GPU_PBS_QUERY_UPDATE_PROJECTION')
    if (budget._number(value['CPU_s']) > snapshot['used'].get('Hopper_CPU_s', 0)
            or value.get('new_storage_current_bytes', 0) > snapshot['used'].get('Hopper_storage_bytes', 0)
            or value.get('storage_reserved_bytes', 0) > snapshot['reserved'].get('Hopper_storage_bytes', 0)):
        _fail('ALL_SITE_PROJECTION_COVERS_ACTUAL_HOPPER_CPU_AND_STORAGE')
    requests = set();jobs = set();reserved_gpu = 0
    for item in value['PBS_submissions']:
        if item['request_id'] in requests or item.get('job_id') and item['job_id'] in jobs:
            _fail('NO_DUPLICATE_NATIVE_SUBMISSIONS')
        requests.add(item['request_id'])
        if item.get('job_id'):
            jobs.add(item['job_id'])
        if item['status'] not in TERMINAL:
            if not item.get('job_id'):
                _fail('UNKNOWN_QSUB_INTENT_CANNOT_BE_FREE_OR_REPLAYED')
            reserved_gpu += budget._number(item['reserved_H200_s'], integer=True)+budget._number(item['accounting_pad_s'], integer=True)
    if reserved_gpu != value['reserved'].get('H200_s', 0):
        _fail('FULL_ACTIVE_NATIVE_ALLOCATION_RESERVATIONS')
    replica_key = progress.get('replica_run_key')
    keys = {}
    for candidate, row in progress['candidates'].items():
        native_key = replica_key if candidate == 'REPLICA' else candidate
        step = value.get('training_runs', {}).get(native_key, {}).get('last_native_step', 0)
        if row['optimizer_updates_committed'] != step:
            _fail('OWNER_PROGRESS_MATCHES_ACTUAL_CHARGED_NATIVE_TRAINING_STEPS')
        if native_key:
            keys[native_key] = step
    if (set(value.get('training_runs', {}))-set(keys)
            or sum(keys.values()) != value['optimizer_updates']):
        _fail('NO_OMITTED_OR_DOUBLE_COUNTED_ACTUAL_TRAINING_RUNS')
    return value, digest


def recompute(root, *, request_bytes, snapshot_bytes, progress_bytes, purpose, now_epoch):
    """Read sealed original evidence and current ledger, then recompute and fail closed."""
    if purpose not in ('PREPARE', 'SUBMIT', 'NATIVE_PRETRAIN'):
        _fail('EXPLICIT_PREQUEUE_OR_FRESH_NATIVE_PRETRAIN_PURPOSE')
    request, request_sha, manifest, index, files = _inputs(root, request_bytes)
    snapshot, snapshot_sha = budget._read(snapshot_bytes)
    progress, progress_sha = budget._read(progress_bytes)
    if snapshot['campaign_run_uuid'] != index['campaign_run_uuid'] or snapshot['T0_epoch'] != index['T0_epoch']:
        _fail('FRESH_METADATA_CANNOT_RESET_OR_CHANGE_CAMPAIGN')
    if purpose != 'NATIVE_PRETRAIN':
        budget._fresh(request['created_epoch'], now_epoch, 120)
    ledger, ledger_sha = _ledger(root, snapshot, progress)
    assumptions, _ = budget._read(files[index['assumptions']])
    train_jobs, _ = budget._read(files[index['training_jobs']])
    binding, binding_sha = budget._read(files[index['request_work_binding']])
    if (binding.get('schema') != 'PI0294_PV1_FUTURE_REQUEST_WORK_BINDING_V1'
            or binding.get('stage') != request['stage']
            or request['stage'] not in ('P2_PRIMARY', 'P4_REPLICA', 'P4_DEV')):
        _fail('ONLY_EXACT_IMPLEMENTED_PRE_FINAL_REQUEST_WORK')
    if request['stage'] in ('P2_PRIMARY', 'P4_REPLICA'):
        block = request['training_block'];replica = request['stage'] == 'P4_REPLICA'
        native_key = ('REPLICA_' if replica else '')+block['arm']
        start = request['expected_native_start_step'];stop = block['stop_after_step']
        if (type(replica) is not bool or block.get('replica', False) is not replica
                or block.get('probe_only', False) is not False
                or type(start) is not int or type(stop) is not int or not 0 <= start < stop <= 4096
                or binding.get('arm') != block['arm'] or binding.get('replica') is not replica
                or binding.get('start_step') != start or binding.get('stop_step') != stop
                or ledger.get('training_runs', {}).get(native_key, {}).get('last_native_step', 0) != start
                or replica and progress.get('replica_run_key') != native_key):
            _fail('EXACT_NATIVE_BLOCK_START_STOP_ARM_REPLICA_NO_PROBE_ONLY')
        candidate = 'REPLICA' if replica else block['arm']
        block_raw = _file(root, 'plans/PRIMARY_TRAIN_BLOCK.json')
        if (manifest['execution']['sealed_inputs'].get('plans/PRIMARY_TRAIN_BLOCK.json') != _digest(block_raw)
                or budget._read(block_raw)[0] != block
                or binding.get('forecast_candidate') != candidate):
            _fail('ORIGINAL_SEALED_TRAINING_BLOCK_BYTES')
    else:
        candidate = binding.get('forecast_candidate')
        if binding.get('evaluation_candidate') != request['evaluation_candidate']:
            _fail('EXACT_CURRENT_DEVELOPMENT_CANDIDATE')
    case = _case_cost(index, files, assumptions, now_epoch)
    output = budget.forecast(snapshot=snapshot, progress_bytes=progress_bytes,
        baseline_admission_bytes=files[index['baseline_admission']], case_cost=case,
        assumptions=assumptions, training_jobs=train_jobs['jobs'],
        replica_mode=index['replica_mode'], now_epoch=now_epoch)
    matches = [job for job in output['mandatory_jobs'] if job['work_id'] == binding['work_id']]
    if len(matches) != 1:
        _fail('CURRENT_WORK_ID_MUST_BE_ONE_REQUIRED_FORECAST_JOB')
    job = matches[0]
    prefix = ('TRAIN:' if request['stage'] in ('P2_PRIMARY', 'P4_REPLICA') else 'DEV:')+candidate+':'
    if (not job['work_id'].startswith(prefix) or job['walltime_upper_s'] != request['walltime_s']
            or job['resources']['Hopper_storage_bytes'] < request['additional_output_upper_bytes']):
        _fail('CURRENT_REQUEST_WALL_OUTPUT_AND_WORK_ID_ENVELOPE')
    if request['stage'] in ('P2_PRIMARY', 'P4_REPLICA') and job['resources']['optimizer_updates'] != stop-start:
        _fail('CURRENT_FORECAST_BLOCK_MUST_MATCH_ACTUAL_UPDATE_INTERVAL')
    if not output['admissible_forecast']:
        _fail('NEGATIVE_RECOMPUTED_RUNWAY_NO_NEW_CANDIDATE')
    return dict(schema='PI0294_PV1_RECOMPUTED_FUTURE_ADMISSION_V1',
        status='PASS_RECOMPUTED_FORECAST_NOT_NATIVE_ALLOCATION_OR_GO', purpose=purpose,
        run_uuid=request['run_uuid'], campaign_run_uuid=request['campaign_run_uuid'],
        request_sha256=request_sha, request_id=request['request_id'], submission_nonce=request['submission_nonce'],
        observed_epoch=now_epoch, snapshot_sha256=snapshot_sha, progress_sha256=progress_sha,
        actual_Hopper_ledger_sha256=ledger_sha, request_work_binding_sha256=binding_sha,
        work_id=binding['work_id'], forecast=output,
        upstream_site_lease_scope_stage_and_native_effect_guards_still_required=True)


def release_attestation(root, *, request_bytes, snapshot_bytes, progress_bytes,
                        native_job_bytes, job_id, phase_started_epoch, now_epoch):
    """Build a new owner proof AFTER the requested pre-training phase began."""
    budget._fresh(phase_started_epoch, now_epoch, 120)
    request, request_sha = budget._read(request_bytes)
    native, native_sha = budget._read(native_job_bytes)
    snapshot, _ = budget._read(snapshot_bytes)
    progress, _ = budget._read(progress_bytes)
    for value in (snapshot, progress):
        if not phase_started_epoch <= value['observed_epoch'] <= now_epoch:
            _fail('PREQUEUE_METADATA_NOT_FRESH_NATIVE_PRETRAIN_PROOF')
    if (native.get('job_state') != 'R' or native.get('Job_Name') != request['job_name']
            or native.get('project') != request['project']
            or not native_request_binding(native, request)):
        _fail('EXACT_RUNNING_JOB_REQUEST_AND_NONCE')
    proof = recompute(root, request_bytes=request_bytes, snapshot_bytes=snapshot_bytes,
        progress_bytes=progress_bytes, purpose='NATIVE_PRETRAIN', now_epoch=now_epoch)
    ledger, _ = budget._read((Path(root).parent.parent/'HOPPER_RESOURCE_LEDGER.json').read_bytes())
    matches = [s for s in ledger['PBS_submissions'] if s['request_id'] == request['request_id']]
    if (len(matches) != 1 or matches[0].get('job_id') != job_id
            or matches[0].get('request_sha256') != request_sha or matches[0]['status'] in TERMINAL):
        _fail('ACTUAL_SINGLE_LIVE_REGISTERED_REQUEST_JOB')
    return dict(schema='PI0294_PV1_FRESH_NATIVE_PRETRAIN_FUTURE_ATTESTATION_V1',
        owner='EXISTING_LOCAL_PC', authority_release=AUTHORITY,
        run_uuid=request['run_uuid'], campaign_run_uuid=request['campaign_run_uuid'],
        job_id=job_id, request_id=request['request_id'], submission_nonce=request['submission_nonce'],
        request_sha256=request_sha, phase='BEFORE_FIRST_UPDATE_IN_THIS_NATIVE_BLOCK',
        phase_started_epoch=phase_started_epoch, observed_epoch=now_epoch,
        raw_snapshot_base64=base64.b64encode(snapshot_bytes).decode(),
        raw_progress_base64=base64.b64encode(progress_bytes).decode(),
        raw_native_job_base64=base64.b64encode(native_job_bytes).decode(),
        native_job_sha256=native_sha, recomputed_proof=proof)


def verify_release(root, *, request_bytes, attestation_bytes, expected_job_id,
                   phase_started_epoch, now_epoch):
    """Recompute from raw inputs again; never trust the stored PASS or forecast."""
    value, _ = budget._read(attestation_bytes);request, request_sha = budget._read(request_bytes)
    if (value.get('schema') != 'PI0294_PV1_FRESH_NATIVE_PRETRAIN_FUTURE_ATTESTATION_V1'
            or value.get('owner') != 'EXISTING_LOCAL_PC' or value.get('authority_release') != AUTHORITY
            or value.get('run_uuid') != request['run_uuid']
            or value.get('campaign_run_uuid') != request['campaign_run_uuid']
            or value.get('job_id') != expected_job_id or value.get('request_sha256') != request_sha
            or value.get('request_id') != request['request_id'] or value.get('submission_nonce') != request['submission_nonce']
            or value.get('phase') != 'BEFORE_FIRST_UPDATE_IN_THIS_NATIVE_BLOCK'
            or value.get('phase_started_epoch') != phase_started_epoch):
        _fail('NO_REUSED_PREQUEUE_OTHER_JOB_OR_OTHER_BLOCK_ATTESTATION')
    budget._fresh(value['observed_epoch'], now_epoch, 120)
    try:
        snapshot = base64.b64decode(value['raw_snapshot_base64'], validate=True)
        progress = base64.b64decode(value['raw_progress_base64'], validate=True)
        native = base64.b64decode(value['raw_native_job_base64'], validate=True)
    except (ValueError, TypeError) as exc:
        raise PermissionError('PV1_FUTURE_ADMISSION_ACTUAL_RAW_RELEASE_INPUT_BYTES') from exc
    if _digest(native) != value.get('native_job_sha256'):
        _fail('ORIGINAL_NATIVE_JOB_READBACK_BYTES')
    current = release_attestation(root, request_bytes=request_bytes, snapshot_bytes=snapshot,
        progress_bytes=progress, native_job_bytes=native, job_id=expected_job_id,
        phase_started_epoch=phase_started_epoch, now_epoch=now_epoch)
    return current['recomputed_proof']


def needs_forecast(request):
    if request['stage']=='B0_DEV':return False
    # Recovery-only archived draws neither create a candidate nor apply updates.
    # Their mature query/CPU/allocation caps remain mandatory, including unknowns.
    return not (request['stage'] in ('P2_PRIMARY','P4_REPLICA')
        and request.get('training_block',{}).get('probe_only') is True)


def prequeue(root,request,purpose,*,request_bytes=None):
    """Recompute against fresh sole-owner files; never accept a PASS summary."""
    if not needs_forecast(request):return None
    raw=request_bytes if request_bytes is not None else (json.dumps(request,indent=2,allow_nan=False)+'\n').encode()
    if budget._read(raw)[0]!=request:_fail('EXACT_PREQUEUE_REQUEST_BYTES')
    return recompute(root,request_bytes=raw,snapshot_bytes=_file(root,SNAPSHOT),
        progress_bytes=_file(root,PROGRESS),purpose=purpose,now_epoch=time.time())


def wait_native_release(root,directory,allocation,guard,*,now=time.time,sleep=time.sleep):
    """Pause before new optimization/evaluation for a fresh same-job owner proof."""
    from scripts.pi0292.durable import write
    from .visibility import read_visible
    root,directory=Path(root),Path(directory)
    request_bytes=_file(root,str((directory/'REQUEST.json').relative_to(root)))
    request,_=budget._read(request_bytes)
    if not needs_forecast(request):return None
    phase_started=now();ready=directory/'PV1_FUTURE_PHASE_READY.json'
    if ready.exists():raise FileExistsError('PV1_PRESERVE_FIRST_FUTURE_PHASE_NO_BLIND_RESTART')
    write(ready,dict(schema='PI0294_PV1_FUTURE_PHASE_READY_V1',run_uuid=root.name,
        job_id=allocation['job_id'],request_id=request['request_id'],
        submission_nonce=request['submission_nonce'],request_sha256=_digest(request_bytes),
        phase='BEFORE_FIRST_UPDATE_OR_DEVELOPMENT_QUERY_IN_THIS_NATIVE_BLOCK',
        phase_started_epoch=phase_started,new_updates=0,new_queries=0))
    ready.chmod(0o400)
    target=directory/'PV1_FUTURE_OWNER_ATTESTATION.private.json'
    while now()<min(phase_started+175,allocation['allocation_end_epoch']-2700):
        guard()
        if read_visible(target) is not None:
            proof=verify_release(root,request_bytes=request_bytes,
                attestation_bytes=_file(root,str(target.relative_to(root))),
                expected_job_id=allocation['job_id'],phase_started_epoch=phase_started,now_epoch=now())
            output=directory/'PV1_FUTURE_NATIVE_RECOMPUTATION.json'
            if output.exists():raise FileExistsError('PV1_PRESERVE_FIRST_NATIVE_FUTURE_PROOF')
            write(output,proof);output.chmod(0o400)
            return proof
        sleep(1)
    raise TimeoutError('PV1_FRESH_NATIVE_FUTURE_OWNER_PROOF_REQUIRED_NO_NEW_UPDATES_OR_QUERIES')
