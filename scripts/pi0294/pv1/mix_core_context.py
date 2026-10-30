"""C/D registered neural child on the existing Core entry and LOCAL ledger.

This validates cold assets and actual registration, never GPU/model permission.
The closed training bytes are checked again before accepting its inference view.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import mix_core_accounting as c, mix_core_identity as identity
from . import mix_accounting as a, successor_cd as s, mix_stage_handoff as handoff
from .evaluation_core_scope import member, PARENT_BINDING_SHA, ORIGINAL_NUMERICAL_TABLE_SHA, COSMOS_PROCESSOR_PINS
from .successor_core_context import ASSET_IDENTITY_SHA

CAPTURE = 'plans/V5_CD_CLOSED_TRAINING_CAPTURE.private.json'
LOCAL_BINDING = 'plans/V5_CD_CORE_LEDGER_BINDING.private.json'


def need(ok, message):
    if not ok:
        raise PermissionError('V5_CD_CORE_CONTEXT_' + message)


def captured_endpoint(value):
    """Decode every actual terminal member, not a caller's endpoint PASS flag."""
    need(value.get('schema') == 'V5_CD_REAL_TRAINING_TERMINAL_BYTES_V1', 'FULL_CAPTURE_SCHEMA')
    fields = {'settled_bytes', 'lease_bytes', 'closed_bytes', 'owner_exit_bytes', 'OS_exit_bytes'}
    need(type(value.get('terminal_capture_base64')) is dict and
         set(value['terminal_capture_base64']) == fields, 'ALL_REAL_TERMINAL_MEMBERS')
    captured = {n:base64.b64decode(raw, validate=True) for n, raw in value['terminal_capture_base64'].items()}
    raw = base64.b64decode(value['global_base64'], validate=True)
    proof = handoff.closed_endpoint(global_bytes=raw, arm=value['arm'], **captured)
    need(value.get('handoff') == proof, 'INDEPENDENT_RECOMPUTED_HANDOFF_NOT_PREPARED')
    return proof


def d25_request_marker(local,request_id):
    """Select the paid D25 route for its exact cold pre-grant successor too."""
    from . import mix_core_d25_local_successor as paid
    from .mix_core_d25_admission_repair import validate as repair
    original=paid.validated(local)
    if original is None:return None
    continuation=repair(local,original)
    allowed=(original['replacement_request_id'],)+((continuation['replacement_request_id'],)if continuation else())
    return original if request_id in allowed else None


def validate(root, *, closeout=False):
    need(type(closeout) is bool, 'EXPLICIT_CLOSEOUT_PHASE')
    root = Path(root)
    need(root.is_absolute() and root.is_dir() and not root.is_symlink()
         and root.stat().st_uid == os.getuid() == 2009 and os.uname().nodename == 'corelab-4080s',
         'ACTUAL_CORE_ACCOUNT_AND_REGISTERED_ROOT')
    for name in ('RUN_MANIFEST.json', 'OUTPUT_BINDING.private.json'):
        p = member(root, name)
        need(p.is_file() and p.stat().st_uid == 2009 and not p.stat().st_mode & 0o222, 'IMMUTABLE_IDENTITY:' + name)
    m = read(root/'RUN_MANIFEST.json'); b = read(root/'OUTPUT_BINDING.private.json'); identity.cold_profile(m)
    need(root.parent.name == 'neural-children' and m['run_uuid'] == root.name
         and m['output_root'] == str(root) and b['run_root'] == str(root) and b['run_uuid'] == root.name
         and b['uid'] == 2009 and b['role'] == c.ROLE+'__'+root.name
         and Path(b['registered_parent']).resolve() == root.parent.resolve()
         and m['execution']['private_path_role'] == b['role']
         and m['execution']['output_binding_sha256'] == sha(root/'OUTPUT_BINDING.private.json'),
         'ORIGINAL_NEURAL_NAMESPACE_AND_EXACT_NEW_REGISTERED_ROLE')
    parent = root.parent.parent
    need(sha(parent/'OUTPUT_BINDING.private.json') == PARENT_BINDING_SHA, 'SAME_ORIGINAL_PARENT')
    pb = read(parent/'OUTPUT_BINDING.private.json')
    need(pb['registry'] == b['registry'], 'ONE_EXISTING_CANONICAL_REGISTRY')
    from . import node_local_custody as custody
    custody.registration(root,m,b,legacy_read=read)
    code = member(root, m['code']['capsule_relative']); seal = member(root, m['code']['source_seal_relative'])
    need(Path(__file__).resolve() == (code/'scripts/pi0294/pv1/mix_core_context.py').resolve()
         and sha(seal) == m['code']['source_seal_sha256'], 'EXECUTE_EXACT_OWN_IMMUTABLE_CAPSULE')
    worker = code.parent/'WORKER_RUN_MANIFEST.json'
    need(os.environ.get('PV1_RUN_MANIFEST') == str(worker) and not worker.is_symlink()
         and not worker.stat().st_mode & 0o222 and worker.read_bytes() == (root/'RUN_MANIFEST.json').read_bytes(),
         'STANDARD_ENTRY_WORKER_MANIFEST_NO_PATH_ONLY_OVERRIDE')
    source = read(seal)
    need(source['source_sha'] == m['code']['vla_commit'], 'ONE_ENGINEERING_SOURCE')
    for name, pin in source['files'].items():
        need(sha(member(code, name)) == pin, 'SOURCE_MEMBER:' + name)
    frozen = m['code']['frozen_numerical_files']
    need(len(frozen) == 15 and hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
         == ORIGINAL_NUMERICAL_TABLE_SHA and all(source['files'].get(n) == pin and sha(member(code, n)) == pin
         for n, pin in frozen.items()), 'ALL_ORIGINAL_NUMERICAL_PHYSICS_S1_SCORE_FILES')
    for name, pin in m['execution']['sealed_inputs'].items():
        p = member(root, name)
        need(p.is_file() and p.stat().st_uid == 2009 and not p.stat().st_mode & 0o222 and sha(p) == pin,
             'ACTUAL_IMMUTABLE_INPUT:' + name)
    def sealed(name):
        need(name in m['execution']['sealed_inputs'], 'PRESEALED_REQUIRED_MEMBER:' + name)
        return read(member(root, name))
    binding = sealed(LOCAL_BINDING); path = Path(binding['canonical_LOCAL_ledger']).resolve()
    need(path.is_relative_to((parent/'control').resolve()) and path.name == 'LOCAL_RESOURCE_LEDGER.private.json',
         'CANONICAL_LOCAL_NOT_A_NEW_CAMPAIGN_LEDGER')
    local = read(path); marker = s.validated(local)
    need(marker == sealed('V5_CD_SCOPE_BINDING.json')
         and local['unknown_effects'] == local['pending_updates'] == local['pending_queries'] == [],
         'ADOPTED_ACTUAL_SCOPE_AND_UNKNOWN_FENCES')
    paths = list((root/'pbs').glob('alloc-*/REQUEST.json'))
    need(len(paths) == 1, 'ONE_EXACT_IMMUTABLE_CURRENT_REQUEST')
    request = read(paths[0]); identity.request_identity(m, request, sha(root/'RUN_MANIFEST.json'))
    need(request['T_CD'] == marker['T_CD'] and request['deadline_epoch'] == marker['deadline_epoch'], 'ACTUAL_CLOCK_NOT_CALLER_CHOSEN')
    row = local.get(c.FIELD, {}).get(request['request_id'])
    need(type(row) is dict and row['request_sha256'] == sha(paths[0])
         and row['source_sha'] == request['source_sha'] and row['child_run_uuid'] == root.name
         and row['status'] in c.STATES and row['run_manifest_sha256'] == sha(root/'RUN_MANIFEST.json'),
         'ACTUAL_ESCROW_ROW_BINDS_REQUEST_AND_MANIFEST')
    from . import mix_core_premodel_repair as repair
    from . import mix_core_d25_local_successor as d25_local
    d25_marker=d25_request_marker(local,request['request_id'])
    exact_d25=d25_marker is not None
    if exact_d25:
        need(sealed('plans/V5_CD_D25_FINITE_LOCAL_SUCCESSOR.private.json')==d25_marker
             and m.get('D25_local_successor_sha256')==d25_local.digest(d25_marker)
             and request.get('D25_local_successor_sha256')==d25_local.digest(d25_marker),
             'EXACT_D25_LOCAL_SUCCESSOR_NOT_C50_CREDIT')
        from .mix_core_d25_cold_source import sealed_context
        sealed_context(root,m,request,local,d25_marker,read,sha)
    elif repair.output_credit({**local},row):
        need(sealed('plans/V5_CD_SAME_PANEL_PREPAID_IO.private.json')==repair.validated(local), 'EXACT_SAME_PANEL_PAID_IO_BINDING')
        from . import mix_core_recovery as recovered
        continuation=recovered.validate(local)
        if continuation is not None:need(sealed('plans/V5_CD_CORE_RECOVERY.private.json')==continuation,'ACTUAL_RECOVERY_CUSTODY_BINDING')
        from . import mix_core_prebind_terminal as prebind
        next_continuation=prebind.continuation(local)
        if next_continuation is not None:need(sealed('plans/V5_CD_KNOWN_ZERO_PREBIND_CONTINUATION.private.json')==next_continuation,'ACTUAL_KNOWN_ZERO_PREBIND_CONTINUATION')
        from . import mix_core_caps_continuation as caps_cont
        cap_retry=caps_cont.validate(local)
        if cap_retry is not None:need(sealed('plans/V5_CD_ZERO_MODEL_CAP_PRECEDENCE_CONTINUATION.private.json')==cap_retry,'ACTUAL_CAP_PRECEDENCE_COLD_CONTINUATION')
        rpc_retry=caps_cont.validate_RPC(local)
        if rpc_retry is not None:need(sealed('plans/V5_CD_ZERO_MODEL_REGISTERED_RPC_AFFINITY_CONTINUATION.private.json')==rpc_retry,'ACTUAL_REGISTERED_RPC_AFFINITY_COLD_CONTINUATION')
        peer_retry=caps_cont.validate_peer(local)
        if peer_retry is not None:need(sealed('plans/V5_CD_ZERO_MODEL_PHYSICAL_PEER_IDENTITY_CONTINUATION.private.json')==peer_retry,'ACTUAL_PHYSICAL_PEER_IDENTITY_COLD_CONTINUATION')
        desktop_retry=caps_cont.validate_recorded_desktop(local)
        if desktop_retry is not None:need(sealed('plans/V5_CD_ZERO_MODEL_RECORDED_DESKTOP_EPOCH_CONTINUATION.private.json')==desktop_retry,'ACTUAL_RECORDED_DESKTOP_EPOCH_COLD_CONTINUATION')
        bound_retry=caps_cont.validate_bound_age(local)
        if bound_retry is not None:need(sealed('plans/V5_CD_ZERO_MODEL_POST600_BOUND_AGE_CONTINUATION.private.json')==bound_retry,'ACTUAL_POST600_BOUND_AGE_CONTINUATION')
        from scripts.pi0294.generalization import minimal_gpu_repair
        if minimal_gpu_repair.validate(local)is not None:need(sealed('plans/V5_CD_GPU148_AUTHORITY.private.json')==local[minimal_gpu_repair.FIELD],'EXACT_ADOPTED_GPU148_AUTHORITY')
        limit_retry=caps_cont.validate_rpc_limit(local)
        if limit_retry is not None:need(sealed('plans/V5_CD_ZERO_INTEGRATION_RPC_CPU_LIMIT_CONTINUATION.private.json')==limit_retry,'ACTUAL_RPC_CPU_LIMIT_CONTINUATION')
        control_retry=caps_cont.validate_production_control(local)
        if control_retry is not None:need(sealed('plans/V5_CD_ZERO_INTEGRATION_PRODUCTION_RPC_CONTROL_CONTINUATION.private.json')==control_retry,'ACTUAL_PRODUCTION_RPC_CONTROL_CONTINUATION')
    from .mix_core_partial_resume import validate_inputs
    validate_inputs(root,m,local)
    proof = captured_endpoint(sealed(CAPTURE)); arm = request['evaluation_candidate']
    need(proof['arm'] == arm and proof['checkpoint_manifest_sha256'] == request['checkpoint_manifest_sha256']
         and a.sha(a.raw(proof)) == request['training_handoff_sha256'], 'TRUE_CLOSED_COMPLETE_1024_ENDPOINT')
    checkpoint = sealed('plans/EVALUATION_CHECKPOINT.private.json')
    need(checkpoint['candidate'] == arm and checkpoint['global_step'] == 1024
         and checkpoint['checkpoint_manifest_sha256'] == proof['checkpoint_manifest_sha256']
         and checkpoint['full_resume_files_present'] is False
         and Path(checkpoint['checkpoint']).resolve() == (root/'model_views'/arm).resolve(),
         'FIXED_1024_INFERENCE_VIEW_NOT_A_NEW_TRAIN_OR_OLD_AB_MODEL')
    from .mix_evaluation import session_spec, candidate_plan, stage, CORE_CONTROL
    control_selector=m['execution'].get('production_RPC_control')
    need(control_selector in (None,CORE_CONTROL),'KNOWN_PRESEALED_PRODUCTION_CONTROL_VERSION')
    session_spec(root, arm+'-DEV-N294101')
    population=sealed('plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json')
    need(sealed('plans/SOURCE_EVALUATION_PLAN.json')==candidate_plan(population['groups'],arm)
         and sealed('PV1_STAGE_BINDING.json')==stage(arm,original_core_control=control_selector==CORE_CONTROL)
         and m['execution']['sealed_inputs']['plans/SOURCE_EVALUATION_PLAN.json']==request['planned_panel_sha256'],
         'SAME_ORIGINAL_PANEL_AND_FROZEN_CONTROL_STAGE')
    sealed('plans/INITIAL_CORR_PARAMETER_INVENTORY.json'); sealed('plans/PRIOR_EVALUATION_OUTCOMES.json')
    need(m['execution']['sealed_inputs'].get('plans/ASSET_IDENTITY.private.json') == ASSET_IDENTITY_SHA,
         'EXACT_FROZEN_ASSET_MAP')
    custody.native_assets(root,m,expected_original_sha256=ASSET_IDENTITY_SHA)
    bindings = sealed('NEURAL_BINDINGS.private.json'); inputs = sealed('plans/PV1_TRAIN_INPUTS.private.json')
    need(m['code']['official_gr00t_commit'] == '51d4c89f72fda44cbf77285c6a8114b52676b8a1'
         and Path(bindings['source']).is_dir() and Path(bindings['python']).is_file()
         and Path(bindings['cosmos']).name == '9ce19a195e423419c349abfc86fd07178b230561'
         and all(sha(Path(bindings['cosmos'])/n) == pin for n, pin in COSMOS_PROCESSOR_PINS.items()),
         'PINNED_EXISTING_OFFICIAL_RUNTIME_AND_PROCESSOR')
    from scripts.pi0294.e1.scope import DATA, STATS
    from scripts.pi0294.dec1.authority import BANK
    need(sha(Path(bindings['dataset'])/'DATASET_MANIFEST.json') == DATA
         and sha(Path(bindings['dataset'])/'statistics.json') == STATS
         and sha(Path(inputs['new_dataset'])/'DATASET_MANIFEST.json') == BANK
         and sha(Path(inputs['new_dataset'])/'statistics.json') == STATS, 'ORIGINAL_TRAIN_METADATA_AND_R0')
    lease = read(b['lease']); now = time.time()
    need(lease['status'] == 'ACTIVE' and lease['run_uuid'] == root.name
         and lease['authority_release'] == s.TASK_SOURCE and lease['deadline_epoch'] == marker['deadline_epoch']
         and marker['T_CD'] <= now and (closeout or now < marker['deadline_epoch']),
         'ACTUAL_SUCCESSOR_LEASE_NOT_EXPIRED_OLD_AB_LEASE')
    infra = custody.entry_directory(root,m,b)
    need(sha(infra/'enter.sh') == m['execution']['entry_sha256']
         and sha(infra/'doctor.py') == m['execution']['doctor_sha256'], 'DEPLOYED_STANDARD_ENTRY_AND_DOCTOR')
    if not closeout:
        need(not any(p.exists() for p in (root/'STOP.json', parent/'STOP.json', root/'SOURCE_PHYSICS_STOP.json',
             root/'SOURCE_SAFETY_STOP_FROM_CORE.json')), 'ALL_ORIGINAL_STOP_FENCES')
    return m, bindings, inputs, sealed('PV1_STAGE_BINDING.json'), marker
