"""Existing owner's finite qualification phase; no scheduler or heartbeat owner."""
import json
from pathlib import Path
import re
import time

from scripts.pi0292.relay_protocol import digest
from .transport_fixture import BASE
from .transport_fixture_router import route


def logged_call(session,args):
    """Read only the one new JSONL wire record written by the mature client."""
    path=Path(session.metrics);offset=path.stat().st_size if path.exists() else 0
    packet=session.call(args)
    with path.open('rb') as stream:stream.seek(offset);lines=stream.read().splitlines()
    if len(lines)!=1:raise PermissionError('ONE_EXACT_SUCCESSFUL_CLIENT_WIRE_RECORD')
    return dict(packet=packet,wire=json.loads(lines[0]))


def drive(sessions,fault_sessions,plan,registry,emit,certify_fault,certify_nominal,publish,heartbeat,*,registry_sha256,
          fixture_base=BASE,reference_proof_sha256=None):
    closed=set()
    try:
        if (fixture_base not in (BASE,'PV1-PREMODEL')
                or fixture_base=='PV1-PREMODEL' and not re.fullmatch('[0-9a-f]{64}',reference_proof_sha256 or '')
                or fixture_base==BASE and reference_proof_sha256 is not None):
            raise PermissionError('EXACT_VERSIONED_FIXTURE_NAMESPACE_AND_REFERENCE_BINDING')
        return _drive(sessions,fault_sessions,plan,registry,emit,certify_fault,certify_nominal,publish,heartbeat,
            registry_sha256,closed,fixture_base,reference_proof_sha256)
    finally:
        for session in [*sessions.values(),*fault_sessions.values()]:
            if id(session) not in closed:session.close();closed.add(id(session))


def _drive(sessions,fault_sessions,plan,registry,emit,certify_fault,certify_nominal,publish,heartbeat,registry_sha256,closed,
           fixture_base,reference_proof_sha256):
    if set(sessions)!={'core','hopper'} or set(fault_sessions)!={'late_response','process_exit'}:
        raise PermissionError('FIXED_EXISTING_TWO_SSH_CHANNELS_AND_TWO_ISOLATED_FAULT_CHANNELS')
    if (not re.fullmatch('[0-9a-f]{64}',registry_sha256)
            or not re.fullmatch(re.escape(fixture_base)+r'-ALLOCATION-(?:0[1-9]|1[0-9]|2[0-4])',registry['label'])
            or any(s.role!=role or s.label!=registry['label'] for role,s in sessions.items())
            or len({id(s) for s in [*sessions.values(),*fault_sessions.values()]})!=4):
        raise PermissionError('EXACT_DISJOINT_EXISTING_FIXTURE_CHANNEL_IDENTITIES')
    for name,session in fault_sessions.items():
        expected=registry['label'].replace(fixture_base,fixture_base+('-FAULT-LATE' if name=='late_response' else '-FAULT-EXIT'),1)
        if session.role!='hopper' or session.label!=expected:
            raise PermissionError('ISOLATED_FAULT_NAMESPACE_NOT_A_PRODUCTION_SESSION')
    def call(role,args):
        heartbeat();return logged_call(sessions[role],args)
    rows=route(call,plan,registry,emit)
    if len(rows)!=80:raise PermissionError('EIGHTY_REAL_NOMINAL_CYCLES_REQUIRED')
    nominal={}
    for role,session in sessions.items():
        session.close();closed.add(id(session));heartbeat();nominal[role]=certify_nominal(role,session)
        if nominal[role]['native_process_terminal_verified'] is not True:
            raise PermissionError('ACTUAL_NOMINAL_FIXTURE_RPC_PROCESS_MUST_CLOSE_BEFORE_MODEL')
    faults={}
    for name in ['late_response','process_exit']:
        session=fault_sessions[name];expected=fixture_base+('-FAULT-LATE' if name=='late_response' else '-FAULT-EXIT')
        if session.role!='hopper' or session.label!=registry['label'].replace(fixture_base,expected,1):
            raise PermissionError('ISOLATED_FAULT_NAMESPACE_NOT_A_PRODUCTION_SESSION')
        heartbeat();began=time.monotonic();error=None
        try:session.call(dict(operation='observation',sequence=0))
        except (TimeoutError,ConnectionError,BrokenPipeError) as exc:error=type(exc).__name__
        if error is None:raise PermissionError('ACTUAL_LATE_OR_EXIT_FAULT_DID_NOT_STOP_CLIENT')
        elapsed=time.monotonic()-began
        if name=='late_response' and elapsed<20:raise PermissionError('LATE_FAULT_MUST_TEST_REAL_ORIGINAL_TWENTY_SECOND_TIMEOUT')
        rejected=False
        try:session.call(dict(operation='observation',sequence=0))
        except RuntimeError as exc:rejected='POISONED_FILE_RPC_SESSION_NO_NEW_REQUEST_OR_REPLAY' in str(exc)
        if not rejected:raise PermissionError('POISONED_FIXTURE_CHANNEL_MUST_REFUSE_SECOND_REQUEST_BEFORE_SEND')
        session.close();closed.add(id(session));heartbeat();native=certify_fault(name,session)
        if (native['native_process_terminal_verified'] is not True or native['actual_fault_requests']!=1
                or native['job_id']!=registry['job_id'] or native['label']!=session.label
                or native['source_sha']!=registry['hopper_source_sha']
                or native['actual_server_exit']!=(0 if name=='late_response' else 72)
                or not re.fullmatch('[0-9a-f]{64}',native['native_fault_intent_sha256'])):
            raise PermissionError('EXACT_FAULT_PROCESS_TERMINAL_AND_ONE_RAW_REQUEST_REQUIRED')
        faults[name]=dict(native,status='PASS',failed_closed=True,unknown_effects_resent=0,
            second_call_blocked_before_send=True,actual_client_error=error,actual_client_wall_s=elapsed)
    value=dict(schema='PI0294_PV1_OWNER_TRANSPORT_FAULTS_V1' if fixture_base=='PV1-PREMODEL' else 'MP1_N1_OWNER_TRANSPORT_FAULTS_V1',
        run_uuid=registry['run_uuid'],label=registry['label'],
        job_id=registry['job_id'],registry_sha256=registry_sha256,nominal_cycles=len(rows),
        nominal_cycles_sha256=digest(rows),nominal_terminals=nominal,faults=faults,heartbeat_writer=False,new_models=0,new_physics=0)
    if reference_proof_sha256 is not None:value['reference_proof_sha256']=reference_proof_sha256
    publish(value);return value
