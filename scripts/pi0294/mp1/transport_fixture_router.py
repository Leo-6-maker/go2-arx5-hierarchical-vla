"""Finite local-PC two-SSH relay of recorded data, never closed-loop evaluation."""
import hashlib
import json
import re
import time

from scripts.pi0292.relay_protocol import unseal
from .transport_fixture import CYCLES,receive_query,check_response,packet_identity


def verify_cycle(row,query,response,ack):
    calls=row['wire_calls'];sequence=row['sequence']
    prefix=[('core','observation'),('hopper','query')];suffix=[('core','install'),('core','ack')]
    if (len(calls)<5 or [(m['role'],m['operation']) for m in calls[:2]]!=prefix
            or [(m['role'],m['operation']) for m in calls[-2:]]!=suffix
            or any((m['role'],m['operation'])!=('hopper','response') for m in calls[2:-2])):
        raise PermissionError('EXACT_NOMINAL_WIRE_OPERATION_ORDER_NO_DUPLICATE_QUERY_OR_INSTALL')
    expected=[]
    expected.append((dict(operation='observation',sequence=sequence),query))
    expected.append((dict(operation='query',sequence=sequence,query=query),dict(status='FIXTURE_QUERY_RESERVED_NO_MODEL',request_sha256=query['sha256'])))
    expected.extend((dict(operation='response',sequence=sequence),response if i==len(calls)-3 else None) for i in range(2,len(calls)-2))
    expected.append((dict(operation='install',sequence=sequence,bundle=response),ack))
    expected.append((dict(operation='ack',sequence=sequence),ack))
    seen=set()
    for metrics,(args,value) in zip(calls,expected):
        key=(metrics['role'],metrics['rpc_id'])
        if (key in seen or metrics['effect_retried'] is not False or metrics['call_watchdog_s']!=20
                or not 0<metrics['wall_s']<=20):raise PermissionError('ORIGINAL_SINGLE_SEND_WIRE_WATCHDOG_AND_UNIQUE_RPC')
        seen.add(key)
        request=(json.dumps(dict(rpc_id=metrics['rpc_id'],args=args),allow_nan=False)+'\n').encode()
        reply=json.dumps(dict(rpc_id=metrics['rpc_id'],ok=True,value=value),allow_nan=False).encode()
        if (hashlib.sha256(request).hexdigest()!=metrics['request_sha256'] or hashlib.sha256(reply).hexdigest()!=metrics['response_sha256']
                or len(request)!=metrics['request_bytes'] or len(reply)!=metrics['response_bytes']):
            raise ValueError('RAW_WIRE_HASH_BYTES_MUST_JOIN_EVERY_REAL_FIXTURE_OPERATION')
    if row['request_sha256']!=calls[1]['request_sha256'] or row['response_sha256']!=calls[-3]['response_sha256']:
        raise ValueError('CYCLE_QUERY_AND_FINAL_RESPONSE_WIRE_IDENTITIES')


def route(call,plan,registry,emit,*,clock=time.monotonic,now=time.time,sleep=time.sleep):
    """call returns the actual envelope and the existing stdio client's wire log."""
    rows=[]
    for sequence in range(CYCLES):
        wire=[]
        def invoke(role,args):
            started=clock()
            if now()>=registry['deadline_epoch']:raise TimeoutError('FINITE_NATIVE_TRANSPORT_FIXTURE_DEADLINE')
            result=call(role,args);elapsed=clock()-started
            packet=result['packet'];metrics=result['wire']
            if (packet.get('ok') is not True or metrics['operation']!=args['operation']
                    or packet['rpc_id']!=metrics['rpc_id'] or metrics['effect_retried'] is not False
                    or metrics['call_watchdog_s']!=20 or not 0<metrics['wall_s']<=20
                    or not 0<=elapsed<=20):
                raise PermissionError('ACTUAL_SUCCESSFUL_PRODUCTION_STDIO_WIRE_RECORD_REQUIRED')
            raw_request=(json.dumps(dict(rpc_id=packet['rpc_id'],args=args),allow_nan=False)+'\n').encode()
            raw_response=json.dumps(packet,allow_nan=False).encode()
            if (hashlib.sha256(raw_request).hexdigest()!=metrics['request_sha256']
                    or hashlib.sha256(raw_response).hexdigest()!=metrics['response_sha256']
                    or len(raw_request)!=metrics['request_bytes'] or len(raw_response)!=metrics['response_bytes']):
                raise ValueError('FIXTURE_ENVELOPE_MUST_EQUAL_RECORDED_RAW_WIRE_BYTES')
            wire.append(dict(metrics,role=role));return packet['value']
        query=invoke('core',dict(operation='observation',sequence=sequence))
        body=receive_query(query,registry,plan,now=now,clock=clock,sleep=sleep)
        if body['query_uuid']!=packet_identity(registry,sequence):raise PermissionError('FIXTURE_QUERY_ID')
        invoke('hopper',dict(operation='query',sequence=sequence,query=query));query_wire=wire[-1]
        while True:
            if now()>=body['expires_epoch']:raise TimeoutError('ORIGINAL_SIXTY_SECOND_FIXTURE_QUERY_NO_RETRY')
            response=invoke('hopper',dict(operation='response',sequence=sequence))
            if response is not None:break
            sleep(.05)
        response_wire=wire[-1];check_response(response,query,registry,plan)
        installed=invoke('core',dict(operation='install',sequence=sequence,bundle=response))
        ack=invoke('core',dict(operation='ack',sequence=sequence))
        a=unseal(ack)
        if (installed!=ack or a['fixture_install_count']!=1 or a['physical_installs']
                or a['unknown_effects_resent'] or a['query_uuid']!=body['query_uuid']
                or a['request_sha256']!=query['sha256'] or a['response_sha256']!=response['sha256']):
            raise PermissionError('EXACT_ONCE_CORE_FIXTURE_ACK_NOT_PHYSICAL_ACK')
        row=dict(sequence=sequence,query_uuid=body['query_uuid'],RGB_payload_bytes=plan['RGB_payload_bytes'],
            maximum_RPC_s=max(v['wall_s'] for v in wire),fixture_install_count=1,ACK_matches_request=True,
            request_sha256=query_wire['request_sha256'],response_sha256=response_wire['response_sha256'],
            query_packet_sha256=query['sha256'],response_packet_sha256=response['sha256'],
            core_ACK_sha256=ack['sha256'],wire_calls=list(wire),physical_installs=0,unknown_effects_resent=0)
        if any(not re.fullmatch('[0-9a-f]{64}',row[k]) for k in ['request_sha256','response_sha256']):
            raise ValueError('ACTUAL_FIXTURE_WIRE_HASHES')
        verify_cycle(row,query,response,ack)
        # A completion may be unknown after a lost reply; never resubmit it.
        invoke('hopper',dict(operation='ack',sequence=sequence,bundle=dict(cycle=row,ack=ack)))
        emit(row);rows.append(row)
    return rows
