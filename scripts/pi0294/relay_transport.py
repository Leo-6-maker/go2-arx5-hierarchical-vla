"""Local owner orchestration of the existing ACK exchange, never physics/model."""
from pathlib import Path
import time

from scripts.pi0292.durable import read,write
from scripts.pi0292.relay_protocol import RelayExchange,PROTOCOL,sealed,unseal,exact_keys


def teacher_action(query,teacher):
    body=unseal(query)
    if body['identity']['profile']!='TEACHER_REFERENCE' or body['kind']!='student_request' or body['protocol']!=PROTOCOL:
        raise PermissionError('ACTUAL_TEACHER_QUALIFICATION_ONLY_NOT_MODEL_POLICY')
    exact_keys(teacher,('observation_sha256','trajectory_eef10','source'))
    if teacher['observation_sha256']!=body['observation_sha256'] or teacher['source']!='ACTUAL_CURRENT_FEEDBACK_TEACHER_NOT_MODEL':
        raise ValueError('TEACHER_MUST_BIND_THIS_FRESH_OBSERVATION')
    return sealed(dict(protocol=PROTOCOL,kind='action',run_uuid=body['run_uuid'],episode_uuid=body['episode_uuid'],
        sequence=body['sequence'],tick=body['tick'],payload={k:body[k] for k in
        ('query_uuid','reference_uuid','observation_sha256','anchor_sha256','identity')}|
        dict(trajectory_eef10=teacher['trajectory_eef10'],offsets_ns=[i*50000000 for i in range(40)])))


def run_episode(directory,case,core,hopper,*,now=time.time,sleep=time.sleep):
    """Injected endpoints must implement bounded actual SSH reads/writes.

    A dispatched command is never resent after an uncertain network outcome.
    Read-only polling is allowed; generation and command sends are one-shot.
    """
    directory=Path(directory)
    exchange=RelayExchange(directory,run_uuid=case['run_uuid'],episode_uuid=case['episode_uuid'],
        identity=case['identity'],now=now())
    state=dict(status='ACTUAL_ROUTE_START_NOT_TASK_SUCCESS',episode_uuid=case['episode_uuid'],
        profile=case['identity']['profile'],fresh_observations=0,commands_sent=0,acknowledgements=0,
        network_outcome_unknown=False,model_calls_local=0,physics_integrations_local=0)
    path=directory/'ROUTE.json'
    write(path,state)
    try:
        for sequence in range(80):
            end=min(case['deadline_epoch'],now()+60)
            packet=None
            while packet is None and now()<end:
                packet=core.observation(case,sequence)
                if packet is None:sleep(.05)
            if packet is None:raise TimeoutError('NO_FRESH_CORE_OBSERVATION')
            query=exchange.observe(packet,now());state['fresh_observations']+=1
            # Durable exchange state already contains the query before network dispatch.
            state['status']='GENERATION_DISPATCH_INTENT_NO_RESEND';write(path,state)
            action=hopper.query(case,query,core.teacher(case,sequence) if case['identity']['profile']=='TEACHER_REFERENCE' else None)
            if action is None:
                stop=min(end,unseal(query)['expires'])
                while action is None and now()<stop:
                    action=hopper.response(case,unseal(query)['query_uuid'])
                    if action is None:sleep(.05)
            if action is None:raise TimeoutError('GENERATION_RESPONSE_UNKNOWN_NO_RESAMPLE')
            command=exchange.action(action,now())
            state['status']='COMMAND_DISPATCH_INTENT_NO_REPLAY';write(path,state)
            core.install(case,sequence,dict(command=command,action=action,request=query))
            state['commands_sent']+=1;state['status']='WAIT_ACTUAL_PREFIX_ACK';write(path,state)
            ack=None;end=min(case['deadline_epoch'],unseal(query)['expires'])
            while ack is None and now()<end:
                ack=core.ack(case,sequence)
                if ack is None:sleep(.05)
            if ack is None:raise TimeoutError('PHYSICAL_PROGRESS_UNKNOWN_DO_NOT_RESEND')
            exchange.acknowledge(ack,now());state['acknowledgements']+=1
            write(path,state)
        if exchange.state['phase']!='DONE' or exchange.state['tick']!=16000:
            raise ValueError('ALL_EIGHTY_ACTUAL_PREFIX_ACKS_REQUIRED')
        state['status']='PASS_ACTUAL_MESSAGE_ACK_CHAIN_PENDING_INDEPENDENT_PHYSICAL_AUDIT'
        return state
    except BaseException as error:
        state.update(status='STOPPED_PRESERVED_NO_AUTOMATIC_RETRY',
            failure=dict(type=type(error).__name__,error=str(error)),
            network_outcome_unknown=exchange.state['phase'] in ('WAIT_ACTION','WAIT_ACK'))
        raise
    finally:
        write(path,state);exchange.close()
