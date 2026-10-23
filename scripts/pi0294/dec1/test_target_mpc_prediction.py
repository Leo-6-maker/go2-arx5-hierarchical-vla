"""CPU fixtures for provided WBC prediction identity; never native safety GO."""
from array import array
from copy import deepcopy
import hashlib
import json
import pytest
from . import target_mpc_prediction as unit
from .test_target_time_transaction import fixture,events
from .target_time_transaction import prepare


def case(mode='I_PROGRESS',controller_delta=0.):
    raw,proposal,args,*_=fixture(mode,controller_time=controller_delta);args['source_mpc_s']=0.
    prepared=prepare(raw,proposal,**args);trace=events(prepared)
    row=trace[1];observation=dict(time=0.,state={'value':array('f',[.1]*12).tolist()},input={'value':[0.]*12},mode=0)
    policy=dict(controller_type=2,init_observation=deepcopy(observation),
        plan_target_trajectories=dict(time_trajectory=row['times'].copy(),
            state_trajectory=[{'value':x.copy()} for x in row['poses']],
            input_trajectory=[{'value':x.copy()} for x in row['inputs']]),
        time_trajectory=[i*.05 for i in range(21)],state_trajectory=[{'value':array('f',[.1]*12).tolist()} for _ in range(21)],
        input_trajectory=[{'value':[0.]*12} for _ in range(21)],post_event_indices=[],
        mode_schedule=dict(event_times=[],mode_sequence=[0]),data=[],performance_indices={})
    return prepared,trace,observation,policy


def invoke(prepared,trace,observation,policy,**kw):
    raw=json.dumps(policy,separators=(',',':'),allow_nan=False).encode()
    return unit.bind(prepared,trace,raw,expected_policy_sha256=kw.get('sha',hashlib.sha256(raw).hexdigest()),
        actual_observation=observation)


@pytest.mark.parametrize('mode',['I_TIME','I_PROGRESS'])
@pytest.mark.parametrize('controller_delta',[0.,.0125,.025])
def test_installed_target_and_independent_observation_bind_original_native_prediction(mode,controller_delta):
    prepared,trace,obs,p=case(mode,controller_delta);before=deepcopy(p)
    r=invoke(prepared,trace,obs,p)
    assert r.wire_sha256==prepared.wire_sha256 and r.command_id==prepared.command_id
    assert len(r.state_trajectory)==21 and len(r.state_trajectory[0])==12
    assert not r.native_prefix_safety_qualified and not r.physics_advance_authorized
    assert p==before


@pytest.mark.parametrize('fault',['sha','no_ack','stale_reference','target_clock','target_pose','BASE',
    'target_input','predicted_as_actual','observation_clock','observation_mode','policy_state_width',
    'policy_input_width','short_horizon','unsorted_times','mode_schedule','controller_unknown','boolean_state','extra_field','nonfinite_nested','boolean_mode','boolean_target_time','unserialized_double'])
def test_old_wrong_or_partial_native_policy_is_not_current_reference_evidence(fault):
    prepared,trace,obs,p=case();kw={}
    if fault=='sha':kw['sha']='0'*64
    elif fault=='no_ack':trace.pop()
    elif fault=='stale_reference':trace[1]['ref_id']='0'*64
    elif fault=='target_clock':p['plan_target_trajectories']['time_trajectory'][0]=.001
    elif fault=='target_pose':p['plan_target_trajectories']['state_trajectory'][5]['value'][0]+=.001
    elif fault=='BASE':p['plan_target_trajectories']['state_trajectory'][0]['value']+=[0.]*7
    elif fault=='target_input':p['plan_target_trajectories']['input_trajectory'][0]['value'][0]=1.
    elif fault=='predicted_as_actual':p['init_observation']['state']['value'][0]=.2
    elif fault=='observation_clock':obs['time']=.026;p['init_observation']['time']=.026
    elif fault=='observation_mode':obs['mode']=True;p['init_observation']['mode']=True
    elif fault=='policy_state_width':p['state_trajectory'][0]['value'].pop()
    elif fault=='policy_input_width':p['input_trajectory'][0]['value'].pop()
    elif fault=='short_horizon':p['time_trajectory']=[x*.1 for x in p['time_trajectory']]
    elif fault=='unsorted_times':p['time_trajectory'][4]=p['time_trajectory'][3]
    elif fault=='mode_schedule':p['mode_schedule']['mode_sequence']=[1]
    elif fault=='controller_unknown':p['controller_type']=0
    elif fault=='boolean_state':p['state_trajectory'][0]['value'][0]=True
    elif fault=='extra_field':p['source_S1']='PASS'
    elif fault=='nonfinite_nested':p['performance_indices']['cost']=float('inf')
    elif fault=='boolean_mode':p['mode_schedule']['mode_sequence']=[False]
    elif fault=='boolean_target_time':p['plan_target_trajectories']['time_trajectory'][0]=False
    elif fault=='unserialized_double':p['state_trajectory'][0]['value'][0]=.1
    with pytest.raises((PermissionError,ValueError)):invoke(prepared,trace,obs,p,**kw)


def test_duplicate_trace_fields_cannot_select_a_different_policy():
    prepared,trace,obs,p=case();raw=json.dumps(p).encode();raw=b'{"controller_type":1,'+raw[1:]
    with pytest.raises(PermissionError,match='NO_DUPLICATE_JSON_FIELDS'):
        unit.bind(prepared,trace,raw,expected_policy_sha256=hashlib.sha256(raw).hexdigest(),actual_observation=obs)
