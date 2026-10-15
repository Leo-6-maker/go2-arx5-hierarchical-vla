"""Finite full-body data shape/causality fixtures, not a bank qualification."""
import copy
import numpy as np
from .night_r1_data import counts,observed_state19,committed_series


def contracts():
    rows=[]
    def good(name,fn):fn();rows.append(dict(name=name,status='PASS_FIXTURE_ONLY'))
    def bad(name,fn):
        try:fn()
        except (ValueError,KeyError,PermissionError):rows.append(dict(name=name,status='PASS_EXPECTED_REJECTION'))
        else:raise AssertionError(name)
    for d in (32,64,96,128):
        def check(d=d):assert counts(d)==(20*d,20*d+1,20*d-39)
        good('VARIABLE_T_'+str(d),check)
    bad('FRACTIONAL_DURATION',lambda:counts(32.))
    bad('UNAUTHORIZED_DURATION',lambda:counts(256))
    def state():
        p=np.eye(4);p[2,3]=.3;s=observed_state19(np.eye(4),.5,p,[.01,.02,.03],p)
        assert s.shape==(19,) and np.array_equal(s[10:13],np.zeros(3))
        assert np.array_equal(s[13:16],[.01,.02,.03]) and s[16]==.3
    good('MEASURED_19D_STATE_NO_TASK_ORACLE',state)
    bad('OVERSHOOT_NOT_CLIPPED_INTO_VALID_STATE',lambda:observed_state19(np.eye(4),1.01,np.eye(4),[0,0,0],np.eye(4)))
    n=640;samples=[dict(time_s=i*.05,native_step=2500+25*i,active_intent_sequence=None if i==0 else (i-1)//8,executed_prefix_index=None if i==0 else (i-1)%8) for i in range(n+1)]
    records=[]
    for i in range(0,n,8):
        action=np.zeros((40,13));action[:,9]=.5
        records.extend([dict(phase='SUBMITTED_REQUIRES_INSTALL_ACK',sequence=i//8,index=i,action_EEF13=action.tolist(),base_valid=True,wire_sha256=str(i),canonical_safety_gate=dict(decision='ACCEPT')),
            dict(phase='INSTALL_ACK',wire_sha256=str(i),native_step=samples[i]['native_step'])])
    def actual():assert committed_series(samples,records,32).shape==(640,13)
    good('EVERY_20HZ_COMMAND_COMES_FROM_EXECUTED_INSTALLED_PREFIX',actual)
    bad('INCOMPLETE_TASK_NOT_EXPORTABLE',lambda:committed_series(samples[:-1],records,32))
    bad('UNACKNOWLEDGED_PREVIEW_NOT_LABEL',lambda:committed_series(samples,[r for r in records if r['phase']!='INSTALL_ACK'],32))
    changed=copy.deepcopy(records);changed[0]['base_valid']=False
    bad('ABSENT_BASE_IS_NOT_ZERO_SUPERVISION',lambda:committed_series(samples,changed,32))
    changed_samples=copy.deepcopy(samples);changed_samples[9]['executed_prefix_index']=8
    bad('PREVIEW_TAIL_NOT_EXECUTED_LABEL',lambda:committed_series(changed_samples,records,32))
    return dict(status='PASS_DATA_CONTRACT_FIXTURES_ONLY',cases=rows,actual_bank=False,official_consumer='NOT_RUN',statistics_fit=0,
        new_physics=0,new_render=0,neural_models=0)
