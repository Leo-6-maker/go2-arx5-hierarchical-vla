"""Synthetic protocol corruption tests, not native capability evidence."""
import copy
import hashlib
import json

import numpy as np
import pytest

from .base_intent import eef_document
from .task_path_audit import check_transactions,check_qualification,audit_episode
from go2arx5_vla.task_data.codec import pose10


def fixture():
    samples=[];records=[];planner=[]
    for i in range(641):
        samples.append(dict(time_s=i*.05,raw_time_s=5+i*.05,native_step=2500+i*25,
            active_intent_sequence=None if i==0 else 2+(i-1)//8,
            executed_prefix_index=None if i==0 else(i-1)%8))
    for j in range(80):
        source=round((5+j*.4)*1e9)/1e9
        poses=np.repeat(np.eye(4)[None],40,axis=0);poses[:,0,3]=np.arange(40)*.001+j*.01
        action=np.stack([pose10(p,1.)for p in poses]);document=eef_document(action)
        document['_pi0294']=dict(episode='SYNTHETIC',sequence=j+2,source_mpc_s=source,ttl_s=.4,operation='install')
        text=json.dumps(document);ref=hashlib.sha256(text.encode()).hexdigest()
        intent=dict(index=j*8,sequence=j+2,gripper_sequence=j,wire_sha256=ref,
            action_EEF10=action.tolist(),base_valid=False,prefix_points=8,execute_physics_steps=200,
            record_every_physics_steps=25,actual_observation_time_s=samples[j*8]['raw_time_s'],source_mpc_s=source)
        records.extend([dict(intent,phase='SUBMISSION_INTENT_BEFORE_EFFECT'),dict(intent,phase='SUBMITTED_REQUIRES_INSTALL_ACK'),
            dict(index=j*8,phase='INSTALL_ACK',wire_sha256=ref,native_step=2500+j*200)])
        points=document['ee']['waypoints']
        planner.extend([dict(kind='accept',ref_id=ref,raw=text,source_mpc_s=source,receiver_mpc_s=source),
            dict(kind='installed',ref_id=ref,bound_start=source,observation_time_s=source,
                times=(source+np.arange(41)*.05).tolist(),
                poses=np.asarray([p['position']+p['quaternion_xyzw']for p in points],dtype=np.float32).tolist())])
    planner.extend([dict(kind='accept',ref_id='qualification',raw='not-a-task',source_mpc_s=5.,receiver_mpc_s=5.),
        dict(kind='installed',ref_id='qualification',bound_start=5.,observation_time_s=5.)])
    return samples,records,planner


def test_prefix8_has_640_observed_intervals_and_no_unexecuted_suffix_labels():
    s,r,p=fixture();v=check_transactions(s,r,p,episode='SYNTHETIC')
    assert v['submitted_and_installed']==80 and v['completed_intervals']==640
    assert len(v['intervals'])==640 and {i['prefix_index']for i in v['intervals']}==set(range(8))
    assert v['intervals'][7]['action_EEF10']==r[1]['action_EEF10'][7]
    assert v['intervals'][8]['action_EEF10']==r[4]['action_EEF10'][0]
    assert {x['selected_MPC_policy_consumption']for x in v['intervals']}=={'UNKNOWN'}


@pytest.mark.parametrize('fault',['labels','source','stale','ack','prefix','record_order','pre_intent','duplicate','missing','extra_reject','installed_position','installed_clock'])
def test_incomplete_or_corrupted_transactions_cannot_pass(fault):
    s,r,p=copy.deepcopy(fixture())
    if fault=='labels':r[1]['action_EEF10'][0][0]+=.1
    elif fault=='source':r[1]['source_mpc_s']+=.05
    elif fault=='stale':p[0]['receiver_mpc_s']+=.026
    elif fault=='ack':r[2]['native_step']+=25
    elif fault=='prefix':s[8]['executed_prefix_index']=8
    elif fault=='record_order':r[0],r[1]=r[1],r[0]
    elif fault=='pre_intent':r[0]['wire_sha256']='other'
    elif fault=='duplicate':p.append(p[0])
    elif fault=='missing':r.pop()
    elif fault=='extra_reject':p.append(dict(kind='reject'))
    elif fault=='installed_position':p[1]['poses'][0][0]+=.001
    elif fault=='installed_clock':p[1]['times'][0]+=.05
    with pytest.raises(ValueError):check_transactions(s,r,p,episode='SYNTHETIC')


def test_even_consistent_rehash_cannot_add_a_BASE_label():
    s,r,p=fixture();d=json.loads(p[0]['raw']);d['base']={'frame':'world'}
    text=json.dumps(d);ref=hashlib.sha256(text.encode()).hexdigest()
    for row in r[:3]:row['wire_sha256']=ref
    p[0].update(raw=text,ref_id=ref);p[1]['ref_id']=ref
    with pytest.raises(ValueError,match='NO_BASE'):check_transactions(s,r,p,episode='SYNTHETIC')


def test_native_republication_retains_one_absolute_reference_without_new_action():
    s,r,p=fixture();republished=copy.deepcopy(p[1]);republished['observation_time_s']+=.1
    p.insert(2,republished)
    proof=check_transactions(s,r,p,episode='SYNTHETIC')
    assert proof['submitted_and_installed']==80
    assert proof['native_task_install_publications']==81
    assert proof['completed_intervals']==640


@pytest.mark.parametrize('fault',['values','reanchored','expired','order'])
def test_republication_cannot_change_reference_or_reanchor_its_clock(fault):
    s,r,p=fixture();row=copy.deepcopy(p[1]);row['observation_time_s']+=.1
    if fault=='values':row['poses'][0][0]+=.001
    elif fault=='reanchored':row['bound_start']+=.1
    elif fault=='expired':row['observation_time_s']+=.4
    elif fault=='order':row['observation_time_s']-=.2
    p.insert(2,row)
    with pytest.raises(ValueError):check_transactions(s,r,p,episode='SYNTHETIC')


def test_qualification_uses_actual_new_native_modes_after_cancel(tmp_path):
    d=dict(_pi0294=dict(sequence=0,operation='install',source_mpc_s=5.));text=json.dumps(d);ref=hashlib.sha256(text.encode()).hexdigest()
    q=dict(status='PASS_CURRENT_FEEDBACK_INSTALL_CANCEL_HOLD_AT_EXACT_PAUSE',native_step=2500,
        physics_integrations_during_qualification=0,ROS_sequence_offset=2,first_gripper_sequence=0,hold_ref_id=ref)
    (tmp_path/'CURRENT_NATIVE_QUALIFICATION.json').write_text(json.dumps(q))
    p=[dict(kind='accept',ref_id=ref,raw=text,monotonic_s=10),dict(kind='installed',ref_id=ref,monotonic_s=11),dict(kind='lifecycle',reason='CANCEL',monotonic_s=12)]
    r=[dict(kind='mode',monotonic_s=13,msg=dict(data='HOLD')),dict(kind='mode',monotonic_s=14,msg=dict(data='WBC'))]
    assert check_qualification(tmp_path,p,r)==ref
    r[-1]['monotonic_s']=9
    with pytest.raises(ValueError,match='ACKNOWLEDGEMENTS'):check_qualification(tmp_path,p,r)


def test_engineering_prefix_has_no_task_score_or_demonstration(tmp_path):
    folder=tmp_path/'SYNTHETIC';folder.mkdir()
    (folder/'CARD.json').write_text(json.dumps(dict(task='T2',id=folder.name)))
    result=dict(status='ENGINEERING_OR_NATIVE_SAFETY_INCOMPLETE',source_sha='a'*40,error='serialization')
    (folder/'RESULT.json').write_text(json.dumps(result))
    proof,arrays=audit_episode(folder,expected_source_sha='a'*40)
    assert arrays is None and proof['quality']is None and not proof['demonstration_admitted']
    assert proof['publication_count_from_empty_log']=='UNKNOWN_NOT_INFERRED'
    with pytest.raises(ValueError,match='RUNNING_SOURCE'):audit_episode(folder,expected_source_sha='b'*40)
