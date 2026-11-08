"""Continuous label arrays and fixed exposure after native-proof verification.

These pure transformations grant no BC eligibility. A separate sealed exporter
must verify every real capture, consumed command, native exit and population.
"""
import numpy as np
from . import task_learning_grasp_v1 as task
from .task_learning_sequence_v1 import map_policy_frame


def packet_arrays(checked,current_policy_states10,frame):
    """States already use the original shadow policy frame; map labels once."""
    states=np.asarray(current_policy_states10,float)
    task.need(states.shape==(15,10)and np.isfinite(states).all()
        and np.all((states[:,9]>=0)&(states[:,9]<=1)), 'FIFTEEN_CURRENT_ACTUAL_POLICY_STATES')
    task.need(np.array_equal(checked['native_command_anchor'],np.arange(0,600,40))
        and checked['policy_absolute10'].shape==checked['installed_audit_only'].shape==(600,10)
        and checked['valid_window'].shape==checked['stages'].shape==(15,),
        'FIFTEEN_REAL_ANCHORS_IN_ONE_EPISODE_NOT_DENSE_FAKE_QUERIES')
    mapped=map_policy_frame(checked,frame)
    return dict(actual_state10=states.copy(),committed_absolute10=mapped['policy_absolute10'],
        installed_consumed_50ms_audit_only=mapped['installed_audit_only'],
        native_command_anchor=checked['native_command_anchor'].copy(),valid_window=checked['valid_window'].copy(),
        expert_stage_audit_only=checked['stages'].copy(),instruction=np.asarray(task.LANGUAGE),
        label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'),
        command_period_s=np.asarray(.05),native_dt_s=np.asarray(.002))


def exposure_schedule(packet):
    """32 visits per episode; 8 boundary visits, 24 other-stage visits."""
    valid=np.asarray(packet['valid_window']);stages=np.asarray(packet['expert_stage_audit_only'])
    task.need(valid.shape==stages.shape==(15,)and valid.dtype==np.bool_,'EXPLICIT_REAL_DECISION_MASK')
    eligible=np.flatnonzero(valid).tolist();critical=[]
    for stage in('CLOSE','LIFT','HOLD'):
        found=[i for i in eligible if stages[i]==stage]
        if found:critical.append(found[0])
    other=[i for i in eligible if i not in critical]
    task.need(bool(critical)and bool(other),'REAL_BOUNDARY_AND_OTHER_PHASE_COVERAGE')
    return [critical[i%len(critical)]for i in range(8)]+[other[i%len(other)]for i in range(24)]


def fit_statistics(episodes):
    from scripts.pi0292.train_adapter import TAG
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10,pose10
    from scripts.pi0294.multitask.source_export import statistics
    states=[];actions=[];relative=[]
    for episode in episodes:
        row=episode['row']
        if row['split']!='TRAIN':continue
        task.need(row['purpose']=='CONTINUOUS_TEACHER_DEMONSTRATION',
            'CONSTRUCTION_QUALIFICATION_OR_OLD_SHORT_DATA_NEVER_NEW_TRAIN_STATISTICS')
        p=episode['packet']
        for i in np.flatnonzero(p['valid_window']):
            atom=int(p['native_command_anchor'][i]);s=p['actual_state10'][i:i+1];a=p['committed_absolute10'][None,atom:atom+40]
            task.need(a.shape==(1,40,10),'NO_PARTIAL_TAIL_OR_CROSS_EPISODE_H40')
            states.append(s);actions.append(a)
            relative.append(pose10(np.linalg.inv(matrix_from10(s))[:,None]@matrix_from10(a),a[...,9])[...,:9])
    task.need(bool(states),'NONEMPTY_UNIQUE_TRUE_TRAIN_DECISIONS_BEFORE_OVERSAMPLING')
    s,a,r=np.concatenate(states),np.concatenate(actions),np.concatenate(relative)
    return {TAG:dict(state=dict(eef_9d=statistics(s[:,:9],0),gripper_position=statistics(s[:,9:],0)),
        action=dict(eef_9d=statistics(a[...,:9],(0,1)),gripper_position=statistics(a[...,9:],(0,1))),
        relative_action=dict(eef_9d=statistics(r,(0,1))))}
