"""Portable two-action exit pilot. No future outcome or torch in native inputs."""
import numpy as np

OFFSETS=np.asarray([[0.,0.,0.],[0.,.001,0.]])


def features(rows):
    if len(rows)<9 or rows[-1]['native_end_step']!=4103:
        raise ValueError('ACTUAL_4103_CONTACT_DECISION_HISTORY_REQUIRED')
    values=[]
    for row in rows[-9:]:
        relative=np.asarray(row['TCP_m'])-np.asarray(row['object_pose'])[:3,3]
        force=max(e['max_pad_cube_normal_force_N']for e in row['native_contact_events'])
        values.extend(np.r_[relative,row['finger_q'],force])
    result=np.asarray(values,dtype=np.float32)
    if result.shape!=(54,)or not np.isfinite(result).all():raise ValueError('FINITE_CAUSAL_CONTACT_SENSOR_INPUT_REQUIRED')
    return result


def logits(arrays,x):
    value=(np.asarray(x,dtype=np.float32)-arrays['mean'])/arrays['scale']
    for layer in (0,2,4):
        value=value@arrays[f'w{layer}'].T+arrays[f'b{layer}']
        if layer<4:value=np.tanh(value)
    if value.shape[-1]!=2 or not np.isfinite(value).all():raise ValueError('ONLY_TWO_FINITE_QUALIFIED_ACTION_LOGITS')
    return value


def choose(adapter,episode,rows):
    from pathlib import Path
    from .storage import digest
    from scripts.pi0292.durable import write
    s=adapter.spec;step=adapter.observe()['step']
    if step!=4103:raise ValueError('PAUSED_CONTACT_CHOICE_AT_4103_REQUIRED')
    with np.load(s['contact_policy_numpy'],allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
    x=features(rows);scores=logits(arrays,x)
    action=int(scores.argmax())if episode['method']=='TRAINED_CONTACT_POLICY'else 1
    result=dict(schema='WM02_PILOT1_CONTACT_POLICY_DECISION_V1',native_step=step,method=episode['method'],
                selected_action=action,selected_offset_xyz_m=OFFSETS[action].tolist(),logits=scores.tolist(),
                actual_prefix_features=x.tolist(),future_response_observed=False,
                policy_sha256=digest(s['contact_policy_numpy']),task='ORIGINAL_TOUCH_WITHDRAW_EXIT_ONLY',
                teacher_is_empirical_TRAIN_return=True,strong_rule_may_equal_policy=True)
    write(adapter.output/'CONTACT_POLICY_DECISION.private.json',result)
    if adapter.observe()['step']!=step:raise RuntimeError('PHYSICS_ADVANCED_DURING_CONTACT_CHOICE')
    return OFFSETS[action].copy()
