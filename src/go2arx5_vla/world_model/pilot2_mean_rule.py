"""Parent-balanced response rule fit only to real preregistered TRAIN actions."""
import numpy as np

SCHEMA='WM02_PILOT2_PARENT_BALANCED_MEAN_RESPONSE_V1'


def fit(records, parents, task_sha256):
    if not parents or len(set(parents))!=len(parents) or not records:
        raise ValueError('DECLARED_UNIQUE_PHYSICAL_TRAIN_PARENTS_REQUIRED')
    names=set();group={}
    for r in records:
        if (r['split']!='TRAIN' or r['parent'] not in parents or r['run'] in names
                or type(r['action']) is not int or r['action'] not in range(4)
                or type(r['repeat']) is not int or r['repeat'] not in (0,1)):
            raise ValueError('ONLY_UNIQUE_PREREGISTERED_TRAIN_ACTIONS_AND_RESTARTS_REQUIRED')
        value=np.asarray(r['response_xy_m'],float)
        if value.shape!=(2,) or not np.isfinite(value).all():
            raise ValueError('ACTUAL_FINITE_TCP_RESPONSE_IN_METERS_REQUIRED')
        key=(r['parent'],r['action'],r['repeat'])
        if key in group:raise ValueError('DUPLICATE_PARENT_ACTION_RESTART')
        group[key]=value;names.add(r['run'])
    if set(group)!={(p,a,r)for p in parents for a in range(4)for r in range(2)}:
        raise ValueError('COMPLETE_FOUR_ACTION_TWO_RESTART_PANEL_PER_TRAIN_PARENT_REQUIRED')
    values=np.asarray([[[group[(p,a,r)]for r in range(2)]for a in range(4)]for p in parents])
    means=values.mean(2)
    return dict(schema=SCHEMA,fit_split='TRAIN',task_sha256=task_sha256,
        physical_parents=list(parents),episodes=len(records),restarts_per_action=2,
        action_order=['HOLD','PLUS_X_1MM','MINUS_X_1MM','PLUS_Y_1MM'],
        mean_response_xy_m=means.mean(0).tolist(),
        parent_mean_response_xy_m={p:means[i].tolist()for i,p in enumerate(parents)},
        same_action_restart_gap_m=np.linalg.norm(values[:,:,0]-values[:,:,1],axis=-1).tolist(),
        exact_fork_oracle=False,rule_fit_changes_task_or_score=False,
        weighting='EQUAL_PHYSICAL_PARENT_THEN_EQUAL_TWO_REAL_RESTARTS')


def responses(spec,task):
    """The historical task mean remains the default; never mutate its score."""
    from pathlib import Path
    import json
    from .storage import digest
    if not spec.get('train_mean_response_model'):
        return np.asarray(task['mean_response_xy_m'],float),None
    p=Path(spec['train_mean_response_model'])
    if digest(p)!=spec['train_mean_response_model_sha256']:
        raise ValueError('FROZEN_TRAIN_MEAN_RESPONSE_MODEL_CHANGED')
    value=json.loads(p.read_text());mean=np.asarray(value['mean_response_xy_m'],float)
    if (value.get('schema')!=SCHEMA or value.get('fit_split')!='TRAIN'
            or value.get('task_sha256')!=spec['task_sha256'] or value.get('exact_fork_oracle')is not False
            or value.get('rule_fit_changes_task_or_score')is not False or mean.shape!=(4,2)
            or not np.isfinite(mean).all()):
        raise ValueError('TRAIN_ONLY_RESPONSE_MODEL_SAME_ORIGINAL_TASK_REQUIRED')
    return mean,digest(p)
