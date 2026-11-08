"""Small preregistered image-information probe, not a GR00T policy score."""
import numpy as np
from .task_learning_grasp_v1 import need


def contract():
    return dict(schema='TASK_LEARNING_CURRENT_RGB_INFORMATION_PROBE_V1',
        population='ALL_RESET_READY_FIRST_DECISIONS_FROM_TWENTY_REQUESTS; LATER_TASK_FAILURES_INCLUDED',
        minimum_observed_requests=18,minimum_distinct_layouts=9,
        anchor='FIRST_REAL_DECISION_AT_NATIVE_STEP_5703',
        features='LARGEST_4_CONNECTED_RED_COMPONENT_XY_CENTROID_IN_EACH_UNCHANGED_RGB',
        red_threshold=dict(min_R=120,min_R_to_G_ratio=1.5,min_R_to_B_ratio=1.5),
        minimum_component_pixels_per_camera=32,
        labels='ACTUAL_TEACHER_FIRST_CHUNK_XY_DISPLACEMENT_ONLY',
        learner='RIDGE_1E_MINUS_3_WITH_TRAIN_FOLD_ONLY_STANDARDIZATION',
        split='LEAVE_ONE_WHOLE_LAYOUT_OUT; BOTH_RESTARTS_HELD_OUT_TOGETHER',
        baseline='TRAIN_FOLD_MEAN_DISPLACEMENT',
        information_probe_gate=dict(maximum_XY_RMSE_m=.001,maximum_fraction_of_baseline_RMSE=.5),
        current_RGB_only=True,privileged_scene_features=False,GR00T_updates=0,
        interpretation='Image information for local XY disambiguation only; not policy competence, recovery observability, broad generalization or task success.')


def red_centroid(rgb):
    rgb=np.asarray(rgb)
    need(rgb.shape==(256,256,3)and rgb.dtype==np.uint8,'UNCHANGED_CURRENT_RGB256_UINT8')
    r,g,b=(rgb[:,:,i].astype(float)for i in range(3))
    mask=(r>=120)&(r>=1.5*g)&(r>=1.5*b)
    visited=np.zeros((256,256),bool);largest=[]
    for y,x in zip(*np.nonzero(mask)):
        if visited[y,x]:continue
        visited[y,x]=True;component=[(int(y),int(x))];cursor=0
        while cursor<len(component):
            yy,xx=component[cursor];cursor+=1
            for ny,nx in((yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)):
                if 0<=ny<256 and 0<=nx<256 and mask[ny,nx]and not visited[ny,nx]:
                    visited[ny,nx]=True;component.append((ny,nx))
        if len(component)>len(largest):largest=component
    if len(largest)<32:return dict(valid=False,pixels=len(largest),centroid_xy=None)
    xy=np.asarray(largest,float).mean(axis=0)[::-1]/255.
    return dict(valid=True,pixels=len(largest),centroid_xy=xy.tolist())


def leave_layout_out(features,displacements,layouts):
    x,y=np.asarray(features,float),np.asarray(displacements,float);groups=np.asarray(layouts)
    n=len(groups)
    need(18<=n<=20 and x.shape==(n,4)and y.shape==(n,2)and groups.shape==(n,)
        and np.isfinite(x).all()and np.isfinite(y).all()
        and 9<=len(set(groups))<=10 and all(1<=np.sum(groups==v)<=2 for v in set(groups)),
        'ALL_RESET_READY_REQUESTS_GROUPED_BY_LAYOUT_WITH_REAL_VALID_CAMERA_FEATURES')
    predictions=np.empty_like(y);baseline=np.empty_like(y);per_layout=[]
    for group in sorted(set(groups)):
        train=groups!=group;held=~train
        center=x[train].mean(axis=0);scale=x[train].std(axis=0)
        scale=np.where(scale>1e-12,scale,1.)
        a=(x[train]-center)/scale;b=(x[held]-center)/scale;mean=y[train].mean(axis=0)
        weight=np.linalg.solve(a.T@a+1e-3*np.eye(4),a.T@(y[train]-mean))
        predictions[held]=b@weight+mean;baseline[held]=mean
        per_layout.append(dict(layout=str(group),restart_count=int(held.sum()),
            XY_RMSE_m=float(np.sqrt(np.mean(np.sum((predictions[held]-y[held])**2,axis=1)))),
            baseline_XY_RMSE_m=float(np.sqrt(np.mean(np.sum((baseline[held]-y[held])**2,axis=1))))))
    error=float(np.sqrt(np.mean(np.sum((predictions-y)**2,axis=1))))
    constant=float(np.sqrt(np.mean(np.sum((baseline-y)**2,axis=1))))
    return dict(contract=contract(),layouts=len(set(groups)),requested=20,observed_requests=n,XY_RMSE_m=error,
        baseline_XY_RMSE_m=constant,per_layout=per_layout,
        information_probe_passes=error<=.001 and constant>0 and error<=.5*constant,
        GR00T_visual_conditioning_proven=False,recovery_observability_proven=False,
        policy_task_success_proven=False,new_native_integrations=0)
