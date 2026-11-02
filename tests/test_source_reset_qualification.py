"""Affected camera/noise contracts. These CPU checks are not native admission."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pi0294.multitask import source_reset_qualification as q
from scripts.pi0294.multitask.book_policy import initial_camera_qualification
from scripts.pi0294.multitask.shared_evaluation import SharedStudent


def cameras():
    state=dict(safety_valid=True,native_grasp=False,object_points=np.tile([0.,0.,1.],(8,1)))
    images={n:np.zeros((256,256,3),dtype=np.uint8)for n in('body_rgb','wrist_rgb')}
    cal={n:dict(object_points_uv=np.tile([128.,128.],(8,1)),T_world_optical=np.eye(4))for n in images}
    return state,images,cal


def test_original_initial_gate_and_exact_pair_without_rendering_or_steps():
    state,images,cal=cameras()
    assert initial_camera_qualification(state,images,cal)['original_dual_camera_qualified']
    assert q.LIMITS==dict(source_constructors=12,RGB_pairs=8,physics_integrations=0,
                          model_loads=0,generation_queries=0,optimizer_updates=0)
    for name in images:
        other=deepcopy(cal);other[name]['object_points_uv'][:5]=[-1,-1]
        with pytest.raises(PermissionError,match='CAMERA_VIEW'):initial_camera_qualification(state,images,other)
    with pytest.raises(PermissionError,match='DUAL_CAMERA'):initial_camera_qualification(state,{}, {})
    with pytest.raises(PermissionError,match='DUAL_CAMERA'):
        initial_camera_qualification(state,dict(images,third_rgb=images['body_rgb']),cal)
    for fault in('unsafe','grasp','dtype','shape'):
        s,a,b=cameras()
        if fault=='unsafe':s['safety_valid']=False
        elif fault=='grasp':s['native_grasp']=True
        elif fault=='dtype':a['body_rgb']=a['body_rgb'].astype(np.float32)
        else:a['wrist_rgb']=a['wrist_rgb'][:128]
        with pytest.raises(PermissionError):initial_camera_qualification(s,a,b)


@pytest.mark.parametrize('task',q.TASKS)
def test_every_noise_query_uses_original_condition_keys(task):
    c=dict(task=task,id='BOOK_DEV000'if task=='BOOK'else'FIXED',scene_group='F0/000',
           noise_namespace='PI0294/PV1/DEV/N294101')
    s=SharedStudent(None,c,None,None)
    assert q.verify_noise(c,s)['all80_original_keys_match']
    s.seed_namespace='CHANGED'
    with pytest.raises(PermissionError,match='CONDITION_NOISE'):q.verify_noise(c,s)
    bad=SimpleNamespace(seed_namespace=SharedStudent(None,c,None,None).seed_namespace,
                        generation_seed=lambda n:0)
    with pytest.raises(PermissionError,match='CONDITION_NOISE'):q.verify_noise(c,bad)


def test_qualification_cannot_overwrite_prior_output(tmp_path):
    p=tmp_path/'native.json';q.put(p,dict(status='retained'))
    with pytest.raises(PermissionError,match='OUTPUT_EXISTS'):q.put(p,dict(status='replacement'))
    assert q.read(p)==dict(status='retained')
