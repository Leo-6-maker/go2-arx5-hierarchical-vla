"""NumPy-only independent task candidate inference and full-reference preview.

No object pose, teacher, score or future response enters the policy. Native
certificates and live safety remain the executor's responsibility.
"""
from pathlib import Path
import hashlib
import numpy as np
from . import stable_task_adapter_v1 as adapter
from .storage import digest

KEYS={'schema','mean','scale','hidden.weight','hidden.bias','output.weight','output.bias'}
SHAPES={'mean':(116,),'scale':(116,),'hidden.weight':(32,116),'hidden.bias':(32,),
        'output.weight':(3,32),'output.bias':(3,)}


def need(ok,why):
    if not ok:raise ValueError('STABLE_TASK_POLICY_'+why)


def load_export(descriptor):
    need(set(descriptor)=={'path','sha256'},'EXACT_MODEL_DESCRIPTOR')
    path=Path(descriptor['path'])
    need(path.is_absolute()and path.resolve()==path and not path.is_symlink()
         and path.stat().st_size<=128*1024 and digest(path)==descriptor['sha256'],
         'PINNED_CANONICAL_SMALL_EXPORT')
    with np.load(path,allow_pickle=False)as saved:
        need(set(saved.files)==KEYS,'EXACT_EXPORT_MEMBERS')
        arrays={key:saved[key].copy()for key in saved.files}
    need(str(arrays['schema'])==adapter.SCHEMA,'EXACT_TRAINED_CANDIDATE_SCHEMA')
    for key,shape in SHAPES.items():
        need(arrays[key].shape==shape and arrays[key].dtype==np.float32
             and np.isfinite(arrays[key]).all(),'FINITE_TRAINED_ARRAY_'+key)
    need((arrays['scale']>0).all(),'POSITIVE_TRAIN_NORMALIZER')
    return arrays


def predict_from_observation(spec,observation,arrays):
    from .stable_task_dataset_v1 import actual_features
    root=Path(spec['registered_root'])
    x,descriptor=actual_features(root,spec,digest(root/'EXECUTION_SPEC.json'),
        dict(student_observation=observation))
    delta=np.asarray(adapter.numpy_predict(arrays,x),float)
    need(delta.shape==(3,)and np.isfinite(delta).all()
         and np.linalg.norm(delta)<=adapter.STEP_M+1e-10,'UNALTERED_BOUNDED_CANDIDATE_OUTPUT')
    return delta,dict(observation=descriptor,features_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        feature_schema=adapter.SCHEMA,policy='INDEPENDENT_LEARNED_TASK_ADAPTER',
        privileged_policy_inputs=False,teacher_fallback=False,GR00T_weights_changed=False)


def build_for_delta(actual,reference,base,delta):
    from ..trajectory.so3 import rotation_matrix_to_rot6d_rows
    from .fusion_trajectory_adapter import TrajectoryAdaptation
    actual,reference,base,delta=(np.asarray(v,float)for v in (actual,reference,base,delta))
    need(actual.shape==reference.shape==(4,4)and base.shape==(7,)and delta.shape==(3,)
         and all(np.isfinite(v).all()for v in (actual,reference,base,delta))
         and np.linalg.norm(delta)<=adapter.STEP_M+1e-10,'BOUNDED_ACTUAL_STATE_COMMAND')
    u=np.linspace(0.,1.,40);w=u**3*(10+u*(-15+6*u))
    raw=np.empty((40,10));raw[:,:3]=actual[:3,3]+w[:,None]*delta
    raw[:,3:9]=rotation_matrix_to_rot6d_rows(actual[:3,:3]);raw[:,9]=1.
    program=TrajectoryAdaptation(raw,actual,np.eye(4),actual,reference,base,1.,time_stretch_factor=2)
    need(program.end_atom+8<=176,'COMPLETE_REFERENCE_FITS_PREREGISTERED_SUFFIX')
    return program,dict(original_physical_H40=raw,reference_delta_m=delta,
        adaptation=program.evidence(),raw_reference_is_GR00T=False,
        candidate_command_clipped=False,native_execution_authorized=False,
        original_native_certificates_required=True)
