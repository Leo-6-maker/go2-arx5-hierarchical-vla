"""FP64 descriptive decomposition of one complete, fixed eight-noise XYZ bank."""
import numpy as np


def decompose(predicted_xyz, teacher_xyz, valid_rotation):
    predicted = np.asarray(predicted_xyz, dtype=np.float64)
    teacher = np.asarray(teacher_xyz, dtype=np.float64)
    rotation = np.asarray(valid_rotation, dtype=bool)
    assert predicted.shape == (8,40,3) and teacher.shape == (40,3) and rotation.shape == (8,)
    assert np.isfinite(teacher).all()
    valid = np.isfinite(predicted).all(axis=(1,2)) & rotation
    if not valid.all():
        return dict(status='UNAVAILABLE_COMPLETE_BANK_HAS_INVALID_OUTPUT', K=8,
            invalid_draws=np.flatnonzero(~valid).tolist(), reduced_K=False)
    error = predicted-teacher[None,:,:]
    mean = error.mean(axis=0)
    mse = np.sum(error**2,axis=2).mean(axis=0)
    mean_squared = np.sum(mean**2,axis=1)
    dispersion = np.sum((error-mean[None,:,:])**2,axis=2).mean(axis=0)
    residual = float(np.max(abs(mse-mean_squared-dispersion)))
    assert np.allclose(mse,mean_squared+dispersion,rtol=1e-12,atol=1e-15)
    regions = {}
    for name,index in [('first',[0]),('endpoint_0p4',[8]),('prefix',np.arange(9)),('suffix',np.arange(9,40)),('full',np.arange(40))]:
        regions[name] = dict(MSE_m2=float(mse[index].mean()), empirical_mean_error_squared_m2=float(mean_squared[index].mean()),
            within_bank_dispersion_m2=float(dispersion[index].mean()))
    return dict(status='FINITE_FIXED_K8_DECOMPOSITION', K=8, units='m2', regions=regions,
        per_horizon_MSE_m2=mse.tolist(), per_horizon_mean_error_squared_m2=mean_squared.tolist(),
        per_horizon_dispersion_m2=dispersion.tolist(), mean_error_vector_m=mean.tolist(),
        identity_max_abs_m2=residual, population_bias_claim=False, action_averaging_executed=False)


def displacement(current_xyz, teacher_endpoint_xyz, predicted_endpoint_xyz):
    target=np.asarray(teacher_endpoint_xyz,dtype=np.float64)-np.asarray(current_xyz,dtype=np.float64)
    generated=np.asarray(predicted_endpoint_xyz,dtype=np.float64)-np.asarray(current_xyz,dtype=np.float64)
    assert target.shape==generated.shape==(3,) and np.isfinite(target).all() and np.isfinite(generated).all()
    t,p=float(np.linalg.norm(target)),float(np.linalg.norm(generated))
    return dict(teacher_displacement_vector_m=target.tolist(), predicted_displacement_vector_m=generated.tolist(),
        teacher_displacement_m=t,predicted_displacement_m=p,teacher_hold_legacy_1e12=t<=1e-12,
        direction_reporting_guard_m=1e-9, direction_cosine=None if min(t,p)<=1e-9 else float(np.dot(target,generated)/(t*p)),
        signed_teacher_direction_progress_m=None if t<=1e-9 else float(np.dot(target,generated)/t),
        reporting_guard_is_not_a_task_success_rule=True)


def self_check():
    teacher=np.zeros((40,3)); bank=np.zeros((8,40,3))
    bank[:,:,0]=.003+np.arange(8)[:,None]*.001
    bank[:,8,1]=.004
    result=decompose(bank,teacher,np.ones(8,dtype=bool))
    expected=np.mean(np.sum(bank[:,8]**2,axis=1))
    assert abs(result['regions']['endpoint_0p4']['MSE_m2']-expected)<1e-18
    assert result['regions']['endpoint_0p4']['MSE_m2'] != result['regions']['prefix']['MSE_m2']
    constant=np.ones((8,40,3))*.002
    assert decompose(constant,teacher,np.ones(8,dtype=bool))['regions']['full']['within_bank_dispersion_m2']==0
    invalid=bank.copy();invalid[4,17,2]=np.nan
    assert decompose(invalid,teacher,np.ones(8,dtype=bool))['invalid_draws']==[4]
    rotations=np.ones(8,dtype=bool);rotations[6]=False
    assert decompose(bank,teacher,rotations)['invalid_draws']==[6]
    try:decompose(bank[:7],teacher,rotations[:7])
    except AssertionError:pass
    else:raise AssertionError('Never reduce K or replace a missing draw')
    assert displacement([0,0,0],[0,0,0],[1,0,0])['direction_cosine'] is None
    opposite=displacement([0,0,0],[.01,0,0],[-.02,0,0])
    assert opposite['direction_cosine']==-1 and opposite['signed_teacher_direction_progress_m']==-.02
    print('PASS_FP64_K8_IDENTITY_ENDPOINT_PREFIX_INVALID_DENOMINATOR_AND_DIRECTION_GUARD')


if __name__=='__main__':self_check()
