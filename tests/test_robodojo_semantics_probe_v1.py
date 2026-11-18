import numpy as np
import json
import pytest
from scripts.sla.robodojo_semantics_probe_v1 import temporal_alignment, necessary_width_error_bounds


def test_temporal_shift_reports_values_not_command_provenance():
    state=np.arange(42,dtype=float).reshape(14,3)
    action=np.vstack([state[1:],state[-1:]])
    rows=temporal_alignment(action,state)
    assert [r['state_index_minus_action_index']for r in rows if r['all_rows_equal']]==[1]
    assert next(r for r in rows if r['state_index_minus_action_index']==1)['compared_rows']==13


def test_error_bound_distinguishes_fixed_context_from_two_free_outputs():
    step=np.r_[np.full(20,.042),np.full(20,.002)]
    assert necessary_width_error_bounds(step,.042)['speed_max_error_lower_bound_m'] >= .019-1e-12
    reversal=np.r_[np.linspace(.002,.042,20),np.linspace(.042,.002,20)]
    assert necessary_width_error_bounds(reversal,.002)['monotonic_max_error_lower_bound_m'] >= .02-1e-12
    assert necessary_width_error_bounds(np.full(40,.02),.02)['combined_max_error_lower_bound_m']==0
    # A permissible linear monotonic path has a zero necessary lower bound.
    assert necessary_width_error_bounds(.02+.0001*np.arange(1,41),.02)['combined_max_error_lower_bound_m']==0
    json.dumps(necessary_width_error_bounds(np.full(40,.02),np.float64(.02)),allow_nan=False)


def test_numeric_probe_is_fully_json_serializable_with_real_hdf5_types(tmp_path):
    h5py=pytest.importorskip('h5py')
    from scripts.sla.robodojo_semantics_probe_v1 import numeric_probe
    path=tmp_path/'numeric.hdf5'
    with h5py.File(path,'w')as f:
        for g in ('action','state'):
            for arm in ('left','right'):
                pose=np.zeros((100,7));pose[:,3]=1
                f[g+'/'+arm+'_ee_poses']=pose
                f[g+'/'+arm+'_ee_joint_states']=np.linspace(.2,.8,100).reshape(-1,1)
                f[g+'/'+arm+'_arm_joint_states']=np.zeros((100,6))
    result=numeric_probe(path,30.)
    assert len(result['windows'])==8
    assert result['context_semantics']=='SOURCE_NOMINAL_GRID_ONLY_NOT_ACTUAL_EMITTED_REFERENCE'
    json.dumps(result,allow_nan=False)
