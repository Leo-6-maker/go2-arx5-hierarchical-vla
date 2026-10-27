"""Actual saved NumPy draw files; these are CPU fixtures, not GR00T output."""
import numpy as np
import pytest
from scripts.pi0294.generalization import minimal_inference_export as export


def packet():
    return dict(physical_H40=np.arange(400,dtype=np.float64).reshape(40,10),
        normalized=np.arange(5280,dtype=np.float32).reshape(1,40,132),
        noise_seed=np.asarray(29400,dtype=np.uint64),
        noise_fingerprint_json=np.asarray('{"sha256":"frozen-test-noise"}'),
        observation_fingerprint=np.asarray('same-test-input'),
        checkpoint_files_fingerprint=np.asarray('full-wrapped-files'))


def test_full_export_comparison_allows_distinct_current_checkpoint_identity(tmp_path):
    before=packet();after=packet();after['checkpoint_files_fingerprint']=np.asarray('ordinary-export-files')
    left=tmp_path/'trained.npz';right=tmp_path/'reloaded.npz';np.savez(left,**before);np.savez(right,**after)
    result=export.compare_saved_draws(left,right)
    assert result['status']=='ACTUAL_SAVED_SAME_INPUT_NOISE_OUTPUTS_EXACT'and result['new_queries']==0
    assert result['reference_sha256']!=result['reloaded_sha256']


@pytest.mark.parametrize('kind',['physical','padded-action','noise','input','nonfinite','dtype','shape'])
def test_saved_output_corruption_is_not_reconstruction_equivalence(tmp_path,kind):
    before=packet();after=packet()
    if kind=='physical':after['physical_H40'][39,9]+=1
    elif kind=='padded-action':after['normalized'][0,39,131]+=1
    elif kind=='noise':after['noise_seed']=np.asarray(29401,dtype=np.uint64)
    elif kind=='input':after['observation_fingerprint']=np.asarray('other-input')
    elif kind=='nonfinite':before['physical_H40'][0,0]=after['physical_H40'][0,0]=np.inf
    elif kind=='dtype':after['normalized']=after['normalized'].astype(np.float64)
    else:before['normalized']=after['normalized']=np.zeros((1,40,10),dtype=np.float32)
    left=tmp_path/'trained.npz';right=tmp_path/'reloaded.npz';np.savez(left,**before);np.savez(right,**after)
    with pytest.raises(PermissionError):export.compare_saved_draws(left,right)


def test_one_draw_or_alias_cannot_prove_full_reload(tmp_path):
    left=tmp_path/'trained.npz';np.savez(left,**packet());alias=tmp_path/'alias.npz';alias.symlink_to(left)
    with pytest.raises(PermissionError):export.compare_saved_draws(left,left)
    with pytest.raises(PermissionError):export.compare_saved_draws(left,alias)
