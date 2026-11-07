"""Clock mismatch is an unscored exposure, never repaired retrospective input."""
import copy
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_program_forecast_score_v1 as old
from go2arx5_vla.world_model import fusion_program_forecast_score_v2 as new
from test_fusion_program_forecast_score_v1 import fixture


def test_exact_exposure_scores_are_identical_to_original():
    values=fixture();a,pa,ma=old.join(*values);b,pb,mb=new.join(*values)
    assert old.score(a,pa)==new.score(b,pb)
    assert mb['actual_exposure_mismatches']==[]
    assert ma['saved_forecasts']==mb['saved_forecasts']==3


def test_real_six_ms_late_source_clock_is_retained_not_scored_or_repredicted():
    s,p,d,g,docs=fixture();before=copy.deepcopy(docs)
    g['windows'][7]['clock']['source_mpc_s']-=.006
    # Preserve the actual older source across the corresponding publication.
    for index in range(8,14):g['windows'][index]['clock']['source_mpc_s']-=.006
    # Later histories record the older actual reference ages too. Regenerate
    # only the fixture's later saved causal input before its synthetic effect.
    from go2arx5_vla.world_model.fusion_program_forecast_v1 import causal_record
    r=causal_record(g['windows'][14]['actual_history_before_atom'],g['windows'][6:14],d['nominal_reference_path'],install_atom=14)
    docs[14].update(r);g['shadow_forecast_records'][2]['feature_sha256']=r['feature_sha256']
    with pytest.raises(ValueError):old.join(s,p,d,g,docs)
    data,pred,meta=new.join(s,p,d,g,docs)
    assert meta['excluded_actual_exposure_forecast_atoms']==[7]
    assert data['program_atom'].tolist()==[0,14]
    assert data['valid'][0].sum()==7
    assert meta['actual_exposure_mismatches'][0]['actual_install_source_age_s']==pytest.approx(.006)
    np.testing.assert_array_equal(docs[7]['X'],before[7]['X'])
    assert meta['actual_clock_labels_or_predictions_rewritten']is False
    assert meta['actual_model_evaluations_by_scorer']==0


@pytest.mark.parametrize('fault',['input','preview','saved_clock','model'])
def test_exposure_exclusion_cannot_hide_changed_prediction_inputs(fault):
    s,p,d,g,docs=fixture();g['windows'][7]['clock']['source_mpc_s']-=.006
    if fault=='input':docs[7]['X']=np.asarray(docs[7]['X']).copy();docs[7]['X'][0]+=.001
    elif fault=='preview':docs[7]['planned_H40']=np.asarray(docs[7]['planned_H40']).copy();docs[7]['planned_H40'][0,0,0]+=.001
    elif fault=='saved_clock':docs[7]['prospective_first_install_source_age_s']=.006
    elif fault=='model':docs[7]['models']=copy.deepcopy(docs[7]['models']);docs[7]['models'][0]['numpy_sha256']='f'*64
    with pytest.raises(ValueError):new.join(s,p,d,g,docs)
