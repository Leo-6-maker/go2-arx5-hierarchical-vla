"""Saved predictions must precede effects and match the whole actual exposure."""
import copy
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_program_forecast_score_v1 as scorer
from go2arx5_vla.world_model.fusion_program_forecast_v1 import causal_record,RIDGE_ARMS
from go2arx5_vla.world_model.fusion_program_response_data_v1 import ARMS
from test_fusion_live_program_replay_v1 import artifacts


def fixture():
    spec,prefix,decision,segment=artifacts()
    spec['registered_root']='SYNTHETIC_NATIVE_RUN'
    spec['response_learning_assignment']=dict(schema='WM02_PROGRAM_RESPONSE_ASSIGNMENT_V1',
        split='DEV',parent='SYNTHETIC_DEV_PARENT',panel_sha256='c'*64)
    models=[dict(arm=a,seed=1,sha256='a'*64)for a in ARMS]+[
        dict(arm=a,sha256='a'*64)for a in RIDGE_ARMS]
    spec['program_response_forecast']=dict(schema='WM02_PROGRAM_RESPONSE_SHADOW_FORECAST_V2',
        mode='SHADOW_NO_REFERENCE_CHANGE',seeds=[1],models=models)
    labels=['1-'+a for a in ARMS]+list(RIDGE_ARMS)
    identities=[dict(label=l,arm=m['arm'],seed=m.get('seed'),numpy_sha256=m['sha256'])for l,m in zip(labels,models)]
    saved={};descriptors=[]
    for atom in (0,7,14):
        h=decision['actual_history']if atom==0 else segment['windows'][atom]['actual_history_before_atom']
        past=(prefix['windows']+segment['windows'][:atom])[-8:]
        r=causal_record(h,past,decision['nominal_reference_path'],install_atom=atom)
        saved[atom]=dict(schema='WM02_LIVE_PROGRAM_PRE_EFFECT_FORECAST_V1',**r,
            native_effects_since_anchor=0,reference_changes=0,forecast_is_native_qualification=False,
            models=identities,predictions={l:np.zeros((16,8))for l in labels+list(scorer.BASELINES)})
        descriptors.append(dict(program_atom=atom,feature_sha256=r['feature_sha256']))
    segment['shadow_forecast_records']=descriptors
    return spec,prefix,decision,segment,saved


def test_actual_join_separates_first_install_and_preserves_partial_labels():
    inputs=fixture();data,pred,meta=scorer.join(*inputs)
    result=scorer.score(data,pred)
    assert meta['observed_forecast_anchors']==3 and meta['actual_model_evaluations_by_scorer']==0
    assert data['valid'][-1].sum()==2
    assert result['1-RAW']['FIRST_INSTALL_H1_H16']['16']['anchors']==1
    assert result['1-RAW']['by_phase']['ALL']['16']['physical_runs']==1


@pytest.mark.parametrize('fault',['future_labels','feature','preview','seed_missing','identity','effect','missing_record'])
def test_posthoc_or_changed_forecast_cannot_be_scored(fault):
    s,p,d,g,docs=fixture()
    if fault=='future_labels':docs[0]['labels']=[]
    elif fault=='feature':docs[0]['X']=np.asarray(docs[0]['X']).copy();docs[0]['X'][0]+=1
    elif fault=='preview':docs[0]['planned_H40']=np.asarray(docs[0]['planned_H40']).copy();docs[0]['planned_H40'][0,39,0]+=.001
    elif fault=='seed_missing':del docs[0]['predictions']['1-SCALED']
    elif fault=='identity':docs[0]['models']=copy.deepcopy(docs[0]['models']);docs[0]['models'][0]['numpy_sha256']='b'*64
    elif fault=='effect':docs[0]['native_effects_since_anchor']=1
    elif fault=='missing_record':del docs[14]
    with pytest.raises(ValueError):scorer.join(s,p,d,g,docs)


def test_later_changed_consumed_preview_masks_truth_without_fabricating_prediction():
    s,p,d,g,docs=fixture()
    g['windows'][3]['full_final_installed_H40']=copy.deepcopy(g['windows'][3]['full_final_installed_H40'])
    g['windows'][3]['full_final_installed_H40'][39][0]+=.001
    data,_,_=scorer.join(s,p,d,g,docs)
    assert data['valid'][0].sum()==3


def test_forecast_for_unexecuted_install_remains_an_unscored_attempt():
    s,p,d,g,docs=fixture()
    last=copy.deepcopy(docs[14]);last.update(program_atom=21,native_step=34303+25*21)
    docs[21]=last
    g['shadow_forecast_records'].append(dict(program_atom=21,feature_sha256=last['feature_sha256']))
    data,_,meta=scorer.join(s,p,d,g,docs)
    assert meta['saved_forecasts']==4 and meta['unobserved_install_forecast_atoms']==[21]
    assert len(data['X'])==3
