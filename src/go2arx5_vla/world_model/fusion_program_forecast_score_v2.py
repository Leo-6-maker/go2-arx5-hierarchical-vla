"""Versioned scoring that retains actual install-clock exposure mismatches.

The caller authenticates original CLOSED specs, artifacts and forecast bytes.
This pure join neither generates predictions nor grants native execution.
"""
import hashlib
import numpy as np
from . import fusion_live_program_replay_v1 as replay
from .fusion_program_forecast_v1 import causal_record, RIDGE_ARMS
from .fusion_program_response_data_v1 import ARMS, metrics_by_phase, metrics

BASELINES=('ZERO_DISPLACEMENT','PAST_CONSTANT_VELOCITY','TRAIN_MEAN_RESPONSE')


def _need(ok,why):
    if not ok:raise ValueError('SAVED_PROGRAM_FORECAST_'+why)


def join(spec,prefix,decision,segment,saved):
    """Accept authenticated original documents; never recompute model outputs.

    ``saved`` maps publication atoms to the exact original saved JSON documents.
    Forecasts for rejected installs with no subsequent effect remain unscored.
    Every observed label still checks the complete published preview and clock.
    """
    config=spec['program_response_forecast']
    _need(config['schema']=='WM02_PROGRAM_RESPONSE_SHADOW_FORECAST_V2'
          and config['mode']=='SHADOW_NO_REFERENCE_CHANGE'
          and replay.assignment(spec)['split']=='DEV','FROZEN_V2_DEV_ONLY')
    seeds=config['seeds']
    labels=[str(s)+'-'+a for s in seeds for a in ARMS]+list(RIDGE_ARMS)
    expected=set(labels+list(BASELINES))
    descriptors=segment['shadow_forecast_records']
    atoms=[d['program_atom']for d in descriptors]
    _need(len(atoms)==len(set(atoms)) and set(atoms)==set(saved),'EVERY_SAVED_ATTEMPT_RETAINED_ONCE')
    _need(atoms==sorted(atoms) and all(type(a)is int and a>=0 and a%7==0 for a in atoms),
          'ORIGINAL_PUBLICATION_GRID')
    rows=[];predictions={name:[]for name in labels+list(BASELINES)};unobserved=[];identities=None;exposure_mismatches=[]
    observed=segment.get('windows',[])
    for atom,descriptor in zip(atoms,descriptors):
        doc=saved[atom]
        _need(doc.get('schema')=='WM02_LIVE_PROGRAM_PRE_EFFECT_FORECAST_V1'
              and doc.get('program_atom')==atom and doc.get('native_step')==replay.B+25*atom
              and doc.get('native_effects_since_anchor')==0 and doc.get('reference_changes')==0
              and doc.get('future_observations_read')is False
              and doc.get('forecast_is_native_qualification')is False
              and 'labels'not in doc,'ORIGINAL_PRE_EFFECT_DOCUMENT_NOT_POSTHOC_PREDICTION')
        _need(set(doc['predictions'])==expected,'ALL_SEEDS_ARMS_RIDGES_AND_STRONG_BASELINES')
        ids=doc['models']
        _need([i['label']for i in ids]==labels,'REGISTERED_MODEL_ORDER')
        for actual,registered in zip(ids,config['models']):
            _need(actual['arm']==registered['arm'] and actual['numpy_sha256']==registered['sha256']
                  and (actual['arm']in RIDGE_ARMS or actual['seed']==registered['seed']),
                  'SAME_FROZEN_MODEL_IDENTITIES')
        if identities is None:identities=ids
        _need(ids==identities,'NO_MODEL_CHANGE_DURING_PHYSICAL_RUN')
        arrays={k:np.asarray(v,float)for k,v in doc['predictions'].items()}
        _need(all(v.shape==(16,8)and np.isfinite(v).all()for v in arrays.values()),'FINITE_H16_OUTPUTS')
        if atom>=len(observed):
            unobserved.append(atom)
            continue
        row=replay.extract(spec,prefix,decision,segment,install_atom=atom)
        x,names=replay.flatten_features(row)
        history=decision['actual_history']if atom==0 else observed[atom]['actual_history_before_atom']
        past=(prefix['windows']+observed[:atom])[-8:]
        prior=causal_record(history,past,decision['nominal_reference_path'],install_atom=atom)
        forecast_x=np.asarray(prior['X'],float)
        pin=hashlib.sha256(forecast_x.astype('<f8').tobytes()).hexdigest()
        _need(np.array_equal(doc['X'],forecast_x) and doc['feature_names']==list(names)
              and doc['feature_sha256']==descriptor['feature_sha256']==pin
              and np.array_equal(doc['planned_H40'],prior['planned_H40'])
              and doc['prospective_first_install_source_age_s']==0.,'EXACT_CAUSAL_FEATURE_AND_FULL_PREVIEW_JOIN')
        if not np.array_equal(x,forecast_x):
            from .fusion_program_response_data_v1 import FEATURE_SLICES
            from scripts.pi0294.task_clock import actual_source_age
            age=float(row['features']['prospective_install_clock'][0])
            clock=observed[atom]['clock']
            actual_age=actual_source_age(clock['native_start_s'],clock['source_mpc_s'])
            expected_actual=forecast_x.copy()
            expected_actual[FEATURE_SLICES['planned_install_source_ages_s'].start]=age
            sl=FEATURE_SLICES['prospective_install_clock']
            expected_actual[sl]=[age,.4-age]
            _need(age>0 and age==actual_age and np.array_equal(x,expected_actual),
                  'NONCLOCK_FEATURE_CHANGE_CANNOT_BECOME_EXPOSURE_EXCLUSION')
            exposure_mismatches.append(dict(program_atom=atom,
                reason='ACTUAL_INSTALL_CLOCK_DIFFERS_FROM_SAVED_PROSPECTIVE_CLOCK',
                saved_prospective_age_s=0.,actual_install_source_age_s=age,
                original_clock=clock,forecast_feature_sha256=pin,
                actual_feature_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),
                predictions_recomputed=False,actual_clock_or_labels_rewritten=False))
            continue
        phase=('POST_QUERY_HOLD_CONTROL'if decision['adaptation'].get('post_query_HOLD_control')is True
               else'PROGRAM'if atom<=decision['adaptation']['clock']['end_atom']else'POST_PROGRAM_HOLD')
        rows.append(dict(X=x,Y=row['labels']['TCP_SE3_and_finger_displacement'],
            valid=row['labels']['observed_horizon_mask'],program_atom=atom,record_phase=phase))
        for name in predictions:predictions[name].append(arrays[name])
    _need(bool(rows),'NO_OBSERVED_FORECAST_ANCHORS')
    data={k:np.stack([r[k]for r in rows])for k in rows[0]}
    assignment=replay.assignment(spec)
    data.update(split=np.full(len(rows),'DEV'),parent=np.full(len(rows),assignment['parent']),
                run_id=np.full(len(rows),spec['registered_root']))
    predictions={k:np.stack(v)for k,v in predictions.items()}
    return data,predictions,dict(saved_forecasts=len(atoms),observed_forecast_anchors=len(rows),
        unobserved_install_forecast_atoms=unobserved,models=identities,
        scoring_recipe='WM02_SAVED_FORECAST_EXPOSURE_MATCH_V2',
        actual_exposure_mismatches=exposure_mismatches,
        excluded_actual_exposure_forecast_atoms=[v['program_atom']for v in exposure_mismatches],
        actual_clock_labels_or_predictions_rewritten=False,
        actual_model_evaluations_by_scorer=0,checkpoint_updates_by_scorer=0,
        hidden_controller_exact_fork=False,correlated_anchors_are_independent=False)


def score(data,predictions):
    """Report query-join, moving and HOLD subsets without selecting a winner."""
    ids=np.arange(len(data['X']))
    first=ids[data['program_atom']==0]
    return {name:dict(by_phase=metrics_by_phase(pred,data,ids),
                      FIRST_INSTALL_H1_H16=metrics(pred,data,first))
            for name,pred in predictions.items()}
