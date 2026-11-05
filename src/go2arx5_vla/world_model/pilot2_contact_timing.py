"""First state change on the declared forecast grid, with misses kept visible.

This is not continuous native first-passage time. An event may change and return
between the six observed horizons; a direct point-forecast head cannot locate
that transient. Timing error is conditional on detection, never zero for misses.
"""
import numpy as np
from .pilot2_contact_model import EVENTS


def transition_timing(probability, data, ids):
    probability=np.asarray(probability,float);truth=np.asarray(data['events'])
    valid=np.asarray(data['valid'],bool);current=np.asarray(data['current_events'])>.5
    horizons=np.asarray(data['horizons']);ids=np.asarray(ids,int)
    if (probability.shape!=truth.shape or truth.shape!=(len(valid),len(horizons),4)
            or current.shape!=(len(valid),4) or not np.isfinite(probability).all()
            or np.any((probability<0)|(probability>1)) or np.any(np.diff(horizons)<=0)):
        raise ValueError('FINITE_EVENT_PROBABILITY_AND_ORDERED_ACTUAL_FORECAST_GRID_REQUIRED')
    result=dict(schema='WM02_PILOT2_GRID_EVENT_TRANSITION_TIMING_V1',
        clock='FIRST_DIFFERENT_EVENT_ON_DECLARED_FORECAST_GRID_NOT_NATIVE_FIRST_PASSAGE',
        atomic_dt_s=.05,forecast_horizons_atoms=horizons.tolist(),
        overlapping_anchors_are_not_independent_trials=True,events={})
    for k,name in enumerate(EVENTS):
        rows=[]
        for i in ids:
            available=np.flatnonzero(valid[i])
            if not len(available):continue
            actual=available[(truth[i,available,k]>.5)!=current[i,k]]
            predicted=available[(probability[i,available,k]>=.5)!=current[i,k]]
            actual_time=float(horizons[actual[0]]*.05)if len(actual)else None
            predicted_time=float(horizons[predicted[0]]*.05)if len(predicted)else None
            rows.append(dict(parent=data['identities'][i]['parent'],initial=bool(current[i,k]),
                actual=actual_time,predicted=predicted_time))
        def summarize(selected):
            actual=sum(r['actual']is not None for r in selected)
            predicted=sum(r['predicted']is not None for r in selected)
            both=[r for r in selected if r['actual']is not None and r['predicted']is not None]
            errors=np.asarray([r['predicted']-r['actual']for r in both])
            return dict(valid_anchors=len(selected),actual_grid_transitions=actual,predicted_grid_transitions=predicted,
                detected_actual_transitions=len(both),missed_actual_transitions=actual-len(both),
                false_transition_predictions=predicted-len(both),
                no_observed_grid_transition= len(selected)-actual,
                detection_recall=len(both)/actual if actual else None,
                detection_precision=len(both)/predicted if predicted else None,
                detected_only_timing_MAE_s=float(np.abs(errors).mean())if len(errors)else None,
                detected_only_mean_signed_error_s=float(errors.mean())if len(errors)else None,
                detected_only_timing_p90_abs_error_s=float(np.quantile(np.abs(errors),.9))if len(errors)else None)
        value=summarize(rows);value['by_initial_state']={
            label:summarize([r for r in rows if r['initial']==initial])
            for label,initial in [('INITIALLY_FALSE_ENTRY',False),('INITIALLY_TRUE_EXIT',True)]}
        value['by_parent']={p:summarize([r for r in rows if r['parent']==p])for p in sorted({r['parent']for r in rows})}
        recalls=[v['detection_recall']for v in value['by_parent'].values()if v['detection_recall']is not None]
        value['parent_equal_detection_recall']=float(np.mean(recalls))if recalls else None
        result['events'][name]=value
    return result
