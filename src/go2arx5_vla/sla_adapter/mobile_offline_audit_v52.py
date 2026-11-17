"""CPU-only summaries of sealed predictions; no inference or native admission."""
import numpy as np

CLOSE_EXPECTED={'TRAIN96':20,'CN38':36}


def distribution(values):
    values=np.asarray(values,dtype=float).reshape(-1)
    if not len(values):return None
    if not np.isfinite(values).all():raise ValueError('FINITE_OFFLINE_METRIC_REQUIRED')
    return dict(n=len(values),minimum=float(values.min()),median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(values.max()))


def bank_summary(rows):
    """Rows contain original evaluator evidence and independently aligned truth.

    For ROLL96, EEF deviation is only to saved B0 predictions, never expert
    truth or a behavioral-cloning label. None of these predictions is executed.
    """
    endpoints=[r['endpoint']for r in rows]
    error=[r['EEF_errors_m']for r in rows if r.get('EEF_errors_m')is not None]
    close=[r for r in rows if r.get('phase')=='CLOSE']
    events=[r for r in close if r['width']['expert_crossing_in_window']is not None]
    missing=[r['id']for r in close if r['width']['missed_in_window_crossing']]
    late=[r['id']for r in events if r['width']['crossing_error_s']is None or r['width']['crossing_error_s']>.10000000001]
    return dict(n=len(rows),full_H40_pass=sum(r['envelope']['original_full_H40_pass']for r in rows),
        failed_ids=[r['id']for r in rows if not r['envelope']['original_full_H40_pass']],
        start_acceleration_m_s2=distribution([r['start']for r in endpoints]),
        end_acceleration_m_s2=distribution([r['end']for r in endpoints]),
        interior_acceleration_m_s2=distribution([r['interior']for r in endpoints]),
        start_margin_to_original_025=distribution([.25-r['start']for r in endpoints]),
        end_margin_to_original_025=distribution([.25-r['end']for r in endpoints]),
        previous_command_seam_abs_speed_m_s=distribution([abs(r['previous_command_seam_speed_m_s'])for r in rows]),
        nominal_H40_progress_m=distribution([r['envelope']['paths']['nominal']['endpoint_progress_m']for r in rows]),
        EEF_point_error_m=distribution(error),CLOSE_rows=len(close),annotated_CLOSE_events=len(events),
        missed_CLOSE_ids=missing,CLOSE_outside100ms_ids=late,
        terminal_intent_mismatches=[r['id']for r in rows if r.get('terminal_intent_matches')is False],
        actual_execution=False)


def audit(train,cn,roll,*,saved_reject_ids):
    if len(saved_reject_ids)!=3 or len(set(saved_reject_ids))!=3:raise ValueError('EXACT_THREE_HISTORICAL_REJECTION_INPUTS')
    if any(len({r['id']for r in rows})!=len(rows)for rows in(train,cn,roll)):
        raise ValueError('DISTINCT_FIXED_OFFLINE_CONDITIONS_REQUIRED')
    summaries={k:bank_summary(rows)for k,rows in(('TRAIN96',train),('CN38',cn),('ROLL96',roll))}
    lookup={r['id']:r for r in roll};rejections=[dict(id=key,present=key in lookup,passed=key in lookup and bool(lookup[key]['envelope']['original_full_H40_pass']))for key in saved_reject_ids]
    close_complete=all(summaries[k]['CLOSE_rows']==n and summaries[k]['annotated_CLOSE_events']==n for k,n in CLOSE_EXPECTED.items())
    gates=dict(TRAIN96_all=len(train)==96 and summaries['TRAIN96']['full_H40_pass']==96,
        CN38_180_all=len(cn)==180 and summaries['CN38']['full_H40_pass']==180,
        saved_reject3_all=all(r['passed']for r in rejections),
        zero_CLOSE_omissions=close_complete and all(not summaries[k]['missed_CLOSE_ids']for k in CLOSE_EXPECTED),
        CLOSE_within100ms_no_regression=close_complete and all(not summaries[k]['CLOSE_outside100ms_ids']for k in CLOSE_EXPECTED))
    return dict(schema='SLA_MOBILE_V52_SEALED_PREDICTION_CPU_AUDIT',banks=summaries,saved_reject3=rejections,
        promotion_offline_gates=gates,all_offline_promotion_gates=all(gates.values()),
        CLOSE_threshold_rule='EXISTING_100MS_STANDARD_RETAINED_NOT_PER_EVENT_ERROR_TUNING',
        EEF_metrics_are_report_only=True,ROLL96_EEF_comparator='SAVED_B0_PREDICTION_NOT_SAME_STATE_EXPERT',
        new_generations=0,new_updates=0,new_native_requests=0,new_integrations=0)
