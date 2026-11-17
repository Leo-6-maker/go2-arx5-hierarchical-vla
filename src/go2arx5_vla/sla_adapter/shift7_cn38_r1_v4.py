"""Fixed V4 episode selection and development gates; no model or native API."""
import numpy as np

PHASES = ('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')
F1_INDICES = (1,5,8,12)
CONTRACT = dict(plan_sha='a51f7a02d23414304f631b8720aec5fcb95d5705',
    first_predictions=80, full_predictions=180, first_close=16, full_close=36,
    max_crossing_error_s=.100000001, max_seam_speed_m_s=.04*(1+1e-10),
    EEF_p95_deterioration_max_m=.0005, EEF_point_max_deterioration_max_m=.001,
    terminal_intent_threshold_m=.022,
    terminal_rule_source='Existing width_metrics 22mm crossing and V4 no missed open/close; terminal width error report-only',
    EEF_distribution='All 40 per-knot position errors pooled over identical conditions; numpy linear percentile',
    full_bank_quality_rule='Same fixed relative quality gates as first80; no post-result tolerance change',
    native_admission=False, new_updates=0)


def panel(bank):
    entries = bank['records']
    families = {f:sorted({r['episode'] for r in entries if r['family']==f}) for f in ('F1','F2')}
    if len(families['F1']) != 14 or len(families['F2']) != 4 or len(entries) != 90:
        raise ValueError('EXACT_F1_14_F2_4_90_ANCHORS_REQUIRED')
    first = [families['F1'][i] for i in F1_INDICES] + families['F2']
    rest = [ep for ep in families['F1'] if ep not in first]
    rows = []
    for wave, episodes in ((1,first),(2,rest)):
        for ep in episodes:
            for phase in PHASES:
                matches = [r for r in entries if r['episode']==ep and r['phase']==phase]
                if len(matches)!=1:
                    raise ValueError('ONE_SEALED_LOWER_MEDIAN_ANCHOR_PER_PHASE_REQUIRED')
                row=matches[0]
                for noise in (0,1):
                    rows.append(dict(wave=wave, entry=row['index'], episode=ep,
                        family=row['family'],phase=phase,noise=noise))
    if len(rows)!=180 or sum(r['wave']==1 for r in rows)!=80:
        raise ValueError('EXACT80_PLUS100_NO_ADDITIONAL_REPLAY')
    return rows


def evaluate(rows, *, expected):
    close=[r for r in rows if r['phase']=='CLOSE']
    count=16 if expected==80 else 36
    if expected not in (80,180) or len(rows)!=expected or len(close)!=count:
        raise ValueError('EXACT_PREDECLARED_GATE_DENOMINATOR_REQUIRED')
    errors=np.asarray([r['EEF_position_errors_m'] for r in rows])
    baseline=np.asarray([r['HYBRID_EEF_position_errors_m'] for r in rows])
    if errors.shape!=(expected,40) or baseline.shape!=errors.shape or not np.isfinite([errors,baseline]).all():
        raise ValueError('SAME_CONDITION_FINITE40_KNOT_GEOMETRY_REQUIRED')
    p95=float(np.percentile(errors,95));old95=float(np.percentile(baseline,95))
    maximum=float(errors.max());oldmax=float(baseline.max())
    checks=dict(all_H40=all(r['envelope']['original_full_H40_pass'] and r['original_compiler_pass'] for r in rows),
        CLOSE_events=all(r['width']['expert_crossing_in_window'] is not None
            and r['width']['crossing_error_s'] is not None
            and r['width']['crossing_error_s']<=CONTRACT['max_crossing_error_s'] for r in close),
        zero_missed_intent=all(r['terminal_intent_matches'] for r in rows),
        zero_seam_violation=all(abs(r['coordination']['previous_actual_command_to_first_speed_m_s'])<=CONTRACT['max_seam_speed_m_s'] for r in rows),
        EEF_p95=p95-old95<=CONTRACT['EEF_p95_deterioration_max_m'],
        EEF_single_point_max=maximum-oldmax<=CONTRACT['EEF_point_max_deterioration_max_m'])
    return dict(predictions=len(rows),CLOSE_count=len(close),checks=checks,GO=all(checks.values()),
        EEF_position_p95_m=p95,HYBRID_EEF_position_p95_m=old95,p95_deterioration_m=p95-old95,
        EEF_max_m=maximum,HYBRID_EEF_max_m=oldmax,max_deterioration_m=maximum-oldmax,
        full_H40_pass=sum(r['envelope']['original_full_H40_pass'] for r in rows),
        original_compiler_pass=sum(r['original_compiler_pass'] for r in rows),
        failed_indices=[i for i,r in enumerate(rows) if not r['envelope']['original_full_H40_pass']],
        by_family={f:dict(n=sum(r['family']==f for r in rows),H40_pass=sum(r['family']==f and r['envelope']['original_full_H40_pass'] for r in rows))for f in ('F1','F2')},
        physical_admission=False)
