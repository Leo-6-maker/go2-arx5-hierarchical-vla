"""V4 bounded continuous qualification scoring. Cold and hot never pooled."""
import numpy as np

CRITERIA=dict(unified_requests=32,first_request_cold=True,hot_requests=31,
    nominal_release_period_s=.5,warm_latency_p95_s=.35,valid_request_latency_strict_s=.4,
    release_slot_deadline_s=.5,archive_max_pending=4,archive_max_reserved_bytes=4*1024**2,
    cold_clock='WALL_CLOCK_STRICT_NOT_EXTENDED_COLD_LATE_RESPONSE_REJECTED',
    hot_schedule='AFTER_COLD_TERMINAL_FIRST_HOT_RELEASE_PLUS500MS_THEN_FIXED31_RELEASES',
    input_latency_start='RECORDED_FILE_ADAPTER_DELIVERED_BYTES_AVAILABLE_NOT_LIVE_CAMERA',
    all_requests_clock='WALL_CLOCK_STRICT',mock_atoms_are_not_realtime_physics=True)


def quantiles(values):
    return None if not values else np.percentile(values,[50,95,99,100]).tolist()


def evaluate(rows, archived):
    if len(rows)!=32 or [r['index']for r in rows]!=list(range(32)):
        raise ValueError('EXACT32_NO_REPLACEMENT_NO_EXTRA_REQUEST')
    hot=rows[1:];latencies=[r['input_available_to_terminal_s']for r in hot]
    accepted=[r for r in rows if r['ACK']]
    checks=dict(hot_all_ACK=len([r for r in hot if r['ACK']])==31,
        all_full_H40=all(r['full_H40_pass']for r in rows),
        warm_p95=float(np.percentile(latencies,95))<=.35,
        valid_max_strict=all(r['input_available_to_terminal_s']<.4 for r in accepted),
        no_lost_release_slot=all(r['release_lateness_s']<.5 for r in hot),
        no_queue_backpressure=all(r['reservation_wait_capacity_s']==0 for r in hot),
        all_raw_outputs_exact=all(r['raw_output_transport_exact']for r in rows),
        all_archived=archived['archived']==32 and archived['peak_pending']<=4,
        no_unprotected_continuation=all(r['guard_violation_count']==0 for r in rows))
    return dict(checks=checks,GO=all(checks.values()),cold=rows[0],warm_n=31,warm_latency_p50_p95_p99_max_s=quantiles(latencies),
        capture_including_file_delivery_p50_p95_p99_max_s=quantiles([r['capture_to_terminal_s']for r in hot]),
        release_lateness_p50_p95_p99_max_s=quantiles([r['release_lateness_s']for r in hot]),
        warm_actual_dispatch_intervals_s=[hot[i]['dispatch_monotonic']-hot[i-1]['dispatch_monotonic']for i in range(1,len(hot))],
        raw_output_H40=sum(r['full_H40_pass']for r in rows),strict_ACK=sum(bool(r['ACK'])for r in rows),
        TTL_rejections=sum(r['TTL_rejected']for r in rows),cold_is_separately_reported=True,
        wall_clock_qualification_scope='FINITE_RECORDED_RGB_MOCK_INTERFACE_ONLY',native_admission=False)
