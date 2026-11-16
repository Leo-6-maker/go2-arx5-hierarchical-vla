"""Nonphysical unified-H40 request contract. No actuator or simulator API.

GPU branch inference is qualified separately. This interface can reject a late
response, but cannot certify the existing native consumer's STOP/TTL behavior.
"""
from .hybrid_eef_grip_v1 import compose
from .gripper_time_v1 import reference_width
import math


def predict_request(eef_provider, grip_provider, *, state, images, language, frame,
                    noise_id, previous_reference, request_id, deadline_monotonic,
                    now, live_state):
    reference_width(previous_reference)
    def check():
        current=live_state()
        wall=now()
        if (not all(math.isfinite(float(x))for x in (wall,deadline_monotonic,current['query_native_s']))
                or current['stop'] or current['cancelled'] or wall>=deadline_monotonic
                or current['request_id']!=request_id or current['consumed_atoms']!=0
                or abs(current['query_native_s']-previous_reference['query_native_s'])>1e-8):
            raise RuntimeError('CANCELLED_EXPIRED_OR_PARTLY_CONSUMED_HYBRID_REQUEST')
    check()
    eef=eef_provider.predict(state,images,language,frame,noise_id=noise_id)
    check()
    grip=grip_provider.predict(state,images,language,frame,noise_id=noise_id,
                               previous_reference=previous_reference)
    check()
    action=compose(eef['physical_H40'],grip['physical_H40'])
    # One response contains both channels. No standalone gripper command exists.
    return dict(schema='MFREE_HYBRID_EEF_GRIP_UNIFIED_RESPONSE_V1',request_id=request_id,
        physical_H40=action,EEF_source='FM500',grip_source='FIXED_REGISTERED_TIME_PACKAGE',
        independent_gripper_write=False,native_execution_admitted=False)
