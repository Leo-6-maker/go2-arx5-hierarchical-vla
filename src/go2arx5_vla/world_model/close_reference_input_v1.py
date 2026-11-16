"""Observable command history for the cold time-package transport.

This records a command actually sent by consume(), never achieved aperture,
expert phase, future reference, or scene truth. It grants no native admission.
"""
import numpy as np
from go2arx5_vla.sla_adapter.gripper_time_v1 import reference_width

TIME_METHOD = 'CLOSE_TIME500'
REFERENCE_FIELDS = ('previous_width_m', 'previous_source_native_s', 'query_native_s')


def remember(adapter, row, gripper, after):
    if adapter.spec['mobile_condition']['method'] != TIME_METHOD:
        return
    adapter.close_previous_reference = dict(width_m=.044 * float(row[9]),
        source_native_s=float(gripper['source_native_s']), end_native_step=int(after['step']))


def query_reference(adapter, packet, sample):
    previous = adapter.close_previous_reference
    if previous['end_native_step'] != packet['step'] or sample['native_step'] != packet['step']:
        raise ValueError('PREVIOUS_COMMAND_END_AND_PAUSED_OBSERVATION_MUST_MATCH')
    result = dict(width_m=previous['width_m'], source_native_s=previous['source_native_s'],
                  query_native_s=float(packet['time_s']))
    reference_width(result)
    return result


def reference_arrays(context):
    reference_width(context)
    return dict(zip(REFERENCE_FIELDS, (np.asarray(context[k], dtype=np.float64)
        for k in ('width_m', 'source_native_s', 'query_native_s'))))


def prediction(provider, method, data, language, noise_id):
    extra = {}
    if method == TIME_METHOD:
        values = []
        for key in REFERENCE_FIELDS:
            value = np.asarray(data[key])
            if value.shape != ():
                raise ValueError('SCALAR_OBSERVABLE_PREVIOUS_REFERENCE_REQUIRED')
            values.append(float(value))
        context = dict(zip(('width_m', 'source_native_s', 'query_native_s'), values))
        reference_width(context)
        extra['previous_reference'] = context
    return provider.predict(data['state_eef10'], [data['body_rgb'], data['wrist_rgb']],
        language, data['T_policy_from_mount'], noise_id=noise_id, **extra)
