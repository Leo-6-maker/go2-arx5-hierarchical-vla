"""Observable command history for the cold time-package transport.

This records a command actually sent by consume(), never achieved aperture,
expert phase, future reference, or scene truth. It grants no native admission.
"""
import numpy as np
from go2arx5_vla.sla_adapter.gripper_time_v1 import reference_width

TIME_METHOD = 'CLOSE_TIME500'
SHIFT7_METHOD = 'SHIFT7_SINGLE'
SHIFT7_NATIVE_VERSION = 'MFREE_SHIFT7_PAUSED_V5'
SHIFT7_SIM_VERSION = 'MFREE_SHIFT7_SIM_ONLY_V51'
SHIFT7_V52_VERSION = 'MFREE_SHIFT7_SIM_ONLY_V52'
SHIFT7_V6_T_VERSION = 'MFREE_B1_FULL_H40_SIM_ONLY_V6'
SHIFT7_HISTORY_VERSIONS = (SHIFT7_NATIVE_VERSION, SHIFT7_SIM_VERSION, SHIFT7_V52_VERSION, SHIFT7_V6_T_VERSION)
REFERENCE_FIELDS = ('previous_width_m', 'previous_source_native_s', 'query_native_s')


def remember(adapter, row, gripper, after):
    method = adapter.spec['mobile_condition']['method']
    if method not in (TIME_METHOD, SHIFT7_METHOD):
        return
    if method == SHIFT7_METHOD:
        if (adapter.spec.get('mobile_experiment_version') not in SHIFT7_HISTORY_VERSIONS
                or adapter.spec['mobile_condition'].get('mode') != 'M_FREE'
                or getattr(adapter, 'shift7_history_invalidated', False)):
            raise ValueError('SHIFT7_EMISSION_HISTORY_REQUIRES_NEW_MFREE_NATIVE_IDENTITY')
        row = np.asarray(row)
        source, end = float(gripper['source_native_s']), after['step']
        if (row.shape != (10,) or not np.isfinite(row).all() or not 0 <= row[9] <= 1
                or type(end) is not int or not np.isfinite([source, after['time_s']]).all()
                or abs(after['time_s'] - end * .002) > 1e-8
                or abs(after['time_s'] - source - .05) > 1e-8):
            raise ValueError('SHIFT7_PREVIOUS_COMMAND_REQUIRES_COMPLETE_25_TICK_ATOM_ACK')
        previous = getattr(adapter, 'close_previous_reference', None)
        if previous is not None and end != previous['end_native_step'] + 25:
            raise ValueError('SHIFT7_CONFIRMED_COMMAND_HISTORY_MUST_ADVANCE_ONE_ATOM')
    adapter.close_previous_reference = dict(width_m=.044 * float(row[9]),
        source_native_s=float(gripper['source_native_s']), end_native_step=int(after['step']))


def query_reference(adapter, packet, sample):
    if adapter.spec['mobile_condition']['method'] == SHIFT7_METHOD:
        if (adapter.spec.get('mobile_experiment_version') not in SHIFT7_HISTORY_VERSIONS
                or getattr(adapter, 'shift7_history_invalidated', False)
                or not isinstance(getattr(adapter, 'close_previous_reference', None), dict)):
            raise ValueError('SHIFT7_CONFIRMED_EMISSION_HISTORY_MISSING_OR_INVALIDATED')
    previous = adapter.close_previous_reference
    if previous['end_native_step'] != packet['step'] or sample['native_step'] != packet['step']:
        raise ValueError('PREVIOUS_COMMAND_END_AND_PAUSED_OBSERVATION_MUST_MATCH')
    result = dict(width_m=previous['width_m'], source_native_s=previous['source_native_s'],
                  query_native_s=float(packet['time_s']))
    reference_width(result)
    return result


def invalidate(adapter):
    # Invalidate before attempting cancel/close: a missing acknowledgement must
    # not leave a reusable history. Old methods retain exactly their old state.
    if adapter.spec.get('mobile_condition', {}).get('method') == SHIFT7_METHOD:
        adapter.close_previous_reference = None
        adapter.shift7_history_invalidated = True


def reference_arrays(context):
    reference_width(context)
    return dict(zip(REFERENCE_FIELDS, (np.asarray(context[k], dtype=np.float64)
        for k in ('width_m', 'source_native_s', 'query_native_s'))))


def prediction(provider, method, data, language, noise_id):
    extra = {}
    if method in (TIME_METHOD, SHIFT7_METHOD):
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
