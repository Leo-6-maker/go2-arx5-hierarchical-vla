"""Bind a cold original-SDK diagnostic to the already bound native policy.

No process dispatch, model construction, native loading or physics permission.
The C++ reader consumes a dedicated original SDK problem; it must NEVER share
a mutable precomputation cache with an active solver. Native build/header ABI,
whole-body guards, install/STOP and current admission remain separate checks.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import re

from .target_mpc_prediction import BoundMPCPrediction

FROZEN_TASK_SHA256 = '0b69bd58db364a01e87331b0e4e777a9684fda18a4f67fa60aee55174f72dcca'
MOBILE_SDK_SHA256 = 'dee326b07daca9a3c61cf2baf187520e551e4120ed4f22e94d078dd689b3235b'
TERMS = ('jointLimitsCost', 'selfCollision')


def need(ok, why):
    if not ok:
        raise PermissionError('DEC1_SDK_' + why)


def cache_only_client_config(frozen):
    """New client-only derived bytes; never overwrite the frozen WBC config.

    The actual task file defaults to recompileLibraries=true. This ONE build
    cache switch is changed for a future admitted client, preserving every
    numerical setting and all other bytes. Missing/stale AD libraries must
    fail admission; this utility does not grant permission to rebuild them.
    """
    need(type(frozen) is bytes and hashlib.sha256(frozen).hexdigest() == FROZEN_TASK_SHA256,
         'EXACT_FROZEN_TASK_BYTES')
    expression = rb'(?m)^(\s*recompileLibraries[ \t]+)true([ \t]*\r?)$'
    need(len(re.findall(expression, frozen)) == 1, 'EXACT_SINGLE_BUILD_CACHE_SWITCH')
    derived, count = re.subn(expression, rb'\g<1>false\g<2>', frozen)
    need(count == 1, 'ONE_CLIENT_ONLY_OVERLAY')
    return derived


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def request(prediction):
    need(isinstance(prediction, BoundMPCPrediction), 'ORIGINAL_BOUND_NATIVE_POLICY')
    need(_digest(prediction.policy_sha256) and _digest(prediction.wire_sha256), 'EXPLICIT_POLICY_AND_INSTALL_HASHES')
    times, states, inputs = prediction.time_trajectory, prediction.state_trajectory, prediction.input_trajectory
    need(2 <= len(times) <= 4096 and len(states) == len(inputs) == len(times), 'BOUNDED_ALIGNED_NATIVE_KNOTS')
    need(all(type(t) in (int, float) and math.isfinite(t) for t in times)
         and all(b > a for a, b in zip(times, times[1:])), 'FINITE_STRICT_NATIVE_CLOCK')
    for rows in (states, inputs):
        need(all(len(row) == 12 and all(type(v) in (int, float) and math.isfinite(v) for v in row)
                 for row in rows), 'FINITE_ORIGINAL_12D_STATE_INPUT')
    return dict(schema='DEC1_ORIGINAL_SDK_KNOT_REQUEST_V1',
                policy_sha256=prediction.policy_sha256, wire_sha256=prediction.wire_sha256,
                frozen_task_sha256=FROZEN_TASK_SHA256, mobile_sdk_sha256=MOBILE_SDK_SHA256,
                terms=list(TERMS), times=list(times), states=[list(x) for x in states],
                inputs=[list(x) for x in inputs])


def _pairs(rows):
    result = {}
    for k, v in rows:
        need(k not in result, 'DUPLICATE_REPLY_FIELD')
        result[k] = v
    return result


@dataclass(frozen=True)
class BoundSdkSamples:
    policy_sha256: str
    wire_sha256: str
    reply_sha256: str
    minimum_joint_limit_cost: float
    maximum_joint_limit_cost: float
    minimum_self_collision_residual: float
    sampling: str = 'ORIGINAL_NATIVE_POLICY_KNOTS_ONLY'
    native_prefix_safety_qualified: bool = False
    physics_advance_authorized: bool = False


def bind_samples(prediction, reply_bytes, *, expected_reply_sha256):
    """Retain signed collision residuals and the original joint BOX COST.

    Call only on owner-retained output from the qualified client. This wire
    check cannot attest executable identity, native execution or continuous
    safety. The box cost includes an offset and is not a signed feasibility
    margin. Neither its zero/sign nor a small tracking cost is a safety gate.
    """
    wanted = request(prediction)
    need(type(reply_bytes) is bytes and len(reply_bytes) <= 4 * 1024**2
         and hashlib.sha256(reply_bytes).hexdigest() == expected_reply_sha256, 'EXACT_OWNER_RETAINED_REPLY')
    reply = json.loads(reply_bytes, object_pairs_hook=_pairs,
                       parse_constant=lambda _: need(False, 'NONFINITE_JSON'))
    need(type(reply) is dict and set(reply) == {'request', 'samples'}, 'EXACT_REPLY_FIELDS')
    need(json.dumps(reply['request'], sort_keys=True, allow_nan=False) ==
         json.dumps(wanted, sort_keys=True, allow_nan=False), 'EXACT_POLICY_KNOTS_AND_INSTALL_IDENTITY')
    samples = reply['samples']
    need(type(samples) is list and len(samples) == len(wanted['times']), 'NO_MISSING_POLICY_KNOT')
    values = {term: [] for term in TERMS}
    widths = {}
    for i, sample in enumerate(samples):
        need(type(sample) is dict and set(sample) == {'time', *TERMS}
             and type(sample['time']) in (int, float) and sample['time'] == wanted['times'][i],
             'EXACT_NATIVE_KNOT_AND_TERMS')
        cost = sample['jointLimitsCost']
        need(type(cost) in (int, float) and math.isfinite(cost), 'FINITE_ORIGINAL_BOX_COST_NOT_MARGIN')
        values['jointLimitsCost'].append(cost)
        for term in ('selfCollision',):
            vector = sample[term]
            need(type(vector) is list and 0 < len(vector) <= 4096
                 and all(type(v) in (int, float) and math.isfinite(v) for v in vector),
                 'FINITE_NONEMPTY_ORIGINAL_CONSTRAINT_VECTOR')
            widths.setdefault(term, len(vector))
            need(len(vector) == widths[term], 'STABLE_FROZEN_CONSTRAINT_DIMENSION')
            values[term].extend(vector)
    return BoundSdkSamples(prediction.policy_sha256, prediction.wire_sha256, expected_reply_sha256,
                           min(values['jointLimitsCost']), max(values['jointLimitsCost']),
                           min(values['selfCollision']))
