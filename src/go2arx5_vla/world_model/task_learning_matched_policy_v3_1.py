"""Eight explicit matched configurations; raw and composed commands stay distinct."""
from .task_learning_student_v1 import compile_proposal as raw_compile
from .task_learning_endpoint_policy_v3 import compile_proposal as basis_compile

PROTOCOL='CORE_MATCHED_WEIGHTS_PROCESSOR_SMOOTH_TRAIN_NOT_DEV_TEST_V3_1'
METHODS=('B0_RAW','B0_PLUS_ENDPOINT_BASIS','B0_TARGET_PROCESSOR_RAW','B0_TARGET_PROCESSOR_PLUS_ENDPOINT_BASIS',
         'SFT2000_RAW','SFT2000_PLUS_ENDPOINT_BASIS','C500_RAW','C500_PLUS_ENDPOINT_BASIS')


def configuration(method):
    if method not in METHODS:raise ValueError('One of the eight preregistered matched methods required')
    basis=method.endswith('_PLUS_ENDPOINT_BASIS')
    family=method[:-len('_PLUS_ENDPOINT_BASIS')]if basis else method[:-len('_RAW')]
    return family,basis


def compile_proposal(action,pin,*,method,consumed_nominal_grip,**context):
    _,basis=configuration(method)
    if basis:
        program,proof=basis_compile(action,pin,consumed_nominal_grip=consumed_nominal_grip,**context)
    else:
        program,proof=raw_compile(action,pin,**context)
        proof.update(explicit_decoder_applied=False,GR00T_raw_H40_consumed=True)
    proof.update(composed_policy_method=method,matched_comparison=True)
    return program,proof
