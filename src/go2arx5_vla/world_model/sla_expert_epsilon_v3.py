"""Run the frozen expert with the new gripper representation in native simulation."""
import numpy as np

from ..sla_adapter.gripper_latent_v2 import EPSILON, VERSION, decode_latent, label_latent

SCHEMA = 'SLA_LOGIT_EPSILON_NATIVE_EXPERT_VALIDATION_V2'


def configuration(spec):
    value = spec.get('sla_expert_epsilon_v3')
    if value is None:
        return False
    from .corelab_expert_semantics_v3 import ROLE
    if (value != dict(schema=SCHEMA, codec=VERSION, epsilon=EPSILON)
            or spec.get('task_learning_role') != ROLE
            or spec.get('task_learning_teacher_mode') != 'NORMAL_TEACHER_V7'):
        raise ValueError('EXACT_FROZEN_TRAIN_EXPERT_LOGIT_EPSILON_QUALIFICATION_REQUIRED')
    return True


def transform(program):
    from .task_learning_grasp_v1 import FiftyMsProgram

    raw = program.decoded_H40.copy()
    original = raw[:, 9].copy()
    latent = label_latent(original)
    raw[:, 9] = decode_latent(latent)
    error = float(np.max(np.abs(raw[:, 9] - original)))
    if error > EPSILON + np.finfo(np.float64).eps:
        raise ValueError('NATIVE_EXPERT_NOMINAL_GRIPPER_EPSILON_EXCEEDED')
    adjusted = FiftyMsProgram(raw, program.decision_TCP_target, program.initial_reference_TCP_target,
                              program.base_pose_hold)
    return adjusted, dict(schema=SCHEMA, codec=VERSION, epsilon=EPSILON,
                          original_nominal=original.tolist(), latent=latent.tolist(),
                          decoded_nominal=raw[:, 9].tolist(), maximum_nominal_bias=error,
                          active_pose_coordinates_exact=True)
