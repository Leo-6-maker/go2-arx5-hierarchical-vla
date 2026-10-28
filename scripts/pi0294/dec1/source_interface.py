"""DEC1 source full-H40 gates and explicit unsupported preview-shadow boundary.

No constructor/integrator runs at import. The source single-pose predictor
cannot certify preview9 dispatch; native preview-shadow qualification is absent.
"""
import numpy as np
from scripts.pi0292.eval_contract import FRAME,TCP,gate_action
from go2arx5_vla.trajectory.so3 import quat_to_matrix,rotation_matrix_to_rot6d_rows
from go2arx5_vla.adapters.execution_progress_binding import NativeShadowReceipt,NativeShadowUnsupported


def mapped_eef10(proposal):
    c=proposal.mapped_full
    if c.source_frame!=FRAME or c.tcp_semantic_id!=TCP:
        raise PermissionError('DEC1_SOURCE_MAPPED_FRAME_TCP_MUST_MATCH_ORIGINAL_GATE')
    return np.column_stack((c.eef_position,
        rotation_matrix_to_rot6d_rows(quat_to_matrix(c.eef_orientation)),c.gripper))


def full_h40_validators(*, actual_state10, observation_time, record_gate):
    """Raw and mapped numerical legality use the original full-H40 gate; not physical S1."""
    def check(action,identifier,kind):
        result=gate_action(action,actual_state10(),observation_time(),identifier)
        if not isinstance(result,dict) or result.get('decision') not in ('ACCEPT','WARN','REJECT'):
            raise PermissionError('DEC1_ORIGINAL_FULL_H40_GATE_MALFORMED_RESULT')
        record_gate(dict(kind=kind,identifier=identifier,result=result))
        if result['decision'] not in ('ACCEPT','WARN'):
            raise PermissionError('DEC1_ORIGINAL_FULL_H40_GATE_REJECT:'+kind)
    def raw(action):check(action,'DEC1_RAW_NATIVE_H40','RAW_H40')
    def mapped(proposal):check(mapped_eef10(proposal),proposal.command_id,'MAPPED_H40')
    return raw,mapped


class SourcePrefixShadow:
    """Fail closed: original source Engine.step consumes one pose, not preview9.

    Original s1_preflight.predict checks 8 single poses via 200 Engine.step
    calls. It cannot certify dispatch of nine-point WBCPreview objects.
    Neither an equal anchor nor a proposal digest proves preview equivalence.
    A qualified native preview9 shadow/dispatch ABI is still required.
    """
    supported=False
    native_status='NOT_SUPPORTED_NINE_POINT_PREVIEW_ABI'

    def __init__(self,engine,*,episode_id,executor_id,guard,charge,record):
        self.engine=engine;self.episode_id=episode_id;self.executor_id=executor_id
        self.guard=guard;self.charge=charge;self.record=record

    def __call__(self,proposal):
        self.guard()
        self.record(dict(kind='SOURCE_PREVIEW_SHADOW_NOT_SUPPORTED',
            command_id=proposal.command_id,status=self.native_status,
            original_single_pose_predictor_called=False,shadow_integrations=0))
        return NativeShadowReceipt(self.episode_id,self.executor_id,proposal.command_id,
            'UNSUPPORTED:'+proposal.command_id,0,decision='NOT_SUPPORTED')

    def bind_before_dispatch(self,proposal):
        raise NativeShadowUnsupported('DEC1_SOURCE_PREVIEW9_NATIVE_ABI_NOT_SUPPORTED')
