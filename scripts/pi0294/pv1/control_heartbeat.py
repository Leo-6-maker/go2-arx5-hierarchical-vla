"""Explicit cold opt-in: RPC reads owner liveness; it cannot renew that liveness."""
from pathlib import Path
import time

from scripts.pi0292.durable import write
from scripts.pi0294.e1.control_read import read
from scripts.pi0294.mp1.owner_heartbeat import read_live_owner
from .decision import AUTHORITY

POLICY='PV1_LOCAL_OWNER_HEARTBEAT_ONLY_V1'


def owner_only(root,manifest):
    """False preserves historical RPC heartbeat semantics for old capsules."""
    policy=manifest.get('execution',{}).get('owner_heartbeat_policy')
    if policy is None:return False
    stage=read(Path(root)/'PV1_STAGE_BINDING.json')
    from . import mix_vanda_eval_identity as colocated
    authority=colocated.s.TASK_SOURCE if colocated.typed(manifest) else AUTHORITY
    if (policy!=POLICY or stage.get('owner_heartbeat_policy')!=POLICY
            or manifest.get('owner')!='EXISTING_LOCAL_PC'
            or manifest.get('authority',{}).get('protocol_ref')!=authority
            or stage.get('authority_release')!=authority):
        raise PermissionError('PV1_PRESEALED_SINGLE_OWNER_HEARTBEAT_POLICY_REQUIRED')
    return True


def verify_without_renewing(root,run_uuid):
    proof=read_live_owner(root,run_uuid)
    if proof['heartbeat_written'] is not False or proof['watchdog_s']!=300:
        raise PermissionError('PV1_RPC_CANNOT_RENEW_OR_EXTEND_ORIGINAL_OWNER_WATCHDOG')
    return proof


def keepalive(root,role,*,readonly_owner):
    if type(readonly_owner) is not bool or role not in ('core','hopper'):
        raise PermissionError('PV1_EXPLICIT_CONTROL_HEARTBEAT_MODE_AND_ROLE')
    if role=='core':return None
    root=Path(root)
    if readonly_owner:return verify_without_renewing(root,root.name)
    write(root/'OWNER_HEARTBEAT.json',dict(run_uuid=root.name,observed_epoch=time.time()))
    return None
