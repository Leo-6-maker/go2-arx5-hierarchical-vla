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
    from scripts.pi0294.generalization import minimal_core_accounting as first512
    if first512.typed(manifest):
        if (policy!=POLICY or stage.get('owner_heartbeat_policy')!=POLICY
                or manifest.get('owner')!='EXISTING_LOCAL_PC'
                or manifest.get('scope')not in(first512.ROLE,first512.PHYSICAL_ROLE)
                or manifest['authority'].get('scope_grant_sha256')!=first512.GRANT_SHA
                or stage.get('scope_grant_sha256')!=first512.GRANT_SHA
                or stage.get('task')!=first512.plan.STAGE
                or manifest['authority'].get('protocol_ref')!=first512.campaign.TASK_SOURCE
                or stage.get('authority_release')!=first512.campaign.TASK_SOURCE):
            raise PermissionError('RAW512_EXACT_SINGLE_OWNER_HEARTBEAT_AUTHORITY')
        return True
    from . import mix_vanda_eval_identity as colocated
    from . import mix_core_identity as cd
    from .successor_core_context import EVAL_ROLE
    successor=colocated.typed(manifest) or cd.typed(manifest) or manifest.get('scope')==EVAL_ROLE
    authority=cd.s.TASK_SOURCE if successor else AUTHORITY
    if successor:
        roles=(colocated.NEURAL_ROLE,colocated.PHYSICAL_ROLE) if colocated.typed(manifest) else (cd.c.ROLE,EVAL_ROLE)
        if (manifest.get('scope') not in roles
                or manifest.get('authority',{}).get('scope_grant_sha256')!=cd.s.GRANT_SHA
                or stage.get('scope_grant_sha256')!=cd.s.GRANT_SHA
                or stage.get('task')!=cd.s.TASK):
            raise PermissionError('PV1_EXACT_SUCCESSOR_OWNER_HEARTBEAT_AUTHORITY')
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
