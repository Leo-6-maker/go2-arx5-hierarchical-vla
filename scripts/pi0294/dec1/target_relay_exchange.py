"""Fresh target binding for the unchanged original durable relay operations.

The original relay is one of the sealed source glue members. Do not modify its
bytes or widen its source contract ABI. This cold subclass only installs the
separate target contract before the first observation. All action gates,
durable generation/command intents, ACK checks and stop behavior are inherited.
It provides no native admission, execution, retry or safety qualification.
"""
from pathlib import Path
from copy import deepcopy
import math
from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0292.relay_protocol import RelayExchange,PROTOCOL,uuid_string,validate_identity
from .target_mapped_ack import FixedTargetTimeAckContract


class TargetRelayExchange(RelayExchange):
    def __init__(self,directory,*,run_uuid=None,episode_uuid=None,identity=None,now=None,mapped_ack_contract=None):
        if not isinstance(mapped_ack_contract,FixedTargetTimeAckContract):
            raise TypeError('EXPLICIT_COLD_TARGET_ACK_CONTRACT_REQUIRED')
        c=mapped_ack_contract.condition
        if run_uuid is None or (Path(directory)/'STATE.json').exists():
            raise FileExistsError('NEVER_REOPEN_OR_RESTART_EXISTING_TARGET_EXCHANGE')
        if c['run_uuid']!=run_uuid or c['episode_uuid']!=episode_uuid or c['identity']!=identity:
            raise ValueError('COLD_TARGET_EXCHANGE_EXACT_OWNER_CONDITION_REQUIRED')
        if not isinstance(run_uuid,str) or not run_uuid:raise ValueError('RUN_UUID_REQUIRED')
        uuid_string(episode_uuid);validate_identity(identity)
        if type(now) not in (int,float) or not math.isfinite(now):raise ValueError('WALL_CLOCK_REQUIRED')
        # Only fresh construction differs from the sealed source constructor:
        # pin the target ABI in the FIRST durable state. Calling super first
        # would transiently leave an unpinned RAW state after a lost return.
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True)
        self.lock=OwnerLock(self.root/'OWNER.lock');self.path=self.root/'STATE.json'
        try:
            if self.path.exists():raise FileExistsError('NEVER_REOPEN_OR_RESTART_EXISTING_TARGET_EXCHANGE')
            self.mapped_ack_contract=mapped_ack_contract
            self.state=dict(protocol=PROTOCOL,run_uuid=run_uuid,episode_uuid=episode_uuid,
                identity=deepcopy(identity),started=now,last_wall=now,deadline=now+900,
                phase='WAIT_OBSERVATION',sequence=0,tick=0,history=[],used_snapshots=[],pending=None,
                mapped_ack_condition_sha256=mapped_ack_contract.condition_sha256)
            write(self.path,self.state)
        except BaseException:
            self.close()
            raise
