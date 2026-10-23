"""Fresh target binding for the unchanged original durable relay operations.

The original relay is one of the sealed source glue members. Do not modify its
bytes or widen its source contract ABI. This cold subclass only installs the
separate target contract before the first observation. All action gates,
durable generation/command intents, ACK checks and stop behavior are inherited.
It provides no native admission, execution, retry or safety qualification.
"""
from pathlib import Path
from scripts.pi0292.durable import write
from scripts.pi0292.relay_protocol import RelayExchange
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
        # The original constructor validates UUID/model/wall clock and owns its
        # lock. It receives no source contract or fabricated source S1 wrapper.
        super().__init__(directory,run_uuid=run_uuid,episode_uuid=episode_uuid,identity=identity,now=now)
        try:
            self.mapped_ack_contract=mapped_ack_contract
            self.state['mapped_ack_condition_sha256']=mapped_ack_contract.condition_sha256
            write(self.path,self.state)
        except BaseException:
            self.close()
            raise
