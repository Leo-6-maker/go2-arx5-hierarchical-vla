"""Three-session suffix using the original GR00T Trainer/restore/export methods.

No entry point or native request adapter is supplied by this module. Central
accounting, owner tickets and the terminal auditor must select this explicit
type before it can run; the old five-session worker is not a continuation.
"""
from pathlib import Path
from . import minimal_execution as e,minimal_remaining_contract as r
from .minimal_worker import Worker as OriginalWorker,once
from .minimal_remaining_effects import RemainingEffects
from .minimal_remaining_inputs import Inputs


class RemainingWorker(OriginalWorker):
    evidence_name='MINIMAL_THREE_REMAINING_SESSIONS'

    def __init__(self,root,request_path,allocation,ledger,*,guard):
        e.need(isinstance(ledger,RemainingEffects),'REMAINDER_WORKER_EXPLICIT_OWN_THREE_SESSION_EFFECTS')
        super().__init__(root,request_path,allocation,ledger,guard=guard)

    def prepare_inputs(self):
        self.saved=Inputs(self.root,self.manifest,guard=self.guard)
        e.need(self.saved.raw_checkpoint==self.raw_checkpoint
            and self.saved.binding==self.ledger.prefix_binding,
            'REMAINDER_WORKER_SAME_RAW_PARENT_AND_EXACT_AUDITED_PREFIX')

    def raw_audit(self):return self.saved.verify_current()

    def phase(self,sid,phase):
        self.saved.verify_current()
        return super().phase(sid,phase)

    def zero_draws(self,sid):
        e.need(sid==r.sessions()[0]['id'],'REMAINDER_WORKER_ONLY_E2_SAVED_ZERO_INPUTS')
        from scripts.pi0294.e1.model import digest
        before=self.ledger.guard()
        fingerprint=digest({k:v for k,v in self.zero_payload.items()if k!='snapshot_uuid'})
        proof=self.saved.E2_zeros(input_fingerprint=fingerprint)
        e.need(self.ledger.guard()==before,'REMAINDER_WORKER_NO_DEBIT_OR_NEW_QUERY_LINK_FOR_SAVED_ZEROS')
        once(self.evidence/sid/'SAVED_ZERO_INPUTS_READONLY.json',proof)
        return [Path(row['path'])for row in proof['original_draws']]
