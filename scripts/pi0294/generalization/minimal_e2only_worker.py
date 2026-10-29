"""Original full GR00T/Trainer, only the unfinished formal E2 session."""
from . import minimal_e2only_contract as c,minimal_execution as e
from .minimal_worker import Worker
from .minimal_e2only_effects import E2OnlyEffects


def recovery_save_headroom(root,request,step,*,guard):
    """Reserve a complete original save, keeping every previous state."""
    import os
    from pathlib import Path
    from .minimal_e2only_execution import request as check_request
    from .minimal_train_session import save_headroom
    check_request(request);policy=c.recovery_policy();base=Path(root)/'training'/c.sessions()[0]['id']
    e.need(type(step)is int and 0<step<=512 and not base.is_symlink(),
        'E2ONLY_RECOVERY_OWN_FRESH_STEP')
    states=list(base.glob('checkpoint-*'))
    e.need(len(states)<policy['max_full_states']and not(base/('checkpoint-'+str(step))).exists()
        and all(p.is_dir()and not p.is_symlink()and p.name.split('-')[-1].isdigit()
            and 0<int(p.name.split('-')[-1])<step and(p/'CHECKPOINT_COMPLETE.json').is_file()for p in states),
        'E2ONLY_TWO_COMPLETE_STATES_NO_REPLAY_OR_DELETION')
    proof=save_headroom(root,c.sessions()[0]['arm'],next_upper_bytes=policy['per_full_state_upper_bytes'],guard=guard)
    # Keep the final export and all own code/control/temp in the same70GB.
    used=sum(p.stat().st_size for p in Path(root).rglob('*')if p.is_file()and not p.is_symlink())
    e.need(used+policy['per_full_state_upper_bytes']+policy['inference_upper_bytes']<=policy['root_upper_bytes'],
        'E2ONLY_RECOVERY_AND_FINAL_INFERENCE_FIT_SAME70GB')
    return dict(proof,existing_full_states=len(states),step=step)


class E2OnlyWorker(Worker):
    evidence_name='MINIMAL_E2_ONLY_FORMAL512'
    def prepare_inputs(self):
        from .minimal_saved512_inputs import Inputs
        self.saved=Inputs(self.root,self.manifest,guard=self.guard)
        e.need(self.adapter.VERSION==c.VERSION and self.saved.raw_checkpoint==self.raw_checkpoint
            and isinstance(self.ledger,E2OnlyEffects),'E2ONLY_ORIGINAL_RAW_PARENT_AND_MATCHED_DATA')
    def raw_audit(self):return self.saved.verify_current()
    def phase(self,sid,phase):
        self.saved.verify_current();return super().phase(sid,phase)
