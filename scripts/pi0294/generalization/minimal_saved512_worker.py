"""Original GR00T/Trainer methods, own-E1 inference only then fresh-E2 training."""
import gc,time
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import minimal_execution as e,minimal_saved512_contract as c,minimal_model as model,minimal_train_session as training
from . import minimal_inference_export as export,minimal_probes as language
from .minimal_worker import Worker as OriginalWorker,once
from .minimal_saved512_effects import Saved512Effects


class Saved512Worker(OriginalWorker):
    evidence_name='MINIMAL_SAVED512_COMPLETION'
    def __init__(self,root,request_path,allocation,ledger,*,guard):
        e.need(isinstance(ledger,Saved512Effects),'SAVED512_EXPLICIT_OWN_EFFECTS_NOT_REMAINDER_ALIAS')
        super().__init__(root,request_path,allocation,ledger,guard=guard)
    def prepare_inputs(self):
        self.saved=self.ledger.saved
        e.need(self.saved.raw_checkpoint==self.raw_checkpoint and self.saved.binding==self.ledger.binding,
            'SAVED512_EXACT_SAME_RAW_PARENT_AND_AUDITED_INPUT_BINDING')
    def raw_audit(self):return self.saved.verify_current()
    def phase(self,sid,phase):
        self.saved.verify_current();return super().phase(sid,phase)

    def finalize_saved_E1(self,sid):
        e.need(sid==c.sessions()[0]['id']and self.ledger.guard()['sessions'][sid]['optimizer_updates']==0,
            'SAVED512_E1_NO_NEW_OPTIMIZER_BEFORE_RESTORE')
        cp,target=self.saved.E1_paths();proof=self.phase(sid,'RESTORE_SAVED512')
        # Build only; train() is never invoked for the old own512. The mature
        # loader verifies every wrapped parameter/buffer after the real load.
        trainer,_=training.build_trainer(self.root,self.instance,self.allocation,self.ledger,sid,guard=self.guard)
        frozen=read(cp/'FROZEN_PARAMETER_IDENTITY.json')
        e.need(trainer.frozen_baseline==frozen,'SAVED512_RAW_E1_FROZEN_BASELINE_EQUALS_ORIGINAL_OWN512')
        once(self.evidence/sid/'OWN512_INFERENCE_RESTORE_INTENT.json',dict(
            saved_result_sha256=c.RESULT,wrapped_checkpoint_manifest_sha256=c.E1_CP,
            release_sha256=e.sha(e.raw(proof)),observed_epoch=time.time(),new_optimization_authorized=False))
        self.ledger.charge('model_loads',1)
        trainer._load_from_checkpoint(str(cp))
        current=model.bind_saved_checkpoint(self.instance,cp,guard=self.guard)
        e.need(current['identity']['global_step']==512 and current['identity']['mode']=='E1_ADAPTER_ONLY',
            'SAVED512_ACTUAL_OWN_WRAPPED512_RESTORED_NOT_RAW_OR_QUALIFICATION')
        del trainer;gc.collect();rows=self.language_draws(sid)
        proof=self.phase(sid,'OFFICIAL_RELOAD');self.unload();self.saved.verify_current()
        receipt=read(target/'CHECKPOINT_COMPLETE.json')
        e.need(sha(target/'CHECKPOINT_COMPLETE.json')==c.E1_EXPORT and receipt['wrapped_checkpoint_manifest_sha256']==c.E1_CP,
            'SAVED512_ORIGINAL_ORDINARY_EXPORT_NOT_REWRITTEN')
        for name,pin in receipt['files'].items():
            path=target/name;w=self.saved.paths.get(str(path))
            e.need(w is not None and w['sha256']==pin,'SAVED512_ORDINARY_EVERY_MEMBER_ALREADY_FULL_STREAMED')
        once(self.evidence/sid/'ORIGINAL_ORDINARY_RELOAD_INTENT.json',dict(export_manifest_sha256=c.E1_EXPORT,
            saved_result_sha256=c.RESULT,release_sha256=e.sha(e.raw(proof)),automatic_retry=False))
        self.ledger.charge('model_loads',1)
        identity=dict(schema='ARX5_MINIMAL_ACTUAL_ORDINARY_FULL_INFERENCE_IDENTITY_V1',mode='E1_ADAPTER_ONLY',global_step=512,
            checkpoint_manifest_sha256=c.E1_EXPORT,files=receipt['files'],statistics_sha256=e.R0,
            raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,full_resume=False,official_inference=True)
        from scripts.pi0294.e1.model import load
        self.instance=load(self.root,target,identity,trainable=False);self.instance.parameter_contract=None
        config=self.instance.model.config
        e.need(config.action_horizon==40 and config.max_action_dim==132 and config.max_state_dim==132
            and config.max_num_embodiments==32 and self.instance.model.action_head.num_inference_timesteps==4
            and self.instance.model.action_head.state_dropout_prob==0,'SAVED512_ORDINARY_ORIGINAL_INFERENCE_GRAPH')
        case=self.language_plan['cases'][0];payload=language.payload_for_case(case,self.capture,self.reader)
        output=self.draw(sid,'official-reloaded',payload,case['noise_seed'])
        equality=export.compare_saved_draws(self.root/rows[0]['relative_path'],output)
        once(self.evidence/sid/'OFFICIAL_INFERENCE_RECONSTRUCTION_VERIFIED.json',dict(export_manifest_sha256=c.E1_EXPORT,
            original_full_checkpoint_manifest_sha256=c.E1_CP,saved_state_audit_sha256=c.RESULT,equality=equality))
        return dict(saved_own512_restore=current,language_probes=rows,official_inference_manifest_sha256=c.E1_EXPORT,
            official_output_equality=equality,original_TRAIN_probes_sha256=c.E1_TRAIN,original_TRAIN_probes_reexecuted=0,
            original_training_job_id=c.scope.JOB,new_optimizer_updates=0)

    def run(self):
        e.need(self.ledger.guard()['active_session']is None,'SAVED512_NO_RUNTIME_RESTART')
        self.audit=self.raw_audit();once(self.evidence/'RAW_PARENT_COMPLETE_STREAM_AUDIT.json',self.audit)
        try:
            for spec in self.ledger.plan():
                sid=spec['id'];self.ledger.start_session(sid);self.load_raw(sid,self.phase(sid,'MODEL_RAW'))
                result=self.finalize_saved_E1(sid)if spec['inference_finalization_only']else self.formal(sid)
                self.unload();row=self.ledger.guard()['sessions'][sid]
                e.need(row['model_loads']==(3 if spec['inference_finalization_only']else 2),
                    'SAVED512_ALL_ACTUAL_OWN_RAW_SAVED_AND_ORDINARY_LOADS')
                receipt=dict(schema='ARX5_MINIMAL_RUNTIME_SESSION_TERMINAL_V1',status='COMPLETE_SAVED_REQUIRES_INDEPENDENT_OWNER_AUDIT',
                    session_id=sid,arm=spec['arm'],request_sha256=self.allocation['request_sha256'],source_sha=self.request['source_sha'],
                    raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,
                    optimizer_updates=row['optimizer_updates'],query_attempts=row['generation_queries'],cursor=spec['sample_visits'],
                    model_load_attempts=row['model_loads'],actual_session_evidence=result,behavior_success_not_measured=True)
                p=self.evidence/sid/'SESSION_TERMINAL.json';once(p,receipt);self.ledger.complete_session(sid,p)
            return self.ledger.summary()
        finally:self.unload()
