"""Five sequential sessions called by the original registered A40 worker.

No submission, new owner, heartbeat, resource reservation or permission is
created here. The existing owner must supply the actual allocation and guard,
consume the13 finite phase tickets, and independently audit the native terminal.
"""
import gc,json,os,time
from pathlib import Path
from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.pbs_owner import publish_release_record as once
from scripts.pi0294.e1 import model as original_model
from . import minimal_execution as e,minimal_model as model,minimal_train_session as training
from . import minimal_probes as language,minimal_session_gate as gates,minimal_inference_export as export
from .minimal_effects import Effects

need=e.need


def seal_checkpoint(root,session_id,step,expected_manifest_sha256,*,guard):
    """Seal only this session's freshly complete outputs, never input aliases.

    The original Trainer already streamed/bound its saved members. This adds
    metadata protection, without another full tensor stream. The independent
    terminal audit must still hash all full-state members after native Exit0.
    """
    import stat
    root=Path(root);spec=next(s for s in e.native_sessions()if s['id']==session_id)
    need(step==spec['optimizer_limit']and step in(2,512)and callable(guard),'WORKER_EXACT_OWN_SAVE_ENDPOINT')
    guard();base=root/'training'/session_id;cp=base/('checkpoint-'+str(step))
    need(not any(p.is_symlink()for p in(root/'training',base,cp))and cp.is_dir()
        and cp.resolve().is_relative_to(root.resolve()),'WORKER_FULL_SAVE_IS_NOT_A_RAW_PARENT_ALIAS')
    complete=cp/'CHECKPOINT_COMPLETE.json';need(sha(complete)==expected_manifest_sha256,'WORKER_FRESH_CLOSED_SAVE_MANIFEST')
    value=read(complete);need(value['global_step']==step and value['training_identity']==spec['arm']
        and value['full_resume']is True and value['qualification_only']is(not spec['formal_candidate']),
        'WORKER_SAME_ARM_COMPLETE_QUALIFICATION_OR_FIXED512')
    paths=list(cp.rglob('*'));members={}
    for path in paths:
        st=path.lstat();need(not stat.S_ISLNK(st.st_mode)and st.st_uid==os.getuid()
            and(stat.S_ISREG(st.st_mode)or stat.S_ISDIR(st.st_mode)),'WORKER_ONLY_OWN_REGULAR_SAVED_MEMBERS')
        if stat.S_ISREG(st.st_mode):members[path.relative_to(cp).as_posix()]=path
    need(set(members)==set(value['files'])|{'CHECKPOINT_COMPLETE.json'},'WORKER_NO_UNLISTED_FULL_SAVE_MEMBER')
    for path in members.values():path.chmod(0o400)
    need(sha(complete)==expected_manifest_sha256,'WORKER_SEAL_PRESERVES_FULL_SAVE_BYTES')
    guard();return dict(checkpoint_manifest_sha256=expected_manifest_sha256,sealed_regular_files=len(members),
        mode='0400',tensor_bytes_restreamed=0,independent_native_terminal_stream_still_required=True)


class Worker:
    evidence_name='MINIMAL_FIVE_SESSIONS'

    def prepare_inputs(self):pass

    def raw_audit(self):return model.audit_raw_parent(self.raw_checkpoint,guard=self.guard)

    def __init__(self,root,request_path,allocation,ledger,*,guard):
        need(isinstance(ledger,Effects)and callable(guard),'WORKER_ORIGINAL_TYPED_EFFECTS_AND_NATIVE_GUARD')
        self.root=Path(root);self.request_path=Path(request_path);self.allocation=allocation
        self.ledger=ledger;self.guard=guard;self.guard();self.request=ledger.request
        from . import minimal_execution_adapter as adapters
        self.adapter=adapters.request(self.request)
        from scripts.pi0294.dec1.a40_training_runtime import native_job
        job,_,_=native_job(self.request)
        need(job==allocation['job_id']==os.environ.get('PBS_JOBID'),'WORKER_OWN_ACTUAL_RUNNING_ALLOCATION')
        self.manifest=self.sealed('RUN_MANIFEST.json',manifest_member=True)
        need(self.manifest['scope']==e.scope.SCOPE and adapters.manifest(self.manifest) is self.adapter
            and sha(self.root/'RUN_MANIFEST.json')==self.request['run_manifest_sha256'],
            'WORKER_EXPLICIT_NEW_SCOPE_NOT_CD_OR_GEN')
        self.bindings=self.sealed('NEURAL_BINDINGS.private.json')
        self.raw_plan=self.sealed('plans/MINIMAL_RAW_PARENT_AUDIT_PLAN.private.json')
        need(sha(self.root/'plans/MINIMAL_RAW_PARENT_AUDIT_PLAN.private.json')==self.request['raw_parent_audit_plan_sha256']
            and self.raw_plan['raw_parent_files_identity']==e.training.RAW_PARENT_FILES_ID
            and self.raw_plan['files']==model.FILES,'WORKER_SEALED_EXACT_RAW_PARENT_STREAM_PLAN')
        self.raw_checkpoint=Path(self.raw_plan['checkpoint'])
        self.capture=self.sealed('plans/MINIMAL_SIX_PRECONTACT16_TRAIN_CAPTURE.private.json')
        self.language_plan=self.sealed('plans/MINIMAL_LANGUAGE24_PANEL.private.json')
        self.panel=self.sealed('plans/MINIMAL_TWELVE_PANEL.json')
        need(self.language_plan==language.prepare(self.capture,self.panel),'WORKER_ORIGINAL_FROZEN_LANGUAGE24_INPUTS')
        self.directory=self.request_path.parent;self.audit=None;self.instance=None
        self.prepare_inputs()
        evidence=self.root/'evidence'/self.evidence_name;need(not evidence.exists(),'WORKER_NO_RESTART_OR_MODEL_REPLAY')
        evidence.mkdir(parents=True,mode=0o700);self.evidence=evidence
        self.reader=training.dataset(self.root,evidence/'language-reader')
        first=self.language_plan['cases'][0]
        self.zero_payload=language.payload_for_case(first,self.capture,self.reader)

    def sealed(self,name,*,manifest_member=False):
        from scripts.pi0294.dec1.a40_training_context import sealed
        path=sealed(self.root,name,parse=False)
        if not manifest_member:
            need(self.manifest['execution']['sealed_inputs'].get(name)==sha(path),'WORKER_PRESEALED_INPUT:'+name)
        return read(path)

    def phase(self,sid,phase):
        return gates.wait_release(self.root,self.directory,self.allocation,self.ledger,sid,phase,guard=self.guard)

    def raw_allocation(self,proof):
        need(proof['phase']in('MODEL_RAW','RESTORE_FULL')and proof['job_id']==self.allocation['job_id']
            and proof['request_sha256']==self.allocation['request_sha256'],
            'WORKER_NATIVE_PHASE_DOES_NOT_AUTHORIZE_ANOTHER_JOB')
        # This is a consumer view of the actual persisted owner release, never
        # an independently created permission. wait_release checked its READY,
        # request, current effects, whole budget, queue and project proof.
        return dict(self.allocation,status=model.LOAD_STATUS,request_id=self.request['request_id'],
            submission_nonce=self.request['submission_nonce'],observed_epoch=proof['observed_epoch'],
            native_proof=proof['native_proof'],actual_session_release_sha256=e.sha(e.raw(proof)))

    def load_raw(self,sid,proof):
        need(self.instance is None,'WORKER_UNLOAD_PREVIOUS_FULL_MODEL_BEFORE_NEW_LOAD')
        self.guard();folder=self.evidence/sid;folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        name='MODEL_RAW_LOAD_INTENT.json'if proof['phase']=='MODEL_RAW'else'RESTORE_RAW_CONSTRUCTION_INTENT.json'
        once(folder/name,dict(session_id=sid,request_sha256=self.allocation['request_sha256'],
            release_sha256=e.sha(e.raw(proof)),observed_epoch=time.time(),automatic_retry=False))
        self.ledger.charge('model_loads',1)
        arm=next(s for s in e.native_sessions()if s['id']==sid)['arm']
        self.instance=model.load(self.root,self.request_path,self.raw_allocation(proof),self.bindings,
            checkpoint=self.raw_checkpoint,audit=self.audit,mode=arm,guard=self.guard)

    def unload(self):
        if self.instance is not None:
            model.unload(self.instance);self.instance=None
        gc.collect()

    def draw(self,sid,name,payload,seed):
        self.guard();folder=self.evidence/sid/'draws';folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        target=folder/(name+'.npz');need(not target.exists(),'WORKER_NO_QUERY_OUTPUT_REPLAY')
        # The original TRAIN probes restore the caller's training flag. The
        # upstream get_action calls head.get_action directly, bypassing the
        # head's forward hook. Fix the whole inference graph before seeding,
        # including shared attention dropout; never change training_step.
        self.instance.model.eval()
        need(not any(module.training for module in self.instance.model.modules()),
             'WORKER_COMPLETE_EVAL_GRAPH_BEFORE_FIXED_NOISE_DRAW')
        original_model.fixed_draw(self.root,sid,name,self.instance,payload,seed,target,self.ledger)
        target.chmod(0o400);return target

    def zero_draws(self,sid):
        paths=[]
        for index in range(15):
            path=self.draw(sid,'zero-'+str(index).zfill(2),self.zero_payload,29400+index);paths.append(path)
            if sid!=e.native_sessions()[0]['id']:
                reference=self.evidence/e.native_sessions()[0]['id']/'draws'/path.name
                export.compare_saved_draws(reference,path)
        return paths

    def language_draws(self,sid):
        rows=[]
        for case in self.language_plan['cases']:
            payload=language.payload_for_case(case,self.capture,self.reader)
            path=self.draw(sid,case['id'],payload,case['noise_seed'])
            rows.append(dict(id=case['id'],input_sha256=case['semantic_input_sha256'],
                noise_seed=case['noise_seed'],arrays_sha256=sha(path),relative_path=str(path.relative_to(self.root))))
        once(self.evidence/sid/'LANGUAGE24_SAVED.json',dict(status='ACTUAL_TRAIN_LANGUAGE24_NOT_TASK_SUCCESS',records=rows))
        return rows

    def original_probe_reader(self):
        """Original COV reader/reference only; never used by the12-parent Trainer."""
        from scripts.pi0294.pv1.mix_training import dataset as original_dataset
        b=self.sealed('plans/MINIMAL_ORIGINAL_PROBE_READER.private.json');inputs=b['inputs']
        need(b['reference_only_not_training']is True and b['arm']=='C50'and b['seed']==20261019
             and b['route']=='COVERAGE_FIRST'
             and inputs==self.sealed('plans/PV1_TRAIN_INPUTS.private.json'),
             'WORKER_ORIGINAL_COV_PROBE_REFERENCE_ONLY_NOT_NEW_TRAINING_POPULATION')
        # C50 is a training recipe, not a valid PV1Dataset constructor arm.
        # The mature factory binds C50 to COVERAGE_FIRST/B_COVERAGE before
        # installing its exact C50 sampler and initializing the same reader.
        return original_dataset(self.root,b['arm'],self.evidence/'original-probe-reader')

    def qualification(self,sid):
        self.zero_draws(sid)
        trainer,trained=training.run_training(self.root,self.instance,self.allocation,self.ledger,sid,
            guard=self.guard,authorize_training=lambda _:self.phase(sid,'FIRST_UPDATE'))
        cp=self.root/'training'/sid/'checkpoint-2'
        sealed=seal_checkpoint(self.root,sid,2,trained['checkpoint_manifest_sha256'],guard=self.guard)
        reference=self.draw(sid,'step2-reference',self.zero_payload,29400)
        proof=self.phase(sid,'RESTORE_FULL')
        del trainer;gc.collect();self.unload()
        self.load_raw(sid,proof)
        restored,context=training.build_trainer(self.root,self.instance,self.allocation,self.ledger,sid,
            guard=self.guard,restoring_qualification=True)
        once(self.evidence/sid/'FULL_QUALIFICATION_STATE_RESTORE_INTENT.json',dict(session_id=sid,
            checkpoint_manifest_sha256=sha(cp/'CHECKPOINT_COMPLETE.json'),observed_epoch=time.time(),automatic_retry=False))
        self.ledger.charge('model_loads',1) # Actual full saved model reload, in addition to the fresh raw construction.
        result=restored.restore_qualification_for_audit(cp,guard=self.guard)
        current=model.bind_saved_checkpoint(self.instance,cp,guard=self.guard)
        reloaded=self.draw(sid,'step2-restored',self.zero_payload,29400)
        equality=export.compare_saved_draws(reference,reloaded)
        del restored;gc.collect()
        return dict(training=trained,save_metadata_seal=sealed,restore=result,full_current_identity=current,output_equality=equality)

    def formal(self,sid):
        trainer,trained=training.run_training(self.root,self.instance,self.allocation,self.ledger,sid,
            guard=self.guard,authorize_training=lambda _:self.phase(sid,'FIRST_UPDATE'))
        del trainer;gc.collect()
        sealed=seal_checkpoint(self.root,sid,512,trained['checkpoint_manifest_sha256'],guard=self.guard)
        from scripts.pi0294.pv1 import probes
        data=self.original_probe_reader();result=probes.run_minimal512(self.root,sid+'-512',self.instance,data,self.ledger,self.guard)
        need(result['complete']is True and result['new_queries']==12,'WORKER_ALL_ORIGINAL_TWELVE_TRAIN_PROBES')
        del data;gc.collect();language_rows=self.language_draws(sid)
        proof=self.phase(sid,'OFFICIAL_EXPORT')
        target,receipt=export.save(self.root,self.instance,self.allocation,self.ledger,sid,guard=self.guard,phase_proof=proof)
        self.unload();self.instance=export.load(self.root,target,receipt,self.ledger,sid,guard=self.guard)
        case=self.language_plan['cases'][0];payload=language.payload_for_case(case,self.capture,self.reader)
        reloaded=self.draw(sid,'official-reloaded',payload,case['noise_seed'])
        reference=self.root/language_rows[0]['relative_path'];equality=export.compare_saved_draws(reference,reloaded)
        once(self.evidence/sid/'OFFICIAL_INFERENCE_RECONSTRUCTION_VERIFIED.json',dict(export_manifest_sha256=sha(target/'CHECKPOINT_COMPLETE.json'),equality=equality))
        return dict(training=trained,save_metadata_seal=sealed,TRAIN_probes=result,language_probes=language_rows,
            official_inference_manifest_sha256=sha(target/'CHECKPOINT_COMPLETE.json'),official_output_equality=equality)

    def run(self):
        need(self.ledger.guard()['active_session']is None,'WORKER_FRESH_OWN_SESSION_START')
        audit=self.raw_audit()
        once(self.evidence/'RAW_PARENT_COMPLETE_STREAM_AUDIT.json',audit);self.audit=audit
        try:
            for spec in self.ledger.plan():
                sid=spec['id'];self.ledger.start_session(sid);proof=self.phase(sid,'MODEL_RAW');self.load_raw(sid,proof)
                if spec['optimizer_limit']==0:
                    zeros=self.zero_draws(sid);language_rows=self.language_draws(sid)
                    result=dict(zero_step_files=[dict(path=str(p.relative_to(self.root)),sha256=sha(p))for p in zeros],language_probes=language_rows)
                elif not spec['formal_candidate']:result=self.qualification(sid)
                else:result=self.formal(sid)
                self.unload();row=self.ledger.guard()['sessions'][sid]
                need(row['model_loads']==dict(zip([s['id']for s in self.ledger.plan()],self.ledger.load_limits()))[sid],
                     'WORKER_ALL_REAL_RAW_FULL_RESTORE_AND_OFFICIAL_RELOAD_ATTEMPTS')
                receipt=dict(schema='ARX5_MINIMAL_RUNTIME_SESSION_TERMINAL_V1',
                    status='COMPLETE_SAVED_REQUIRES_INDEPENDENT_OWNER_AUDIT',session_id=sid,arm=spec['arm'],
                    request_sha256=self.allocation['request_sha256'],source_sha=self.request['source_sha'],
                    raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,
                    optimizer_updates=row['optimizer_updates'],query_attempts=row['generation_queries'],cursor=spec['sample_visits'],
                    model_load_attempts=row['model_loads'],actual_session_evidence=result,behavior_success_not_measured=True)
                path=self.evidence/sid/'SESSION_TERMINAL.json';once(path,receipt);self.ledger.complete_session(sid,path)
            return self.ledger.summary()
        finally:self.unload()
