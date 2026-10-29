"""Post-F audit of actual new outputs and the original audited saved-E1 input."""
from copy import deepcopy
import json,os,subprocess
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from . import minimal_execution as e,minimal_native_audit as original,minimal_inference_export as export,minimal_probes as language
from . import minimal_saved512_contract as c,minimal_terminal as terminal
from .minimal_saved512_effects import Saved512Effects
from .minimal_saved512_inputs import Inputs

AUDIT='PASS_INDEPENDENT_SAVED_E1_512_INFERENCE_AND_RAW_E2_512_COMPLETE_STATE_AND_OUTPUTS'


def audit(root,request_path,*,native,inventory_sha256):
    root,request_path=Path(root),Path(request_path);r=read(request_path);job=native['job_id']
    e.need(os.getuid()==22182 and os.environ.get('CUDA_VISIBLE_DEVICES')==''and native['job_state']=='F'
        and type(native['Exit_status'])is int,'SAVED512_AUDIT_NATIVE_F_ONLY_NO_CUDA')
    p=subprocess.run(['/opt/pbs/bin/qstat','-x','-f','-F','json',job],capture_output=True,text=True,timeout=25)
    e.need(p.returncode==0 and not p.stderr,'SAVED512_AUDIT_ACTUAL_NATIVE_F_QUERY')
    actual=json.loads(p.stdout)['Jobs'][job]
    e.need(actual==native['raw_native']and actual['job_state']=='F'and actual['Exit_status']==native['Exit_status']
        and actual['Variable_List']['PV1_REQUEST_ID']==r['request_id']
        and actual['Variable_List']['PV1_SUBMISSION_NONCE']==r['submission_nonce'],'SAVED512_AUDIT_EXACT_CURRENT_NATIVE_IDENTITY')
    d=request_path.parent;final=read(d/'PBS_PROCESS_FINAL.json');wrapper=read(d/'PBS_WRAPPER_FINAL.json')
    e.need(final['job_id']==wrapper['job_id']==job and final['request_id']==r['request_id']
        and final['request_sha256']==sha(request_path)and final['source_sha']==r['source_sha']
        and wrapper['returncode']==native['Exit_status'],'SAVED512_AUDIT_ACTUAL_SAME_WORKER_WRAPPER_FINALS')
    ledger=Saved512Effects(root,read(d/'ALLOCATION_ADMISSION.json'),request_path=request_path,guard=lambda:None)
    ledger.check(read(ledger.path));effects=original.match_saved_effects(ledger.summary(),final['effects'])
    base=dict(request_sha256=sha(request_path),source_sha=r['source_sha'],job_id=job,
        effect_summary_sha256=e.sha(e.raw(effects)),inventory_sha256=inventory_sha256,
        raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,
        new_model_loads=0,new_queries=0,new_updates=0,new_physics=0,saved512_result_sha256=c.RESULT,
        saved512_OS_exit_sha256=c.OS_EXIT,E1_new_optimizer_calls=0)
    if native['Exit_status']!=0 or not effects['all_saved512_sessions_runtime_complete']:
        return dict(base,status=terminal.PARTIAL,preserved_session_states=effects['sessions'])
    saved=ledger.saved;saved.verify_current();stream=original.StreamHashes();folder=root/'evidence/MINIMAL_SAVED512_COMPLETION'
    from .minimal_train_session import dataset
    from scripts.pi0294.e1.model import digest
    data=dataset(root,root/'evidence/independent-terminal-reader');capture=read(root/'plans/MINIMAL_SIX_PRECONTACT16_TRAIN_CAPTURE.private.json')
    plan=read(root/'plans/MINIMAL_LANGUAGE24_PANEL.private.json');panel=read(root/'plans/MINIMAL_TWELVE_PANEL.json')
    e.need(plan==language.prepare(capture,panel),'SAVED512_AUDIT_SAME_FROZEN_LANGUAGE_INPUTS')
    payloads={x['id']:language.payload_for_case(x,capture,data)for x in plan['cases']}
    wire=lambda p:digest({k:v for k,v in p.items()if k!='snapshot_uuid'})
    endpoints={};ids=[]
    for spec in c.sessions():
        sid=spec['id'];sf=folder/sid;receipt=read(sf/'SESSION_TERMINAL.json');row=effects['sessions'][sid]
        e.need(stream(sf/'SESSION_TERMINAL.json')==row['terminal_receipt_sha256']and receipt['request_sha256']==sha(request_path)
            and receipt['source_sha']==r['source_sha']and receipt['session_id']==sid
            and receipt['optimizer_updates']==spec['optimizer_limit']and receipt['query_attempts']==spec['query_limit'],
            'SAVED512_AUDIT_OWN_RUNTIME_RECEIPT_AND_EFFECT_DEBITS')
        if spec['inference_finalization_only']:
            evidence=receipt['actual_session_evidence'];cp=deepcopy(saved.result['saved_states'][sid]);oldcp,inf=saved.E1_paths()
            e.need(evidence['original_training_job_id']==c.scope.JOB and evidence['new_optimizer_updates']==0
                and evidence['original_TRAIN_probes_reexecuted']==0 and evidence['original_TRAIN_probes_sha256']==c.E1_TRAIN
                and evidence['saved_own512_restore']['identity']['checkpoint_manifest_sha256']==c.E1_CP
                and evidence['saved_own512_restore']['identity']['global_step']==512
                and evidence['saved_own512_restore']['actual_parameter_identity']==read(oldcp/'MODEL_PARAMETER_STATE_IDENTITY.json'),
                'SAVED512_AUDIT_REAL_IN_MEMORY_OWN512_RESTORE_WITHOUT_OPTIMIZATION')
            official_manifest=c.E1_EXPORT
        else:
            cp=original.checkpoint(root,spec,dict(r,job_id=job),data,stream)
            oldq=saved.science/'training/E2_SHARED_ACTION_HEAD_QUALIFY2/checkpoint-2'
            e.need(cp.pop('frozen_parameter_identity')==read(oldq/'FROZEN_PARAMETER_IDENTITY.json'),
                'SAVED512_AUDIT_E2_FRESH_RAW_FROZEN_BASELINE_NO_QUALIFICATION_INHERITANCE')
            pp=root/'evidence'/(sid+'-512')/'TRAIN_PROBES';probes=read(pp/'RESULT.json');declared=read(root/'plans/PV1_FIXED_TRAIN_PROBES.json')
            e.need(probes['complete']is True and len(probes['records'])==12
                and probes['checkpoint_manifest_sha256']==cp['checkpoint_manifest_sha256']
                and probes['probe_plan_sha256']==sha(root/'plans/PV1_FIXED_TRAIN_PROBES.json')
                and[x['id']for x in probes['records']]==[x['id']for x in declared['windows']]
                and all(x['split']=='TRAIN'for x in probes['records']),'SAVED512_AUDIT_ORIGINAL12_TRAIN_POPULATION')
            stream(pp/'RESULT.json')
            for x in probes['records']:e.need(stream(pp/(x['id']+'.npz'))==x['arrays_sha256'],'SAVED512_AUDIT_TRAIN_DRAW_BYTES')
            inf=root/'inference'/spec['arm'];ordinary=read(inf/'CHECKPOINT_COMPLETE.json');original.own_members(inf,ordinary,stream)
            e.need(ordinary['wrapped_checkpoint_manifest_sha256']==cp['checkpoint_manifest_sha256']and ordinary['global_step']==512,
                'SAVED512_AUDIT_ORDINARY_E2_EXPORT_OF_OWN_FORMAL512')
            official_manifest=stream(inf/'CHECKPOINT_COMPLETE.json')
        recorded=read(sf/'LANGUAGE24_SAVED.json');e.need(len(recorded['records'])==24,'SAVED512_AUDIT_ALL24_LANGUAGE_DRAWS')
        for case,rowp in zip(plan['cases'],recorded['records']):
            e.need(rowp['id']==case['id']and rowp['input_sha256']==case['semantic_input_sha256']
                and rowp['arrays_sha256']==original.draw(sf/'draws'/(case['id']+'.npz'),stream,seed=case['noise_seed'],input_fingerprint=wire(payloads[case['id']])),
                'SAVED512_AUDIT_ACTUAL_LANGUAGE_INPUT_NOISE_OUTPUT')
        first=plan['cases'][0];out=sf/'draws/official-reloaded.npz'
        original.draw(out,stream,seed=first['noise_seed'],input_fingerprint=wire(payloads[first['id']]))
        equality=export.compare_saved_draws(sf/'draws'/(first['id']+'.npz'),out)
        verified=read(sf/'OFFICIAL_INFERENCE_RECONSTRUCTION_VERIFIED.json')
        e.need(verified['export_manifest_sha256']==official_manifest and verified['equality']==equality,
            'SAVED512_AUDIT_ACTUAL_STRICT_SAME_NOISE_RECONSTRUCTION_PAIR')
        cp.update(TRAIN_probes=12,language_probes=24,official_inference_manifest_sha256=official_manifest,
            original_training_job_id=c.scope.JOB if spec['inference_finalization_only']else job,
            current_job_optimizer_updates=spec['optimizer_limit'])
        endpoints[spec['arm']]=cp;ids.append(sid)
    witnesses=stream.witnesses();saved.verify_current()
    return dict(base,status=AUDIT,audited_session_ids=ids,endpoints=endpoints,streamed_witnesses=witnesses,
        saved_E1_full_stream_input_audit_sha256=c.RESULT,old_failed_language_queries_preserved_not_promoted=True,
        same15_zero_step_raw_wrapped_arms_exact=True,two_full_optimizer_scheduler_RNG_sampler_restore_pairs_exact=True,
        original_qualification_evidence_reused_not_reexecuted=True,official_full_model_reconstruction_output_pairs_exact=True,
        frozen_parameters_and_buffers_unchanged=True)
