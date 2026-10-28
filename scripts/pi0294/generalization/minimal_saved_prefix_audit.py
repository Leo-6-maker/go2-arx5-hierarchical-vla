"""Bounded registered CPU audit of the saved1504095 prefix; no neural calls.

Original7c checkpoint/optimizer/RNG/draw readers supply the scientific checks.
The new role writes only its own immutable evidence and progress files. It
does not resume a Trainer, fabricate a completed session, or grant a suffix.
"""
import argparse,hashlib,json,os,re,resource,sys,time,traceback
from pathlib import Path

SCHEMA='ARX5_FIRST512_REGISTERED_SAVED_PREFIX_CPU_AUDIT_V1'
GRANT='c2733caa01de7a85419d9fd46f2e315f8bea89d892e43b6b1945f6af319bc20c'
FIELD='ARX5_first512_saved_prefix_one_completion_authority'
SOURCE='7c7e77d4b83a7b3a30f336a6f53f4788942c2373'
REQUEST='a6b3936e-f273-4491-b43b-b8168a6d0f3b'
JOB='1504095.stdct-mgmt-02'
MANIFEST='08237bb3ce533cb833a348c6cddcc1e53c7e2950e7d0a5a7a21d8f62e9d559a0'
REQUEST_SHA='d90e62a0b7a55fc620ecd666ae9a626a33e68ea67eacfde873a5db34b9196832'
FINAL='7025d34565df67c4eff1312ab5cf0ee0c745107e85442a72a8fad0ec8830682c'
WRAPPER='9cd33de39b544a232f28f29b81b468509b3484ba77335034fb81e8affae0525e'
raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())


def need(value,message):
    if not value:raise PermissionError('SAVED_PREFIX_AUDIT_'+message)


def validate_plan(p):
    need(p['schema']==SCHEMA and p['scope_grant_sha256']==GRANT
        and p['science_source_sha']==SOURCE and p['science_manifest_sha256']==MANIFEST
        and p['request_id']==REQUEST and p['request_sha256']==REQUEST_SHA and p['job_id']==JOB,
        'EXACT_ORIGINAL_NONZERO_PREFIX_NOT_ANOTHER_CHECKPOINT')
    need(p['CPU_upper_s']==480 and p['walltime_upper_s']==600
        and p['new_model_loads']==p['new_queries']==p['new_updates']==p['new_physics']==0
        and p['formal_candidate']is False and p['completed_session_reexecution']is False,
        'ONLY_BOUNDED_SAVED_STATE_READER')
    need(re.fullmatch('[0-9a-f]{40}',p['source_sha'])is not None and p['source_sha']!=SOURCE
        and re.fullmatch('[0-9a-f]{64}',p['auditor_sha256'])is not None
        and all(Path(p[k]).is_absolute()for k in('metadata_root','science_root','registry','lease'))
        and p['metadata_root']!=p['science_root']and p['role']and p['run_uuid'],
        'NEW_REGISTERED_PRIVATE_METADATA_ROLE_NOT_REWRITTEN_OLD_ROOT')
    need(set(p['old_processes'])=={'owner','wrapper','prior_reader','prior_reader_wrapper'}
        and all(type(v['pid'])is type(v['procstart'])is int and v['pid']>0 and v['procstart']>0
                for v in p['old_processes'].values()),'EXACT_ALL_PREVIOUS_NATIVE_ACTORS')
    return p


def run(root):
    root=Path(root);manifest_bytes=(root/'RUN_MANIFEST.json').read_bytes();manifest=json.loads(manifest_bytes)
    p=validate_plan(manifest['plan']);need(root==Path(p['metadata_root']),'CANONICAL_OWN_OUTPUT_ROOT')
    need(os.getuid()==22182 and os.uname().nodename.startswith('stdct-login-')
        and not os.environ.get('PBS_JOBID')and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
        'ACTUAL_LOGIN_CPU_ROLE_NO_CUDA_OR_ALLOCATION')
    need(sha(Path(__file__).read_bytes())==p['auditor_sha256'],'EXACT_PUBLISHED_CPU_AUDITOR')
    binding=read(root/'OUTPUT_BINDING.private.json')
    need(binding['role']==p['role']and binding['run_uuid']==p['run_uuid']
        and binding['run_root']==str(root)and binding['lease']==p['lease']
        and binding['registry']==p['registry'],'EXACT_REGISTERED_BINDING')
    rows=read(p['registry'])['entries']
    need(sum(v.get('role')==p['role']and v.get('path')==str(root)and v.get('run_uuid')==p['run_uuid']for v in rows)==1,
        'ACTUAL_CANONICAL_ROLE_REGISTRATION')
    started=time.monotonic();cpu=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(479,480))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
    sequence=0
    def put(name,value):
        fd=os.open(root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
        with os.fdopen(fd,'wb')as f:f.write(raw(value));f.flush();os.fsync(f.fileno())
    def progress(stage,**values):
        nonlocal sequence
        sequence+=1;put('PROGRESS_'+str(sequence).zfill(4)+'.json',dict(stage=stage,
            elapsed_s=time.monotonic()-started,CPU_s=time.process_time()-cpu,**values))
    def guard():
        need(time.monotonic()-started<600 and time.time()<manifest['deadline_epoch'],'ORIGINAL_SCOPE_AND_FINITE_CPU_WALLTIME')
        lease=read(p['lease'])
        need(lease['status']=='ACTIVE'and lease['run_uuid']==p['run_uuid']
            and lease['manifest_sha256']==sha(manifest_bytes)and not(root/'STOP.json').exists(),
            'OWN_ACTIVE_CPU_LEASE_AND_STOP')
    try:
        guard();science=Path(p['science_root']);sm=read(science/'RUN_MANIFEST.json')
        need(sha((science/'RUN_MANIFEST.json').read_bytes())==MANIFEST
            and sm['code']['vla_commit']==SOURCE,'OLD_SEALED_SOURCE_MANIFEST_UNCHANGED')
        code=science/sm['code']['capsule_relative'];seal_bytes=(science/'SOURCE_SEAL.json').read_bytes();seal=json.loads(seal_bytes)
        need(sha(seal_bytes)==sm['code']['source_seal_sha256']and seal['source_sha']==SOURCE,
             'ACTUAL_ORIGINAL_READER_SOURCE_SEAL')
        for name,pin in seal['files'].items():
            path=code/name;need(not path.is_symlink()and sha(path.read_bytes())==pin,'SEALED_READER_SOURCE_FILE:'+name)
        progress('ALL_ORIGINAL_SCIENCE_READER_SOURCE_FILES_REHASHED',source_files=len(seal['files']))
        sys.path[:0]=[str(code),str(code/'src')]
        # Disable only the CPU observer's package update check, not training.
        os.environ['NO_ALBUMENTATIONS_UPDATE']='1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1);need(not torch.cuda.is_initialized(),'NO_CUDA_CONTEXT')
        from scripts.pi0294.generalization import minimal_native_audit as audit,minimal_execution as e
        from scripts.pi0294.generalization import minimal_model as model,minimal_probes as language,minimal_inference_export as export
        from scripts.pi0294.generalization.minimal_train_session import dataset
        from scripts.pi0294.generalization.minimal_effects import Effects
        from scripts.pi0294.dec1.readonly_checkpoint_hash import StreamHashes
        from scripts.pi0294.e1.model import digest
        gb=(science.parent.parent/'HOPPER_RESOURCE_LEDGER.json').read_bytes();g=json.loads(gb);m=g[FIELD]
        need(m['grant_sha256']==GRANT and m['native_admission']is False
            and m['phase']=='CONTROL_AND_SAVED_PREFIX_AUDIT_ONLY'
            and m['control_paid_once']['Hopper_CPU_s']==3600,'REAL_ONCE_PAID_CPU_CONTROL_SCOPE')
        old=read(science/'control/registered-dispatch-owner01/SOURCE_REQUEST.private.json');allocation=Path(old['directory'])
        request=read(allocation/'REQUEST.json');need(sha((allocation/'REQUEST.json').read_bytes())==REQUEST_SHA,'ORIGINAL_REQUEST_BYTES')
        need(sha((allocation/'PBS_PROCESS_FINAL.json').read_bytes())==FINAL
            and sha((allocation/'PBS_WRAPPER_FINAL.json').read_bytes())==WRAPPER,'KNOWN_NATIVE_WORKER_WRAPPER_TERMINAL')
        for root_name in('science','metadata'):
            origin=science if root_name=='science'else science/'control/registered-dispatch-owner01'
            need(read(read(origin/'OUTPUT_BINDING.private.json')['lease'])['status']=='CLOSED','ORIGINAL_'+root_name.upper()+'_LEASE_CLOSED')
        for actor in p['old_processes'].values():
            f=Path('/proc',str(actor['pid']),'stat');fields=f.read_text().rsplit(')',1)[1].split()if f.exists()else None
            need(not(fields and fields[0]not in('Z','X')and int(fields[19])==actor['procstart']),'NO_PRIOR_LIVE_PROCSTART')
        final=read(allocation/'PBS_PROCESS_FINAL.json');native_admission=read(allocation/'ALLOCATION_ADMISSION.json')
        ledger=Effects(science,native_admission,request_path=allocation/'REQUEST.json',guard=lambda:None)
        effects=audit.match_saved_effects(ledger.summary(),final['effects'])
        need((effects['optimizer_updates_committed'],effects['query_attempts'],effects['model_load_attempts'])==(2,71,5),
             'KNOWN_ACTUAL_PARTIAL_EFFECTS_NO_NEW_SESSION_OR_REPLAY')
        progress('ORIGINAL_F1_CLOSED_SOURCE_AND_EFFECTS_VERIFIED')
        data=dataset(science,root/'dataset-reader');progress('ORIGINAL_TWELVE_PARENT_READER_READY')
        folder=science/'evidence/MINIMAL_FIVE_SESSIONS'
        capture=read(science/'plans/MINIMAL_SIX_PRECONTACT16_TRAIN_CAPTURE.private.json')
        plan=read(science/'plans/MINIMAL_LANGUAGE24_PANEL.private.json');panel=read(science/'plans/MINIMAL_TWELVE_PANEL.json')
        need(plan==language.prepare(capture,panel),'EXACT_ORIGINAL_TRAIN_LANGUAGE_INPUTS')
        payloads={c['id']:language.payload_for_case(c,capture,data)for c in plan['cases']}
        fingerprint=lambda value:digest({k:v for k,v in value.items()if k!='snapshot_uuid'})
        first=fingerprint(payloads[plan['cases'][0]['id']])
        class ProgressStream(StreamHashes):
            def __call__(self,path):
                guard();value=super().__call__(path)
                progress('OWN_IMMUTABLE_FILE_STREAMED',path=str(path),sha256=value,bytes_read=self.bytes_read)
                return value
        stream=ProgressStream();pairs=[]
        for spec in e.native_sessions()[:3]:
            sf=folder/spec['id']
            for i in range(15):
                path=sf/'draws'/('zero-'+str(i).zfill(2)+'.npz')
                audit.draw(path,stream,seed=29400+i,input_fingerprint=first)
                if spec['optimizer_limit']:pairs.append(export.compare_saved_draws(folder/e.native_sessions()[0]['id']/'draws'/path.name,path))
        e0=read(folder/'E0_RAW_DIAGNOSTIC/SESSION_TERMINAL.json');language_rows=e0['actual_session_evidence']['language_probes']
        need(len(language_rows)==len(plan['cases'])==24,'ACTUAL_E0_ALL24_LANGUAGE_OUTPUTS')
        for case,row in zip(plan['cases'],language_rows):
            path=science/row['relative_path'];need(row['id']==case['id']
                and row['arrays_sha256']==audit.draw(path,stream,seed=case['noise_seed'],input_fingerprint=fingerprint(payloads[case['id']])),
                'ACTUAL_MATCHED_E0_LANGUAGE_DRAW_BYTES')
        progress('ALL69_SAVED_ZERO_AND_LANGUAGE_DRAWS_AND30_EQUALITY_PAIRS_VERIFIED')
        spec=e.native_sessions()[1];sf=folder/spec['id'];receipt=read(sf/'SESSION_TERMINAL.json')
        need(stream(sf/'SESSION_TERMINAL.json')==effects['sessions'][spec['id']]['terminal_receipt_sha256']
            and receipt['request_sha256']==REQUEST_SHA and receipt['source_sha']==SOURCE
            and receipt['optimizer_updates']==2 and receipt['query_attempts']==17,'REAL_E1_COMPLETE_NATIVE_SESSION_RECEIPT')
        checkpoint=audit.checkpoint(science,spec,dict(request,job_id=JOB),data,stream)
        put('E1_CHECKPOINT_STATE_FIRST_PASS.json',checkpoint);progress('E1_FULL_STATE_AND_ACTUAL_SERIALIZED_OPTIMIZER_SCHEDULER_RNG_VERIFIED')
        restore=receipt['actual_session_evidence']['restore']
        need(restore['status']=='PASS_ORIGINAL_TRAINER_OWN_QUALIFICATION_FULL_STATE_RESTORE_ONLY'
            and restore['optimizer_scheduler_model_RNG_sampler_checked']is True and restore['step']==2
            and restore['cursor']==64 and restore['new_optimizer_updates']==restore['new_queries']==0,
            'REAL_NATIVE_FULL_TRAINER_SAVE_AND_RESTORE_RECEIPT')
        for name in('step2-reference','step2-restored'):
            audit.draw(sf/'draws'/(name+'.npz'),stream,seed=29400,input_fingerprint=first)
        restored=export.compare_saved_draws(sf/'draws/step2-reference.npz',sf/'draws/step2-restored.npz')
        guard();progress('INDEPENDENT_SECOND_STREAM_OF_EVERY_CHECKPOINT_MEMBER_STARTED')
        witnesses=stream.witnesses();put('FULL_STREAMED_PREFIX_WITNESSES.json',witnesses)
        progress('EVERY_CHECKPOINT_MEMBER_AND_OUTPUT_INDEPENDENT_SECOND_STREAM_COMPLETE')
        raw_plan=read(science/'plans/MINIMAL_RAW_PARENT_AUDIT_PLAN.private.json')
        raw_audit=model.audit_raw_parent(raw_plan['checkpoint'],guard=guard)
        put('POST_F_LOGIN_RAW_PARENT_FULL_STREAM_AUDIT.json',raw_audit)
        model.unchanged_raw_parent(raw_plan['checkpoint'],raw_audit)
        guard();audit.match_saved_effects(ledger.summary(),effects)
        need(not torch.cuda.is_initialized(),'NO_CUDA_DURING_FULL_SAVED_STATE_AUDIT')
        result=dict(schema=SCHEMA,status='PASS_INDEPENDENT_ACTUAL_E0_E1_COMPLETE_SAVED_PREFIX_AND_E2_ZERO_OUTPUTS',
            observed_epoch=time.time(),job_id=JOB,request_id=REQUEST,science_source_sha=SOURCE,
            saved_E1_checkpoint=checkpoint,original_E1_terminal_sha256=sha((sf/'SESSION_TERMINAL.json').read_bytes()),
            saved_E0_terminal_sha256=sha((folder/'E0_RAW_DIAGNOSTIC/SESSION_TERMINAL.json').read_bytes()),
            actual_E1_restore=restore,restored_pair=restored,matched_zero_pairs=len(pairs),
            full_streamed_witnesses_sha256=sha((root/'FULL_STREAMED_PREFIX_WITNESSES.json').read_bytes()),
            raw_parent_post_F_audit_sha256=sha((root/'POST_F_LOGIN_RAW_PARENT_FULL_STREAM_AUDIT.json').read_bytes()),
            raw_parent_revision=raw_audit['revision'],bytes_streamed=witnesses['bytes_streamed'],
            raw_parent_stream_bytes=sum(x['stat']['bytes']for x in raw_audit['files'].values()),
            completed_prefix=['E0_RAW_DIAGNOSTIC','E1_ADAPTER_ONLY_QUALIFY2'],E2_saved_zero_queries=15,
            E2_optimizer_calls=0,formal_E1_E2_steps=[0,0],formal_initialization_from_qualification=False,
            original_effect_summary_sha256=e.sha(e.raw(effects)),new_model_loads=0,new_queries=0,new_updates=0,
            new_physics=0,CUDA_initialized=False,CPU_actual_s=time.process_time()-cpu,wall_seconds=time.monotonic()-started,
            completion_native_admission=False)
        put('RESULT.json',result);return result
    except BaseException as error:
        put('FAILED_PRESERVED.json',dict(error_type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc()[-6000:],CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-started,
            completed_E0_E1_not_reexecuted=True,new_model_loads=0,new_queries=0,new_updates=0,new_physics=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args();run(args.root)
