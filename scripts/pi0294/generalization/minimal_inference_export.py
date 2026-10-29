"""One-way full ordinary GR00T export after an own fixed512 session.

This is a library used by the existing registered worker, not a launcher or a
permission source. It preserves the full wrapped resume checkpoint. Model
reload and the same-input/seed output comparison remain actual charged calls.
"""
import json,os,shutil,time
from pathlib import Path
from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.pbs_owner import publish_release_record as once
from . import minimal_execution as e,minimal_train_session as session
from .minimal_adapter import consume_for_official_inference

SCHEMA='ARX5_MINIMAL_OWN512_ORDINARY_FULL_INFERENCE_EXPORT_V1'
need=e.need


def check_phase(proof,allocation,session_id):
    need(proof.get('status')=='PASS_MINIMAL_SESSION_OFFICIAL_EXPORT_NATIVE'
        and proof.get('session_id')==session_id and proof.get('phase')=='OFFICIAL_EXPORT'
        and proof.get('job_id')==allocation['job_id']
        and proof.get('request_sha256')==allocation['request_sha256']
        and 0<=time.time()-proof.get('observed_epoch',0)<=120,
        'EXPORT_ACTUAL_ORIGINAL_OWNER_SAME_SESSION_FRESH_RELEASE')


def save(root,instance,allocation,ledger,session_id,*,guard,phase_proof):
    """No extra full model copy; consume only the fully saved trained wrapper.

    The caller first releases its Trainer/optimizer references. This function
    does not load a model, call inference, train, remove inputs or replace an
    existing export. All exported weights are ordinary upstream category tables.
    """
    from .minimal_effects import Effects
    root=Path(root);spec=next(s for s in e.native_sessions()if s['id']==session_id)
    need(isinstance(ledger,Effects)and callable(guard)and spec['formal_candidate'],
         'EXPORT_ONLY_TYPED_OWN_FORMAL512_SESSION')
    guard();check_phase(phase_proof,allocation,session_id);value=ledger.guard();row=value['sessions'][session_id]
    need(value['active_session']==session_id and row==dict(row,optimizer_updates=512,
        generation_queries=36,model_loads=1)and value['pending_updates']==value['unknown_effects']==[],
        'EXPORT_AFTER_COMPLETE512_AND_ORIGINAL36_PROBES_NOT_QUALIFICATION')
    identity=instance.identity;contract=instance.parameter_contract
    need(identity.get('schema')=='ARX5_MINIMAL_ACTUAL_WRAPPED_CHECKPOINT_IDENTITY_V1'
        and identity.get('global_step')==512 and identity.get('mode')==spec['arm']
        and identity.get('qualification_only')is False and identity.get('statistics_sha256')==e.R0
        and identity.get('raw_parent_files_identity')==e.training.RAW_PARENT_FILES_ID
        and contract.arm==spec['arm'],'EXPORT_EXACT_CURRENT_WRAPPED_FORMAL_STATE')
    full=root/'training'/session_id/'checkpoint-512';before=sha(full/'CHECKPOINT_COMPLETE.json')
    need(before==identity['checkpoint_manifest_sha256']and read(full/'CHECKPOINT_COMPLETE.json')['full_resume']is True,
         'EXPORT_KEEP_COMPLETE_OWN_OPTIMIZER_SCHEDULER_RNG_SAMPLER')
    session.save_headroom(root,spec['arm'],next_upper_bytes=14000000000,guard=guard)
    target=root/'inference'/spec['arm'];need(not target.exists()and not target.is_symlink(),'EXPORT_NO_OVERWRITE_OR_REPLAY')
    evidence=root/'evidence'/session_id;need(not evidence.is_symlink(),'EXPORT_OWN_EVIDENCE_DIRECTORY');evidence.mkdir(parents=True,exist_ok=True,mode=0o700)
    once(evidence/'OFFICIAL_EXPORT_INTENT.json',dict(schema=SCHEMA,
        request_sha256=allocation['request_sha256'],session_id=session_id,
        wrapped_checkpoint_manifest_sha256=before,phase_proof_sha256=e.sha(e.raw(phase_proof)),
        observed_epoch=time.time(),automatic_retry=False))
    target.mkdir(parents=True,mode=0o700);model=consume_for_official_inference(contract)
    instance.parameter_contract=None
    model.save_pretrained(target,safe_serialization=True,max_shard_size='5GB')
    instance.processor.save_pretrained(target/'processor')
    statistics=Path(instance.bindings['dataset'])/'statistics.json'
    need(sha(statistics)==e.R0,'EXPORT_ORIGINAL_R0_BYTES');shutil.copyfile(statistics,target/'statistics.json')
    once(target/'FULL_SOURCE_CHECKPOINT_REFERENCE.json',dict(schema=SCHEMA,arm=spec['arm'],global_step=512,
        cursor=16384,scheduler_horizon=1024,raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,
        statistics_sha256=e.R0,source_sha=ledger.request['source_sha'],
        wrapped_checkpoint_manifest_sha256=before,full_resume_checkpoint_retained=True,
        ordinary_upstream_full_model=True,adapter_only_file=False))
    files={}
    for path in sorted(target.rglob('*')):
        guard();need(not path.is_symlink(),'EXPORT_NO_INPUT_ALIAS_OR_HIDDEN_TENSOR_LINK')
        if path.is_file():files[path.relative_to(target).as_posix()]=sha(path)
    need({'config.json','statistics.json','FULL_SOURCE_CHECKPOINT_REFERENCE.json'}<=set(files)
        and any(n.endswith('.safetensors')for n in files)and any(n.startswith('processor/')for n in files),
        'EXPORT_FULL_MODEL_CONFIG_PROCESSOR_AND_R0')
    receipt=dict(schema=SCHEMA,status='ACTUAL_FULL_OFFICIAL_MODEL_SAVED_RELOAD_AND_OUTPUT_CHECK_PENDING',
        arm=spec['arm'],global_step=512,cursor=16384,scheduler_horizon=1024,files=files,
        wrapped_checkpoint_manifest_sha256=before,statistics_sha256=e.R0,
        raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,full_resume=False,
        original_full_resume_retained=True,query_comparison_complete=False)
    once(target/'CHECKPOINT_COMPLETE.json',receipt)
    for path in target.rglob('*'):
        if path.is_file():path.chmod(0o400)
    need(sha(full/'CHECKPOINT_COMPLETE.json')==before,'EXPORT_DID_NOT_CHANGE_FULL_RESUME_CHECKPOINT')
    guard();return target,receipt


def load(root,target,receipt,ledger,session_id,*,guard):
    """Charge one actual ordinary reload; no raw/qualification checkpoint alias."""
    from scripts.pi0294.e1 import model as original
    root,target=Path(root),Path(target);guard();spec=next(s for s in e.native_sessions()if s['id']==session_id)
    need(spec['formal_candidate']and target==root/'inference'/spec['arm']
        and receipt==read(target/'CHECKPOINT_COMPLETE.json')and receipt['schema']==SCHEMA
        and receipt['arm']==spec['arm']and receipt['global_step']==512 and not receipt['full_resume'],
        'EXPORT_RELOAD_ONLY_OWN_IMMUTABLE_ORDINARY_FULL_MODEL')
    for name,pin in receipt['files'].items():
        path=target/name;need(path.is_file()and not path.is_symlink()
            and not path.stat().st_mode&0o222 and sha(path)==pin,'EXPORT_RELOAD_FULL_MEMBER:'+name)
    once(root/'evidence'/session_id/'OFFICIAL_RELOAD_INTENT.json',dict(session_id=session_id,
        export_manifest_sha256=sha(target/'CHECKPOINT_COMPLETE.json'),observed_epoch=time.time(),automatic_retry=False))
    ledger.charge('model_loads',1)
    identity=dict(schema='ARX5_MINIMAL_ACTUAL_ORDINARY_FULL_INFERENCE_IDENTITY_V1',mode=spec['arm'],
        global_step=512,checkpoint_manifest_sha256=sha(target/'CHECKPOINT_COMPLETE.json'),
        files=receipt['files'],statistics_sha256=e.R0,raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,
        full_resume=False,official_inference=True)
    instance=original.load(root,target,identity,trainable=False)
    config=instance.model.config
    need(config.action_horizon==40 and config.max_action_dim==132 and config.max_state_dim==132
        and config.max_num_embodiments==32 and instance.model.action_head.num_inference_timesteps==4
        and instance.model.action_head.state_dropout_prob==0,'EXPORT_ORIGINAL_H40_PAD132_CATEGORY32_AND_FOUR_STEP_GRAPH')
    instance.parameter_contract=None;guard();return instance


def compare_saved_draws(reference,current):
    """Compare actual charged outputs, not model labels or wrapper flags."""
    import numpy as np
    reference,current=Path(reference),Path(current)
    need(reference!=current and reference.is_file()and current.is_file()
        and not reference.is_symlink()and not current.is_symlink(),'EXPORT_TWO_DISTINCT_ACTUAL_DRAW_FILES')
    with np.load(reference,allow_pickle=False)as left,np.load(current,allow_pickle=False)as right:
        def exact(keys):
            return all(k in left and k in right and left[k].dtype==right[k].dtype
                and left[k].shape==right[k].shape and np.array_equal(left[k],right[k])for k in keys)
        # Reject a different actual noise draw before attributing output
        # inequality to reconstructed weights. Seed labels alone are weaker.
        inputs=('noise_seed','noise_fingerprint_json','observation_fingerprint')
        need(exact(inputs),
            'EXPORT_ACTUAL_SAME_INPUT_AND_NOISE_REQUIRED_BEFORE_OUTPUT_COMPARISON')
        need(exact(('physical_H40','normalized')),
            'EXPORT_ACTUAL_FULL_MODEL_SAME_INPUT_NOISE_OUTPUT_MUST_BE_EXACT')
        need(left['physical_H40'].shape==(40,10)and left['normalized'].shape==(1,40,132)
            and np.isfinite(left['physical_H40']).all()and np.isfinite(left['normalized']).all(),
            'EXPORT_FULL_ORIGINAL_OUTPUT_SHAPES_FINITE')
    return dict(status='ACTUAL_SAVED_SAME_INPUT_NOISE_OUTPUTS_EXACT',reference_sha256=sha(reference),
        reloaded_sha256=sha(current),new_queries=0,behavior_success_not_measured=True)
