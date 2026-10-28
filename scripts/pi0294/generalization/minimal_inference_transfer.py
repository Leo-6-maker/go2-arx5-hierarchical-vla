"""Closed first512 ordinary-model delivery through the existing HF format.

Pure preparation/restoration only. Neither a runtime COMPLETE flag nor a
qualification checkpoint is a deliverable. No network, model, simulator,
ledger transition or permission is implemented here. The actual original
owner's full audit, settlement, own lease close and natural OS0 precede use.
"""
import base64
import json
import math
import re
from pathlib import PurePosixPath

from scripts.pi0294.pv1 import evaluation_transfer as original
from . import minimal_execution as e, minimal_inference_export as export
from . import minimal_remaining_execution as remaining

SCHEMA='ARX5_MINIMAL_CLOSED_OWN512_MODEL_ONLY_TRANSFER_V1'
RESTORE='ARX5_MINIMAL_CLOSED_OWN512_PRIVATE_RESTORATION_V1'
SMALL=original.SMALL+('FULL_SOURCE_CHECKPOINT_REFERENCE.json',)
need=e.need


def ordinary_members(complete,index,reference,*,arm,endpoint):
    """An ordinary exported full model, bound to its separately saved state."""
    need(arm in e.training.ARMS,'TRANSFER_EXPLICIT_MINIMAL_ARM')
    need(complete.get('schema')==export.SCHEMA and complete.get('arm')==arm
        and type(complete.get('global_step'))is int and complete['global_step']==512
        and complete.get('cursor')==16384 and complete.get('scheduler_horizon')==1024
        and complete.get('full_resume')is False and complete.get('original_full_resume_retained')is True
        and complete.get('statistics_sha256')==e.R0
        and complete.get('raw_parent_files_identity')==e.training.RAW_PARENT_FILES_ID,
        'TRANSFER_OWN512_OFFICIAL_EXPORT_NOT_WRAPPED_OR_QUALIFICATION')
    need(endpoint.get('global_step')==512 and endpoint.get('cursor')==16384
        and endpoint.get('scheduler_horizon')==1024 and endpoint.get('qualification_only')is False
        and endpoint.get('full_resume')is True and endpoint.get('TRAIN_probes')==12
        and endpoint.get('language_probes')==24 and endpoint.get('statistics_sha256')==e.R0
        and endpoint.get('raw_parent_files_identity')==e.training.RAW_PARENT_FILES_ID
        and complete.get('wrapped_checkpoint_manifest_sha256')==endpoint.get('checkpoint_manifest_sha256'),
        'TRANSFER_EXACT_AUDITED_FULL_STATE_AND_ORIGINAL_PROBES')
    need(reference.get('schema')==export.SCHEMA and reference.get('arm')==arm
        and reference.get('global_step')==512 and reference.get('cursor')==16384
        and reference.get('scheduler_horizon')==1024
        and reference.get('wrapped_checkpoint_manifest_sha256')==endpoint['checkpoint_manifest_sha256']
        and reference.get('statistics_sha256')==e.R0
        and reference.get('raw_parent_files_identity')==e.training.RAW_PARENT_FILES_ID
        and reference.get('full_resume_checkpoint_retained')is True
        and reference.get('ordinary_upstream_full_model')is True
        and reference.get('adapter_only_file')is False,'TRANSFER_SOURCE_REFERENCE_NOT_OTHER_ARM')
    files=complete.get('files',{})
    need(type(files)is dict and all(type(n)is str and n==PurePosixPath(n).as_posix()
        and not PurePosixPath(n).is_absolute()and '..'not in PurePosixPath(n).parts
        and '\\'not in n and type(h)is str and re.fullmatch('[0-9a-f]{64}',h)
        for n,h in files.items()),'TRANSFER_SAFE_EXACT_MEMBER_NAMES_AND_SHA')
    shards={n for n in files if n.endswith('.safetensors')};weights=index.get('weight_map')
    need(shards and type(weights)is dict and weights
        and all(type(k)is str and k and type(v)is str for k,v in weights.items())
        and set(weights.values())==shards
        and all(re.fullmatch(r'model-[0-9]{5}-of-[0-9]{5}\.safetensors',n)for n in shards)
        and set(files)==set(SMALL)|shards and files['statistics.json']==e.R0
        and files['processor/statistics.json']==e.R0,'TRANSFER_ONLY_FULL_INFERENCE_NO_OPTIMIZER_RNG_OR_DATA')
    return {n:files[n]for n in sorted(files)}


def closed_endpoint(global_bytes,lease_bytes,settled_bytes,owner_observation,*,arm):
    need(all(type(x)is bytes for x in(global_bytes,lease_bytes,settled_bytes)),
        'TRANSFER_ACTUAL_CLOSED_GLOBAL_LEASE_AND_SETTLEMENT_BYTES')
    g=json.loads(global_bytes);m=g.get(remaining.FIELD,{})
    rows=[r for r in g['PBS_submissions']if r.get('platform_training_version')==remaining.VERSION]
    need(len(rows)==1 and rows[0].get('status')==e.STATES[2]
        and rows[0].get('scientific_endpoint_complete')is True and m.get('status')=='COMPLETE'
        and g['pending_updates']==g['unknown_effects']==[],
        'TRANSFER_RUNNING_PARTIAL_OR_RUNTIME_ONLY_IS_NOT_CLOSED')
    remaining.validate(g)  # Original full history/physical/terminal consumer.
    row=rows[0];t=m['terminal'];audit=t['audit'];settled=json.loads(settled_bytes);lease=json.loads(lease_bytes)
    need(all(settled[k]==t[k]for k in('native','effects','audit'))
        and lease.get('status')=='CLOSED'and lease.get('run_uuid')==row['run_uuid']
        and lease.get('manifest_sha256')==row['run_manifest_sha256']
        and lease.get('terminal_receipt_sha256')==e.sha(settled_bytes),
        'TRANSFER_ORIGINAL_ONCE_SETTLED_AND_OWN_CLOSED_LEASE')
    obs=owner_observation;os_final=obs.get('OS_EXIT.json',{});process=obs.get('CONTINUATION_PROCESS.json',{})
    observed_effects=obs.get('effects',{})
    need(obs.get('GLOBAL_sha256')==e.sha(global_bytes)
        and obs.get('same_wrapper_live')is False and obs.get('same_owner_live')is False
        and all(observed_effects.get(k)==row[k]for k in('request_id','request_sha256','job_id','source_sha'))
        and process.get('science_source_sha')==row['source_sha']
        and process.get('controller_source_sha')==os_final.get('source_sha')
        and type(os_final.get('source_sha'))is str and re.fullmatch('[0-9a-f]{40}',os_final['source_sha'])
        and type(process.get('pid'))is int and process['pid']>0
        and str(process.get('procstart','')).isdigit()
        and type(os_final.get('returncode'))is int and os_final['returncode']==0
        and os_final.get('foreground_wait_completed')is True
        and type(os_final.get('finished_epoch'))in(int,float)
        and math.isfinite(os_final['finished_epoch'])
        and type(lease.get('closed_epoch'))in(int,float)and math.isfinite(lease['closed_epoch'])
        and os_final['finished_epoch']>=lease['closed_epoch']>=row['settled_epoch']
        and type(obs.get('epoch'))in(int,float)and math.isfinite(obs['epoch'])
        and obs['epoch']>=os_final['finished_epoch'],
        'TRANSFER_ACTUAL_ORIGINAL_OWNER_NATURAL_OS0_NOT_TIMEOUT')
    need(arm in e.training.ARMS and arm in audit['endpoints'],
        'TRANSFER_ORIGINAL_MATCHED512_ARM')
    return dict(GLOBAL_sha256=e.sha(global_bytes),request_id=row['request_id'],
        job_id=row['job_id'],source_sha=row['source_sha'],run_manifest_sha256=row['run_manifest_sha256'],
        arm=arm,endpoint=audit['endpoints'][arm],audit=audit,
        lease_sha256=e.sha(lease_bytes),settled_sha256=e.sha(settled_bytes))


def prepare(global_bytes,lease_bytes,settled_bytes,owner_observation,small_bytes,*,arm):
    proof=closed_endpoint(global_bytes,lease_bytes,settled_bytes,owner_observation,arm=arm)
    need(type(small_bytes)is bytes,'TRANSFER_ACTUAL_SMALL_METADATA_BYTES');small=json.loads(small_bytes)
    need(small.get('schema')=='ARX5_MINIMAL_CLOSED512_INFERENCE_SMALL_READBACK_V1'
        and small.get('GLOBAL_sha256')==proof['GLOBAL_sha256']and small.get('arm')==arm
        and small.get('tensor_payloads_read')is False,'TRANSFER_SAME_CLOSED_ENDPOINT_SMALL_READ')
    root=PurePosixPath(small['official_root'])
    need(root.is_absolute()and '..'not in root.parts and root.parts[-2:]==('inference',arm),
        'TRANSFER_EXACT_AUDITED_OWN_ORDINARY_EXPORT_ROOT')
    readback=small.get('files',{});need(set(readback)==set(SMALL)|{'CHECKPOINT_COMPLETE.json'},
        'TRANSFER_ALL_AND_ONLY_SMALL_METADATA')
    bodies={}
    for name,row in readback.items():
        b=base64.b64decode(row['base64'],validate=True)
        need(type(row['bytes'])is int and 0<len(b)==row['bytes']<2*1024**2
            and e.sha(b)==row['sha256'],'TRANSFER_FULL_SMALL_BYTES:'+name);bodies[name]=b
    complete=json.loads(bodies['CHECKPOINT_COMPLETE.json']);cp=proof['endpoint']
    need(e.sha(bodies['CHECKPOINT_COMPLETE.json'])==cp['official_inference_manifest_sha256'],
        'TRANSFER_OFFICIAL_EXPORT_WAS_INDEPENDENTLY_AUDITED')
    reference=json.loads(bodies['FULL_SOURCE_CHECKPOINT_REFERENCE.json'])
    need(reference.get('source_sha')==proof['source_sha'],'TRANSFER_EXACT_TRAINING_SOURCE')
    members=ordinary_members(complete,json.loads(bodies['model.safetensors.index.json']),reference,arm=arm,endpoint=cp)
    witness=proof['audit']['streamed_witnesses']
    need(witness.get('all_paths_rechecked')is True and witness.get('all_unique_inodes_independently_rehashed')is True,
        'TRANSFER_ORIGINAL_FULL_STREAM_AND_FINAL_INDEPENDENT_REHASH')
    rows={}
    for row in witness['paths']:
        p=PurePosixPath(row['path'])
        if root not in p.parents:continue
        name=p.relative_to(root).as_posix();need(name not in rows,'TRANSFER_NO_DUPLICATE_WITNESS');rows[name]=row
    need(set(rows)==set(members)|{'CHECKPOINT_COMPLETE.json'}
        and all(rows[n]['sha256']==h for n,h in dict(members,**{'CHECKPOINT_COMPLETE.json':cp['official_inference_manifest_sha256']}).items())
        and all(type(r['bytes'])is int and r['bytes']>0 for r in rows.values()),
        'TRANSFER_ALL_ORDINARY_EXPORT_MEMBERS_PHYSICALLY_AUDITED')
    need(all(e.sha(b)==rows[n]['sha256']and len(b)==rows[n]['bytes']for n,b in bodies.items()),
        'TRANSFER_SMALL_METADATA_EQUALS_INDEPENDENT_STREAM')
    public_files={};rewritten={};restoration={}
    for name,digest in members.items():
        if name in original.REWRITTEN:
            b=original.encode(original.public_config(json.loads(bodies[name]),processor=name.startswith('processor/')))
            rewritten[name]=base64.b64encode(b).decode();restoration[name]=base64.b64encode(bodies[name]).decode()
            public_files[name]=dict(bytes=len(b),sha256=e.sha(b),kind='DECLARED_COSMOS_PATH_REBIND')
        else:
            if name in bodies:original.reject_private(json.loads(bodies[name]))
            public_files[name]=dict(bytes=rows[name]['bytes'],sha256=digest,kind='UNCHANGED_ORIGINAL_BYTES')
    public=dict(schema=SCHEMA,candidate=arm,checkpoint_step=512,cursor=16384,scheduler_horizon=1024,
        official_inference_manifest_sha256=cp['official_inference_manifest_sha256'],
        wrapped_checkpoint_manifest_sha256=cp['checkpoint_manifest_sha256'],statistics_sha256=e.R0,
        files=public_files,payload_bytes=sum(r['bytes']for r in public_files.values()),
        tensor_bytes=sum(r['bytes']for n,r in public_files.items()if n.endswith('.safetensors')),
        full_resume_payload=False,original_complete_state_retained_at_source=True,numerical_tensor_changes=0,
        source_tensor_streaming_verified_this_plan=False,uploaded=False,downloaded=False,native_admission=False,
        model_loaded=False,physics_authorized=False,new_training_authorized=False,
        publication_requirements=['registered_source_destination_and_full_remaining_resources',
            'full_tensor_SHA_before_after_upload_and_download','original_license_and_notices',
            'fixed_HF_revision_remote_full_readback','exact_private_metadata_restoration',
            'separate_current_Core_original_600s80cycles_and_original24_admission'])
    original.reject_private(public)
    private=dict(schema=RESTORE,candidate=arm,public_files=public_files,public_plan=public,
        original_metadata_base64=restoration,original_complete_base64=base64.b64encode(bodies['CHECKPOINT_COMPLETE.json']).decode(),
        original_source_reference_base64=base64.b64encode(bodies['FULL_SOURCE_CHECKPOINT_REFERENCE.json']).decode(),
        audited_endpoint=cp,closed_evidence=dict(
            global_base64=base64.b64encode(global_bytes).decode(),lease_base64=base64.b64encode(lease_bytes).decode(),
            settled_base64=base64.b64encode(settled_bytes).decode(),owner_observation=owner_observation))
    return dict(public_plan=public,public_rewritten_metadata_base64=rewritten,private_restoration=private,
        resource_counters_changed=False,HF_IO_performed=False)


def restore_original(public_files,bundle,*,arm):
    need(arm in e.training.ARMS and bundle.get('schema')==RESTORE and bundle.get('candidate')==arm,
        'TRANSFER_EXACT_PRIVATE_RESTORATION_ARM')
    closed=bundle['closed_evidence'];proof=closed_endpoint(
        base64.b64decode(closed['global_base64'],validate=True),
        base64.b64decode(closed['lease_base64'],validate=True),
        base64.b64decode(closed['settled_base64'],validate=True),closed['owner_observation'],arm=arm)
    need(proof['endpoint']==bundle['audited_endpoint'],'TRANSFER_REVALIDATE_ORIGINAL_CLOSED_ENDPOINT_ON_RECEIPT')
    plan=bundle['public_plan'];need(plan.get('schema')==SCHEMA and plan.get('candidate')==arm
        and plan.get('files')==bundle['public_files']and set(public_files)==set(bundle['public_files'])
        and plan.get('official_inference_manifest_sha256')==bundle['audited_endpoint'].get('official_inference_manifest_sha256')
        and plan.get('wrapped_checkpoint_manifest_sha256')==bundle['audited_endpoint'].get('checkpoint_manifest_sha256'),
        'TRANSFER_EXACT_RECEIVED_PLAN_AND_MEMBER_SET')
    restored={}
    for name,body in public_files.items():
        row=bundle['public_files'][name]
        need(original._download_identity(body)==(row['bytes'],row['sha256']),
            'TRANSFER_DOWNLOADED_PUBLIC_BYTES:'+name);restored[name]=body
    need(set(bundle['original_metadata_base64'])==set(original.REWRITTEN),
        'TRANSFER_ONLY_ORIGINAL_TWO_METADATA_RESTORATIONS')
    restored.update({n:base64.b64decode(b,validate=True)for n,b in bundle['original_metadata_base64'].items()})
    complete_raw=base64.b64decode(bundle['original_complete_base64'],validate=True)
    need(e.sha(complete_raw)==plan['official_inference_manifest_sha256'],'TRANSFER_ORIGINAL_COMPLETE_BYTES')
    reference_raw=base64.b64decode(bundle['original_source_reference_base64'],validate=True)
    need(e.sha(reference_raw)==json.loads(complete_raw)['files']['FULL_SOURCE_CHECKPOINT_REFERENCE.json'],
        'TRANSFER_ORIGINAL_SOURCE_REFERENCE_BYTES')
    index=restored['model.safetensors.index.json']
    if not isinstance(index,bytes):
        need(index.stat().st_size<2*1024**2,'TRANSFER_BOUNDED_DOWNLOADED_INDEX');index=index.read_bytes()
    members=ordinary_members(json.loads(complete_raw),json.loads(index),json.loads(reference_raw),
        arm=arm,endpoint=bundle['audited_endpoint'])
    need(set(restored)==set(members)and all(original._download_identity(restored[n])[1]==h for n,h in members.items()),
        'TRANSFER_RESTORED_EXACT_ORDINARY_MODEL_PROCESSOR_AND_R0')
    return restored
