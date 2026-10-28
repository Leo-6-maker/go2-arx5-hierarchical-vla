"""Exact C/D1024 inference subset via the existing public exporter/restorer.

Pure byte checks only. The existing owner must perform registered source reads,
resource admission, fixed-revision transfer and independent stream readback.
Optimizer/RNG/scheduler/full-state history stays at its original source.
"""
import base64
import json
from pathlib import Path
import re

from . import evaluation_transfer as original, mix_core_context as context
from . import mix_accounting as a, successor_cd as s


def need(ok, reason):
    if not ok:raise PermissionError('V5_CD_INFERENCE_DELIVERY_'+reason)


def endpoint(capture_bytes):
    need(type(capture_bytes)is bytes,'ACTUAL_CLOSED_TRAINING_CAPTURE_BYTES')
    capture=json.loads(capture_bytes);proof=context.captured_endpoint(capture)
    settled=json.loads(base64.b64decode(capture['terminal_capture_base64']['settled_bytes'],validate=True))
    return capture,proof,settled['audit']


def prepare(capture_bytes,small_bytes,*,arm):
    """Bind actual full-state audit witnesses and post-closure metadata bytes."""
    capture,proof,audit=endpoint(capture_bytes)
    need(arm in ('C50','D25')and proof['arm']==arm,'SAME_PREDECLARED_ARM')
    need(type(small_bytes)is bytes,'ACTUAL_POST_CLOSURE_SMALL_READ_BYTES')
    small=json.loads(small_bytes)
    need(small.get('schema')=='V5_CD_CLOSED1024_SMALL_METADATA_READBACK_V1'
         and small.get('arm')==arm and small.get('training_capture_sha256')==a.sha(capture_bytes)
         and small.get('training_handoff_sha256')==a.sha(a.raw(proof))
         and small.get('checkpoint_manifest_sha256')==proof['checkpoint_manifest_sha256']
         and small.get('tensor_payloads_read')is False,'EXACT_CLOSED_ENDPOINT_READBACK')
    witness=audit['streamed_witness']
    need(witness.get('all_paths_rechecked')is True
         and witness.get('all_unique_inodes_independently_rehashed')is True,
         'REAL_FULL_STATE_STREAM_AND_INDEPENDENT_FINAL_WITNESS')
    cp=Path(audit['checkpoint_root']);rows=witness['paths'];members={}
    need(cp.is_absolute()and '..'not in cp.parts,'EXACT_AUDITED_CHECKPOINT_ROOT')
    for row in rows:
        p=Path(row['path'])
        if not p.is_relative_to(cp):continue  # The audit also hashes original probes.
        name=p.relative_to(cp).as_posix()
        need(name not in members and type(row.get('bytes'))is int and row['bytes']>0
             and type(row.get('sha256'))is str and re.fullmatch('[0-9a-f]{64}',row['sha256']),
             'UNIQUE_BOUNDED_ORIGINAL_AUDITED_MEMBER')
        members[name]=row
    readback=small.get('files',{})
    need(set(readback)==set(original.SMALL)|{'CHECKPOINT_COMPLETE.json'},'ALL_SEVEN_METADATA_MEMBERS')
    raw={}
    for name,record in readback.items():
        need(type(record.get('bytes'))is int and 0<record['bytes']<2*1024**2,'BOUNDED_SMALL_FILE')
        b=base64.b64decode(record['base64'],validate=True)
        need(len(b)==record['bytes']and a.sha(b)==record['sha256'],'FULL_SMALL_READBACK_SHA:'+name)
        raw[name]=b
    complete_raw=raw['CHECKPOINT_COMPLETE.json'];complete=json.loads(complete_raw)
    need(a.sha(complete_raw)==proof['checkpoint_manifest_sha256']
         and complete.get('origin_job_id')==proof['job_id']
         and complete.get('origin_source_sha')==proof['source_sha']
         and set(members)==set(complete['files']),'SAME_ALL_FIFTEEN_AUDITED_TRAINING_MEMBERS')
    need(all(members[n]['sha256']==h for n,h in complete['files'].items()),'ALL_FIFTEEN_STREAMED_DIGESTS')
    subset=original.model_members(complete,json.loads(raw['model.safetensors.index.json']),expected_step=1024)
    need(all(a.sha(b)==complete['files'][n]and len(b)==members[n]['bytes']
             for n,b in raw.items()if n!='CHECKPOINT_COMPLETE.json'),'METADATA_MATCHES_INDEPENDENT_AUDIT')
    public_files={};rewritten={};restoration={};tensor_bytes=0
    for name,digest in subset.items():
        if name in original.REWRITTEN:
            value=original.public_config(json.loads(raw[name]),processor=name.startswith('processor/'))
            public_raw=original.encode(value)
            rewritten[name]=base64.b64encode(public_raw).decode()
            restoration[name]=base64.b64encode(raw[name]).decode()
            public_files[name]=dict(bytes=len(public_raw),sha256=a.sha(public_raw),kind='DECLARED_COSMOS_PATH_REBIND')
        else:
            if name in raw:original.reject_private(json.loads(raw[name]))
            public_files[name]=dict(bytes=members[name]['bytes'],sha256=digest,kind='UNCHANGED_ORIGINAL_BYTES')
            if name.endswith('.safetensors'):tensor_bytes+=members[name]['bytes']
    public=dict(schema='V5_CD_CLOSED1024_MODEL_ONLY_TRANSFER_PLAN_V1',task=s.TASK,candidate=arm,
        checkpoint_step=1024,checkpoint_manifest_sha256=proof['checkpoint_manifest_sha256'],
        statistics_sha256=a.STATS,training_handoff_sha256=a.sha(a.raw(proof)),files=public_files,
        tensor_bytes=tensor_bytes,payload_bytes=sum(x['bytes']for x in public_files.values()),
        full_resume_payload=False,original_complete_state_retained_at_source=True,numerical_tensor_changes=0,
        source_tensor_streaming_verified_this_plan=False,uploaded=False,downloaded=False,
        native_admission=False,model_loaded=False,physics_authorized=False,new_training_authorized=False,
        publication_requirements=['fresh_full_resources_and_registered_storage','source_and_destination_full_tensor_SHA',
            'original_upstream_license_and_notices','fixed_revision_remote_full_readback',
            'exact_private_metadata_restoration','separate_current_native_Core_qualification'])
    original.reject_private(public)
    private=dict(schema='V5_CD_EXACT_PRIVATE_INFERENCE_RESTORATION_V1',candidate=arm,
        checkpoint_manifest_sha256=proof['checkpoint_manifest_sha256'],public_files=public_files,
        original_metadata_base64=restoration,original_complete_base64=base64.b64encode(complete_raw).decode(),
        closed_training_capture_base64=base64.b64encode(capture_bytes).decode(),
        closed_metadata_readback_base64=base64.b64encode(small_bytes).decode())
    return dict(public_plan=public,public_rewritten_metadata_base64=rewritten,private_restoration=private,
        resource_counters_changed=False,HF_IO_performed=False)


def restore_original(public_files,private_bundle,*,arm):
    """Recompile closed evidence before exact existing streamed download checks."""
    b=private_bundle
    need(b.get('schema')=='V5_CD_EXACT_PRIVATE_INFERENCE_RESTORATION_V1'
         and b.get('candidate')==arm,'ONLY_SAME_CLOSED_CD_MODEL')
    prepared=prepare(base64.b64decode(b['closed_training_capture_base64'],validate=True),
        base64.b64decode(b['closed_metadata_readback_base64'],validate=True),arm=arm)
    need(prepared['private_restoration']==b,'EXACT_RECOMPILED_PRIVATE_RESTORATION')
    need(set(public_files)==set(b['public_files']),'EXACT_INFERENCE_SUBSET_NO_TRAINING_STATE')
    restored={}
    for name,value in public_files.items():
        metadata=b['public_files'][name]
        need(original._download_identity(value)==(metadata['bytes'],metadata['sha256']),
             'FULL_DOWNLOADED_PUBLIC_BYTES:'+name)
        restored[name]=value
    for name,encoded in b['original_metadata_base64'].items():restored[name]=base64.b64decode(encoded,validate=True)
    complete=json.loads(base64.b64decode(b['original_complete_base64'],validate=True))
    for name,value in restored.items():
        need(original._download_identity(value)[1]==complete['files'][name],'EXACT_RESTORED_ORIGINAL_BYTES:'+name)
    return restored
