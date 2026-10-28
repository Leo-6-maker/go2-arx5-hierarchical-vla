"""Prepare exact saved A/B inference delivery; no HF IO or execution permission.

The public copy sanitizes two Cosmos path fields using the existing exporter.
The private restoration bundle restores their exact original bytes on receipt,
so the original model/processor/relay identity remains unchanged. Optimizer,
scheduler, RNG and training arguments remain in the full Vanda checkpoint.
"""
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from scripts.pi0294.public_checkpoint import public_config, reject_private
from scripts.pi0294.e1.scope import STATS
from .evaluation_platform import ENDPOINTS

CAPTURE_SHA='589564c55dcfd12ef14349c8a7e0d32da93c978dbf615d14ad7f45ef71afa078'
SMALL=('config.json','model.safetensors.index.json','statistics.json',
       'processor/processor_config.json','processor/statistics.json','processor/embodiment_id.json')
REWRITTEN=('config.json','processor/processor_config.json')


def need(ok,why):
    if not ok:raise PermissionError('PV1_EVAL_TRANSFER_'+why)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def model_members(complete,index,*,expected_step=4096):
    """Reject index escapes and unbound/missing shards before creating a plan."""
    files=complete.get('files',{})
    need(type(files) is dict and len(files)==15,'ORIGINAL_COMPLETE_15_MEMBER_INVENTORY')
    need(all(type(n) is str and not PurePosixPath(n).is_absolute()
             and '..' not in PurePosixPath(n).parts and '\\' not in n
             and type(h) is str and re.fullmatch('[0-9a-f]{64}',h) for n,h in files.items()),
         'RELATIVE_MEMBERS_AND_EXACT_HASHES')
    need(type(expected_step)is int and expected_step in (1024,4096)
         and complete.get('global_step')==expected_step and type(complete['global_step']) is int
         and complete.get('full_resume') is True,'EXACT_COMPLETE_SAVED_STEP_NOT_NEW_TRAINING')
    weights=index.get('weight_map')
    need(type(weights) is dict and weights and all(type(k) is str and k for k in weights)
         and all(type(n) is str for n in weights.values()),
         'ACTUAL_SAFETENSORS_INDEX')
    shards={n for n in files if n.endswith('.safetensors')}
    need(shards and set(weights.values())==shards
         and all(re.fullmatch(r'model-[0-9]{5}-of-[0-9]{5}\.safetensors',n) for n in shards),
         'EXACT_ALL_SHARDS_NO_PATH_OR_UNREGISTERED_TENSOR')
    need(set(SMALL)<=set(files) and files['statistics.json']==STATS,
         'ORIGINAL_PROCESSOR_AND_R0')
    return {n:files[n] for n in sorted(shards|set(SMALL))}


def _download_identity(value):
    if type(value) is bytes:return len(value),sha(value)
    need(isinstance(value,Path) and value.is_file() and not value.is_symlink(),
         'REGULAR_DOWNLOADED_FILE_OR_METADATA_BYTES')
    state=value.stat();digest=hashlib.sha256()
    with value.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):digest.update(chunk)
    after=value.stat()
    need((state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns)==
         (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'DOWNLOAD_CHANGED_DURING_READ')
    return state.st_size,digest.hexdigest()


def restore_original(public_files,private_bundle,*,candidate):
    """Pure byte verification; caller still needs registered IO and admission."""
    if candidate in ('E1_ADAPTER_ONLY','E2_SHARED_ACTION_HEAD'):
        from scripts.pi0294.generalization.minimal_inference_transfer import restore_original as restore_minimal
        return restore_minimal(public_files,private_bundle,arm=candidate)
    if candidate in ('C50','D25'):
        from .mix_evaluation_transfer import restore_original as restore_cd
        return restore_cd(public_files,private_bundle,arm=candidate)
    need(candidate in ('A_CONTROL','B_COVERAGE'),'ONLY_APPROVED_SAVED_AB')
    b=private_bundle
    need(b.get('schema')=='PV1_EXACT_PRIVATE_INFERENCE_RESTORATION_V1'
         and b.get('candidate')==candidate and b.get('checkpoint_manifest_sha256')==ENDPOINTS[candidate],
         'EXACT_PRIVATE_RESTORATION_IDENTITY')
    original=base64.b64decode(b['original_complete_base64'],validate=True)
    need(sha(original)==ENDPOINTS[candidate],'ORIGINAL_COMPLETE_BYTES')
    complete=json.loads(original)
    need(set(public_files)==set(b['public_files']),'EXACT_PUBLIC_MEMBER_SET')
    restored={}
    for name,raw in public_files.items():
        expected=b['public_files'][name]
        need(_download_identity(raw)==(expected['bytes'],expected['sha256']),
             'DOWNLOADED_PUBLIC_BYTES:'+name)
        restored[name]=raw
    need(set(b['original_metadata_base64'])==set(REWRITTEN),'ONLY_TWO_DECLARED_PATH_RESTORATIONS')
    for name,raw in b['original_metadata_base64'].items():
        restored[name]=base64.b64decode(raw,validate=True)
    index=restored['model.safetensors.index.json']
    if isinstance(index,Path):
        need(index.stat().st_size<2*1024**2,'BOUNDED_DOWNLOADED_INDEX')
        index=index.read_bytes()
    members=model_members(complete,json.loads(index))
    need(set(restored)==set(members) and all(_download_identity(restored[n])[1]==h for n,h in members.items()),
         'RESTORED_ORIGINAL_INFERENCE_AND_RELAY_IDENTITY')
    return restored


def prepare(endpoint_capture_bytes,small_capture_bytes,*,candidate):
    """Compile the actual audited-byte subset without reading tensor payloads."""
    need(candidate in ('A_CONTROL','B_COVERAGE'),'ONLY_APPROVED_SAVED_AB')
    need(type(endpoint_capture_bytes) is bytes and sha(endpoint_capture_bytes)==CAPTURE_SHA,
         'EXACT_INDEPENDENT_SAVED_ENDPOINT_CAPTURE')
    captured=json.loads(endpoint_capture_bytes)['endpoints'][candidate]
    small=json.loads(small_capture_bytes)
    need(small['returncode']==0 and small['endpoint_metadata_sha256']==CAPTURE_SHA
         and small['value']['tensor_payloads_read'] is False,'ACTUAL_SMALL_FILE_READBACK')
    selected=small['value']['endpoints'][candidate]
    need(selected['checkpoint_sha256']==captured['checkpoint_sha256']==ENDPOINTS[candidate]
         and captured['lease_status']=='CLOSED','EXACT_CLOSED_SAVED_ENDPOINT')
    readback=selected['files'];raw={}
    need(set(readback)==set(SMALL)|{'CHECKPOINT_COMPLETE.json'},'EXACT_SMALL_MEMBER_SET')
    for n,r in readback.items():
        need(type(r['bytes']) is int and 0<r['bytes']<2*1024**2,'BOUNDED_METADATA')
        raw[n]=base64.b64decode(r['base64'],validate=True)
        expected=ENDPOINTS[candidate] if n=='CHECKPOINT_COMPLETE.json' else captured['complete']['files'][n]
        need(len(raw[n])==r['bytes'] and sha(raw[n])==r['sha256']==expected,'ACTUAL_SMALL_SHA:'+n)
    complete=json.loads(raw['CHECKPOINT_COMPLETE.json'])
    need(complete==captured['complete'],'ORIGINAL_COMPLETE_NOT_SYNTHESIZED')
    members=model_members(complete,json.loads(raw['model.safetensors.index.json']))
    public_files={};rewritten={};restoration={};tensor_bytes=0
    for n,digest in members.items():
        metadata=captured['members_metadata'][n]
        need(metadata['recorded_sha256']==digest and type(metadata['bytes']) is int
             and metadata['bytes']>0,'AUDITED_SOURCE_SIZE_AND_DIGEST:'+n)
        if n in REWRITTEN:
            value=public_config(json.loads(raw[n]),processor=n.startswith('processor/'))
            public_raw=encode(value);rewritten[n]=base64.b64encode(public_raw).decode()
            restoration[n]=base64.b64encode(raw[n]).decode()
            public_files[n]=dict(bytes=len(public_raw),sha256=sha(public_raw),kind='DECLARED_COSMOS_PATH_REBIND')
        else:
            if n in raw:reject_private(json.loads(raw[n]))
            public_files[n]=dict(bytes=metadata['bytes'],sha256=digest,kind='UNCHANGED_ORIGINAL_BYTES')
            if n.endswith('.safetensors'):tensor_bytes+=metadata['bytes']
    public=dict(schema='PV1_SAVED4096_MODEL_ONLY_HF_TRANSFER_PLAN_V1',candidate=candidate,
        checkpoint_manifest_sha256=ENDPOINTS[candidate],statistics_sha256=STATS,
        original_complete_state_retained_at_source=True,full_resume_payload=False,
        files=public_files,tensor_bytes=tensor_bytes,payload_bytes=sum(x['bytes'] for x in public_files.values()),
        numerical_tensor_changes=0,source_tensor_streaming_verified_this_plan=False,
        uploaded=False,downloaded=False,native_admission=False,model_loaded=False,
        physics_authorized=False,new_training_authorized=False,
        publication_requirements=['fresh_complete_resources_and_storage_reservation',
            'stream_all_tensor_SHA_before_and_after_upload','original_upstream_license_and_notices',
            'fixed_HF_revision_full_remote_readback','private_exact_metadata_restoration',
            'separate_current_Core_GPU_and_feedback_qualification'])
    reject_private(public)
    private=dict(schema='PV1_EXACT_PRIVATE_INFERENCE_RESTORATION_V1',candidate=candidate,
        checkpoint_manifest_sha256=ENDPOINTS[candidate],public_files=public_files,
        original_metadata_base64=restoration,original_complete_base64=base64.b64encode(raw['CHECKPOINT_COMPLETE.json']).decode())
    return dict(public_plan=public,public_rewritten_metadata_base64=rewritten,
        private_restoration=private,resource_counters_changed=False,HF_IO_performed=False)
